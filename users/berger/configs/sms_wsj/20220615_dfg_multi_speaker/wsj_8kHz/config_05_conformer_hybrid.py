import copy
from functools import partial
import os
from typing import Any, Dict, Optional
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
import i6_experiments.users.berger.args.jobs.hybrid_args as hybrid_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.conformer_hybrid import (
    make_conformer_hybrid_model,
    make_conformer_hybrid_recog_model,
)
from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import (
    get_returnn_rasr_data_inputs,
)
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)

from .config_00_gmm import py as run_gmm


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

f_name = "gt"

frequency = 8

num_inputs = 40
num_classes = 9001


def run_exp(
    alignments: Dict[str, Any],
    cart_file: tk.Path,
    **kwargs,
) -> SummaryReport:
    am_args = {
        "state_tying": "cart",
        "state_tying_file": cart_file,
    }

    # ********** Data inputs **********
    train_data_inputs, dev_data_inputs, test_data_inputs, _ = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        add_all_allophones=False,
    )

    nn_data_inputs = get_returnn_rasr_data_inputs(
        train_data_inputs=train_data_inputs,
        cv_data_inputs=dev_data_inputs,
        dev_data_inputs=dev_data_inputs,
        test_data_inputs=test_data_inputs,
        alignments=alignments,
        am_args=am_args,
    )

    # ********** Hybrid System **********

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["Conformer_Hybrid", kwargs.get("name_suffix", "")]))

    train_conformer_net, train_python_code = make_conformer_hybrid_model(
        specaug_args={
            "max_time_num": kwargs.get("time_num", 1),
            "max_time": kwargs.get("max_time", 15),
            "max_feature_num": 4,
            "max_feature": 5,
        },
        conformer_args={
            "num_blocks": kwargs.get("num_blocks", 12),
        },
        aux_loss_blocks=kwargs.get("aux_loss_blocks", [(4, 0.3), (8, 0.3)]),
        num_outputs=num_classes,
    )
    train_networks[name] = train_conformer_net

    recog_conformer_net, recog_python_code = make_conformer_hybrid_recog_model(
        conformer_args={
            "num_blocks": kwargs.get("num_blocks", 12),
        },
        num_outputs=num_classes,
    )
    recog_networks[name] = recog_conformer_net

    num_subepochs = kwargs.get("num_subepochs", 180)

    extra_config = None

    nn_args = hybrid_args.get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        search_type=SearchTypes.AdvancedTreeSearch,
        returnn_train_config_args={
            "python_prolog": ["import sys", "sys.setrecursionlimit(3000)"],
            "extra_python": train_python_code,
            "batch_size": 15000,
            "grad_noise": 0.0,
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 8e-4),
            "learning_rate": kwargs.get("learning_rate", 8e-04),
            "n_steps_per_epoch": 1100,
            "base_chunk_size": kwargs.get("chunk_size", 500),
            "extra_config": extra_config,
        },
        returnn_recog_config_args={
            "python_prolog": ["import sys", "sys.setrecursionlimit(3000)"],
            "extra_python": recog_python_code,
        },
        train_args={"partition_epochs": 3},
        prior_args={
            "use_python_control": False,
            "num_classes": num_classes,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "lm_scales": kwargs.get("lm_scales", [16.0]),
            "prior_scales": kwargs.get("prior_scales", [0.7]),
            "use_gpu": True,
            "label_file_blank": False,
            "add_eow": False,
            "add_sow": False,
            "allow_blank": False,
            "label_unit": "hmm",
            "label_scorer_type": "precomputed-log-posterior",
            "rtf": 20,
            "mem": 8,
        },
    )

    sms_wsj_hybrid_system = TransducerSystem(rasr_binary_path=rasr_binary_path)
    init_args = get_init_args(sample_rate_kHz=frequency)
    sms_wsj_hybrid_system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("extract", {"feature_key": f_name, **init_args.feature_extraction_args})
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)
    sms_wsj_hybrid_system.run(nn_steps)

    summary_report = sms_wsj_hybrid_system.get_summary_report()

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summaries/{name}.report", summary_report)

    return summary_report


def py(alignments: Optional[Dict[str, Any]] = None, cart_file: Optional[tk.Path] = None) -> SummaryReport:
    if alignments is None or cart_file is None:
        gmm_outputs = run_gmm()
        alignments = alignments or {key: output.alignments for key, output in gmm_outputs.items()}
        cart_file = (cart_file or gmm_outputs[train_key].crp.acoustic_model_config.state_tying.file,)

    run_exp_partial = partial(run_exp, alignments, cart_file)

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp_partial(), update_structure=True)

    summary_report.merge_report(run_exp_partial(name_suffix="newbob", schedule=LearningRateSchedules.Newbob))

    for time_num in [1, 3]:
        for max_time in [15]:
            summary_report.merge_report(
                run_exp_partial(
                    name_suffix=f"tn-{time_num}_mt-{max_time}",
                    time_num=time_num,
                    max_time=max_time,
                )
            )

    tk.register_report(
        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report",
        summary_report,
    )
    return summary_report
