import copy
import math
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from sisyphus import tk

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel, create_fresh_model_file


def eow_phon_ls960_ce_base(
    model_conf_w2v: Optional[dict] = None,
    train_conf_w2v: Optional[dict] = None,
    train_name_suffix: Optional[str] = None,
    fairseq_root: Optional[tk.Path] = None,
    ):
    if train_name_suffix is None:
        prefix_name = "eow_phon_ce"
    else:
        prefix_name = "eow_phon_ce" + "/" + train_name_suffix
    
    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    # num_epochs = n_updates / (corpus_size / batch_size) = 80k / (100h / 1920s) ~= 427
    num_epochs = 427
    init_lr_scale = 0.01
    final_lr_scale = 0.05
    lr = 3e-5
    if train_conf_w2v is None:
        # default train config
        train_conf_w2v = {
            "optimizer": {"class": "adam", "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, },
            "learning_rates": list(np.linspace(lr * init_lr_scale, lr, int(math.ceil(num_epochs * 0.1))))
                + list(np.linspace(lr, lr, int(math.ceil(num_epochs * 0.4))))
                + list(np.geomspace(lr, final_lr_scale * lr, int(math.ceil(num_epochs * 0.5)))),                 
            # tri-stage lr schedule, see:
            # https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py
            "batch_size": 1920 * 16000 / 8, # batch size: 1920s, 16000 samples per second, accum_grad 8
            "accum_grad_multiple_step": 8,
            "gradient_clip": 1,
            "random_seed": 187,
        }

    if model_conf_w2v is None:
        # default model config
        # see: https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/config/finetuning/base_100h.yaml
        model_conf_w2v = {
            "_name": "wav2vec_ctc",
            "w2v_path": "/u/andreas.pletschko/fairseq/models/wav2vec_small.pt",
            "apply_mask": True,
            "mask_prob": 0.65,
            "mask_channel_prob": 0.5,
            "mask_channel_length": 64,
            "layerdrop": 0.1,
            "activation_dropout": 0.1,
            "feature_grad_mult": 0.0,
            "freeze_finetune_updates": 10000 # was 0 in fairseq config 
        }

    fresh_model = create_fresh_model_file()
    model_conf_w2v["w2v_path"] = fresh_model
    task_conf_w2v = {
        "_name": "audio_finetuning",
        "normalize": False,
        "data": "-", # can be ignored
    }

    net_args_w2v = {
        "model_config_updates": model_conf_w2v,
        "task_config_updates": task_conf_w2v,
        "label_target_size": vocab_size_without_blank,
    }

    network_module_w2v = "wav2vec.w2v_hybrid_wrapper"
    train_args_w2v = {
        # default train config
        "config": train_conf_w2v,
        "network_module": network_module_w2v,
        "net_args": {"w2v_config_updates": net_args_w2v},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module_w2v + ".ls100_24gbgpu"
    train_job = training(training_name, train_data, train_args_w2v, num_epochs=num_epochs,fairseq_root=fairseq_root, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_w2v, with_prior=True, datasets=train_data, get_specific_checkpoint=num_epochs
    )
