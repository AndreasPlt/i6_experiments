from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining
        
# Pretraining
neg_other_trg_pretrain_job = run_fairseq_pretraining(
        exp_name="monophone_negatives_other_target_v1",
        commit="1397363c5c0e3c4e3ab620be562730399c852493",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        negative_sampling_strategy="other_target",
    )
phon_boundary_pretrain_job = run_fairseq_pretraining(
        exp_name="monophone_boundary_masking_v1",
        commit="b768be5b81987364d39a07d1caad2bfe1e956896",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        mask_strategy="phoneme",
        mask_length=1,
    )
neg_other_trg_phon_boundary_pretrain_job = run_fairseq_pretraining(
        exp_name="monophone_negatives_other_target_boundary_masking_v1",
        commit="b768be5b81987364d39a07d1caad2bfe1e956896",
        negative_sampling_strategy="other_target",
        mask_strategy="phoneme",
        mask_length=1,
    )
neg_hard_pretrain_job = run_fairseq_pretraining(
        exp_name="monophone_negatives_hard_v1",
        commit="56acedca3b72c09ec30b7208da0d15ada03d0479",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        negative_sampling_strategy="hard_negatives",
    )


# fairseq root
fairseq_root = get_fairseq_root(fairseq_exe=tk.Path("/usr/bin/python3"))

# Finetuning
base_model_conf = {
    "_name": "wav2vec_ctc",
    "apply_mask": True,
    "mask_prob": 0.65,
    "mask_channel_prob": 0.5,
    "mask_channel_length": 64,
    "layerdrop": 0.1,
    "activation_dropout": 0.1,
    "feature_grad_mult": 0.0,
    "freeze_finetune_updates": 10000,  # was 0 in fairseq config
}
checkpoints = [100, 200, 300, 400, 500, 600]
for checkpoint in checkpoints:
    # negative sampling
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_trg_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_neg_sampling_other_target", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )

    # phoneme boundary masking
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_phoneme_boundary_masking", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )

    # negative sampling + phoneme boundary masking
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_trg_phon_boundary_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join(
            "w2v_neg_sampling_other_target_phoneme_boundary_masking",
            f"checkpoint_{checkpoint}"
            ),
        fairseq_root=fairseq_root,
    )

    # hard negatives
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_hard_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_negatives_hard", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )
