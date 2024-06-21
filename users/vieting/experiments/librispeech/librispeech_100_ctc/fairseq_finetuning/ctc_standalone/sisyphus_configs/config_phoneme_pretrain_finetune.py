from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme \
        import run_fairseq_pretraining_informed, get_fairseq_root

# Pretraining
pretrain_job = run_fairseq_pretraining_informed()

# Get fairseq root
fairseq_root = get_fairseq_root()

# Finetuning
checkpoints = [100, 200, 300, 400, 500, 600]
for checkpoint in checkpoints:
    model_conf_w2v = {
        "_name": "wav2vec_ctc",
        "w2v_path": pretrain_job.out_models[checkpoint].model,
        "apply_mask": True,
        "mask_prob": 0.65,
        "mask_channel_prob": 0.5,
        "mask_channel_length": 64,
        "layerdrop": 0.1,
        "activation_dropout": 0.1,
        "feature_grad_mult": 0.0,
        "freeze_finetune_updates": 10000,  # was 0 in fairseq config
    }
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=f"w2v_neg_sampling_other_target_cp{checkpoint}",
        fairseq_root=fairseq_root,
    )