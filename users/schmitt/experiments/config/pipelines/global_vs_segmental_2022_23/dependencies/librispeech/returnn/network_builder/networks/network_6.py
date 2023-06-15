network = {
    "source": {
        "class": "eval",
        "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
        "from": "log_mel_features",
    },
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},
    "conv0": {
        "class": "conv",
        "from": "source0",
        "padding": "same",
        "filter_size": (3, 3),
        "n_out": 32,
        "activation": "relu",
        "with_bias": True,
    },
    "conv0p": {
        "class": "pool",
        "from": "conv0",
        "pool_size": (1, 2),
        "mode": "max",
        "trainable": False,
        "padding": "same",
    },
    "conv_out": {"class": "copy", "from": "conv0p"},
    "subsample_conv0": {
        "class": "conv",
        "from": "conv_out",
        "padding": "same",
        "filter_size": (3, 3),
        "n_out": 64,
        "activation": "relu",
        "with_bias": True,
        "strides": (3, 1),
    },
    "subsample_conv1": {
        "class": "conv",
        "from": "subsample_conv0",
        "padding": "same",
        "filter_size": (3, 3),
        "n_out": 64,
        "activation": "relu",
        "with_bias": True,
        "strides": (2, 1),
    },
    "conv_merged": {"class": "merge_dims", "from": "subsample_conv1", "axes": "static"},
    "source_linear": {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": "conv_merged",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_ffmod_1_ln": {"class": "layer_norm", "from": "source_linear"},
    "conformer_block_01_ffmod_1_ff1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_ffmod_1_ln",
        "n_out": 1024,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_ffmod_1_relu": {
        "class": "activation",
        "activation": "relu",
        "from": "conformer_block_01_ffmod_1_ff1",
    },
    "conformer_block_01_ffmod_1_square_relu": {
        "class": "eval",
        "eval": "source(0) ** 2",
        "from": "conformer_block_01_ffmod_1_relu",
    },
    "conformer_block_01_ffmod_1_drop1": {
        "class": "dropout",
        "from": "conformer_block_01_ffmod_1_square_relu",
        "dropout": 0.0,
    },
    "conformer_block_01_ffmod_1_ff2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_ffmod_1_drop1",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_ffmod_1_drop2": {
        "class": "dropout",
        "from": "conformer_block_01_ffmod_1_ff2",
        "dropout": 0.0,
    },
    "conformer_block_01_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_ffmod_1_drop2",
    },
    "conformer_block_01_ffmod_1_res": {
        "class": "combine",
        "kind": "add",
        "from": ["conformer_block_01_ffmod_1_half_step", "source_linear"],
        "n_out": 256,
    },
    "conformer_block_01_self_att_ln": {
        "class": "layer_norm",
        "from": "conformer_block_01_ffmod_1_res",
    },
    "conformer_block_01_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "from": "conformer_block_01_self_att_ln",
        "n_out": 32,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
        "clipping": 16,
    },
    "conformer_block_01_self_att": {
        "class": "self_attention",
        "from": "conformer_block_01_self_att_ln",
        "n_out": 256,
        "num_heads": 8,
        "total_key_dim": 256,
        "key_shift": "conformer_block_01_self_att_ln_rel_pos_enc",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)",
    },
    "conformer_block_01_self_att_linear": {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": "conformer_block_01_self_att",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_self_att_dropout": {
        "class": "dropout",
        "from": "conformer_block_01_self_att_linear",
        "dropout": 0.0,
    },
    "conformer_block_01_self_att_res": {
        "class": "combine",
        "kind": "add",
        "from": [
            "conformer_block_01_self_att_dropout",
            "conformer_block_01_ffmod_1_res",
        ],
        "n_out": 256,
    },
    "conformer_block_01_conv_mod_ln": {
        "class": "layer_norm",
        "from": "conformer_block_01_self_att_res",
    },
    "conformer_block_01_conv_mod_pointwise_conv1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_conv_mod_ln",
        "n_out": 512,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
    },
    "conformer_block_01_conv_mod_glu": {
        "class": "gating",
        "from": "conformer_block_01_conv_mod_pointwise_conv1",
        "activation": "identity",
    },
    "conformer_block_01_conv_mod_depthwise_conv2": {
        "class": "conv",
        "from": "conformer_block_01_conv_mod_glu",
        "padding": "same",
        "filter_size": (16,),
        "n_out": 256,
        "activation": None,
        "with_bias": True,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
        "groups": 256,
    },
    "conformer_block_01_conv_mod_bn": {
        "class": "batch_norm",
        "from": "conformer_block_01_conv_mod_depthwise_conv2",
        "momentum": 0.1,
        "epsilon": 0.001,
        "update_sample_only_in_training": True,
        "delay_sample_update": True,
    },
    "conformer_block_01_conv_mod_swish": {
        "class": "activation",
        "activation": "swish",
        "from": "conformer_block_01_conv_mod_bn",
    },
    "conformer_block_01_conv_mod_pointwise_conv2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_conv_mod_swish",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
    },
    "conformer_block_01_conv_mod_drop": {
        "class": "dropout",
        "from": "conformer_block_01_conv_mod_pointwise_conv2",
        "dropout": 0.0,
    },
    "conformer_block_01_conv_mod_res": {
        "class": "combine",
        "kind": "add",
        "from": ["conformer_block_01_conv_mod_drop", "conformer_block_01_self_att_res"],
        "n_out": 256,
    },
    "conformer_block_01_ffmod_2_ln": {
        "class": "layer_norm",
        "from": "conformer_block_01_conv_mod_res",
    },
    "conformer_block_01_ffmod_2_ff1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_ffmod_2_ln",
        "n_out": 1024,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_ffmod_2_relu": {
        "class": "activation",
        "activation": "relu",
        "from": "conformer_block_01_ffmod_2_ff1",
    },
    "conformer_block_01_ffmod_2_square_relu": {
        "class": "eval",
        "eval": "source(0) ** 2",
        "from": "conformer_block_01_ffmod_2_relu",
    },
    "conformer_block_01_ffmod_2_drop1": {
        "class": "dropout",
        "from": "conformer_block_01_ffmod_2_square_relu",
        "dropout": 0.0,
    },
    "conformer_block_01_ffmod_2_ff2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_01_ffmod_2_drop1",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_01_ffmod_2_drop2": {
        "class": "dropout",
        "from": "conformer_block_01_ffmod_2_ff2",
        "dropout": 0.0,
    },
    "conformer_block_01_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_ffmod_2_drop2",
    },
    "conformer_block_01_ffmod_2_res": {
        "class": "combine",
        "kind": "add",
        "from": [
            "conformer_block_01_ffmod_2_half_step",
            "conformer_block_01_conv_mod_res",
        ],
        "n_out": 256,
    },
    "conformer_block_01_ln": {
        "class": "layer_norm",
        "from": "conformer_block_01_ffmod_2_res",
    },
    "conformer_block_01": {"class": "copy", "from": "conformer_block_01_ln"},
    "conformer_block_02_ffmod_1_ln": {
        "class": "layer_norm",
        "from": "conformer_block_01",
    },
    "conformer_block_02_ffmod_1_ff1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_ffmod_1_ln",
        "n_out": 1024,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_02_ffmod_1_relu": {
        "class": "activation",
        "activation": "relu",
        "from": "conformer_block_02_ffmod_1_ff1",
    },
    "conformer_block_02_ffmod_1_square_relu": {
        "class": "eval",
        "eval": "source(0) ** 2",
        "from": "conformer_block_02_ffmod_1_relu",
    },
    "conformer_block_02_ffmod_1_drop1": {
        "class": "dropout",
        "from": "conformer_block_02_ffmod_1_square_relu",
        "dropout": 0.0,
    },
    "conformer_block_02_ffmod_1_ff2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_ffmod_1_drop1",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_02_ffmod_1_drop2": {
        "class": "dropout",
        "from": "conformer_block_02_ffmod_1_ff2",
        "dropout": 0.0,
    },
    "conformer_block_02_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_ffmod_1_drop2",
    },
    "conformer_block_02_ffmod_1_res": {
        "class": "combine",
        "kind": "add",
        "from": ["conformer_block_02_ffmod_1_half_step", "conformer_block_01"],
        "n_out": 256,
    },
    "conformer_block_02_self_att_ln": {
        "class": "layer_norm",
        "from": "conformer_block_02_ffmod_1_res",
    },
    "conformer_block_02_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "from": "conformer_block_02_self_att_ln",
        "n_out": 32,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
        "clipping": 16,
    },
    "conformer_block_02_self_att": {
        "class": "self_attention",
        "from": "conformer_block_02_self_att_ln",
        "n_out": 256,
        "num_heads": 8,
        "total_key_dim": 256,
        "key_shift": "conformer_block_02_self_att_ln_rel_pos_enc",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)",
    },
    "conformer_block_02_self_att_linear": {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": "conformer_block_02_self_att",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_02_self_att_dropout": {
        "class": "dropout",
        "from": "conformer_block_02_self_att_linear",
        "dropout": 0.0,
    },
    "conformer_block_02_self_att_res": {
        "class": "combine",
        "kind": "add",
        "from": [
            "conformer_block_02_self_att_dropout",
            "conformer_block_02_ffmod_1_res",
        ],
        "n_out": 256,
    },
    "conformer_block_02_conv_mod_ln": {
        "class": "layer_norm",
        "from": "conformer_block_02_self_att_res",
    },
    "conformer_block_02_conv_mod_pointwise_conv1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_conv_mod_ln",
        "n_out": 512,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
    },
    "conformer_block_02_conv_mod_glu": {
        "class": "gating",
        "from": "conformer_block_02_conv_mod_pointwise_conv1",
        "activation": "identity",
    },
    "conformer_block_02_conv_mod_depthwise_conv2": {
        "class": "conv",
        "from": "conformer_block_02_conv_mod_glu",
        "padding": "same",
        "filter_size": (16,),
        "n_out": 256,
        "activation": None,
        "with_bias": True,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
        "groups": 256,
    },
    "conformer_block_02_conv_mod_bn": {
        "class": "batch_norm",
        "from": "conformer_block_02_conv_mod_depthwise_conv2",
        "momentum": 0.1,
        "epsilon": 0.001,
        "update_sample_only_in_training": True,
        "delay_sample_update": True,
    },
    "conformer_block_02_conv_mod_swish": {
        "class": "activation",
        "activation": "swish",
        "from": "conformer_block_02_conv_mod_bn",
    },
    "conformer_block_02_conv_mod_pointwise_conv2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_conv_mod_swish",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
        "scale=1.0)",
    },
    "conformer_block_02_conv_mod_drop": {
        "class": "dropout",
        "from": "conformer_block_02_conv_mod_pointwise_conv2",
        "dropout": 0.0,
    },
    "conformer_block_02_conv_mod_res": {
        "class": "combine",
        "kind": "add",
        "from": ["conformer_block_02_conv_mod_drop", "conformer_block_02_self_att_res"],
        "n_out": 256,
    },
    "conformer_block_02_ffmod_2_ln": {
        "class": "layer_norm",
        "from": "conformer_block_02_conv_mod_res",
    },
    "conformer_block_02_ffmod_2_ff1": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_ffmod_2_ln",
        "n_out": 1024,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_02_ffmod_2_relu": {
        "class": "activation",
        "activation": "relu",
        "from": "conformer_block_02_ffmod_2_ff1",
    },
    "conformer_block_02_ffmod_2_square_relu": {
        "class": "eval",
        "eval": "source(0) ** 2",
        "from": "conformer_block_02_ffmod_2_relu",
    },
    "conformer_block_02_ffmod_2_drop1": {
        "class": "dropout",
        "from": "conformer_block_02_ffmod_2_square_relu",
        "dropout": 0.0,
    },
    "conformer_block_02_ffmod_2_ff2": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "conformer_block_02_ffmod_2_drop1",
        "n_out": 256,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
    "conformer_block_02_ffmod_2_drop2": {
        "class": "dropout",
        "from": "conformer_block_02_ffmod_2_ff2",
        "dropout": 0.0,
    },
    "conformer_block_02_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_ffmod_2_drop2",
    },
    "conformer_block_02_ffmod_2_res": {
        "class": "combine",
        "kind": "add",
        "from": [
            "conformer_block_02_ffmod_2_half_step",
            "conformer_block_02_conv_mod_res",
        ],
        "n_out": 256,
    },
    "conformer_block_02_ln": {
        "class": "layer_norm",
        "from": "conformer_block_02_ffmod_2_res",
    },
    "conformer_block_02": {"class": "copy", "from": "conformer_block_02_ln"},
    "encoder": {"class": "copy", "from": "conformer_block_02"},
    "ctc": {
        "class": "softmax",
        "from": "encoder",
        "target": "targets",
        "loss": "ctc",
        "loss_opts": {"beam_width": 1, "use_native": True},
        "loss_scale": 1.0,
    },
    "enc_ctx": {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": "encoder",
        "n_out": 1024,
    },
    "enc_value": {
        "class": "split_dims",
        "axis": "F",
        "dims": (1, 256),
        "from": "encoder",
    },
    "inv_fertility": {
        "class": "linear",
        "activation": "sigmoid",
        "with_bias": False,
        "from": "encoder",
        "n_out": 1,
    },
    "decision": {
        "class": "decide",
        "from": "output",
        "loss": "edit_distance",
        "target": "targets",
    },
    "output": {
        "class": "rec",
        "from": [],
        "unit": {
            "end": {"class": "compare", "kind": "equal", "from": "output", "value": 0},
            "target_embed0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "output",
                "n_out": 640,
                "initial_output": 0,
            },
            "target_embed": {
                "class": "dropout",
                "from": "target_embed0",
                "dropout": 0.0,
                "dropout_noise_shape": {"*": None},
            },
            "s_transformed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "s",
                "n_out": 1024,
            },
            "accum_att_weights": {
                "class": "eval",
                "eval": "source(0) + source(1) * source(2) * 0.5",
                "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                "out_type": {"dim": 1, "shape": (None, 1)},
            },
            "weight_feedback": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "prev:accum_att_weights",
                "n_out": 1024,
            },
            "energy_in": {
                "class": "combine",
                "kind": "add",
                "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                "n_out": 1024,
            },
            "energy_tanh": {
                "class": "activation",
                "activation": "tanh",
                "from": "energy_in",
            },
            "energy": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "energy_tanh",
                "n_out": 1,
            },
            "att_weights": {"class": "softmax_over_spatial", "from": "energy"},
            "att0": {
                "class": "generic_attention",
                "weights": "att_weights",
                "base": "base:enc_value",
            },
            "att": {"class": "merge_dims", "from": "att0", "axes": "except_batch"},
            "s": {
                "class": "rnn_cell",
                "unit": "zoneoutlstm",
                "n_out": 1024,
                "from": ["prev:target_embed", "prev:att"],
                "unit_opts": {
                    "zoneout_factor_cell": 0.15,
                    "zoneout_factor_output": 0.05,
                },
            },
            "readout_in": {
                "class": "linear",
                "activation": None,
                "with_bias": True,
                "from": ["s", "prev:target_embed", "att"],
                "n_out": 1024,
            },
            "readout": {
                "class": "reduce_out",
                "from": "readout_in",
                "num_pieces": 2,
                "mode": "max",
            },
            "output_prob": {
                "class": "softmax",
                "from": "readout",
                "target": "targets",
                "loss": "ce",
                "loss_opts": {"label_smoothing": 0},
            },
            "output": {
                "class": "choice",
                "target": "targets",
                "beam_size": 12,
                "from": "output_prob",
                "initial_output": 0,
            },
        },
        "target": "targets",
        "max_seq_len": "max_len_from('base:encoder')",
    },
    "#config": {"batch_size": 3600000},
    "#copy_param_mode": "subset",
    "stft": {
        "class": "stft",
        "frame_shift": 160,
        "frame_size": 400,
        "fft_size": 512,
        "from": "data:data",
    },
    "abs": {"class": "activation", "from": "stft", "activation": "abs"},
    "power": {"class": "eval", "from": "abs", "eval": "source(0) ** 2"},
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 512,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {"from": "log", "class": "eval", "eval": "source(0) / 2.3026"},
    "log_mel_features": {"class": "copy", "from": "log10"},
}
