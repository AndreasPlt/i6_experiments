from __future__ import annotations
from typing import Dict

import sys
import numpy

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from .conformer_import_moh_att_2023_04_24_BxqgICRSGkgb import Model, MakeModel, from_scratch_training


_ParamMapping = {}  # type: Dict[str,str]


def _add_params():
    # frontend
    for layer_idx in [0, 1, 2]:
        orig_name = "conv0" if layer_idx == 0 else f"subsample_conv{layer_idx - 1}"
        _ParamMapping.update(
            {
                f"encoder.input_layer.conv_layers.{layer_idx}.filter": f"{orig_name}/W",
                f"encoder.input_layer.conv_layers.{layer_idx}.bias": f"{orig_name}/bias",
            }
        )
    _ParamMapping.update(
        {
            "encoder.input_projection.weight": "source_linear/W",
            "enc_ctx.weight": "enc_ctx/W",
            "enc_ctx.bias": "enc_ctx/b",
            "inv_fertility.weight": "inv_fertility/W",
            "target_embed.weight": "output/rec/target_embed0/W",
            "weight_feedback.weight": "output/rec/weight_feedback/W",
            "s_transformed.weight": "output/rec/s_transformed/W",
            "energy.weight": "output/rec/energy/W",
            "readout_in.weight": "output/rec/readout_in/W",
            "readout_in.bias": "output/rec/readout_in/b",
            "output_prob.weight": "output/rec/output_prob/W",
            "output_prob.bias": "output/rec/output_prob/b",
        }
    )
    # conformer
    for layer_idx in range(12):
        # FF
        for sub in [1, 2]:
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/W"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/b"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/W"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/b"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/scale"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/bias"
        # conv
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/b"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/bias"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/b"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/bias"
        # self-att
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.qkv.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att/QKV"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.proj.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_linear/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/bias"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln_rel_pos_enc/encoding_matrix"
        # final layer norm
        _ParamMapping[
            f"encoder.layers.{layer_idx}.final_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.final_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/bias"


_add_params()


def map_param_func_v2(reader, name: str, var: rf.Parameter) -> numpy.ndarray:
    """map params, TF to RF"""
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    from i6_experiments.users.zeyer.returnn.convert.params import numpy as convert_params_np
    from i6_experiments.users.zeyer.returnn.convert.params import tf_to_rf_np as convert_params_tf_to_rf_np

    assert isinstance(reader, CheckpointReader)
    assert isinstance(var, rf.Parameter)

    tf_var_name = name.replace(".", "/")
    if reader.has_tensor(tf_var_name):
        return reader.get_tensor(tf_var_name)

    if name in _ParamMapping:
        var_name = _ParamMapping[name]
        assert reader.has_tensor(var_name)
        value = reader.get_tensor(var_name)
        assert isinstance(value, numpy.ndarray)
        if name.endswith(".filter"):
            value = convert_params_np.convert_tf_conv_to_pt_conv_filter(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.ff_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_ff(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.rec_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_rec(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.bias":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_bias(value, forget_gate_bias=1.0)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if ".conv_block.norm." in name:
        assert name.startswith("encoder.layers.")
        layer_idx = int(name.split(".")[2])
        value = convert_params_tf_to_rf_np.convert_tf_batch_norm_to_rf(
            reader=reader,
            rf_name=name,
            rf_prefix_name=f"encoder.layers.{layer_idx}.conv_block.norm.",
            tf_prefix_name=f"conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/",
            var=var,
        )
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    raise NotImplementedError(f"cannot map {name!r} {var}")


# See comment below, use `py = test_import` to easily run this.
def test_import():
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
    from pprint import pprint

    # Pick some layers to check outputs for equality.
    # See the func tracing logic below, entries in captured_tensors.
    # RETURNN layer name -> trace point in RF/PT model forwarding.
    _layer_mapping = {
        "source": (Model.encode, 0, "source", -1),
        "conv_merged": (ConformerEncoder.__call__, 0, "x_subsample", 0),
        "source_linear": (ConformerEncoder.__call__, 0, "x_linear", 0),
        "conformer_block_01_ffmod_1_drop2": (ConformerEncoderLayer.__call__, 0, "x_ffn1", 0),
        "conformer_block_01_ffmod_1_res": (ConformerEncoderLayer.__call__, 0, "x_ffn1_out", 0),
        "conformer_block_01_self_att_res": (ConformerEncoderLayer.__call__, 0, "x_mhsa_out", 0),
        "conformer_block_01_conv_mod_res": (ConformerEncoderLayer.__call__, 0, "x_conv_out", 0),
        "conformer_block_01_ffmod_2_res": (ConformerEncoderLayer.__call__, 0, "x_ffn2_out", 0),
        "conformer_block_01": (ConformerEncoderLayer.__call__, 1, "inp", 0),
        "encoder": (Model.encode, 0, "enc", 0),
        "inv_fertility": (Model.encode, 0, "inv_fertility", 0),
        "enc_ctx": (Model.encode, 0, "enc_ctx", 0),
        "output/prev:target_embed": (from_scratch_training, 0, "input_embeddings", -1),
        # Note: Some of these commented-out checks are not available anymore because we cleaned up the code.
        # If we want to test this again, we need to re-add the corresponding locals and outputs from rf.scan.
        # "output/weight_feedback": (from_scratch_training, 0, "weight_feedback", 0),
        "output/s": (Model.decode_logits, 0, "s", 0),
        # "output/s_transformed": (from_scratch_training, 0, "s_transformed", 0),
        # "output/energy": (from_scratch_training, 0, "energy", 0),
        # "output/att_weights": (from_scratch_training, 0, "att_weights", 0),
        "output/att": (Model.decode_logits, 0, "att", 0),
        "output/output_prob": (from_scratch_training, 0, "logits", 0),
    }

    from i6_experiments.common.setups.returnn_common import serialization

    exec(serialization.PythonEnlargeStackWorkaroundNonhashedCode.code)

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
    time_dim = Dim(
        name="time",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("time_size", dims=[batch_dim], dtype="int32"),
    )
    target_dim = Dim(name="target", dimension=1030, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    data = Tensor("data", dim_tags=[batch_dim, time_dim])
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor("target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim)

    from ._moh_att_2023_04_24_BxqgICRSGkgb_net_dict import net_dict

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            network=net_dict,
            extern_data={
                "audio_features": {"dim_tags": data.dims},
                "bpe_labels": {"dim_tags": target.dims, "sparse_dim": target.sparse_dim},
            },
        )
    )

    from returnn.tensor.utils import tensor_dict_fill_random_numpy_
    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_
    from returnn.tf.network import TFNetwork
    import tensorflow as tf
    import numpy.testing
    import tempfile
    import atexit
    import shutil

    ckpt_dir = tempfile.mkdtemp("returnn-import-test")
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    print("*** Construct TF graph for old model")
    extern_data = TensorDict()
    extern_data.update(config.typed_dict["extern_data"], auto_convert=True)
    tensor_dict_fill_random_numpy_(
        extern_data, dyn_dim_max_sizes={time_dim: 2000}, dyn_dim_min_sizes={time_dim: 1000}
    )  # raw sample level
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    tf1 = tf.compat.v1
    with tf1.Graph().as_default() as graph, tf1.Session(graph=graph).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        print("*** Random init old model")
        net.initialize_params(session)
        print("*** Save old model to disk")
        net.save_params_to_file(ckpt_dir + "/old_model/model", session=session)

        print("*** Forwarding ...")

        extern_data_tf_raw_dict = net.extern_data.as_raw_tensor_dict()
        assert set(extern_data_tf_raw_dict.keys()) == set(extern_data_numpy_raw_dict.keys())
        feed_dict = {extern_data_tf_raw_dict[k]: extern_data_numpy_raw_dict[k] for k in extern_data_numpy_raw_dict}
        fetches = net.get_fetches_dict()
        old_model_outputs_data = {}
        for old_layer_name, _ in _layer_mapping.items():
            layer = net.get_layer(old_layer_name)
            out = layer.output.copy_as_batch_major()
            old_model_outputs_data[old_layer_name] = out
            fetches["layer:" + old_layer_name] = out.placeholder
            for i, tag in enumerate(out.dim_tags):
                if tag.is_batch_dim():
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.get_dim_value()
                elif tag.dyn_size_ext:
                    old_model_outputs_data[f"{old_layer_name}:size{i}"] = tag.dyn_size_ext
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.placeholder
        old_model_outputs_fetch = session.run(fetches, feed_dict=feed_dict)

    def _make_new_model():
        return MakeModel.make_model(in_dim, target_dim, num_enc_layers=num_layers)

    rf.select_backend_torch()

    print("*** Convert old model to new model")
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=Checkpoint(index_path=tk.Path(ckpt_dir + "/old_model/model.index")),
        make_model_func=_make_new_model,
        map_func=map_param_func_v2,
        epoch=1,
        step=0,
    )
    converter._out_model_dir = tk.Path(ckpt_dir + "/new_model")
    converter.out_checkpoint = tk.Path(ckpt_dir + "/new_model/model.pt")
    converter.run()

    print("*** Create new model")
    new_model = _make_new_model()

    rf.init_train_step_run_ctx(train_flag=False)
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_numpy_raw_dict)
    tensor_dict_numpy_to_torch_(extern_data)

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(ckpt_dir + "/new_model/model.pt")
    pt_module.load_state_dict(checkpoint_state["model"])

    print("*** Forwarding with tracing ...")

    funcs_to_trace_list = [
        Model.encode,
        Model.decode_logits,
        ConformerEncoder.__call__,
        ConformerEncoderLayer.__call__,
        ConformerConvSubsample.__call__,
        from_scratch_training,
    ]
    code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
    captured_tensors = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

    def _trace_func(frame, event, arg):
        """
        Trace func to get intermediate outputs.
        """
        func = code_obj_to_func.get(frame.f_code)
        if func:
            if event == "call":
                captured_tensors.setdefault(func, []).append({})
            else:
                for k, v in frame.f_locals.items():
                    if not isinstance(v, Tensor):
                        continue
                    prev = captured_tensors[func][-1].get(k, None)
                    if prev is None or prev[-1] is not v:
                        print(f"{func.__qualname__} tensor var changed: {k} = {v}")
                        captured_tensors[func][-1].setdefault(k, []).append(v)
            return _trace_func

    sys.settrace(_trace_func)
    from_scratch_training(
        model=new_model,
        data=extern_data["audio_features"],
        data_spatial_dim=time_dim,
        targets=extern_data["bpe_labels"],
        targets_spatial_dim=target_spatial_dim,
    )
    sys.settrace(None)
    pprint(captured_tensors)

    print("*** Getting values from trace ...")
    fetches = {}
    for old_layer_name, new_var_path in _layer_mapping.items():
        new_out = captured_tensors
        try:
            for k in new_var_path:
                new_out = new_out[k]
        except KeyError as exc:
            raise Exception(f"{exc.__class__.__name__} {exc}, new_var_path: {new_var_path}")
        assert isinstance(new_out, Tensor), f"new_out: {new_out}, new_var_path: {new_var_path}"
        old_out = old_model_outputs_data[old_layer_name]
        assert old_out.batch_ndim == new_out.batch_ndim
        mapped_axes = new_out.find_matching_dim_map(old_out, list(range(old_out.batch_ndim)))
        out = new_out.copy_transpose([mapped_axes[i] for i in range(old_out.batch_ndim)])
        fetches["layer:" + old_layer_name] = out.raw_tensor
        for i, tag in enumerate(out.dim_tags):
            if tag.dyn_size_ext:
                fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.raw_tensor
    fetches = {k: v.detach().cpu().numpy() for (k, v) in fetches.items()}
    new_model_outputs_fetch = fetches

    print("*** Comparing ...")
    print("**** target spatial len:", extern_data_numpy_raw_dict["bpe_labels"].shape[1])
    for out_step in range(extern_data_numpy_raw_dict["bpe_labels"].shape[1]):
        for old_layer_name, new_var_path in _layer_mapping.items():
            out = old_model_outputs_data[old_layer_name]
            if out_step > 0 and target_spatial_dim not in out.dim_tags:
                continue
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    old_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    new_v = new_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    numpy.testing.assert_equal(old_v, new_v, err_msg=f"{tag} mismatch")
            old_v = old_model_outputs_fetch["layer:" + old_layer_name]
            new_v = new_model_outputs_fetch["layer:" + old_layer_name]
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext and tag.dyn_size_ext.dim_tags:  # dynamic, and not scalar dyn sizes
                    assert tag.dyn_size_ext.dim_tags == (batch_dim,)  # not implemented otherwise
                    assert out.batch_dim_axis == 0  # not implemented otherwise but should be ensured above
                    size_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    for b in range(old_v.shape[0]):
                        idx = tuple([slice(b, b + 1)] + [slice(None, None)] * (i - 1) + [slice(size_v[b], None)])
                        old_v[idx] = 0
                        new_v[idx] = 0
            print(f"* Comparing {out}: {old_layer_name!r} vs {new_var_path!r}")
            assert old_v.shape == new_v.shape
            if target_spatial_dim in out.dim_tags:
                assert out.get_axis_from_description(target_spatial_dim) == 1  # not implemented otherwise
                out = out.copy_template_excluding_axis(1)
                print("** comparing out_step", out_step, out)
                old_v = old_v[:, out_step]
                new_v = new_v[:, out_step]
            # Using equal_nan=False because we do not want any nan in any of the values.
            rtol, atol = 1e-5, 1e-5
            if numpy.allclose(old_v, new_v, rtol=rtol, atol=atol):
                continue
            print("** not all close. close:")
            # Iterate over all indices, and check if the values are close.
            # If not, add the index to the mismatches list.
            remarks = []
            count_mismatches = 0
            for idx in sorted(numpy.ndindex(old_v.shape), key=sum):
                if numpy.isnan(old_v[idx]) and numpy.isnan(new_v[idx]):
                    remarks.append("[%s]:? (both are nan)" % ",".join([str(i) for i in idx]))
                    count_mismatches += 1
                    continue
                close = numpy.allclose(old_v[idx], new_v[idx], rtol=rtol, atol=atol)
                if not close:
                    count_mismatches += 1
                remarks.append(
                    "[%s]:" % ",".join([str(i) for i in idx])
                    + ("✓" if close else "✗ (%.5f diff)" % abs(old_v[idx] - new_v[idx]))
                )
                if len(remarks) >= 50 and count_mismatches > 0:
                    remarks.append("...")
                    break
            print("\n".join(remarks))
            numpy.testing.assert_allclose(
                old_v,
                new_v,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                err_msg=f"{old_layer_name!r} vs {new_var_path!r} mismatch",
            )
            raise Exception(f"should not get here, mismatches: {remarks}")

    print("*** Done, all correct (!), exit now ***")
    raise SystemExit("done")


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_import
