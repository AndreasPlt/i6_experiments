import torch
import numpy as np
from torchaudio.models.rnnt import RNNT
from returnn.frontend import Tensor
from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.forward.transducer_beam_search import monotonic_timesync_beam_search
from sisyphus import tk
from i6_core.lib.lexicon import Lexicon
from i6_experiments.users.berger.pytorch.helper_functions import map_tensor_to_minus1_plus1_interval
from i6_experiments.users.berger.pytorch.models.conformer_transducer_v2 import (
    FFNNTransducerDecoderOnly,
    FFNNTransducerEncoderOnly,
)


def encoder_forward_step(*, model: FFNNTransducerEncoderOnly, extern_data: TensorDict, **_):
    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)

    audio_feature_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_feature_lengths is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_outputs, encoder_output_lengths = model(
        features=audio_features.to(device),
        features_size=audio_feature_lengths.to(device),
    )  # [B, T, E], [B]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["encoder_outputs"].dims[1].dyn_size_ext.raw_tensor = encoder_output_lengths
    run_ctx.mark_as_output(encoder_outputs, name="encoder_outputs")


def decoder_forward_step(*, model: FFNNTransducerDecoderOnly, extern_data: TensorDict, **_):
    encoder = extern_data["encoder"].raw_tensor
    assert encoder is not None

    history = extern_data["history"].raw_tensor
    assert history is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_outputs, encoder_output_lengths = model(
        features=encoder.to(device),
        features_size=audio_feature_lengths.to(device),
    )  # [B, T, E], [B]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["encoder_outputs"].dims[1].dyn_size_ext.raw_tensor = encoder_output_lengths
    run_ctx.mark_as_output(encoder_outputs, name="encoder_outputs")


def monotonic_timesync_beam_search_forward_step(
    *, model: RNNT, extern_data: TensorDict, lexicon_file: tk.Path, beam_size: int, blank_id: int, **kwargs
):
    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_feature_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_feature_lengths is not None
    audio_feature_lengths = audio_feature_lengths.to(device="cuda")

    seq_tags = extern_data["seq_tag"].raw_tensor
    assert seq_tags is not None

    lexicon = Lexicon()
    lexicon.load(lexicon_file)
    label_list = list(lexicon.phonemes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens_arrays = []
    token_lengths = []

    for b in range(audio_features.size(0)):
        seq_feature_len = audio_feature_lengths[b : b + 1]
        seq_features = audio_features[b : b + 1, : seq_feature_len[0]]
        token_indices, score = monotonic_timesync_beam_search(
            model=model.to(device=device),
            features=seq_features.to(device=device),
            feature_lengths=seq_feature_len.to(device=device),
            blank_id=blank_id,
            beam_size=beam_size,
        )
        tokens_array = np.array([label_list[token_idx] for token_idx in token_indices], dtype="U")
        print(f"Recognized sequence {repr(seq_tags[b])} (score {score}): {tokens_array}")
        tokens_arrays.append(tokens_array)
        token_lengths.append(len(tokens_array))

    max_len = np.max(token_lengths)
    tokens_arrays_padded = [
        np.pad(tokens_array, pad_width=(0, max_len - len(tokens_array))) for tokens_array in tokens_arrays
    ]

    tokens_tensor = Tensor(
        name="tokens", dtype="string", raw_tensor=np.stack(tokens_arrays_padded, axis=0), feature_dim_axis=None
    )
    tokens_len_array = np.array(token_lengths, dtype=np.int32)

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = tokens_len_array

    run_ctx.mark_as_output(tokens_tensor, name="tokens")
