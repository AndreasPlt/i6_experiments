import time
import torch
from torch import nn
from torch.onnx import export
from torchaudio.functional import mask_along_axis

from returnn.torch.engine import TrainCtx


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        lstm_size = 1024
        target_size=12001
        self.blstm1 = nn.LSTM(input_size=50, hidden_size=lstm_size, bidirectional=True, batch_first=False)
        self.blstm_stack = nn.LSTM(input_size=2*lstm_size, hidden_size=lstm_size, bidirectional=True, num_layers=7, batch_first=False)
        self.final_linear = nn.Linear(2*lstm_size, target_size)
        self.lstm_size = lstm_size

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
        audio_features_time_masked_2 = mask_along_axis(audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1)
        audio_features_masked = mask_along_axis(audio_features_time_masked_2, mask_param=10, mask_value=0.0, axis=2)
        audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        blstm_in = torch.swapaxes(audio_features_masked_2, 0, 1)  # [B, T, F] -> [T, B, F]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len)
        blstm_first, _ = self.blstm1(blstm_packed_in)
        blstm_packed_out, _ = self.blstm_stack(blstm_first)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_packed_out, padding_value=0.0, batch_first=False)  # [T, B, F]
        
        logits = self.final_linear(blstm_out)  # [T, B, F]
        logits  = torch.permute(logits, dims=(1, 2, 0))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits
        
scripted_model = None

def train_step(*, model: Model, data, train_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:seq_len"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:seq_len"][indices]

    if scripted_model is None:
        scripted_model = torch.jit.script(model)

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len, batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    train_ctx.mark_as_loss(name="CE", loss=loss)






