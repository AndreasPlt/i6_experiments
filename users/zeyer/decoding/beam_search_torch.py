"""
Beam search
"""

from typing import Sequence, Tuple, List

from dataclasses import dataclass
import functools
import torch

# noinspection PyProtectedMember
from torch.utils import _pytree as pytree

from .interface_torch import LabelScorer


@dataclass
class BeamSearchOpts:
    beam_size: int  # e.g. 12
    length_normalization_exponent: float  # e.g. 1 to enable, 0 to disable
    bos_label: int
    eos_label: int
    num_labels: int


def beam_search(
    label_scorer: LabelScorer,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchOpts,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beam search

    :param label_scorer:
    :param batch_size:
    :param max_seq_len: e.g. use encoder length. shape [Batch]
    :param device:
    :param opts:
    :return: seq_targets, seq_log_prob, out_seq_len:
        seq_targets: [Batch,FinalBeam,OutSeqLen]
        seq_log_prob: [Batch,FinalBeam]
        out_seq_len: [Batch,FinalBeam]
    """
    # Eager-mode implementation of beam search.
    # Initial state.
    beam_size = 1
    state = label_scorer.get_initial_state(batch_size=batch_size, device=device)
    target = torch.full([batch_size, beam_size], opts.bos_label, device=device)
    ended = torch.full([batch_size, beam_size], False, device=device)
    out_seq_len = torch.full([batch_size, beam_size], 0, device=device)
    seq_log_prob = torch.full([batch_size, beam_size], 0.0, device=device)

    masked_finished_log_prob = torch.where(torch.arange(0, opts.num_labels) == opts.eos_label, 0.0, -1.0e30)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        new_state = label_scorer.update_state(prev_state=state, prev_label=target)
        label_log_prob = label_scorer.score(
            prev_state=state, prev_label=target, state=new_state
        )  # [Batch,InBeam,Vocab]

        # Filter out finished beams
        label_log_prob = torch.where(ended, masked_finished_log_prob, label_log_prob)
        seq_log_prob = seq_log_prob[:, :, None] + label_log_prob  # [Batch,InBeam,Vocab]
        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        beam_size = seq_log_prob.shape[1]
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        state = pytree.tree_map(functools.partial(batch_gather, indices=backrefs), new_state)  # [Batch,Beam,...]
        ended = batch_gather(ended, indices=backrefs)  # [Batch,Beam]
        out_seq_len = batch_gather(out_seq_len, indices=backrefs)  # [Batch,Beam]
        i += 1

        ended = ended | (target == opts.eos_label)
        ended = ended | (i >= max_seq_len)[:, None]  # [Batch,Beam]
        if ended.all():
            break
        out_seq_len = out_seq_len + torch.where(ended, 0, 1)

        if i > 1 and opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= torch.where(
                ended,
                (i / (i - 1)) ** opts.length_normalization_exponent,
                1.0,
            )

    if i > 0 and opts.length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** opts.length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = torch.arange(beam_size)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: [Batch,FinalBeam] -> Beam
        # backrefs: [Batch,Beam] -> PrevBeam
        seq_targets_.insert(0, batch_gather(target, indices=indices))  # [Batch,FinalBeam]
        indices = batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

    seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch,FinalBeam,OutSeqLen]

    return seq_targets, seq_log_prob, out_seq_len


# noinspection PyShadowingBuiltins
def top_k_nd(
    source: torch.Tensor, *, k: int, dim: Sequence[int], sorted: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    # Derived from returnn.torch.frontend._backend.TorchBackend.top_k.
    # Move axis to the end, in the right order.
    dim = [(d + source.ndim) % source.ndim for d in dim]
    source = source.permute([d for d in range(source.ndim) if d not in dim] + list(dim))
    source_flat = source.flatten(start_dim=source.ndim - len(dim))
    values, indices = torch.topk(source_flat, k=k, dim=-1, largest=True, sorted=sorted)
    indices_out = []
    for i in reversed(list(range(len(dim)))):
        a_dim = source.shape[source.ndim - len(dim) + i]
        indices_out = indices % a_dim
        indices = indices // a_dim
        indices_out.insert(0, indices_out)
    return values, indices_out


def batch_gather(values: torch.Tensor, *, indices: torch.Tensor) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...]
    """
    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[0]
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]
    else:
        indices_flat = indices.flatten(1)
    out = torch.gather(values, dim=1, index=indices_flat.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(1)
    else:
        out = out.unflatten(1, indices.shape[1:])
    return out
