import torch


def blank_collapse_init(emission, blank_threshold, blank_id=0):
    # prob = torch.nn.functional.softmax(emission, dim=-1)
    blanks = emission[:, blank_id] > blank_threshold
    first_blanks = 0
    u, c = torch.unique_consecutive(blanks, dim=0, return_counts=True)
    u = u.tolist()
    c = c.tolist()
    cc = []
    for j in range(len(c)):
        if u[j]:  # if blank
            if j == 0:
                first_blanks = c[j]
            elif j == len(c) - 1:
                break
            else:
                cc.append(c[j])
        else:
            cc += [1] * c[j]
    if len(cc) == 0:  # case: every frame is a blank frame
        cc = [0]
        first_blanks = 0
    indices = torch.cumsum(torch.tensor(cc), dim=0) - 1 + first_blanks
    new_emission = emission[indices]

    return new_emission, indices


def blank_collapse_single(logprobs, threshold, blank_idx):
    blanks = logprobs[:, blank_idx] > threshold  # [T,]
    _, counts = torch.unique_consecutive(blanks, return_counts=True)

    blank_begin, blank_end = blanks[0].item(), blanks[-1].item()
    initial_blank_cnt = counts[0].item() if blank_begin else 0
    final_blank_cnt = counts[-1].item() if blank_end else 0

    initial_slice = initial_blank_cnt
    final_slice = len(blanks) - final_blank_cnt

    blanks = blanks[initial_slice:final_slice]
    assert not blanks[0].item() and not blanks[-1].item()
    blanks_shift = torch.roll(blanks, shifts=-1)

    # Need to explicitly adjust emission lengths when using ctc_decoder!
    collapsed_logrpobs = logprobs[initial_slice:final_slice]
    collapsed_logprobs = collapsed_logrpobs[~(blanks & blanks_shift)]

    return collapsed_logprobs


def blank_collapse_batched(logprobs, audio_features_len, blank_threshold, blank_idx):
    """
    :param logprobs: softmax-normalized probabilities in log-space, [B, T, V+1]
    :param audio_features_len: length of T as [B]
    :param blank_threshold: collapse threshold probability in log-space
    :param blank_idx: index of blank label, i.e. V+1
    """
    batch_dim, time_dim = logprobs.shape[0], logprobs.shape[1]

    # Global candidates for blank collapse pruning
    blanks = logprobs[:, :, blank_idx] > blank_threshold  # [B, T]

    # For batches, adjust individual lengths by mapping paddings to True values in mask
    audio_lens_mask = (
        torch.arange(time_dim)[None, :] >= audio_features_len[:, None]
    )  # [B, T]
    blanks = blanks | audio_lens_mask  # [B, T]

    # Obtain counts on initial and final blank frames
    sequence_mask, sequence_indices = (~blanks).nonzero(as_tuple=True)  # tuple of [T',]
    _, sequence_bounds = torch.unique(sequence_mask, return_counts=True)  # [B,]

    sequence_bounds = torch.cat(
        (torch.Tensor([0]).to(torch.int), torch.cumsum(sequence_bounds, dim=0))
    )  # [B+1,]

    initial_blank_idx = sequence_indices[sequence_bounds[:-1]]  # [B,]
    final_blank_idx = sequence_indices[(sequence_bounds - 1)[1:]]  # [B,]

    # Logical-and between "blanks" and "blanks_shift" to account for label-blank-label case
    blanks_shift = torch.roll(blanks, shifts=-1, dims=1)  # [B, T]

    # Logical-or between "(blanks & blanks_shift)" and "bounds_mask" to restore proper lengths
    bounds_mask = torch.arange(time_dim).repeat(batch_dim, 1)  # [B, T]
    bounds_mask_initial = bounds_mask < initial_blank_idx[:, None]  # [B, T]
    bounds_mask_final = bounds_mask > final_blank_idx[:, None]  # [B, T]
    bounds_mask = bounds_mask_initial | bounds_mask_final  # [B, T]

    # Logical-not to assign True to frames kept
    blanks = ~((blanks & blanks_shift) | bounds_mask)  # [B, T]

    # De-batchify and re-arrange based on changed lengths
    batch_mask, batch_indices = blanks.nonzero(as_tuple=True)
    _, collapsed_audio_features_len = torch.unique(
        batch_mask, return_counts=True
    )  # [B,]

    # Compute new time dimension to restore batching
    collapsed_time_dim = torch.max(collapsed_audio_features_len)  # T''

    # IMPORTANT: After blank collapse, permuting the batch might be necessary due to new audio lengths
    # batch_collapsed_order = torch.argsort(collapsed_audio_features_len, descending=True)

    # Align mask and indices to match the collapsed audio lengths in sorted order
    batch_mask = torch.arange(batch_dim)[:, None].expand(
        batch_dim, collapsed_time_dim
    )  # [B, T'']
    # batch_mask = batch_mask[batch_collapsed_order]  # [B, T'']

    batch_indices = torch.split(
        batch_indices, collapsed_audio_features_len.tolist()
    )  # tuple (B,)
    batch_indices = torch.nn.utils.rnn.pad_sequence(
        batch_indices, batch_first=True
    )  # [B, T'']
    # batch_indices = batch_indices[batch_collapsed_order]  # [B, T'']

    # Restore original order within the batch
    collapsed_logprobs = logprobs[batch_mask, batch_indices]  # [B, T'', V+1]
    # collapsed_logprobs = permuted_logprobs[torch.argsort(batch_collapsed_order)]  # [B, T'', V+1]
    # collapsed_audio_features_len = collapsed_audio_features_len[batch_collapsed_order]  # [B, ]

    return collapsed_logprobs, collapsed_audio_features_len