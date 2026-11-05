"""Additive observed-target statistics, separated by own-history support."""

import torch


SUPPORT_NAMES = ("all", "no_history", "one", "two_to_four", "five_plus")


def history_support_masks(observed_mask, batch, variables):
    if (not torch.is_tensor(observed_mask) or observed_mask.ndim < 3
            or observed_mask.shape[0] != batch or observed_mask.shape[-1] != variables):
        raise ValueError("observed_mask must match target batch and variable axes")
    count = (observed_mask > 0).sum(dim=tuple(range(1, observed_mask.ndim - 1)))
    return torch.stack((torch.ones_like(count, dtype=torch.bool), count == 0, count == 1,
                        (count >= 2) & (count <= 4), count >= 5))


@torch.no_grad()
def observed_measure_groups(value, valid_target, observed_mask=None):
    if (not torch.is_tensor(value) or value.ndim != 3 or value.shape != valid_target.shape
            or valid_target.dtype != torch.bool or value.device != valid_target.device):
        raise ValueError("measure and boolean eligibility must match [batch, time, variable]")
    if observed_mask is None:
        support = torch.ones(1, value.shape[0], value.shape[-1], dtype=torch.bool, device=value.device)
    else:
        if observed_mask.device != value.device:
            raise ValueError("observed_mask and measure must share a device")
        support = history_support_masks(observed_mask, value.shape[0], value.shape[-1])
    eligible = valid_target.unsqueeze(0) & support.unsqueeze(2)
    finite = eligible & torch.isfinite(value).unsqueeze(0)
    clean = torch.where(finite, value.double().unsqueeze(0), 0.)
    # Each array is additive across recorded batches, avoiding averages of means.
    packed = torch.stack((eligible.sum((1, 2)).double(), finite.sum((1, 2)).double(),
                          clean.sum((1, 2)), clean.square().sum((1, 2)))).cpu().tolist()
    rows = {}
    for index, name in enumerate(SUPPORT_NAMES[:len(support)]):
        rows[name] = {"eligible_count": [int(x) for x in packed[0][index]],
                      "finite_count": [int(x) for x in packed[1][index]],
                      "sum": packed[2][index], "square_sum": packed[3][index]}
    return {"schema_version": 1, "variables": value.shape[-1], "groups": rows,
            "definition": "Unweighted finite observed target slots; variable order is original; all is the sum of four disjoint history groups when history is supplied",
            "coverage": "Only diagnostic-recorded batches, not a complete epoch"}
