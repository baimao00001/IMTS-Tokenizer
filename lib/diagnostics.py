"""Tensor statistics, sampled feature geometry and measured parameter updates."""

import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_CHUNK = 262144


def _indices(size, limit, device):
    count = min(size, limit)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == size:
        return torch.arange(size, device=device)
    return torch.linspace(0, size - 1, count, device=device,
                          dtype=torch.float64).round().long()


def _mask_for(value, mask):
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=value.device).detach()
    return torch.broadcast_to(mask, value.shape)


def _chunk(value, start, end):
    # Diagnostic consumers only read chunks; contiguous storage needs no gather.
    if value.is_contiguous() and 0 <= start <= end <= value.numel():
        return value.view(-1)[start:end]
    positions = torch.arange(start, end, device=value.device)
    return torch.take(value, positions)


@torch.no_grad()
def summarize_tensor(value, mask=None, max_elements=4096):
    """Return finite-value moments and sampled quantiles.

    ``mask`` follows normal PyTorch broadcasting and selects nonzero entries.
    Nonfinite mask entries are excluded and counted explicitly. Quantile samples
    are evenly spaced by rank in the selected finite values, not random draws.
    Variance is population variance over finite entries.
    """
    if max_elements < 1:
        raise ValueError("max_elements must be positive")
    value = torch.as_tensor(value).detach()
    if value.is_complex():
        raise TypeError("complex tensor diagnostics are not supported")
    mask = _mask_for(value, mask)
    total = value.numel()
    selected = finite_count = nan_count = posinf_count = neginf_count = 0
    zeros = invalid_mask_count = 0
    mean = m2 = square_sum = 0.0
    minimum, maximum = math.inf, -math.inf
    chunk_counts = []
    for start in range(0, total, _CHUNK):
        end = min(start + _CHUNK, total)
        part = _chunk(value, start, end).double()
        if mask is not None:
            local_mask = _chunk(mask, start, end)
            valid_mask = torch.isfinite(local_mask)
            invalid_mask_count += int((~valid_mask).sum().item())
            part = part[valid_mask & (local_mask != 0)]
        selected += part.numel()
        nan_count += int(torch.isnan(part).sum().item())
        posinf_count += int(torch.isposinf(part).sum().item())
        neginf_count += int(torch.isneginf(part).sum().item())
        finite = part[torch.isfinite(part)]
        count = finite.numel()
        chunk_counts.append((start, end, count))
        if not count:
            continue
        local_mean = float(finite.mean().item())
        local_m2 = float(((finite - local_mean) ** 2).sum().item())
        new_count = finite_count + count
        delta = local_mean - mean
        m2 += local_m2 + delta * delta * finite_count * count / new_count
        mean += delta * count / new_count
        finite_count = new_count
        square_sum += float((finite * finite).sum().item())
        zeros += int((finite == 0).sum().item())
        minimum = min(minimum, float(finite.min().item()))
        maximum = max(maximum, float(finite.max().item()))

    ranks = _indices(finite_count, max_elements, value.device)
    samples, offset = [], 0
    for start, end, count in chunk_counts:
        chosen = ranks[(ranks >= offset) & (ranks < offset + count)] - offset
        offset += count
        if chosen.numel() == 0:
            continue
        part = _chunk(value, start, end).double()
        if mask is not None:
            local_mask = _chunk(mask, start, end)
            part = part[torch.isfinite(local_mask) & (local_mask != 0)]
        part = part[torch.isfinite(part)]
        samples.append(part[chosen].cpu())
    sample = torch.cat(samples) if samples else torch.empty(0, dtype=torch.double)
    quantiles = dict.fromkeys(("q01", "q05", "q25", "q50", "q75", "q95", "q99"))
    if sample.numel():
        levels = torch.tensor([.01, .05, .25, .5, .75, .95, .99], dtype=torch.double)
        quantiles.update(zip(quantiles, torch.quantile(sample, levels).tolist()))
    variance = m2 / finite_count if finite_count else None
    if variance is not None and math.isfinite(variance):
        variance = max(0.0, variance)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "population_elements": total, "selected_elements": selected,
        "excluded_elements": total - selected, "finite_count": finite_count,
        "nan_count": nan_count, "posinf_count": posinf_count,
        "neginf_count": neginf_count, "invalid_mask_count": invalid_mask_count,
        "finite_fraction": finite_count / selected if selected else None,
        "min": minimum if finite_count else None,
        "max": maximum if finite_count else None,
        "mean": mean if finite_count else None, "variance": variance,
        "std": math.sqrt(variance) if variance is not None else None,
        "rms": math.sqrt(square_sum / finite_count) if finite_count else None,
        "l2_norm": math.sqrt(square_sum) if finite_count else None,
        "zero_count": zeros,
        "zero_fraction": zeros / finite_count if finite_count else None,
        "quantiles": quantiles, "sample_elements": sample.numel(),
        "sampled": sample.numel() < finite_count,
        "sampling": "evenly_spaced_selected_finite_ranks",
        "basic_statistics_exact": True,
    }


def _json_safe(value):
    """Return JSON-compatible data and the number of nonfinite replacements."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("scalar payload tensors must have exactly one element")
        value = value.detach().item()
    if value is None or isinstance(value, (str, bool, int)):
        return value, 0
    if isinstance(value, float):
        return (value, 0) if math.isfinite(value) else (None, 1)
    if isinstance(value, Path):
        return str(value), 0
    if isinstance(value, dict):
        result, count = {}, 0
        for key, item in value.items():
            result[str(key)], replacements = _json_safe(item)
            count += replacements
        return result, count
    if isinstance(value, (tuple, list)):
        result, count = [], 0
        for item in value:
            safe, replacements = _json_safe(item)
            result.append(safe)
            count += replacements
        return result, count
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError("unsupported diagnostic payload type: " + type(value).__name__)


def _write_json(handle, record, *, compact=False):
    safe, count = _json_safe(record)
    safe["nonfinite_json_values"] = count
    separators = (",", ":") if compact else None
    handle.write(json.dumps(safe, ensure_ascii=True, allow_nan=False, separators=separators) + "\n")
    handle.flush()


def _key(mapping, name):
    if name not in mapping:
        return name
    index = 2
    while name + "#" + str(index) in mapping:
        index += 1
    return name + "#" + str(index)


def _row_coordinates(shape, rows):
    coordinates = []
    remaining = rows
    for width in reversed(shape):
        if width == 0:
            coordinates.append(torch.zeros_like(remaining))
        else:
            coordinates.append(remaining.remainder(width))
            remaining = torch.div(remaining, width, rounding_mode="floor")
    return tuple(reversed(coordinates))


@torch.no_grad()
def _norm_stats(value):
    value = value.detach()
    squared = 0.0
    finite_count = zero_count = 0
    for start in range(0, value.numel(), _CHUNK):
        part = _chunk(value, start, min(start + _CHUNK, value.numel())).double()
        finite = part[torch.isfinite(part)]
        finite_count += finite.numel()
        zero_count += int((finite == 0).sum().item())
        squared += float((finite * finite).sum().item())
    return {"elements": value.numel(), "finite_count": finite_count,
            "nonfinite_count": value.numel() - finite_count,
            "zero_count": zero_count, "finite_l2_norm": math.sqrt(squared),
            "l2_norm": math.sqrt(squared) if finite_count == value.numel() else None,
            "zero_fraction": zero_count / finite_count if finite_count else None}


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _cosine_ratio(dot, denominator):
    if denominator <= 0 or not (math.isfinite(dot) and math.isfinite(denominator)):
        return None
    return max(-1.0, min(1.0, dot / denominator))


class Diagnostics:
    """Record sampled batches in JSONL with optional tensor snapshots."""

    def __init__(self, output_dir, enabled=True, every=50, first_steps=2,
                 max_elements=4096, snapshots=0, metadata=None, storage="full"):
        if every < 1 or first_steps < 0 or max_elements < 1 or snapshots < 0:
            raise ValueError("invalid diagnostic sampling configuration")
        if storage not in {"full", "compact"}:
            raise ValueError("diagnostic storage must be full or compact")
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.every, self.first_steps = int(every), int(first_steps)
        self.max_elements, self.snapshots = int(max_elements), int(snapshots)
        self.active = False
        self._record = None
        self._batch_open = self._closed = False
        self._hooks, self._before, self._snapshot = [], {}, {}
        self._snapshot_count = self._snapshot_elements = 0
        self._batches = self._events = None
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sampling": {"every": self.every, "first_steps": self.first_steps,
                         "max_elements": self.max_elements,
                         "snapshots": self.snapshots,
                         "snapshot_total_elements_limit": self.max_elements * 32,
                         "training_index": "global_step",
                         "evaluation_index": "batch_index"},
            "metadata": metadata or {},
            "parameter_updates": "actual_values_on_deterministic_sample",
            "distribution_normalized_entropy": "entropy_divided_by_log_full_row_width",
            "feature_effective_rank": "exp_entropy_of_centered_singular_value_proportions",
        }
        if self.storage == "compact":
            manifest["storage"] = {"mode": "compact", "compression": "gzip", "compression_level": 1,
                                   "batches": "batches.jsonl.gz", "events": "events.jsonl.gz",
                                   "sampling_unchanged": True}
        with (self.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            _write_json(handle, manifest)
        try:
            if self.storage == "compact":
                self._batches = gzip.open(self.output_dir / "batches.jsonl.gz", "xt", encoding="utf-8", compresslevel=1)
                self._events = gzip.open(self.output_dir / "events.jsonl.gz", "xt", encoding="utf-8", compresslevel=1)
            else:
                self._batches = (self.output_dir / "batches.jsonl").open("x", encoding="utf-8")
                self._events = (self.output_dir / "events.jsonl").open("x", encoding="utf-8")
        except Exception:
            if self._batches is not None:
                self._batches.close()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("diagnostics are closed")

    def begin_batch(self, split, epoch, batch_index, step):
        self._check_open()
        if self._batch_open:
            raise RuntimeError("end_batch must precede the next begin_batch")
        self._batch_open = True
        index = step if str(split).lower() in ("train", "training") else batch_index
        self.active = self.enabled and (batch_index < self.first_steps or index % self.every == 0)
        self._before, self._snapshot = {}, {}
        self._snapshot_elements = 0
        if self.active:
            self._record = {
                "schema_version": SCHEMA_VERSION, "split": str(split),
                "epoch": int(epoch), "batch_index": int(batch_index), "step": int(step),
                "tensors": {}, "gradients": {}, "scalars": {},
                "distributions": {}, "features": {}, "parameters": {},
            }

    def scalar(self, name, value):
        if not self.active:
            return
        safe, count = _json_safe(value)
        self._record["scalars"][_key(self._record["scalars"], name)] = safe
        if count:
            self._record["scalar_nonfinite_values"] = self._record.get("scalar_nonfinite_values", 0) + count

    def tensor(self, name, value, mask=None, grad=False, axes=None):
        if not self.active or value is None:
            return
        key = _key(self._record["tensors"], name)
        summary = summarize_tensor(value, mask, self.max_elements)
        if axes is not None:
            if len(axes) != value.ndim:
                raise ValueError("axis labels must match tensor dimensions")
            summary["axes"] = list(axes)
        summary["requires_grad"] = bool(value.requires_grad)
        self._record["tensors"][key] = summary
        if grad:
            summary["gradient_requested"] = True
            summary["gradient_observed"] = False
            if value.requires_grad:
                detached_mask = mask.detach() if isinstance(mask, torch.Tensor) else mask

                def capture(gradient):
                    if self.active:
                        gradient_key = _key(self._record["gradients"], key)
                        self._record["gradients"][gradient_key] = summarize_tensor(
                            gradient, detached_mask, self.max_elements)
                        summary["gradient_observed"] = True
                    return None

                self._hooks.append(value.register_hook(capture))
        if self._snapshot_count < self.snapshots:
            remaining = self.max_elements * 32 - self._snapshot_elements
            if remaining > 0:
                detached = value.detach()
                positions = _indices(detached.numel(), min(self.max_elements, remaining), detached.device)
                item = {"shape": list(detached.shape), "flat_indices": positions.cpu(),
                        "values": torch.take(detached, positions).cpu().clone()}
                if mask is not None:
                    item["mask"] = torch.take(_mask_for(detached, mask), positions).cpu().clone()
                self._snapshot[key] = item
                self._snapshot_elements += positions.numel()

    @torch.no_grad()
    def distribution(self, name, weights, dim=-1, mask=None):
        if not self.active:
            return
        weights = weights.detach()
        if weights.ndim == 0:
            raise ValueError("distribution requires at least one dimension")
        if not -weights.ndim <= dim < weights.ndim:
            raise IndexError("distribution dimension is outside tensor dimensions")
        dim = dim % weights.ndim
        moved = weights.movedim(dim, -1)
        width = moved.shape[-1]
        row_count = math.prod(moved.shape[:-1])
        rows = _indices(row_count, min(64, self.max_elements), weights.device)
        coordinates = _row_coordinates(moved.shape[:-1], rows)
        allowed = _mask_for(weights, mask)
        allowed = allowed.movedim(dim, -1) if allowed is not None else None
        size = rows.numel()
        mass = torch.zeros(size, dtype=torch.double, device=weights.device)
        squared, weighted_log, maximum, leaked = [torch.zeros_like(mass) for _ in range(4)]
        allowed_count = torch.zeros_like(mass)
        invalid = torch.zeros(size, dtype=torch.bool, device=weights.device)
        for start in range(0, width, min(self.max_elements, _CHUNK)):
            end = min(start + min(self.max_elements, _CHUNK), width)
            part = moved[coordinates + (slice(start, end),)].reshape(size, end - start).double()
            finite = torch.isfinite(part)
            invalid |= ((~finite) | (part < 0)).any(dim=-1)
            positive = torch.where(finite & (part > 0), part, 0)
            mass += positive.sum(dim=-1)
            squared += (positive ** 2).sum(dim=-1)
            weighted_log += torch.where(positive > 0, positive * positive.clamp_min(torch.finfo(torch.double).tiny).log(), 0).sum(dim=-1)
            maximum = torch.maximum(maximum, positive.max(dim=-1).values)
            if allowed is None:
                allowed_count += end - start
            else:
                local_mask = allowed[coordinates + (slice(start, end),)].reshape(size, end - start)
                valid_mask = torch.isfinite(local_mask) & (local_mask != 0)
                allowed_count += valid_mask.sum(dim=-1)
                leaked += torch.where(valid_mask, 0, positive).sum(dim=-1)
        valid = (~invalid) & torch.isfinite(mass) & (mass > 0)
        entropy = torch.full_like(mass, float("nan"))
        entropy[valid] = (mass[valid].log() - weighted_log[valid] / mass[valid]).clamp_min(0)
        effective_support = entropy.exp()
        normalized_entropy = torch.full_like(mass, float("nan"))
        if width > 1:
            normalized_entropy[valid] = entropy[valid] / math.log(width)
        elif width == 1:
            normalized_entropy[valid] = 0
        reported_mass = torch.where(~invalid, mass, float("nan"))
        summary = {
            "shape": list(weights.shape), "dim": dim, "rows": row_count,
            "sampled_rows": size, "sampled": size < row_count,
            "sampling": "evenly_spaced_rows_all_columns",
            "row_width": width, "invalid_rows_in_sample": int(invalid.sum().item()),
            "zero_mass_rows_in_sample": int((mass == 0).sum().item()),
            "nonfinite_mass_rows_in_sample": int((~torch.isfinite(mass)).sum().item()),
            "all_masked_rows_in_sample": int((allowed_count == 0).sum().item()),
        }
        row_metrics = {
            "mass": reported_mass, "normalization_error": (reported_mass - 1).abs(),
            "entropy": entropy, "normalized_entropy": normalized_entropy,
            "effective_support": effective_support,
            "inverse_simpson_support": torch.where(valid, mass ** 2 / squared, float("nan")),
            "max_probability": torch.where(valid, maximum / mass, float("nan")),
            "mask_leakage_mass": torch.where(~invalid, leaked, float("nan")),
            "mask_leakage_fraction": torch.where(valid, leaked / mass, float("nan")),
            "allowed_support": allowed_count,
        }
        for metric, values in row_metrics.items():
            summary[metric] = summarize_tensor(values, max_elements=self.max_elements)
        self._record["distributions"][_key(self._record["distributions"], name)] = summary

    @torch.no_grad()
    def features(self, name, tensor):
        if not self.active:
            return
        value = tensor.detach()
        summary = summarize_tensor(value, max_elements=self.max_elements)
        if value.ndim == 0:
            raise ValueError("features requires a feature dimension")
        width = value.shape[-1]
        row_count = math.prod(value.shape[:-1])
        row_limit = min(64, max(1, int(math.sqrt(self.max_elements))))
        rows = _indices(row_count, row_limit, value.device)
        columns = _indices(width, max(1, self.max_elements // max(1, rows.numel())), value.device)
        coordinates = _row_coordinates(value.shape[:-1], rows)
        if value.ndim == 1:
            matrix = value[columns].unsqueeze(0).double().cpu()
        else:
            matrix = value[tuple(c.unsqueeze(-1) for c in coordinates) + (columns.unsqueeze(0),)].double().cpu()
        finite_rows = torch.isfinite(matrix).all(dim=-1)
        matrix = matrix[finite_rows]
        summary.update({
            "geometry_rows": row_count, "sampled_rows": rows.numel(),
            "finite_sampled_rows": matrix.shape[0], "feature_width": width,
            "sampled_features": columns.numel(),
            "geometry_sampled": rows.numel() < row_count or columns.numel() < width,
            "geometry_sampling": "evenly_spaced_rows_and_feature_columns",
        })
        if matrix.numel():
            scale = matrix.abs().max().clamp_min(torch.finfo(torch.double).tiny)
            normalized = matrix / scale
            normalized_norms = torch.linalg.vector_norm(normalized, dim=-1)
            norms = normalized_norms * scale
            summary["row_norm_estimate"] = summarize_tensor(norms * math.sqrt(width / columns.numel()), max_elements=self.max_elements)
            variances = normalized.var(dim=0, unbiased=False) * scale * scale
            summary["feature_variance"] = summarize_tensor(variances, max_elements=self.max_elements)
            summary["constant_feature_fraction"] = float((variances == 0).double().mean().item())
            nonzero = normalized_norms > 0
            unit = normalized[nonzero] / normalized_norms[nonzero, None]
            similarities = unit @ unit.T
            upper = torch.triu_indices(unit.shape[0], unit.shape[0], offset=1)
            summary["pairwise_cosine"] = summarize_tensor(similarities[upper[0], upper[1]], max_elements=self.max_elements)
            centered = normalized - normalized.mean(dim=0, keepdim=True)
            singular = torch.linalg.svdvals(centered)
            total = singular.sum()
            if total > 0:
                probabilities = singular[singular > 0] / total
                summary["effective_rank"] = float((-(probabilities * probabilities.log()).sum()).exp().item())
                summary["top_singular_value_fraction"] = float((singular[0] / total).item())
            else:
                summary["effective_rank"] = 0.0
                summary["top_singular_value_fraction"] = None
        else:
            summary["effective_rank"] = None
            summary["top_singular_value_fraction"] = None
        self._record["features"][_key(self._record["features"], name)] = summary

    @torch.no_grad()
    def parameters(self, model, when="before"):
        if not self.active:
            return
        if when not in ("before", "after"):
            raise ValueError("when must be 'before' or 'after'")
        if when == "after" and "before" not in self._record["parameters"]:
            raise RuntimeError("parameter updates require a before capture")
        details, groups = {}, {}
        for name, parameter in model.named_parameters():
            if when == "before":
                weight = _norm_stats(parameter)
                gradient = parameter.grad
                if gradient is not None and gradient.is_sparse:
                    raise TypeError("sparse parameter gradients are not supported")
                grad_stats = _norm_stats(gradient) if gradient is not None else None
                details[name] = {"shape": list(parameter.shape), "requires_grad": parameter.requires_grad,
                                 "weight": weight, "gradient": grad_stats,
                                 "missing_gradient": gradient is None,
                                 "grad_to_weight_ratio": _ratio(grad_stats["l2_norm"], weight["l2_norm"]) if grad_stats else None}
                positions = _indices(parameter.numel(), self.max_elements, parameter.device)
                self._before[name] = {
                    "indices": positions, "weight": torch.take(parameter, positions).double().cpu(),
                    "gradient": torch.take(gradient, positions).double().cpu() if gradient is not None else None,
                    "elements": parameter.numel(),
                }
            else:
                before = self._before.get(name)
                if before is None or before["elements"] != parameter.numel():
                    raise RuntimeError("model parameters changed between captures: " + name)
                current = torch.take(parameter, before["indices"]).double().cpu()
                delta = current - before["weight"]
                update = _norm_stats(delta)
                sampled_weight = _norm_stats(before["weight"])
                gradient = before["gradient"]
                cosine = None
                gradient_norm = negative_gradient_dot = None
                if gradient is not None and torch.isfinite(gradient).all() and torch.isfinite(delta).all():
                    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
                    negative_gradient_dot = float(torch.dot(delta, -gradient).item())
                    denominator = torch.linalg.vector_norm(delta) * gradient_norm
                    if denominator > 0:
                        cosine = float(torch.dot(delta, -gradient).div(denominator).clamp(-1, 1).item())
                scale = math.sqrt(parameter.numel() / max(1, delta.numel()))
                details[name] = {
                    "sampled": delta.numel() < parameter.numel(), "elements": parameter.numel(),
                    "sample_elements": delta.numel(), "sample_update": update,
                    "sample_weight_norm": sampled_weight["l2_norm"],
                    "sample_gradient_norm": gradient_norm,
                    "sample_negative_gradient_dot": negative_gradient_dot,
                    "update_norm_estimate": update["l2_norm"] * scale if update["l2_norm"] is not None else None,
                    "update_to_weight_ratio": _ratio(update["l2_norm"], sampled_weight["l2_norm"]),
                    "negative_gradient_cosine": cosine,
                    "changed_fraction": 1 - update["zero_fraction"] if update["zero_fraction"] is not None else None,
                }
            group = name.rsplit(".", 1)[0] if "." in name else "__root__"
            groups.setdefault(group, []).append(name)
        aggregates = {}
        for group, names in dict(groups, __all__=list(details)).items():
            items = [details[name] for name in names]
            if when == "before":
                weights = [item["weight"] for item in items]
                gradients = [item["gradient"] for item in items if item["gradient"] is not None]
                weight_norm = math.sqrt(sum(item["finite_l2_norm"] ** 2 for item in weights))
                grad_norm = math.sqrt(sum(item["finite_l2_norm"] ** 2 for item in gradients))
                weight_nonfinite = sum(item["nonfinite_count"] for item in weights)
                grad_nonfinite = sum(item["nonfinite_count"] for item in gradients)
                aggregates[group] = {
                    "parameters": len(items), "trainable_parameters": sum(item["requires_grad"] for item in items),
                    "missing_gradient_parameters": sum(item["missing_gradient"] and item["requires_grad"] for item in items),
                    "zero_gradient_parameters": sum(item["gradient"] is not None and item["gradient"]["l2_norm"] == 0 for item in items),
                    "gradient_nonfinite_elements": grad_nonfinite, "weight_nonfinite_elements": weight_nonfinite,
                    "weight_norm": weight_norm if not weight_nonfinite else None,
                    "gradient_norm": grad_norm if not grad_nonfinite else None,
                    "grad_to_weight_ratio": _ratio(grad_norm, weight_norm) if not (weight_nonfinite or grad_nonfinite) else None,
                }
            else:
                estimates = [item["update_norm_estimate"] for item in items]
                scales = [item["elements"] / max(1, item["sample_elements"]) for item in items]
                weight_energy = sum(item["sample_weight_norm"] ** 2 * scale
                                    for item, scale in zip(items, scales)
                                    if item["sample_weight_norm"] is not None)
                update_norm = math.sqrt(sum(v * v for v in estimates)) if all(v is not None for v in estimates) else None
                weight_valid = all(item["sample_weight_norm"] is not None for item in items)
                paired = [(item, scale) for item, scale in zip(items, scales)
                          if item["sample_gradient_norm"] is not None]
                paired_update_energy = sum(item["sample_update"]["l2_norm"] ** 2 * scale for item, scale in paired)
                paired_gradient_energy = sum(item["sample_gradient_norm"] ** 2 * scale for item, scale in paired)
                dot = sum(item["sample_negative_gradient_dot"] * scale for item, scale in paired)
                denominator = math.sqrt(paired_update_energy * paired_gradient_energy)
                aggregates[group] = {
                    "parameters": len(items), "sampled": any(item["sampled"] for item in items),
                    "update_norm_estimate": update_norm,
                    "update_to_weight_ratio_estimate": _ratio(update_norm, math.sqrt(weight_energy)) if weight_valid else None,
                    "negative_gradient_cosine_on_observed_gradients": _cosine_ratio(dot, denominator),
                    "parameters_with_observed_gradients": len(paired),
                    "unchanged_parameters_in_sample": sum(item["sample_update"]["l2_norm"] == 0 for item in items),
                    "nonfinite_update_elements_in_sample": sum(item["sample_update"]["nonfinite_count"] for item in items),
                }
        self._record["parameters"][when] = {"by_parameter": details, "by_module": aggregates}
        if when == "after":
            self._before.clear()

    def end_batch(self):
        self._check_open()
        if not self._batch_open:
            return
        try:
            if self.active:
                if self._snapshot:
                    filename = "snapshot_%04d.pt" % self._snapshot_count
                    torch.save({"split": self._record["split"], "step": self._record["step"],
                                "tensors": self._snapshot}, self.output_dir / filename)
                    self._record["snapshot"] = filename
                    self._record["snapshot_elements"] = self._snapshot_elements
                    self._snapshot_count += 1
                _write_json(self._batches, self._record, compact=self.storage == "compact")
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._before.clear()
            self._snapshot.clear()
            self._record = None
            self.active = self._batch_open = False

    def event(self, kind, **payload):
        self._check_open()
        if self.enabled:
            _write_json(self._events, {"schema_version": SCHEMA_VERSION,
                                      "kind": kind, "payload": payload}, compact=self.storage == "compact")

    def prepare_event(self, kind, **payload):
        """Build a JSON-compatible event record for shared storage."""
        self._check_open()
        safe, count = _json_safe({"schema_version": SCHEMA_VERSION, "kind": kind, "payload": payload})
        safe["nonfinite_json_values"] = count
        return safe

    def event_reference(self, kind, reference, **small_summary):
        self._check_open()
        if self.enabled:
            if "record_ref" in small_summary:
                raise ValueError("record_ref is reserved for the supplied reference")
            _write_json(self._events, {"schema_version": SCHEMA_VERSION, "kind": kind,
                                      "payload": {**small_summary, "record_ref": reference}},
                        compact=self.storage == "compact")

    def close(self):
        if self._closed:
            return
        try:
            self.end_batch()
        finally:
            if self._batches is not None:
                self._batches.close()
            if self._events is not None:
                self._events.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
