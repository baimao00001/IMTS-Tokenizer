"""Observed-row caches for readout analysis."""
import json
import os
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _owned_cpu(tensor):
    return tensor.detach().to(device="cpu").contiguous().clone()


class ObservedReadoutCache:
    """Each sample_window_index identifies an input window, not an original record."""

    def __init__(self, output_dir, split, variables, feature_dim, chunk_rows=4096):
        if split not in ("train", "val"):
            raise ValueError("readout feature cache split must be train or val")
        self.variables = _positive_int(variables, "variables")
        self.feature_dim = _positive_int(feature_dim, "feature_dim")
        self.chunk_rows = _positive_int(chunk_rows, "chunk_rows")
        self.output_dir = Path(output_dir).resolve()
        self.directory = self.output_dir / split
        self.split = split
        self.directory.mkdir(parents=True, exist_ok=False)
        self._schema = None
        self._parts = []
        self._buffered = 0
        self._rows = 0
        self._written = 0
        self._windows = 0
        self._batches = 0
        self._variable_rows = torch.zeros(self.variables, dtype=torch.int64)
        self._variable_weight = torch.zeros(self.variables, dtype=torch.float64)
        self._fold_rows = {}
        self._fold_weight = {}
        self._shards = []
        self._finalized = False
        self._write_manifest()

    def _validate(self, features, core, target, mask, query_times, fold_labels, states):
        for name, value in (("features", features), ("core", core), ("target", target),
                            ("mask", mask), ("query_times", query_times)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(name + " must be a tensor")
        if features.ndim != 4 or features.shape[-2:] != (self.variables, self.feature_dim):
            raise ValueError("features must be [batch,time,variables,feature_dim]")
        shape = features.shape[:3]
        b, t, _ = shape
        for name, value in (("core", core), ("target", target), ("mask", mask)):
            if value.shape != shape:
                raise ValueError(name + " must match features [batch,time,variable]")
        if query_times.shape != (b, t):
            raise ValueError("query_times must be [batch,time]")
        for name, value in (("features", features), ("core", core), ("target", target),
                            ("query_times", query_times)):
            if not value.is_floating_point():
                raise TypeError(name + " must have a real floating dtype")
            if value.device != features.device:
                raise ValueError(name + " must share the feature device")
        if mask.device != features.device or mask.is_complex():
            raise ValueError("mask must be real and share the feature device")
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("mask must be finite and nonnegative, including masked positions")
        accumulated_weight = mask.double().sum().cpu() + self._variable_weight.sum()
        if not torch.isfinite(accumulated_weight):
            raise ValueError("weighted cache counts overflow float64")
        if fold_labels is not None:
            if (not isinstance(fold_labels, torch.Tensor) or fold_labels.shape != (b,)
                    or fold_labels.dtype != torch.long or fold_labels.device.type != "cpu"):
                raise ValueError("fold_labels must be a CPU long tensor with shape [batch]")
            if (fold_labels < 0).any():
                raise ValueError("provided fold_labels must be nonnegative")
        if states is not None:
            if (not isinstance(states, torch.Tensor) or states.ndim != 4
                    or states.shape[:3] != shape or states.shape[-1] < 1):
                raise ValueError("states must be [batch,time,variable,state_dim] with state_dim > 0")
            if not states.is_floating_point() or states.device != features.device:
                raise ValueError("states must be real floating tensors on the feature device")
        schema = {
            "features": str(features.dtype), "core": str(core.dtype), "target": str(target.dtype),
            "weight": str(mask.dtype), "query_time": str(query_times.dtype),
            "state": str(states.dtype) if states is not None else None,
            "state_dim": states.shape[-1] if states is not None else None,
            "fold_labels_provided": fold_labels is not None,
        }
        if self._schema is not None and schema != self._schema:
            raise ValueError("cache dtypes, state width/presence and fold presence must stay consistent")
        bi, ti, vi = (mask > 0).nonzero(as_tuple=True)
        # Validate all observed rows before accepting any part of this batch.
        for start in range(0, bi.numel(), self.chunk_rows):
            a, q, v = bi[start:start+self.chunk_rows], ti[start:start+self.chunk_rows], vi[start:start+self.chunk_rows]
            selected = {"features": features[a, q, v], "core": core[a, q, v],
                        "target": target[a, q, v], "query_time": query_times[a, q]}
            if states is not None:
                selected["state"] = states[a, q, v]
            for name, value in selected.items():
                if not torch.isfinite(value).all():
                    raise ValueError("nonfinite observed " + name + " cannot be cached")
        return schema, bi, ti, vi

    @torch.no_grad()
    def update(self, features, core, target, mask, query_times, fold_labels=None, states=None):
        if self._finalized:
            raise RuntimeError("a finalized readout cache cannot accept more batches")
        schema, bi, ti, vi = self._validate(features, core, target, mask, query_times, fold_labels, states)
        self._schema = schema
        window_offset = self._windows
        self._windows += features.shape[0]
        self._batches += 1
        new_paths = []
        cursor = 0
        while cursor < bi.numel():
            length = min(self.chunk_rows - self._buffered, bi.numel() - cursor)
            a, q, v = bi[cursor:cursor+length], ti[cursor:cursor+length], vi[cursor:cursor+length]
            sample_index = _owned_cpu(a)
            payload = {
                "features": _owned_cpu(features[a, q, v]),
                "core": _owned_cpu(core[a, q, v]),
                "target": _owned_cpu(target[a, q, v]),
                "weight": _owned_cpu(mask[a, q, v]),
                "variable": _owned_cpu(v),
                "sample_window_index": sample_index + window_offset,
                "query_time": _owned_cpu(query_times[a, q]),
                "fold": (torch.full((length,), -1, dtype=torch.long, device="cpu")
                         if fold_labels is None else _owned_cpu(fold_labels[sample_index])),
            }
            if states is not None:
                payload["state"] = _owned_cpu(states[a, q, v])
            self._parts.append(payload)
            self._buffered += length
            self._rows += length
            self._variable_rows += torch.bincount(payload["variable"], minlength=self.variables)
            self._variable_weight += torch.bincount(
                payload["variable"], weights=payload["weight"].double(), minlength=self.variables)
            for fold in payload["fold"].unique().tolist():
                chosen = payload["fold"] == fold
                key = str(fold)
                self._fold_rows[key] = self._fold_rows.get(key, 0) + int(chosen.sum())
                self._fold_weight[key] = self._fold_weight.get(key, 0.0) + float(payload["weight"][chosen].double().sum())
            if self._buffered == self.chunk_rows:
                new_paths.append(self._flush())
            cursor += length
        self._write_manifest()
        return new_paths

    def _flush(self):
        payload = {
            name: torch.cat([part[name] for part in self._parts], dim=0).contiguous().clone()
            for name in self._parts[0]
        }
        name = "{:06d}.pt".format(len(self._shards))
        path = self.directory / name
        torch.save(payload, path)
        rows = self._buffered
        self._shards.append({
            "file": name, "rows": rows, "bytes": path.stat().st_size,
            "per_variable_rows": torch.bincount(payload["variable"], minlength=self.variables).tolist(),
            "per_variable_weight": torch.bincount(
                payload["variable"], weights=payload["weight"].double(), minlength=self.variables).tolist(),
            "first_sample_window_index": int(payload["sample_window_index"].min()),
            "last_sample_window_index": int(payload["sample_window_index"].max()),
        })
        self._written += rows
        self._parts.clear()
        self._buffered = 0
        self._write_manifest()
        return (Path(self.split) / name).as_posix()

    def summary(self):
        return {
            "schema_version": 1, "kind": "lossless_observed_readout_rows",
            "status": "finalized" if self._finalized else "open",
            "split": self.split, "variables": self.variables, "feature_dim": self.feature_dim,
            "chunk_rows": self.chunk_rows, "input_batches": self._batches, "input_windows": self._windows,
            "rows": self._rows, "rows_written": self._written, "buffered_rows": self._buffered,
            "per_variable_rows": self._variable_rows.tolist(),
            "per_variable_weight": self._variable_weight.tolist(),
            "weighted_count": float(self._variable_weight.sum()),
            "fold_rows": dict(self._fold_rows), "fold_weight": dict(self._fold_weight),
            "dtypes": dict(self._schema) if self._schema is not None else None,
            "shards": deepcopy(self._shards),
            "manifest": (Path(self.split) / "manifest.json").as_posix(),
            "manifest_relative_to": "output_dir supplied to ObservedReadoutCache",
            "input_layouts": {
                "features": "[batch,time,variables,feature_dim]",
                "core_target_mask": "[batch,time,variables]", "query_times": "[batch,time]",
                "fold_labels": "optional [batch] CPU long",
                "states": "optional [batch,time,variables,state_dim]",
            },
            "definitions": {
                "selection": "Rows with positive mask values, retaining feature dimension and dtype.",
                "weight": "Positive mask value; totals accumulated in float64.",
                "sample_window_index": "Input-window index across collected batches, including empty windows.",
                "fold": "Original-record fold label repeated across its rows; -1 when unavailable.",
                "scope": "Training and validation.",
                "reuse": "Fixed encoder features at observed targets.",
                "state": "Optional context with fixed width and dtype.",
            },
        }

    def _write_manifest(self):
        temporary = self.directory / ".manifest.tmp"
        temporary.write_text(json.dumps(self.summary(), indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.directory / "manifest.json")

    def finalize(self):
        if not self._finalized:
            if self._buffered:
                self._flush()
            self._finalized = True
            self._write_manifest()
        return self.summary()
