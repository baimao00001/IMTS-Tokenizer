"""Deterministic train-record grouping for conditional readout cross-validation."""
import hashlib
import json
import math
from functools import partial
from numbers import Integral, Real

import torch


def _record_key(value):
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return ["int", str(int(value))]
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return ["float", float(value).hex()]
    raise TypeError("readout group IDs must be strings or finite numeric scalars")


def _serialized_key(value):
    return json.dumps(_record_key(value), ensure_ascii=True, separators=(",", ":"))


def requested_group_folds(args):
    folds = getattr(args, "collect_group_folds", 0)
    if isinstance(folds, bool) or not isinstance(folds, Integral) or folds < 0 or folds == 1:
        raise ValueError("collect_group_folds must be zero or an integer >= 2")
    return int(folds)


class TrainingGroupFolds:
    def __init__(self, train_records, folds, dataset, activity_original_ids=None):
        if isinstance(folds, bool) or not isinstance(folds, Integral) or folds < 2:
            raise ValueError("readout grouping requires an integer >= 2 folds")
        self.folds = int(folds)
        self.dataset = str(dataset)
        self._activity_original = None
        if activity_original_ids is not None:
            if dataset != "activity":
                raise ValueError("original activity IDs are only valid for activity")
            originals = {}
            for value in activity_original_ids:
                key = _serialized_key(value)
                printed = str(value)
                if printed in originals and originals[printed] != key:
                    raise ValueError("activity IDs collide after chunk ID formatting")
                originals[printed] = key
            self._activity_original = originals
        if dataset == "activity" and self._activity_original is None:
            raise ValueError("activity grouping requires original pre-chunk IDs")
        keys = [self._key(record[0]) for record in train_records]
        unique = set(keys)
        if len(unique) < self.folds:
            raise ValueError("readout grouping has fewer original train records than folds")
        namespace = "imts-readout-group-folds-v1/" + self.dataset + "/"
        digests = {key: hashlib.sha256((namespace + key).encode("utf-8")).hexdigest() for key in unique}
        ordered = sorted(unique, key=lambda key: (digests[key], key))
        self._assignment = {key: rank % self.folds for rank, key in enumerate(ordered)}
        group_counts = [0] * self.folds
        window_counts = [0] * self.folds
        for fold in self._assignment.values():
            group_counts[fold] += 1
        for key in keys:
            window_counts[self._assignment[key]] += 1
        self.summary = {
            "folds": self.folds,
            "unique_groups": len(unique),
            "windows": len(keys),
            "groups_per_fold": group_counts,
            "windows_per_fold": window_counts,
            "dataset": self.dataset,
            "definition": "Original training record IDs sorted by SHA256, then assigned by rank modulo folds. "
                          "All windows of a record share one fold.",
            "selection_scope": "Readout cross-validation with a fixed neural core.",
        }

    def _key(self, value):
        if self._activity_original is None:
            return _serialized_key(value)
        if not isinstance(value, str):
            raise ValueError("activity chunk ID must be a string")
        original, separator, suffix = value.rpartition("_")
        if not separator or not suffix.isascii() or not suffix.isdecimal() or original not in self._activity_original:
            raise ValueError("activity chunk ID does not match an original record plus chunk suffix")
        return self._activity_original[original]

    def batch_folds(self, records):
        keys = [self._key(record[0]) for record in records]
        if any(key not in self._assignment for key in keys):
            raise ValueError("readout group batch contains an unknown training record")
        return torch.tensor([self._assignment[key] for key in keys], dtype=torch.long, device="cpu")

    def collate(self, collator, records):
        folds = self.batch_folds(records)
        batch = collator(records)
        if "readout_group_fold" in batch:
            raise ValueError("collator already supplied readout_group_fold")
        if batch["data_to_predict"].shape[0] != len(records):
            raise ValueError("readout group labels require collator batch order and size preservation")
        batch["readout_group_fold"] = folds
        return batch


def make_training_group_folds(train_records, args, activity_original_ids=None):
    folds = requested_group_folds(args)
    if not folds:
        return None
    return TrainingGroupFolds(train_records, folds, args.dataset, activity_original_ids)


def wrap_training_collate(collator, grouping):
    return collator if grouping is None else partial(grouping.collate, collator)


def attach_group_summary(data_objects, grouping):
    if grouping is not None:
        data_objects["readout_group_folds"] = grouping.summary
    return data_objects
