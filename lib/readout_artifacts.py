"""Shared storage for repeated readout diagnostics."""

import gzip
import hashlib
import json
from pathlib import Path


SUMMARY_FIELDS = (
    "kind", "definition", "counts", "active_variable_indices", "zero_support_variables",
    "lambda_per_variable", "ridge", "local_ridge", "global_ridge", "support_power",
    "bias_penalty", "center_features", "effective_degrees_of_freedom",
    "bias_effective_degrees_of_freedom", "slope_effective_degrees_of_freedom",
    "active_regularized_condition", "local_normal_gradient_norm",
    "float32_deployed_local_normal_gradient_norm", "shared_normal_gradient_norm",
    "float32_deployed_shared_normal_gradient_norm", "coefficient_norm",
    "local_regularizer", "shared_regularizer",
)

RUN_ANALYSIS_FIELDS = (
    "feature_shift", "extra_features", "context_features", "spectral_features",
    "nested_spectral_features", "input_reliability", "baseline_validation",
)


def compact_readout_run_metadata(output_dir, report):
    """Reference analysis already persisted in the final compact results file."""
    metadata = {name: value for name, value in report.items() if name != "modes"}
    shared = [name for name in RUN_ANALYSIS_FIELDS if name in metadata]
    if not shared:
        return metadata
    directory = Path(output_dir).resolve()
    results = directory / "results.json.gz"
    if results.resolve().parent != directory or not results.is_file():
        raise ValueError("Compact run metadata requires its own completed results.json.gz write")
    for name in shared:
        del metadata[name]
    metadata["analysis_artifact"] = {
        "artifact_format": "json_record_v1", "artifact": "results.json.gz",
        "member": [],
    }
    metadata["analysis_fields"] = shared
    return metadata


def archive_candidate_diagnostics(output_dir, details):
    """Deduplicate candidate diagnostics by content and return a reference."""
    payload = json.dumps(details, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(output_dir) / "fit_diags"
    directory.mkdir(exist_ok=True)
    # Content-derived names let different readout modes share the same diagnostics.
    destination = directory / (digest[:32] + ".json.gz")
    if not destination.exists():
        with destination.open("xb") as handle:
            handle.write(gzip.compress(payload, mtime=0))
    return {
        **{name: details[name] for name in SUMMARY_FIELDS if name in details},
        "artifact": destination.relative_to(output_dir).as_posix(),
        "artifact_format": "gzip_json",
        "uncompressed_bytes": len(payload),
    }
