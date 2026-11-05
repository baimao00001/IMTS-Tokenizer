"""Plain and gzip JSON artifacts with run-relative references."""

import gzip
import io
import json
import math
import os
from pathlib import Path
import tempfile


def find_artifact(path):
    """Prefer an existing plain path, then its gzip counterpart."""
    path = Path(path)
    if path.exists() or path.suffix == ".gz":
        return path
    compressed = Path(str(path) + ".gz")
    return compressed if compressed.exists() else path


def _reject_constant(value):
    raise ValueError("Nonstandard JSON numeric literal: " + value)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number: " + value)
    return number


def _load(stream):
    return json.load(stream, parse_constant=_reject_constant, parse_float=_finite_float)


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_json(path):
    path = find_artifact(path)
    with _open_text(path) as stream:
        return _load(stream)


def write_json(path, already_json_safe_payload, compressed=False, compact=False):
    """Atomically replace one JSON artifact; return its actual output path."""
    path = Path(path)
    if compressed and path.suffix != ".gz":
        path = Path(str(path) + ".gz")
    if not compressed and path.suffix == ".gz":
        raise ValueError("A .gz output requires compressed=True")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as binary:
            if compact:
                payload = json.dumps(already_json_safe_payload, ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")
                if compressed:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=binary, compresslevel=1, mtime=0) as zipped:
                        zipped.write(payload)
                else:
                    binary.write(payload)
            elif compressed:
                with gzip.GzipFile(filename="", mode="wb", fileobj=binary, compresslevel=1, mtime=0) as zipped:
                    with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
                        json.dump(already_json_safe_payload, text, ensure_ascii=True, allow_nan=False,
                                  indent=2)
            else:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="\n") as text:
                    json.dump(already_json_safe_payload, text, ensure_ascii=True, allow_nan=False,
                              indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _reference_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Artifact reference must use a nonempty relative POSIX path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("Artifact reference must remain inside its run")
    root = Path(root).resolve()
    path = (root / path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Artifact reference escaped its run")
    return root, path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def write_reference(root, relative_name, record):
    """Store a JSON record once and return its relative reference."""
    root, path = _reference_path(root, relative_name)
    if path.suffix != ".gz":
        path = Path(str(path) + ".gz")
    if path.exists():
        if _canonical(read_json(path)) != _canonical(record):
            raise ValueError("An existing artifact reference has different content")
    else:
        write_json(path, record, compressed=True, compact=True)
    return {"artifact_format": "json_record_v1", "artifact": path.relative_to(root).as_posix(),
            "member": []}


def resolve_reference(root, ref):
    if not isinstance(ref, dict):
        raise ValueError("Artifact reference must be an object")
    root, path = _reference_path(root, ref.get("artifact"))
    format_ = ref.get("artifact_format")
    if format_ == "json_record_v1":
        record = read_json(path)
    elif format_ == "gzip_json":
        if path.suffix != ".gz":
            raise ValueError("Legacy gzip_json reference must name a gzip file")
        record = read_json(path)
    else:
        raise ValueError("Unknown artifact reference format")
    member = ref.get("member", [])
    if not isinstance(member, list):
        raise ValueError("Artifact member must be a list")
    for key in member:
        if isinstance(record, dict) and isinstance(key, str):
            if key not in record:
                raise ValueError("Artifact member key is absent")
            record = record[key]
        elif isinstance(record, list) and type(key) is int and 0 <= key < len(record):
            record = record[key]
        else:
            raise ValueError("Invalid artifact member traversal")
    return record


def iter_jsonl(path):
    """Stream JSONL and resolve referenced records."""
    path = find_artifact(path)
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
                if not isinstance(record, dict):
                    raise ValueError("A JSONL record must be an object")
                payload = record.get("payload")
                if isinstance(payload, dict) and "record_ref" in payload:
                    complete = resolve_reference(path.parent, payload["record_ref"])
                    if not isinstance(complete, dict) or complete.get("kind") != record.get("kind"):
                        raise ValueError("Referenced event kind differs from its journal")
                    if isinstance(complete.get("payload"), dict) and "record_ref" in complete["payload"]:
                        raise ValueError("Nested journal references are not supported")
                    record = complete
            except (ValueError, TypeError, KeyError, OSError) as error:
                raise ValueError("Invalid artifact at {}:{}".format(path, line_number)) from error
            yield record
