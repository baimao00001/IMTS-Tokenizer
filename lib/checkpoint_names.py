"""Load checkpoint names from earlier releases without changing tensor values."""
import re
import torch


def _name(value):
    value = value.replace("refinement", "correction").replace("refine", "correct").replace("frozen", "fixed")
    return re.sub(r"(?<![A-Za-z])base(?![A-Za-z])", "core", value)


def load_checkpoint(path, **kwargs):
    checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        return checkpoint
    arguments = checkpoint.get("args")
    if isinstance(arguments, dict):
        checkpoint["args"] = {_name(key): value for key, value in arguments.items()}
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        translated = type(state)((_name(key), value) for key, value in state.items())
        if hasattr(state, "_metadata"):
            translated._metadata = {_name(key): value for key, value in state._metadata.items()}
        checkpoint["state_dict"] = translated
    for field in ("training_method", "training_policy"):
        if isinstance(checkpoint.get(field), str):
            checkpoint[field] = _name(checkpoint[field])
    return checkpoint
