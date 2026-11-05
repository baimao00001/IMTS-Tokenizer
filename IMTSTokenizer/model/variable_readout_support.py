"""Shared integration for single-model variable residual heads and diagnostics."""
import math

import torch

from .VariableReadout import VariableReadout


def initialize_variable_readout(model, args, feature_dim):
    mode = getattr(args, "variable_head", "none")
    if mode not in {"none", "fixed", "learned"}:
        raise ValueError("variable_head must be none, fixed or learned")
    model.variable_readout = None
    model._variable_readout_pending = None
    if mode == "none":
        return
    penalty = float(getattr(args, "variable_head_penalty", .01))
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("variable_head_penalty must be finite and nonnegative")
    model.variable_readout = VariableReadout(
        feature_dim, args.num_var, rank=getattr(args, "variable_head_rank", 4),
        learned_projection=mode == "learned",
        training_counts=getattr(args, "training_target_counts", None))


def apply_variable_readout(model, features, shared, recorder=None):
    capture = recorder is not None or getattr(model, "collect_diagnostic_predictions", False)
    if capture:
        model._diagnostic_predictions["shared_readout"] = shared.detach()
    if recorder is not None:
        _record_shared_head_gradients(shared, features, recorder)
    head = model.variable_readout
    if head is None:
        return shared
    if recorder is not None:
        delta, details = head(features, return_details=True)
    else:
        delta = head(features)
    output = shared + delta
    if model.training and torch.is_grad_enabled() and delta.requires_grad:
        model._variable_readout_pending = (
            head, delta, float(getattr(model.args, "variable_head_penalty", .01)))
    if recorder is not None:
        prefix = "model/variable_head"
        model._diagnostic_target_tensors[prefix] = details
        recorder.scalar(prefix + "/coefficient_definition", "Unbounded signed residual coefficients, not probability gates")
        recorder.tensor(prefix + "/coefficients", head.coefficients, grad=True)
        recorder.tensor(prefix + "/projection", head.projection, grad=True)
        recorder.tensor(prefix + "/delta", delta, grad=True)
        recorder.tensor(prefix + "/support_weights", head.support_weights)
        recorder.tensor(prefix + "/training_target_counts", head.training_counts)
        with torch.no_grad():
            directions = head.normalized_projection().detach().double()
            gram = directions.T @ directions
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
            probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            recorder.tensor(prefix + "/projection_gram", gram)
            recorder.tensor(prefix + "/projection_eigenvalues", eigenvalues)
            recorder.scalar(prefix + "/projection_effective_rank", rank if eigenvalues.sum() > 0 else 0.)
            off_diagonal = ~torch.eye(head.rank, device=gram.device, dtype=torch.bool)
            recorder.tensor(prefix + "/projection_column_cosines", gram, mask=off_diagonal)
    return output


def _record_shared_head_gradients(shared, features, recorder):
    if not shared.requires_grad:
        return
    fixed_features = features.detach()
    prefix = "model/shared_head"
    def shared_gradient(gradient):
        if not recorder.active:
            return None
        with torch.no_grad():
            vectors = _shared_head_gradient_vectors(fixed_features, gradient.detach())
            norms = vectors.norm(dim=-1)
            live = norms > 0
            normalized = vectors / torch.where(live, norms, 1.0).unsqueeze(-1)
            cosines = (normalized @ normalized.T).clamp(-1, 1)
            pairs = live[:, None] & live[None, :]
            pairs &= ~torch.eye(len(norms), device=norms.device, dtype=torch.bool)
            recorder.tensor(prefix + "/gradient_by_variable", vectors)
            recorder.tensor(prefix + "/gradient_norm_by_variable", norms)
            recorder.tensor(prefix + "/gradient_cosines", cosines, mask=pairs)
            recorder.scalar(prefix + "/gradient_norms", norms.cpu().tolist())
            recorder.scalar(prefix + "/gradient_variable_summary",
                            _variable_gradient_summary(vectors, norms, cosines, pairs))
            recorder.scalar(prefix + "/gradient_block_max_feature_elements", 262144)
            recorder.scalar(prefix + "/gradient_conflict_fraction",
                            (cosines[pairs] < 0).float().mean() if pairs.any() else None)
            recorder.scalar(prefix + "/gradient_cancellation_ratio",
                            vectors.sum(0).norm() / norms.sum() if norms.sum() > 0 else None)
            recorder.scalar(prefix + "/gradient_definition",
                            "Actual downstream loss gradient times original shared-head features; includes bias; excludes zero-gradient variables; cancellation ratio=norm(sum gradients)/sum gradient norms")
        return None
    recorder._hooks.append(shared.register_hook(shared_gradient))


@torch.no_grad()
def _variable_gradient_summary(vectors, norms, cosines, pairs):
    partner_counts = pairs.sum(-1)
    denominator = partner_counts.clamp_min(1)
    negative_fraction = ((cosines < 0) & pairs).sum(-1).double() / denominator
    mean_cosine = torch.where(pairs, cosines, 0.0).sum(-1) / denominator
    others = vectors.sum(0, keepdim=True) - vectors
    other_norms = others.norm(dim=-1)
    usable = (norms > 0) & (other_norms > 0)
    normalized = vectors / torch.where(norms > 0, norms, 1.0).unsqueeze(-1)
    other_normalized = others / torch.where(other_norms > 0, other_norms, 1.0).unsqueeze(-1)
    other_cosine = (normalized * other_normalized).sum(-1).clamp(-1, 1)
    packed = torch.stack((partner_counts.double(), negative_fraction, mean_cosine,
                          other_cosine, usable.double()), dim=-1).cpu().tolist()
    return [{"variable": index, "partner_count": int(count),
             "negative_fraction": negative if count else None,
             "mean_cos": mean if count else None,
             "cos_with_sum_others": other if valid else None}
            for index, (count, negative, mean, other, valid) in enumerate(packed)]


@torch.no_grad()
def _shared_head_gradient_vectors(features, gradient, max_elements=262144):
    """Bound feature workspace without materializing a full residual*feature tensor."""
    if features.ndim != 4 or gradient.shape != features.shape[:-1]:
        raise ValueError("Shared-head gradients must match [batch,time,variable,feature]")
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be positive")
    batch, times, variables, width = features.shape
    vectors = torch.zeros(variables, width + 1, dtype=torch.float64, device=features.device)
    feature_chunk = min(width, max_elements)
    variable_chunk = min(variables, max(1, max_elements // max(1, feature_chunk)))
    time_chunk = max(1, max_elements // max(1, variable_chunk * feature_chunk))
    dtype = torch.float64 if features.dtype == torch.float64 or gradient.dtype == torch.float64 else torch.float32
    for batch_index in range(batch):
        for start in range(0, times, time_chunk):
            stop = min(start + time_chunk, times)
            for variable in range(0, variables, max(1, variable_chunk)):
                end_variable = min(variable + variable_chunk, variables)
                local_gradient = gradient[batch_index, start:stop, variable:end_variable].to(dtype=dtype)
                vectors[variable:end_variable, -1] += local_gradient.double().sum(0)
                for coordinate in range(0, width, max(1, feature_chunk)):
                    end_coordinate = min(coordinate + feature_chunk, width)
                    block = features[batch_index, start:stop, variable:end_variable,
                                     coordinate:end_coordinate].to(dtype=dtype)
                    summed = torch.einsum("tv,tvd->vd", local_gradient, block)
                    vectors[variable:end_variable, coordinate:end_coordinate] += summed.double()
    return vectors


def take_variable_readout_terms(model):
    """Return pending readout terms and release the model's references."""
    terms = []
    modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
    for module in modules:
        pending = getattr(module, "_variable_readout_pending", None)
        if pending is not None:
            terms.append(pending)
            module._variable_readout_pending = None
    return terms


def variable_readout_penalty(terms, mask, data_obj=None):
    total = None
    for head, delta, strength in terms:
        batches = None
        if data_obj is not None and data_obj.get("loss_reduction") == "global_macro":
            batches = data_obj["n_train_batches"]
        penalty = head.regularization(mask, delta, training_batches=batches) * strength
        total = penalty if total is None else total + penalty
    return total
