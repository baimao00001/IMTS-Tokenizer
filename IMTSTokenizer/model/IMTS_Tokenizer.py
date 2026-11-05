import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .variable_readout_support import initialize_variable_readout, apply_variable_readout
from .legacy_source_bias import first_source_bias


sqrt_2pi = (math.sqrt(2 * math.pi))


def _diagnostic(module):
    recorder = getattr(module, "diagnostics", None)
    if recorder is None:
        recorder = getattr(module, "_diagnostics", None)
    return recorder if recorder is not None and recorder.active else None


def _bind_diagnostics(module, recorder, prefix):
    module._diagnostics = recorder
    module._diagnostic_prefix = prefix


def _norm_ratio(numerator, denominator):
    denominator_norm = denominator.detach().float().norm()
    return torch.where(denominator_norm > 0,
                       numerator.detach().float().norm() / denominator_norm,
                       torch.full_like(denominator_norm, float("nan")))


def _record_scale(recorder, prefix, x, times, mask, relative_times, anchors,
                  exponent, logits, weights, tokens):
    """Inspect observed support separately from padded and entirely empty patches."""
    recorder.tensor(prefix + "/anchors", anchors, grad=True,
                    axes=["singleton", "singleton", "anchor", "singleton", "variable"])
    recorder.tensor(prefix + "/relative_times", relative_times, mask=mask > 0)
    recorder.tensor(prefix + "/kernel_exponent", exponent)
    recorder.tensor(prefix + "/kernel_logits", logits, grad=True)
    recorder.tensor(prefix + "/weights", weights, grad=True,
                    axes=["batch", "patch", "anchor", "time", "variable"])
    recorder.distribution(prefix + "/weight_distribution", weights, dim=-2,
                          mask=(mask > 0).unsqueeze(2))
    recorder.tensor(prefix + "/weighted_tokens", tokens, grad=True,
                    axes=["batch", "variable", "patch", "anchor"])
    recorder.features(prefix + "/token_features", tokens)
    with torch.no_grad():
        observed = mask > 0
        count = observed.sum(dim=-2)
        supported = count > 0
        recorder.tensor(prefix + "/observed_count", count)
        recorder.tensor(prefix + "/observed_fraction", count.float() / max(1, observed.shape[-2]))
        recorder.scalar(prefix + "/empty_patch_variable_fraction", (~supported).float().mean())
        recorder.scalar(prefix + "/single_observation_fraction", (count == 1).float().mean())
        if relative_times.shape[-2]:
            minimum = relative_times.masked_fill(~observed, float("inf")).amin(dim=-2)
            maximum = relative_times.masked_fill(~observed, -float("inf")).amax(dim=-2)
        else:
            minimum = maximum = torch.zeros_like(count, dtype=relative_times.dtype)
        span = torch.where(supported, maximum - minimum, torch.zeros_like(minimum))
        recorder.tensor(prefix + "/observed_time_span", span, mask=supported)
        outside = ((anchors.squeeze(-2) < minimum.unsqueeze(2)) |
                   (anchors.squeeze(-2) > maximum.unsqueeze(2)))
        support = supported.unsqueeze(2).expand_as(outside)
        if support.any():
            recorder.scalar(prefix + "/anchors_outside_observed_range_fraction", outside[support].float().mean())
        leakage_parts, center_parts, spread_parts, distance_parts = [], [], [], []
        for start in range(0, weights.shape[2], 8):
            local_weights = weights[:, :, start:start + 8]
            valid_weights = local_weights * observed.unsqueeze(2)
            mass = valid_weights.sum(dim=-2)
            leakage_parts.append((local_weights * (~observed).unsqueeze(2)).sum(dim=-2))
            mass = mass.clamp_min(1e-12)
            center = (valid_weights * times.unsqueeze(2)).sum(dim=-2) / mass
            center_parts.append(center)
            spread = (valid_weights * (times.unsqueeze(2) - center.unsqueeze(-2)).square()).sum(dim=-2) / mass
            spread_parts.append(spread.clamp_min(0).sqrt())
            distance = relative_times.unsqueeze(2) - anchors[:, :, start:start + 8]
            distance_parts.append((valid_weights * distance.abs()).sum(dim=-2) / mass)
        leakage = torch.cat(leakage_parts, dim=2)
        recorder.tensor(prefix + "/mask_leakage_mass", leakage)
        live = supported.unsqueeze(2).expand_as(leakage)
        if live.any():
            recorder.scalar(prefix + "/mask_leakage_nonempty_mean", leakage[live].mean())
        if (~live).any():
            recorder.scalar(prefix + "/mask_mass_on_empty_patches", leakage[~live].mean())
        recorder.tensor(prefix + "/weighted_observation_time", torch.cat(center_parts, dim=2), mask=live)
        recorder.tensor(prefix + "/weighted_time_std", torch.cat(spread_parts, dim=2), mask=live)
        recorder.tensor(prefix + "/weighted_anchor_distance", torch.cat(distance_parts, dim=2), mask=live)
        overflow_threshold = math.log(torch.finfo(exponent.dtype).max / sqrt_2pi)
        recorder.scalar(prefix + "/exponent_overflow_threshold", overflow_threshold)
        recorder.scalar(prefix + "/exponent_overflow_risk_fraction",
                        (exponent > overflow_threshold).float().mean() if exponent.numel() else None)

        _record_weight_geometry(recorder, prefix, weights, mask)


@torch.no_grad()
def _record_weight_geometry(recorder, prefix, weights, mask):
    """Summarize weight geometry by observed-history support."""
    observed = mask > 0
    count = observed.sum(dim=-2)
    # Spread bounded geometry samples across every axis, including late variables.
    def indices(size, limit):
        return torch.linspace(0, size - 1, min(size, limit),
                              dtype=torch.float64, device=weights.device).round().long()

    b_idx, p_idx, a_idx, t_idx, v_idx = [
        indices(size, limit) for size, limit in zip(weights.shape, (2, 4, 32, 256, 8))
    ]
    coordinates = torch.meshgrid(b_idx, p_idx, a_idx, t_idx, v_idx, indexing="ij")
    sampled = weights[coordinates].double()
    sampled_mask = observed[b_idx[:, None, None, None], p_idx[None, :, None, None],
                            t_idx[None, None, :, None], v_idx[None, None, None, :]]
    sampled = (sampled * sampled_mask.unsqueeze(2)).permute(0, 1, 4, 2, 3)
    sampled_count = count[b_idx[:, None, None], p_idx[None, :, None], v_idx[None, None, :]]
    sampled_time_count = sampled_mask.sum(dim=2)
    finite_sample = torch.isfinite(sampled).all(dim=-1)
    sampled = torch.where(torch.isfinite(sampled), sampled, torch.zeros_like(sampled))
    norms = sampled.norm(dim=-1)
    live_sample = finite_sample & (norms > 0)
    normalized = sampled / norms.clamp_min(torch.finfo(sampled.dtype).tiny).unsqueeze(-1)
    normalized = torch.where(live_sample.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    gram = (normalized @ normalized.transpose(-1, -2)).clamp(0, 1)
    k = gram.shape[-1]
    pair_live = live_sample.unsqueeze(-1) & live_sample.unsqueeze(-2)
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=gram.device)
    pairs = pair_live & off_diagonal
    if pairs.any():
        recorder.scalar(prefix + "/sampled_anchor_weight_cosine", gram[pairs].mean())
        recorder.scalar(prefix + "/sampled_near_duplicate_weight_fraction", (gram[pairs] > 0.99).float().mean())
    active_anchors = live_sample.sum(dim=-1)
    rank = active_anchors.float().square() / gram.square().sum(dim=(-1, -2)).clamp_min(1e-12)
    recorder.tensor(prefix + "/sampled_weight_effective_diversity", rank, mask=active_anchors > 0)
    recorder.scalar(prefix + "/similarity_sampled_anchor_count", k)
    recorder.scalar(prefix + "/similarity_sampling", {
        "scheme": "evenly_spaced_each_axis_including_endpoints",
        "axis_names": ["batch", "patch", "anchor", "time", "variable"],
        "population_shape": list(weights.shape),
        "indices": [index.cpu().tolist() for index in (b_idx, p_idx, a_idx, t_idx, v_idx)],
        "group_support": "full_time_observed_count",
        "similarity_support": "sampled_time_columns_only",
    })
    retention = sampled_time_count.double() / sampled_count.clamp_min(1)
    recorder.tensor(prefix + "/sampled_observation_retention_fraction", retention, mask=sampled_count > 0)

    def grouped_similarity(minimum_count, variable_position=None):
        eligible = sampled_count >= minimum_count
        if variable_position is not None:
            selector = torch.zeros_like(eligible)
            selector[..., variable_position] = True
            eligible = eligible & selector
        chosen_pairs = pairs & eligible[..., None, None]
        chosen_rank = eligible & (active_anchors > 0)
        pair_count = int(chosen_pairs.sum().item())
        return {
            "patch_variables": int(eligible.sum().item()),
            "represented_patch_variables": int(chosen_rank.sum().item()),
            "ordered_anchor_pairs": pair_count,
            "anchor_weight_cosine": float(gram[chosen_pairs].mean().item()) if pair_count else None,
            "near_duplicate_weight_fraction": float((gram[chosen_pairs] > .99).double().mean().item()) if pair_count else None,
            "weight_effective_diversity": float(rank[chosen_rank].mean().item()) if chosen_rank.any() else None,
            "observation_retention_fraction": float(retention[eligible].mean().item()) if eligible.any() else None,
        }

    for threshold in (1, 2, 4):
        label = "support_ge_" + str(threshold)
        grouped = grouped_similarity(threshold)
        recorder.scalar(prefix + "/sampled_" + label + "/summary", grouped)
        for metric in ("anchor_weight_cosine", "near_duplicate_weight_fraction", "weight_effective_diversity"):
            recorder.scalar(prefix + "/sampled_" + label + "/" + metric, grouped[metric])
    recorder.scalar(prefix + "/sampled_weight_similarity_by_variable", {
        str(variable): {"support_ge_" + str(threshold): grouped_similarity(threshold, index)
                        for threshold in (1, 2, 4)}
        for index, variable in enumerate(v_idx.cpu().tolist())
    })

    # Entropy uses complete time rows, with a valid-support denominator.
    support_coordinates = (count > 0).nonzero(as_tuple=False)
    entropy_anchors = indices(weights.shape[2], 4)
    row_limit = max(1, 64 // max(1, len(entropy_anchors)))
    entropy_coordinates = support_coordinates[indices(len(support_coordinates), row_limit)]
    if len(entropy_coordinates):
        eb, ep, ev = entropy_coordinates.unbind(dim=1)
        full_time = torch.arange(weights.shape[3], device=weights.device)
        entropy_weights = weights[eb[:, None, None], ep[:, None, None], entropy_anchors[None, :, None],
                                  full_time[None, None, :], ev[:, None, None]].double()
        entropy_mask = observed[eb[:, None], ep[:, None], full_time[None, :], ev[:, None]]
        entropy_weights = entropy_weights * entropy_mask.unsqueeze(1)
        mass = entropy_weights.sum(dim=-1, keepdim=True)
        probabilities = entropy_weights / mass.clamp_min(torch.finfo(torch.double).tiny)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.double).tiny).log()).sum(dim=-1)
        entropy_count = count[eb, ep, ev].unsqueeze(-1).expand_as(entropy)
        valid_entropy = torch.isfinite(entropy_weights).all(dim=-1) & (mass.squeeze(-1) > 0)
        normalized_entropy = entropy / entropy_count.clamp_min(2).double().log()
        for threshold in (2, 4):
            recorder.tensor(prefix + "/sampled_support_ge_" + str(threshold) + "/normalized_entropy",
                            normalized_entropy, mask=valid_entropy & (entropy_count >= threshold))
        recorder.scalar(prefix + "/support_entropy_sampling", {
            "patch_variable_coordinates": entropy_coordinates.cpu().tolist(),
            "anchor_indices": entropy_anchors.cpu().tolist(),
            "time_columns": "all",
            "denominator": "log_full_observed_count_excluding_padding",
            "population_supported_patch_variables": len(support_coordinates),
            "sampled_rows": int(entropy.numel()),
        })

class input_embedding(nn.Module):
    def __init__(self, args):
        super(input_embedding, self).__init__()
        self.num_mean = args.hid_dim
        self.mean = nn.Parameter(torch.randn(1,args.npatch,self.num_mean, 1, args.num_var))
        self.mlp = nn.Linear(args.hid_dim, args.hid_dim)
        self.query = learn_angle(num_var=args.num_var, dim=args.hid_dim)
        self.layer = nn.ModuleList(
            [
                SelfAttentionLayer(args.hid_dim, args.hid_dim*2,num_heads=8,dropout=args.dropout)
                for _ in range(args.num_layers)
            ]
        )
        self.head = nn.Linear(args.hid_dim, 1)
    def forward(self, x, truth_time_steps, mask):
        mask_bool = mask > 0
        selected_elements = truth_time_steps.masked_select(mask)
        mean_time = selected_elements.mean(dim=-2,keepdim=True).repeat(11,x.shape[-2],1)
        inner_time_residul = truth_time_steps - mean_time
        inner_time = inner_time_residul.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        time_result = torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        return self.mlp(smooth_embedding), mean_time
    def forecasting(self, time_predict, x, truth_time_steps, mask):
        inner_time = truth_time_steps.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        time_result = F.softmax(torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi,dim=-2) 
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        key = smooth_embedding.squeeze(dim=-2)
        query_token = self.query(time_predict).flatten(1,2)
        for attn in self.layer:
            query_token, key = attn(query_token, key, key)
        predict = query_token
        return self.head(predict.reshape(x.shape[0],time_predict.shape[-1],41,-1)).squeeze(dim=-1)
    
    
class gate_embedding(nn.Module):
    def __init__(self, args):
        super(gate_embedding, self).__init__()
        self.num_mean = args.hid_dim
        self.args = args
        self.mean = nn.Parameter(torch.randn(1,args.npatch,self.num_mean, 1, args.num_var))
        self.mlp = nn.Linear(args.hid_dim, args.hid_dim)
        self.query = learn_angle(num_var=args.num_var, dim=args.hid_dim)
        self.layer = nn.ModuleList(
            [
                SelfAttentionLayer(args.hid_dim, args.hid_dim*2,num_heads=8,dropout=args.dropout)
                for _ in range(args.num_layers)
            ]
        )
        self.actibation = nn.GELU()
        self.head = nn.Linear(args.hid_dim, 1)
    def forward(self, x, truth_time_steps, mask):
        mask_bool = mask > 0
        selected_elements = truth_time_steps.masked_select(mask)
        mean_time = selected_elements.mean(dim=-2,keepdim=True).repeat(11,x.shape[-2],1)
        inner_time_residul = truth_time_steps - mean_time
        inner_time = inner_time_residul.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        time_result = torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        return self.mlp(smooth_embedding), mean_time
    def forecasting(self, time_predict, x, truth_time_steps, mask):
        inner_time = truth_time_steps.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        inverted_mask = 1 - mask
        time_result = F.softmax(torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi-1e9*inverted_mask.unsqueeze(dim=2).repeat(1,1,self.args.hid_dim,1,1),dim=-2) 
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        key = smooth_embedding.squeeze(dim=-2)
        query_token = self.query(time_predict)
        for attn in self.layer:
            key, _ = attn(key, key, key)
        key = key.unsqueeze(dim=1).repeat(1,query_token.shape[1],1,1)
        predict = query_token * key
        return self.head(predict.reshape(predict.shape[0],time_predict.shape[-1],self.args.num_var,-1)).squeeze(dim=-1)
    
class add_embedding(nn.Module):
    def __init__(self, args):
        super(add_embedding, self).__init__()
        self.args = args
        self.num_mean = args.hid_dim
        self.mean = nn.Parameter(torch.randn(1,args.npatch,self.num_mean, 1, args.num_var))
        self.mlp = nn.Linear(args.hid_dim, args.hid_dim)
        self.query = learn_angle(num_var=args.num_var, dim=10)
        self.layer = nn.ModuleList(
            [
                SelfAttentionLayer(args.hid_dim, args.hid_dim*2,num_heads=8,dropout=args.dropout)
                for _ in range(args.num_layers)
            ]
        )
        self.head = nn.Linear(args.hid_dim+10, 1)
        self.actibation = nn.GELU()
    def forward(self, x, truth_time_steps, mask):
        mask_bool = mask > 0
        selected_elements = truth_time_steps.masked_select(mask)
        mean_time = selected_elements.mean(dim=-2,keepdim=True).repeat(11,x.shape[-2],1)
        inner_time_residul = truth_time_steps - mean_time
        inner_time = inner_time_residul.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        time_result = torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        return self.mlp(smooth_embedding), mean_time
    def forecasting(self, time_predict, x, truth_time_steps, mask):
        inner_time = truth_time_steps.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1) - self.mean.repeat(x.shape[0],1,1,x.shape[-2],1)
        time_result = F.softmax(torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi,dim=-2) 
        smooth_embedding = torch.sum(time_result * x.unsqueeze(dim=2).repeat(1,1,self.num_mean,1,1), dim=-2,keepdim=False).permute(0,3,1,2)
        key = smooth_embedding.squeeze(dim=-2)
        query_token = self.query(time_predict)
        for attn in self.layer:
            key, _ = attn(key, key, key)
        key = key.unsqueeze(dim=1).repeat(1,query_token.shape[1],1,1)
        predict  = torch.cat((key, query_token), dim=-1)
        return self.head(predict.reshape(32,time_predict.shape[-1],self.args.num_var,-1)).squeeze(dim=-1)
    
class IRAM(nn.Module):
    def __init__(self, num_mean, num_var):
        super(IRAM, self).__init__()
        self.mean = nn.Parameter(torch.randn(1, 1, num_mean, 1, num_var))
    
    def forward(self, x):
        return self.mean
    
class IMTS_Tokenizer(nn.Module):
    def __init__(self, args):
        super(IMTS_Tokenizer, self).__init__()
        if args.multi < 1 or args.hid_dim < 1 or args.num_var < 1:
            raise ValueError("multi, hid_dim, and num_var must be positive integers")
        model_dim = args.hid_dim * (2 ** (args.multi - 1)) * args.multi
        if model_dim % 2:
            raise ValueError("IMTS_Tokenizer feature dimension must be even for sinusoidal encoding")
        if args.num_layers > 0 and model_dim % 8:
            raise ValueError("IMTS_Tokenizer feature dimension must be divisible by 8 attention heads")
        if hasattr(args, "npatch") and args.npatch != 2 ** (args.multi - 1):
            raise ValueError("npatch must equal 2 ** (multi - 1) for hierarchical patch merging")
        self.num_mean = args.hid_dim
        self.args = args
        self.multi_mean = nn.ModuleList(
            [
                 IRAM(self.num_mean * (2 **_), args.num_var)
                for _ in range(args.multi)
            ]
        )
        self.query = learn_angle(num_var=args.num_var, dim=args.hid_dim* (2**(args.multi-1))*args.multi)
        self.layer = nn.ModuleList(
            [
                SelfAttentionLayer(args.hid_dim* (2**(args.multi-1))*args.multi, args.hid_dim* (2**(args.multi-1))*args.multi,num_heads=8,dropout=args.dropout)
                for _ in range(args.num_layers)
            ]
        )
        self.actibation = nn.GELU()
        self.head = nn.Linear(args.hid_dim* (2**(args.multi-1))*args.multi, 1)
        initialize_variable_readout(self, args, self.head.in_features)
        self._readout_features = self._readout_query = self._readout_context = None
        normalization = getattr(args, "qk_normalization", "none")
        if normalization not in {"none", "layer", "adaptive"}:
            raise ValueError("qk_normalization must be none, layer or adaptive")
        for layer in self.layer:
            layer.attn.qk_normalization = normalization
            if normalization == "adaptive":
                layer.attn.qk_mix = nn.Parameter(torch.zeros(()))
        self.source_reliability_enabled = bool(getattr(args, "source_reliability", False))
        self.source_reliability_mode = getattr(args, "source_reliability_mode", "gated")
        if self.source_reliability_enabled:
            if self.source_reliability_mode != "first_bias" or not self.layer:
                raise ValueError("Legacy source_reliability requires first_bias with attention layers")
            for name in ("history", "time_max"):
                if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
                    raise ValueError(name + " must be finite and positive for source_reliability")
            self.source_reliability_weight = nn.Parameter(torch.zeros(4))
            self.source_reliability_logits = nn.Parameter(torch.zeros(args.num_var))
            self.uses_window_start = True
    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        self._readout_features = self._readout_query = self._readout_context = None
        self._variable_readout_pending = None
        self._diagnostic_predictions = {}
        self._diagnostic_target_tensors = {}
        if x.ndim != 4 or truth_time_steps.shape != x.shape or mask.shape != x.shape:
            raise ValueError("values, observation times, and mask must have matching [batch, patch, time, variable] shapes")
        if x.shape[1] != 2 ** (self.args.multi - 1) or x.shape[-1] != self.args.num_var:
            raise ValueError("input patch count or variable count does not match model configuration")
        if time_predict.ndim != 2 or time_predict.shape[0] != x.shape[0]:
            raise ValueError("prediction times must be [batch, horizon] with matching batch size")
        recorder = _diagnostic(self)
        _bind_diagnostics(self.query, recorder, "model/query")
        for layer_index, layer in enumerate(self.layer):
            _bind_diagnostics(layer, recorder, "model/layer_{}".format(layer_index))
        if recorder is not None:
            recorder.tensor("model/input/values", x, mask=mask > 0)
            recorder.tensor("model/input/times", truth_time_steps, mask=mask > 0)
            recorder.tensor("model/input/mask", mask)
            recorder.tensor("model/input/prediction_times", time_predict)
            with torch.no_grad():
                adjacent_valid = (mask[:, :, 1:] > 0) & (mask[:, :, :-1] > 0)
                gaps = truth_time_steps[:, :, 1:] - truth_time_steps[:, :, :-1]
                recorder.tensor("model/input/within_patch_time_gaps", gaps, mask=adjacent_valid)
                if adjacent_valid.any():
                    selected_gaps = gaps[adjacent_valid]
                    recorder.scalar("model/input/negative_time_gap_fraction", (selected_gaps < 0).float().mean())
                    recorder.scalar("model/input/duplicate_time_fraction", (selected_gaps == 0).float().mean())
        source_bias = source_supported = None
        if self.source_reliability_enabled:
            source_bias, source_supported = first_source_bias(
                self, truth_time_steps, mask, window_start, recorder)
        embeddings = []
        for i in range(self.args.multi):
            mean_time = self.multi_mean[i].mean.repeat(x.shape[0], x.shape[1], 1, x.shape[-2], 1)
            b_values = torch.arange(truth_time_steps.shape[-3]).repeat(x.shape[0], 1)
            expanded_b_values = b_values.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, truth_time_steps.shape[-2], truth_time_steps.shape[-1]) * (2 **i) /self.args.multi
            expanded_b_values = truth_time_steps - expanded_b_values.to(truth_time_steps.device)
            expanded_b_values = expanded_b_values.to(truth_time_steps.device)
            inner_time = expanded_b_values.unsqueeze(dim=2).repeat(1, 1, self.num_mean * (2**i), 1, 1) - mean_time
            inverted_mask = 1 - mask
            if recorder is None:
                time_result = F.softmax(
                    torch.exp((inner_time ** 2) * 0.5) * sqrt_2pi - 1e9 * inverted_mask.unsqueeze(dim=2).repeat(1, 1, self.num_mean * (2**i), 1, 1),
                    dim=-2)
            else:
                exponent = (inner_time ** 2) * 0.5
                kernel_logits = torch.exp(exponent) * sqrt_2pi - 1e9 * inverted_mask.unsqueeze(dim=2).repeat(1, 1, self.num_mean * (2**i), 1, 1)
                time_result = F.softmax(kernel_logits, dim=-2)
            smooth_embedding = torch.sum(
                time_result * x.unsqueeze(dim=2).repeat(1, 1, self.num_mean * (2**i), 1, 1),
                dim=-2,
                keepdim=False
            ).permute(0, 3, 1, 2)
            embeddings.append(smooth_embedding.view(smooth_embedding.shape[0],smooth_embedding.shape[1],-1))
            if recorder is not None:
                prefix = "model/scale_{}".format(i)
                _record_scale(recorder, prefix, x, truth_time_steps, mask, expanded_b_values,
                              self.multi_mean[i].mean, exponent, kernel_logits, time_result,
                              smooth_embedding)
                recorder.tensor(prefix + "/flat_embedding", embeddings[-1], grad=True)
                del exponent, kernel_logits
            if i != self.args.multi-1:
                x1 = x[:,::2,:,:]
                x2 = x[:,1::2,:,:]
                x = torch.concat((x1, x2), dim=2)
                truth_time_steps_1 = truth_time_steps[:,::2,:,:]
                truth_time_steps_2 = truth_time_steps[:,1::2,:,:]
                truth_time_steps = torch.concat((truth_time_steps_1, truth_time_steps_2), dim=2)
                mask_1 = mask[:,::2,:,:]
                mask_2 = mask[:,1::2,:,:]
                mask = torch.concat((mask_1,mask_2),dim=2)
        key = torch.concat(embeddings, dim=-1)
        if recorder is not None:
            recorder.tensor("model/multiscale/concatenated", key, grad=True)
            recorder.features("model/multiscale/features", key)
            with torch.no_grad():
                energy = torch.stack([part.detach().float().square().sum() for part in embeddings])
                recorder.tensor("model/multiscale/energy_fraction", energy / energy.sum().clamp_min(1e-12))
                for left in range(len(embeddings)):
                    for right in range(left + 1, len(embeddings)):
                        cosine = F.cosine_similarity(embeddings[left].detach().float(), embeddings[right].detach().float(), dim=-1)
                        nonzero = ((embeddings[left].detach().float().norm(dim=-1) > 0) &
                                   (embeddings[right].detach().float().norm(dim=-1) > 0))
                        recorder.tensor("model/multiscale/coordinate_cosine_{}_{}".format(left, right), cosine, mask=nonzero)
        query_token = self.query(time_predict)
        first_bias_effect = source_value_effect = None
        for layer_index, attn in enumerate(self.layer):
            if self.source_reliability_enabled:
                options = {"source_supported": source_supported} if recorder is not None else {}
                if layer_index == 0:
                    options.update(source_logit_bias=source_bias, source_supported=source_supported)
                if recorder is not None:
                    recorder.scalar("model/layer_{}/attention/source_bias_applied".format(layer_index), layer_index == 0)
                key, _ = attn(key, key, key, **options)
                if layer_index == 0 and recorder is not None:
                    first_bias_effect = getattr(attn.attn, "_diagnostic_source_bias_effect", None)
                    source_value_effect = getattr(attn.attn, "_diagnostic_source_value_effect", None)
            else:
                key, _ = attn(key, key, key)
        readout_context = key
        key = key.unsqueeze(dim=1).repeat(1,query_token.shape[1],1,1)
        predict = query_token * key
        output = self.head(predict.reshape(predict.shape[0],time_predict.shape[-1],self.args.num_var,-1)).squeeze(dim=-1)
        output = apply_variable_readout(self, predict, output, recorder)
        for prefix, effects in (("model/first_source_bias", first_bias_effect),
                                ("model/first_source_values", source_value_effect)):
            if effects is not None:
                self._diagnostic_target_tensors[prefix] = {
                    "observed_measures": {name: value[:, None, :].expand_as(output)
                                          for name, value in effects.items()}}
        if recorder is not None:
            recorder.tensor("model/modulation/key", key, grad=True)
            recorder.tensor("model/modulation/query", query_token, grad=True)
            recorder.tensor("model/modulation/product", predict, grad=True)
            recorder.tensor("model/output/prediction", output, grad=True)
            recorder.features("model/modulation/product_features", predict)
            with torch.no_grad():
                recorder.scalar("model/modulation/product_to_key_norm", _norm_ratio(predict, key))
                recorder.scalar("model/modulation/product_to_query_norm", _norm_ratio(predict, query_token))
                recorder.scalar("model/modulation/query_near_zero_fraction", (query_token.abs() < 1e-3).float().mean())
                recorder.scalar("model/modulation/query_negative_fraction", (query_token < 0).float().mean())
                recorder.scalar("model/modulation/query_abs_gt_one_fraction", (query_token.abs() > 1).float().mean())
                contribution_sample = predict.detach()[:2, :32, :8]
                contributions = contribution_sample * self.head.weight.detach()
                recorder.scalar("model/head/contribution_sampling", "first_2_batches_32_times_8_variables_all_features")
                recorder.scalar("model/head/contribution_sample_shape", list(contribution_sample.shape))
                recorder.tensor("model/head/sampled_feature_contributions", contributions)
                recorder.tensor("model/head/bias", self.head.bias)
                width = embeddings[0].shape[-1]
                for scale_index in range(self.args.multi):
                    contribution = contributions[..., scale_index * width:(scale_index + 1) * width].sum(dim=-1)
                    recorder.tensor("model/head/sampled_coordinate_block_{}_contribution".format(scale_index), contribution)
                magnitude = contributions.abs().sum(dim=-1)
                recorder.tensor("model/head/sampled_contribution_cancellation_ratio", 1 - contributions.sum(dim=-1).abs() / magnitude.clamp_min(1e-12), mask=magnitude > 0)
        # Store readout features after the forecast completes.
        if getattr(self, "capture_readout_features", False):
            self._readout_features = predict
        if getattr(self, "capture_readout_components", False):
            self._readout_query, self._readout_context = query_token, readout_context
        return output
        
        
class learn_angle(nn.Module):
    def __init__(self, num_var,dim=64):
        super().__init__()
        self.num_var = num_var
        self.weight = nn.Parameter(torch.randn(self.num_var, int(dim/2)))
        self.dim=dim
        self.actibation = nn.GELU()
        self.mlp=nn.Linear(dim,dim*2)
        self.mlp2=nn.Linear(dim*2,dim)
    def forward(self, position):
        recorder = _diagnostic(self)
        prefix = getattr(self, "_diagnostic_prefix", "model/query")
        batch_size = position.shape[0]
        # Broadcast channel frequencies to [batch, target_time, variable, dim/2].
        freqs = self.weight.unsqueeze(dim=0).unsqueeze(dim=1)
        freqs = freqs.repeat(batch_size, position.shape[1],1,1)
    
        expanded_position = position.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1,1,self.num_var,int(self.dim/2))
        multiplied = torch.mul(expanded_position, freqs)
        sin_mat = torch.sin(multiplied)
        cos_mat = torch.cos(multiplied)

        # Concatenate sine and cosine features along the encoding dimension.
        result = torch.cat((sin_mat, cos_mat), dim=-1)
        if recorder is not None:
            recorder.tensor(prefix + "/frequencies", self.weight, grad=True)
            recorder.tensor(prefix + "/phase", multiplied, grad=True)
            recorder.tensor(prefix + "/sinusoidal_features", result, grad=True)
        result = self.actibation(result)
        if recorder is not None:
            recorder.tensor(prefix + "/sinusoidal_after_gelu", result, grad=True)
        result = self.mlp(result)
        if recorder is not None:
            recorder.tensor(prefix + "/hidden_pre_gelu", result, grad=True)
        result = self.actibation(result)
        if recorder is not None:
            recorder.tensor(prefix + "/hidden_after_gelu", result, grad=True)
        result = self.mlp2(result)

        if recorder is not None:
            recorder.tensor(prefix + "/encoded_time", result, grad=True)
            recorder.features(prefix + "/features", result)
            with torch.no_grad():
                if result.shape[1] > 1:
                    delta_time = position[:, 1:] - position[:, :-1]
                    delta_feature = (result[:, 1:] - result[:, :-1]).float().norm(dim=-1)
                    positive_gap = delta_time > 0
                    recorder.tensor(prefix + "/consecutive_time_feature_change", delta_feature,
                                    mask=positive_gap.unsqueeze(-1))
                    sensitivity = delta_feature / delta_time.abs().unsqueeze(-1).clamp_min(1e-12)
                    recorder.tensor(prefix + "/consecutive_time_sensitivity", sensitivity,
                                    mask=positive_gap.unsqueeze(-1))

        return result



        
        
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.qk_normalization = "none"

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, source_logit_bias=None,
                source_value_gate=None, source_supported=None, target_query_offset=None):
        self._diagnostic_source_bias_effect = None
        self._diagnostic_target_query_effect = None
        self._diagnostic_source_value_effect = None
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        recorder = _diagnostic(self)
        prefix = getattr(self, "_diagnostic_prefix", "model/attention")

        if self.qk_normalization == "layer":
            query = F.layer_norm(query, (self.model_dim,))
            key = F.layer_norm(key, (self.model_dim,))
            if recorder is not None:
                recorder.tensor(prefix + "/normalized_query_input", query, grad=True)
                recorder.tensor(prefix + "/normalized_key_input", key, grad=True)
        elif self.qk_normalization == "adaptive":
            coefficient = self.qk_mix.tanh()
            query = query + coefficient * (F.layer_norm(query, (self.model_dim,)) - query)
            key = key + coefficient * (F.layer_norm(key, (self.model_dim,)) - key)
            if recorder is not None:
                recorder.scalar(prefix + "/qk_mix_definition", "signed tanh coefficient in [-1,1], not a probability gate")
                recorder.tensor(prefix + "/qk_mix", self.qk_mix, grad=True)
                recorder.tensor(prefix + "/qk_coefficient", coefficient, grad=True)
                recorder.tensor(prefix + "/adaptive_query_input", query, grad=True)
                recorder.tensor(prefix + "/adaptive_key_input", key, grad=True)
        query_delta = None
        if target_query_offset is not None:
            if (target_query_offset.shape != query.shape or query.ndim != 3
                    or target_query_offset.device != query.device or target_query_offset.dtype != query.dtype):
                raise ValueError("target_query_offset must match query [batch, target, feature], device and dtype")
            if recorder is not None:
                recorder.tensor(prefix + "/target_query_offset", target_query_offset, grad=True)
                with torch.no_grad():
                    query_delta = F.linear(target_query_offset.detach(), self.FC_Q.weight.detach())
            # Identity changes Q only, after optional input normalization. The
            # outer residual and the K/V paths still receive their original inputs.
            query = query + target_query_offset
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        if query_delta is not None:
            query_delta = torch.cat(torch.split(query_delta, self.head_dim, dim=-1), dim=0)

        if source_value_gate is not None:
            if source_value_gate.shape != (batch_size, src_length) or value.ndim != 3:
                raise ValueError("source_value_gate requires [batch, source] and 3D attention")
            ungated_value = value
            value = value * source_value_gate.repeat(self.num_heads, 1).unsqueeze(-1)
            if recorder is not None:
                recorder.tensor(prefix + "/source_gated_value", value, grad=True)
                recorder.scalar(prefix + "/gated_to_ungated_value_norm", _norm_ratio(value, ungated_value))

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        unbiased_logits = None
        if source_logit_bias is not None:
            if source_logit_bias.shape != (batch_size, src_length) or attn_score.ndim != 3:
                raise ValueError("source_logit_bias requires [batch, source] and 3D attention")
            if recorder is not None and source_value_gate is None and source_supported is not None:
                unbiased_logits = attn_score.detach()
            attn_score = attn_score + source_logit_bias.repeat(self.num_heads, 1).unsqueeze(1)

        if recorder is not None:
            recorder.tensor(prefix + "/query_projection", query, grad=True)
            recorder.tensor(prefix + "/key_projection", key, grad=True)
            recorder.tensor(prefix + "/value_projection", value, grad=True)
            recorder.tensor(prefix + "/logits", attn_score, grad=True)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
            if unbiased_logits is not None:
                unbiased_logits = unbiased_logits.masked_fill(~mask, -torch.inf)

        identity_logits = None
        if query_delta is not None:
            with torch.no_grad():
                identity_logits = attn_score.detach() - (query_delta @ key.detach()) / self.head_dim**0.5
        attn_score = torch.softmax(attn_score, dim=-1)
        if recorder is not None:
            recorder.tensor(prefix + "/weights", attn_score, grad=True)
            recorder.distribution(prefix + "/distribution", attn_score, dim=-1)
            with torch.no_grad():
                by_head = attn_score.reshape(self.num_heads, batch_size, *attn_score.shape[1:])
                if identity_logits is not None:
                    without_identity = identity_logits.softmax(dim=-1)
                    total_variation = (attn_score - without_identity).abs().sum(dim=-1) / 2
                    effects = {
                        "attention_total_variation": total_variation.reshape(self.num_heads, batch_size, tgt_length).mean(0),
                        "query_offset_norm": target_query_offset.detach().norm(dim=-1),
                    }
                    if source_supported is not None:
                        if source_supported.shape != (batch_size, src_length):
                            raise ValueError("source_supported requires [batch, source]")
                        unsupported = (~source_supported.bool()).repeat(self.num_heads, 1).unsqueeze(1)
                        mass_change = ((attn_score - without_identity) * unsupported).sum(-1)
                        effects["no_history_source_mass_change"] = mass_change.reshape(self.num_heads, batch_size, tgt_length).mean(0)
                    self._diagnostic_target_query_effect = effects
                for head_index in range(self.num_heads):
                    recorder.distribution(prefix + "/head_{}/distribution".format(head_index), by_head[head_index], dim=-1)
                    if tgt_length == src_length:
                        diagonal = by_head[head_index].diagonal(dim1=-2, dim2=-1)
                        recorder.scalar(prefix + "/head_{}/self_mass".format(head_index), diagonal.mean())
                        recorder.scalar(prefix + "/head_{}/cross_mass".format(head_index), 1 - diagonal.mean())
                recorder.tensor(prefix + "/received_mass_by_variable", attn_score.sum(dim=-2))
                if source_supported is not None:
                    if source_supported.shape != (batch_size, src_length) or attn_score.ndim != 3:
                        raise ValueError("source_supported requires [batch, source] and 3D attention")
                    supported = source_supported.bool().repeat(self.num_heads, 1)
                    unsupported_mass = (attn_score * (~supported).unsqueeze(1)).sum(dim=-1)
                    recorder.tensor(prefix + "/mass_to_no_history_sources", unsupported_mass)
                    recorder.scalar(prefix + "/no_history_source_fraction", (~supported).float().mean())
                    recorder.scalar(prefix + "/mass_to_no_history_sources_mean", unsupported_mass.mean())
                    recorder.scalar(prefix + "/source_received_probability_by_variable",
                                    attn_score.mean(dim=(0, 1)).tolist())
                    if unbiased_logits is not None:
                        unadjusted = torch.softmax(unbiased_logits, dim=-1)
                        before = (unadjusted * (~supported).unsqueeze(1)).sum(dim=-1)
                        before = before.reshape(self.num_heads, batch_size, tgt_length).mean(dim=0)
                        after = unsupported_mass.reshape(self.num_heads, batch_size, tgt_length).mean(dim=0)
                        support_fraction = source_supported.bool().float().sum(dim=-1) / max(1, src_length)
                        self._diagnostic_source_bias_effect = {
                            "no_history_mass_before": before.detach(),
                            "no_history_mass_after": after.detach(),
                            "bias_effect": (after - before).detach(),
                            "source_support_fraction": support_fraction[:, None].expand(batch_size, tgt_length).detach(),
                        }
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        if recorder is not None and source_supported is not None:
            with torch.no_grad():
                support = source_supported.bool().repeat(self.num_heads, 1).unsqueeze(1)
                observed = (attn_score.detach() * support) @ value.detach()
                no_history = (attn_score.detach() * (~support)) @ value.detach()
                observed = torch.cat(torch.split(observed, batch_size, dim=0), dim=-1)
                no_history = torch.cat(torch.split(no_history, batch_size, dim=0), dim=-1)
                observed = F.linear(observed, self.out_proj.weight.detach()).double()
                no_history = F.linear(no_history, self.out_proj.weight.detach()).double()
                observed_norm = observed.norm(dim=-1)
                no_history_norm = no_history.norm(dim=-1)
                component_sum = observed_norm + no_history_norm
                summed_norm = (observed + no_history).norm(dim=-1)
                cosine_defined = (observed_norm > 0) & (no_history_norm > 0)
                tiny = torch.finfo(torch.float64).tiny
                cosine = ((observed / observed_norm.clamp_min(tiny).unsqueeze(-1))
                          * (no_history / no_history_norm.clamp_min(tiny).unsqueeze(-1))).sum(-1)
                bias_norm = (self.out_proj.bias.detach().double().norm() if self.out_proj.bias is not None
                             else observed_norm.new_zeros(()))
                effects = {
                    "supported_source_norm": observed_norm,
                    "no_history_source_norm": no_history_norm,
                    "summed_source_norm": summed_norm,
                    "no_history_component_norm_share": torch.where(
                        component_sum > 0, no_history_norm / component_sum.clamp_min(tiny), 0.),
                    "component_cosine": torch.where(cosine_defined, cosine.clamp(-1., 1.), 0.),
                    "component_cosine_defined": cosine_defined.double(),
                    "cancellation_fraction": torch.where(
                        component_sum > 0, (1 - summed_norm / component_sum.clamp_min(tiny)).clamp(0., 1.), 0.),
                    "output_bias_norm": bias_norm.expand_as(observed_norm),
                    "attention_output_norm": out.detach().double().norm(dim=-1),
                }
                self._diagnostic_source_value_effect = effects
                recorder.scalar(prefix + "/source_values/definition",
                                "Actual projected V partitioned by original history support, including any existing value gate; out_proj weight applied separately without its bias. Norm shares are not attention probabilities. Zero components give undefined cosine, encoded as0 with a separate flag.")
                for name, tensor in effects.items():
                    recorder.tensor(prefix + "/source_values/" + name, tensor)

        return out
        
class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, source_logit_bias=None,
                source_value_gate=None, source_supported=None, target_query_offset=None):
        # x: (batch_size, length, model_dim)
        recorder = _diagnostic(self)
        prefix = getattr(self, "_diagnostic_prefix", "model/layer")
        _bind_diagnostics(self.attn, recorder, prefix + "/attention")
        if recorder is not None:
            recorder.tensor(prefix + "/input", query, grad=True)
            recorder.features(prefix + "/input_features", query)
        residual = query
        attention_options = dict(source_logit_bias=source_logit_bias,
                                 source_value_gate=source_value_gate, source_supported=source_supported)
        if target_query_offset is not None:
            attention_options["target_query_offset"] = target_query_offset
        out = self.attn(query, key, value, **attention_options)
        out = self.dropout1(out)
        if recorder is not None:
            recorder.tensor(prefix + "/attention_branch", out, grad=True)
            recorder.scalar(prefix + "/attention_branch_to_residual_norm", _norm_ratio(out, residual))
        out = self.ln1(residual + out)
        if recorder is not None:
            recorder.tensor(prefix + "/after_attention_norm", out, grad=True)

        residual = out
        if recorder is None:
            out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        else:
            out = self.feed_forward[0](out)
            # The following ReLU is in-place, so only summarize its input now.
            recorder.tensor(prefix + "/feedforward_pre_relu", out)
            out = self.feed_forward[1](out)
            recorder.tensor(prefix + "/feedforward_relu", out, grad=True)
            with torch.no_grad():
                recorder.scalar(prefix + "/relu_zero_fraction", (out == 0).float().mean())
                inactive_units = (out == 0).reshape(-1, out.shape[-1]).all(dim=0)
                recorder.scalar(prefix + "/relu_inactive_unit_fraction", inactive_units.float().mean())
            out = self.feed_forward[2](out)
        out = self.dropout2(out)
        if recorder is not None:
            recorder.tensor(prefix + "/feedforward_branch", out, grad=True)
            recorder.scalar(prefix + "/feedforward_branch_to_residual_norm", _norm_ratio(out, residual))
        out = self.ln2(residual + out)
        if recorder is not None:
            recorder.tensor(prefix + "/output", out, grad=True)
            recorder.features(prefix + "/output_features", out)
        
        value_res = value 
        value_out = self.feed_forward(value)
        value_out = self.dropout2(value_out)
        value_out = self.ln2(value_res+value_out)
        if recorder is not None:
            recorder.tensor(prefix + "/unused_value_output", value_out, grad=True)
        

        return out, value_out
