import gc
import math
import numpy as np
import sklearn as sk
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from IMTSTokenizer.model.variable_readout_support import take_variable_readout_terms, variable_readout_penalty
from lib.target_diagnostic_groups import observed_measure_groups

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
	n_data_points = mu_2d.size()[-1]

	if n_data_points > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
		log_prob = gaussian.log_prob(data_2d) 
		log_prob = log_prob / n_data_points 
	else:
		log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
	return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
	n_data_points = masked_data.size()[-1]

	if n_data_points > 0:
		log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
	else:
		log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
	return log_prob



def compute_binary_CE_loss(label_predictions, mortality_label):
	mortality_label = mortality_label.reshape(-1)

	if len(label_predictions.size()) == 1:
		label_predictions = label_predictions.unsqueeze(0)
 
	n_traj_samples = label_predictions.size(0)
	label_predictions = label_predictions.reshape(n_traj_samples, -1)
	
	idx_not_nan = ~torch.isnan(mortality_label)
	if len(idx_not_nan) == 0.:
		print("All are labels are NaNs!")
		ce_loss = torch.Tensor(0.).to(get_device(mortality_label))

	label_predictions = label_predictions[:,idx_not_nan]
	mortality_label = mortality_label[idx_not_nan]

	if torch.sum(mortality_label == 0.) == 0 or torch.sum(mortality_label == 1.) == 0:
		print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

	assert(not torch.isnan(label_predictions).any())
	assert(not torch.isnan(mortality_label).any())

	# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
	mortality_label = mortality_label.repeat(n_traj_samples, 1)
	ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

	# Average over sampled trajectories.
	ce_loss = ce_loss / n_traj_samples
	return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask):
	if (len(label_predictions.size()) == 3):
		label_predictions = label_predictions.unsqueeze(0)

	n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

	# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
	true_label = true_label.repeat(n_traj_samples, 1, 1)

	label_predictions = label_predictions.reshape(n_traj_samples * n_traj * n_tp, n_dims)
	true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

	# choose time points with at least one measurement
	mask = torch.sum(mask, -1) > 0

	# repeat the mask for each label to mark that the label for this time point is present
	pred_mask = mask.repeat(n_dims, 1,1).permute(1,2,0)

	label_mask = mask
	pred_mask = pred_mask.repeat(n_traj_samples,1,1,1)
	label_mask = label_mask.repeat(n_traj_samples,1,1,1)

	pred_mask = pred_mask.reshape(n_traj_samples * n_traj * n_tp,  n_dims)
	label_mask = label_mask.reshape(n_traj_samples * n_traj * n_tp, 1)

	if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
		assert(label_predictions.size(-1) == true_label.size(-1))
		# targets are in one-hot encoding -- convert to indices
		_, true_label = true_label.max(-1)

	res = []
	for i in range(true_label.size(0)):
		pred_masked = torch.masked_select(label_predictions[i], pred_mask[i].bool())
		labels = torch.masked_select(true_label[i], label_mask[i].bool())
	
		pred_masked = pred_masked.reshape(-1, n_dims)

		if (len(labels) == 0):
			continue

		ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
		res.append(ce_loss)

	ce_loss = torch.stack(res, 0).to(get_device(label_predictions))
	ce_loss = torch.mean(ce_loss)
	return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
	# Average within each patient and variable before aggregating variables.
	n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

	res = []
	for i in range(n_traj_samples):
		for k in range(n_traj):
			for j in range(n_dims):
				data_masked = torch.masked_select(data[i,k,:,j], mask[i,k,:,j].bool())
				
				mu_masked = torch.masked_select(mu[i,k,:,j], mask[i,k,:,j].bool())
				# Take mean over the length of sequence
				log_prob = likelihood_func(mu_masked, data_masked, indices = (i,k,j))
				res.append(log_prob)

	res = torch.stack(res, 0).to(get_device(data))
	res = res.reshape((n_traj_samples, n_traj, n_dims))
	# Take mean over the number of dimensions
	res = torch.mean(res, -1)
	res = res.transpose(0,1) # (B, 1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
	
		res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
		res = res.reshape(n_traj_samples, n_traj).transpose(0,1)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std, indices = indices)
		res = compute_masked_likelihood(mu, data, mask, func)
	return res



def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse


def compute_mse(mu, data, mask = None):
	"""Compute mean squared error, optionally over observed entries."""
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		# (n_traj_samples, n_traj, n_timepoints, n_dims) -> (n_traj (bs), 1)
		res = compute_masked_likelihood(mu, data, mask, mse) 
	return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask = None):
	# Compute Poisson likelihood
	# https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
	# Sum log lambdas across all time points
	if mask is None:
		poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
		# Sum over data dims
		poisson_log_l = torch.mean(poisson_log_l, -1)
	else:
		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
		int_lambda = info["int_lambda"]
		f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
		poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
		poisson_log_l = poisson_log_l.permute(1,0)
		
	# poisson_log_l shape: [n_traj_samples, n_traj]
	return poisson_log_l
	

def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape [n_batch, n_tp, n_dim]

	if len(pred_y.shape) == 3: 
		pred_y = pred_y.unsqueeze(dim=0)
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1, n_dim).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1, n_dim).sum(dim=0) # (n_dim, )

	if(reduce == "mean"):
		# Average observed errors within each variable, then across variables.
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / n_avai_var # (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")


def _evaluation_error_sums(truth, pred_y, mask):
	"""Compute per-variable MSE, MAE and MAPE sums with shared intermediates."""
	if len(pred_y.shape) == 3:
		pred_y = pred_y.unsqueeze(dim=0)
	n_dim = pred_y.shape[-1]
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)
	difference = truth_repeated - pred_y
	se_var_sum = ((difference ** 2) * mask).reshape(-1, n_dim).sum(dim=0)
	mask_count = mask.reshape(-1, n_dim).sum(dim=0)
	absolute_error = torch.abs(difference)
	del difference
	ae_var_sum = (absolute_error * mask).reshape(-1, n_dim).sum(dim=0)
	mape_mask = (truth_repeated != 0) * mask
	del mask
	truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
	del truth_repeated
	ape_var_sum = (absolute_error / truth_div * mape_mask).reshape(-1, n_dim).sum(dim=0)
	mask_count_mape = mape_mask.reshape(-1, n_dim).sum(dim=0)
	return se_var_sum, ae_var_sum, ape_var_sum, mask_count, mask_count_mape


def _record_target_tensors(payloads, batch_dict, diagnostics):
	"""Summarize observed-target rows and attach gradient hooks."""
	mask = batch_dict["mask_predicted_data"]
	valid_target = (mask > 0) & torch.isfinite(mask) & torch.isfinite(batch_dict["data_to_predict"])
	for prefix, payload in payloads.items():
		diagnostics.scalar(prefix + "/observed_targets/definition",
			"Finite observed targets only; actual_gradient is captured on the original graph tensor, not a diagnostic slice; hidden geometry samples at most 64 target rows.")
		for name, value in payload.get("observed_measures", {}).items():
			if not torch.is_tensor(value) or value.shape != valid_target.shape:
				raise ValueError("Observed diagnostic measures must match target [B,T,V]")
			diagnostics.tensor(prefix + "/observed_targets/" + name, value, mask=valid_target)
			diagnostics.scalar(prefix + "/observed_targets/" + name + "/groups",
				observed_measure_groups(value, valid_target, batch_dict.get("observed_mask")))
		for name in ("coefficients", "contributions", "hidden"):
			value = payload.get(name)
			if value is None:
				continue
			valid = valid_target
			while valid.ndim < value.ndim - 1:
				valid = valid.unsqueeze(0)
			valid = valid.expand(value.shape[:-1])
			key = prefix + "/observed_targets/" + name
			diagnostics.tensor(key, value, mask=valid.unsqueeze(-1), grad=True)
			labels = tuple(payload.get("branch_names", []))
			if name != "hidden" and value.requires_grad:
				def capture_branches(gradient, key=key, valid=valid, labels=labels, recorder=diagnostics):
					if recorder.active:
						for index in range(gradient.shape[-1]):
							label = labels[index] if index < len(labels) else str(index)
							recorder.tensor(key + "/" + label + "/actual_gradient", gradient.detach()[..., index], mask=valid)
					return None
				diagnostics._hooks.append(value.register_hook(capture_branches))
			with torch.no_grad():
				coordinates = valid.nonzero(as_tuple=True)
				count = coordinates[0].numel()
				if name == "hidden":
					limit = min(count, 64, max(1, int(math.sqrt(diagnostics.max_elements))))
					positions = torch.linspace(0, count - 1, limit, device=value.device).round().long()
					sample = value.detach()[tuple(axis[positions] for axis in coordinates)]
					diagnostics.features(key + "_features", sample)
					diagnostics.scalar(key + "_sampling", {
						"eligible_target_rows": count, "sampled_rows": limit,
						"scheme": "evenly_spaced_valid_target_ranks", "feature_width": value.shape[-1]})
					continue
				variables = coordinates[-1]
				selected = value.detach()[coordinates].double()
				for index in range(value.shape[-1]):
					label = labels[index] if index < len(labels) else str(index)
					# Slices are forward statistics only; backward statistics come from the original tensor hook.
					diagnostics.tensor(key + "/" + label, value.detach()[..., index], mask=valid)
					column = selected[:, index]
					finite = torch.isfinite(column)
					n = torch.bincount(variables[finite], minlength=valid.shape[-1])
					clean = torch.where(finite, column, 0.0)
					summed = torch.bincount(variables, weights=clean, minlength=valid.shape[-1])
					energy = torch.bincount(variables, weights=clean.square(), minlength=valid.shape[-1])
					threshold = payload.get("saturation_threshold") if name == "coefficients" else None
					saturated = None
					if threshold is not None:
						saturated = torch.bincount(variables, weights=(finite & (column.abs() >= threshold)).double(), minlength=valid.shape[-1])
					stats = torch.stack((n.double(), summed, energy)).cpu().tolist()
					saturation_counts = saturated.cpu().tolist() if saturated is not None else None
					by_variable = []
					for variable, (n_value, total, squared) in enumerate(zip(*stats)):
						row = {"variable": variable, "count": int(n_value),
							"mean": total / n_value if n_value else None,
							"rms": math.sqrt(squared / n_value) if n_value else None}
						if saturation_counts is not None:
							row["abs_ge_threshold_fraction"] = saturation_counts[variable] / n_value if n_value else None
						by_variable.append(row)
					diagnostics.scalar(key + "/" + label + "/by_variable", {
						"target_rows": count, "finite_elements": int(n.sum()),
						"saturation_threshold": threshold, "per_variable": by_variable})


def _record_prediction_diagnostics(model, batch_dict, pred_y, diagnostics=None):
	target_tensors = getattr(model, "_diagnostic_target_tensors", {})
	# Release the model's temporary diagnostic references before recording.
	model._diagnostic_target_tensors = {}
	components = getattr(model, "_diagnostic_predictions", {})
	gates = getattr(model, "_diagnostic_gates", {})
	branch_observer = getattr(model, "branch_diagnostics", None)
	if branch_observer is not None:
		if getattr(branch_observer, "accepts_target_tensors", False):
			branch_observer.record(batch_dict, pred_y, components, gates, target_tensors=target_tensors)
		else:
			branch_observer.record(batch_dict, pred_y, components, gates)
	model._diagnostic_predictions = {}
	model._diagnostic_gates = {}
	observer = getattr(model, "prediction_diagnostics", None)
	if observer is not None:
		observer.record(batch_dict, pred_y)
	if diagnostics is None:
		diagnostics = getattr(model, "diagnostics", None)
	if diagnostics is None or not diagnostics.active:
		return
	_record_target_tensors(target_tensors, batch_dict, diagnostics)
	for name in ("observed_data", "observed_tp", "observed_mask", "data_to_predict", "tp_to_predict", "mask_predicted_data"):
		mask = None
		if name == "observed_data":
			mask = batch_dict["observed_mask"]
		elif name == "data_to_predict":
			mask = batch_dict["mask_predicted_data"]
		diagnostics.tensor("batch." + name, batch_dict[name], mask=mask)
	prediction_mask = batch_dict["mask_predicted_data"]
	if pred_y.ndim == 4:
		prediction_mask = prediction_mask.unsqueeze(0).expand_as(pred_y)
	diagnostics.tensor("forecast.prediction", pred_y, mask=prediction_mask, grad=True)
	diagnostics.tensor("forecast.residual", pred_y - batch_dict["data_to_predict"], mask=prediction_mask)
	own_history = None
	if "reference" in gates or any(name in components for name in ("history_mean", "last_observation")):
		history = batch_dict["observed_data"]
		usable = ((batch_dict["observed_mask"] > 0) & torch.isfinite(history)
			& torch.isfinite(batch_dict["observed_tp"]))
		own_history = usable.reshape(history.shape[0], -1, history.shape[-1]).any(dim=1)[:, None, :]
		if pred_y.ndim == 4:
			own_history = own_history.unsqueeze(0)
	for name, part in components.items():
		component_mask = prediction_mask * own_history if name in ("history_mean", "last_observation") else prediction_mask
		diagnostics.tensor("forecast.components/" + name, part, mask=component_mask)
	for name, gate in gates.items():
		gate_mask = prediction_mask * own_history if name == "reference" else prediction_mask
		diagnostics.tensor("forecast.observed_gates/" + name, gate, mask=gate_mask)


def forecast_batch(model, batch_dict):
	arguments = (batch_dict["tp_to_predict"], batch_dict["observed_data"],
		batch_dict["observed_tp"], batch_dict["observed_mask"])
	if getattr(model, "uses_window_start", False):
		return model.forecasting(*arguments, window_start=batch_dict.get("window_start"))
	return model.forecasting(*arguments)


def compute_all_losses(model, batch_dict,data_obj=None, *, metrics=True):
	"""Compute the training loss and optional reporting metrics."""

	pred_y = forecast_batch(model, batch_dict)
	optimization_observer = getattr(model, "optimization_diagnostics", None)
	if optimization_observer is not None:
		optimization_observer.observe_prediction(pred_y, batch_dict)
	readout_terms = take_variable_readout_terms(model)
	_record_prediction_diagnostics(model, batch_dict, pred_y)

	# Compute avg error of each variable first, then compute avg error of all variables
	mse = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MSE", reduce="mean") # a scalar
	rmse = torch.sqrt(mse) if metrics else None
	mae = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="mean") if metrics else None # a scalar

	loss = mse
	if data_obj is not None and data_obj.get("loss_reduction") == "global_macro":
		squared_error, _ = compute_error(batch_dict["data_to_predict"], pred_y,
			mask=batch_dict["mask_predicted_data"], func="MSE", reduce="sum")
		counts = data_obj["training_target_counts"].to(device=pred_y.device, dtype=pred_y.dtype)
		present = counts > 0
		loss = (squared_error[present] / counts[present]).mean() * data_obj["n_train_batches"]
	data_objective = loss
	objective_observer = getattr(model, "objective_diagnostics", None)
	if objective_observer is not None:
		objective_observer.update(batch_dict["mask_predicted_data"], prediction_shape=tuple(pred_y.shape))
	penalty = variable_readout_penalty(readout_terms, batch_dict["mask_predicted_data"], data_obj)
	if penalty is not None:
		loss = loss + penalty
	diagnostics = getattr(model, "diagnostics", None)
	if diagnostics is not None and diagnostics.active:
		diagnostics.scalar("loss/data_objective", data_objective.detach())
		if penalty is not None:
			diagnostics.scalar("loss/variable_head_penalty", penalty.detach())
			diagnostics.scalar("loss/variable_head_reduction", (data_obj or {}).get("loss_reduction", "batch_macro"))
			valid_denominator = torch.isfinite(data_objective.detach()) & (data_objective.detach() > 0)
			diagnostics.scalar("loss/penalty_to_data_ratio",
				penalty.detach() / data_objective.detach() if valid_denominator else None)

	results = {}
	results["loss"] = loss
	if metrics:
		results["mse"] = mse.item()
		results["rmse"] = rmse.item()
		results["mae"] = mae.item()
		if penalty is not None:
			results["variable_head_penalty"] = penalty.detach().item()

	return results

def evaluation(model, dataloader, n_batches, data_obj=None,
	diagnostics=None, epoch=0, step=0, split="val"):

	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0

	if diagnostics is None:
		diagnostics = getattr(model, "diagnostics", None)
	for batch_index in range(n_batches):
		batch_dict = utils.get_next_batch(dataloader)

		if diagnostics is not None:
			diagnostics.begin_batch(split, epoch, batch_index, step)
		try:
			pred_y = forecast_batch(model, batch_dict)
			_record_prediction_diagnostics(model, batch_dict, pred_y, diagnostics)
		finally:
			if diagnostics is not None:
				diagnostics.end_batch()
		
		se_var_sum, ae_var_sum, ape_var_sum, mask_count, mask_count_mape = _evaluation_error_sums(
			batch_dict["data_to_predict"], pred_y, batch_dict["mask_predicted_data"])

		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape

	n_avai_var = torch.count_nonzero(n_eval_samples)
	n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)
	
	# Average observed errors within each variable, then across variables.
	total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["rmse"] = torch.sqrt(total_results["mse"])
	total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape

	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor):
			var = var.item()
		total_results[key] = var

	return total_results

