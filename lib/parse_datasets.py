import os
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.ushcn import *
from lib.mimic import MIMIC
from lib.person_activity import *
from sklearn import model_selection
from lib.run_paths import resolve_output_dir
from lib.data_residency import collate_records_on_device
from lib.readout_group_folds import (
	attach_group_summary, make_training_group_folds, requested_group_folds, wrap_training_collate,
)


def attach_training_objective(data_objects, train_data, args):
	global_macro = getattr(args, "loss_reduction", "batch_macro") == "global_macro"
	want_prior = (getattr(args, "variant", "legacy") in {"local_residual", "local_gated", "local_adaptive"}
		or getattr(args, "collect_reference_prior", False) or getattr(args, "temporal_correction", False)
		or getattr(args, "ridge_mode", "none") != "none")
	want_variable_counts = getattr(args, "variable_head", "none") != "none"
	want_objective_counts = bool(getattr(args, "collect_objective_counts", False))
	if global_macro or want_prior or want_variable_counts or want_objective_counts:
		counts = torch.zeros(data_objects["input_dim"], dtype=torch.float64)
		value_sum = torch.zeros_like(counts)
		for record in train_data:
			times, mask = record[1], record[3]
			# Match the collator's prefix split, including its existing ordering convention.
			first_target = int((times < args.history).sum().item())
			counts += mask[first_target:].sum(dim=0, dtype=torch.float64).cpu()
			if want_prior:
				values = record[2][first_target:]
				value_sum += torch.where(mask[first_target:] > 0, values, 0.0).sum(dim=0, dtype=torch.float64).cpu()
		if not (counts > 0).any():
			raise ValueError("training data has no forecast targets")
		data_objects["training_target_counts"] = counts
		if global_macro:
			data_objects["loss_reduction"] = "global_macro"
		if want_prior:
			prior = value_sum / counts.clamp_min(1)
			if args.dataset != "ushcn":
				minimum = data_objects["data_min"].detach().cpu().double()
				scale = data_objects["data_max"].detach().cpu().double() - minimum
				scale = scale + (scale == 0) * 1e-8
				prior = (prior - minimum) / scale
			data_objects["training_value_prior"] = torch.where(counts > 0, prior, 0.0)
	return data_objects


def parse_datasets(args, patch_ts=False, length_stat=False):

	device = args.device
	dataset_name = args.dataset
	output_dir = resolve_output_dir(Path(__file__).resolve().parents[1], getattr(args, "output_dir", None))
	cache_root = output_dir / "data_cache" / dataset_name

	if dataset_name in ["physionet", "mimic"]:
		storage_device = device
		if dataset_name == "mimic":
			storage = getattr(args, "dataset_storage", "cpu")
			if storage not in {"cpu", "model"}:
				raise ValueError("dataset_storage must be cpu or model")
			storage_device = torch.device("cpu") if storage == "cpu" else device

		# Records contain (record_id, times, values, mask).
		if dataset_name == "physionet":
			total_dataset = PhysioNet('../data/physionet', quantization = args.quantization,
											download=True, n_samples = args.n, device = device, output_root=cache_root)
		elif dataset_name == "mimic":
			total_dataset = MIMIC('../data/mimic/', n_samples = args.n, device = storage_device, output_root=cache_root)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, time_max = get_data_min_max(seen_data, storage_device) # (n_dim,), (n_dim,)
		data_min, data_max, time_max = (value.to(device) for value in (data_min, data_max, time_max))

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		else:
			collate_fn = variable_time_collate_fn
		if dataset_name == "mimic":
			collate_batch = partial(collate_records_on_device, collator=collate_fn, args=args, device=device,
				data_min=data_min, data_max=data_max, time_max=time_max)
		else:
			collate_batch = partial(collate_fn, args=args, device=device,
				data_min=data_min, data_max=data_max, time_max=time_max)

		grouping = make_training_group_folds(train_data, args)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True,
			collate_fn=wrap_training_collate(lambda batch: collate_batch(batch, data_type="train"), grouping))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_batch(batch, data_type="val"))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_batch(batch, data_type="test"))

		data_objects = {
					"dataset_storage_device": str(storage_device),
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max,
					"data_min": data_min,
					"time_max": time_max
					}

		if(length_stat):
			max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return attach_training_objective(attach_group_summary(data_objects, grouping), train_data, args)

	elif dataset_name == "ushcn":
		args.n_months = 48 # months
		args.pred_window = 1 # predict future one month

		total_dataset = USHCN('../data/ushcn/', n_samples = args.n, device = device, output_root=cache_root)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

		if(patch_ts):
			collate_fn = USHCN_patch_variable_time_collate_fn
		else:
			collate_fn = USHCN_variable_time_collate_fn

		train_data = USHCN_time_chunk(train_data, args, device)
		val_data = USHCN_time_chunk(val_data, args, device)
		test_data = USHCN_time_chunk(test_data, args, device)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		grouping = make_training_group_folds(train_data, args)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True,
			collate_fn=wrap_training_collate(lambda batch: collate_fn(batch, args, device, time_max = time_max), grouping))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max,
					"data_min": data_min,
					"time_max": time_max
					}

		if(length_stat):
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return attach_training_objective(attach_group_summary(data_objects, grouping), train_data, args)
		

	elif dataset_name == "activity":
		args.pred_window = 1000 # predict future 1000 ms

		total_dataset = PersonActivity('../data/activity/', n_samples = args.n, download=True, device = device, output_root=cache_root)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, _ = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)
		time_max = torch.tensor(args.history + args.pred_window)
		print('manual set time_max:', time_max)

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		else:
			collate_fn = variable_time_collate_fn

		original_train_ids = [record[0] for record in train_data] if requested_group_folds(args) else None
		train_data = Activity_time_chunk(train_data, args, device)
		val_data = Activity_time_chunk(val_data, args, device)
		test_data = Activity_time_chunk(test_data, args, device)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		grouping = make_training_group_folds(train_data, args, original_train_ids)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True,
			collate_fn=wrap_training_collate(lambda batch: collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max, time_max = time_max), grouping))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "val",
				data_min = data_min, data_max = data_max, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max,
					"data_min": data_min,
					"time_max": time_max
					}

		if(length_stat):
			max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return attach_training_objective(attach_group_summary(data_objects, grouping), train_data, args)
	
