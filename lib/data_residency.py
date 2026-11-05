"""Keep clinical records on the host while preserving device-side collation."""

import torch


def records_to_device(records, device):
    return [
        tuple(value.to(device) if torch.is_tensor(value) else value for value in record)
        for record in records
    ]


def collate_records_on_device(records, collator, args, device, **kwargs):
    # Normalize and construct patches on the requested device.
    return collator(records_to_device(records, device), args, device, **kwargs)
