from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


VALID_PLANES = {"sag", "cor", "tra"}


def parse_case_info(image_path: str) -> dict[str, str]:
    """Parse case, sequence, plane and image IDs from names such as case001_haste-cor_0001.jpg."""
    stem = Path(image_path).stem

    try:
        left, image_id = stem.rsplit("_", 1)
        patient_id, sequence_plane = left.split("_", 1)
        sequence_name, plane = sequence_plane.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(
            "Expected filename format '<patient>_<sequence>-<plane>_<image_id>', "
            f"but got: {stem}"
        ) from exc

    plane = plane.lower()
    if plane not in VALID_PLANES:
        raise ValueError(f"Unknown plane in filename: {stem}")

    return {
        "patient_id": patient_id,
        "sequence_name": sequence_name,
        "plane": plane,
        "image_id": image_id,
        "case_id": f"{patient_id}_{sequence_name}",
    }


def build_foreground_masks(
    masks: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    overlap_mask: bool,
) -> torch.Tensor:
    """Convert YOLO masks to one foreground mask per image, returned as [B, 1, H, W]."""
    masks = masks.float()

    if overlap_mask:
        if masks.ndim != 3 or masks.shape[0] != batch_size:
            raise ValueError(f"Unexpected overlap masks shape: {tuple(masks.shape)}")

        return (masks > 0).float().unsqueeze(1)

    if masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")

    h, w = masks.shape[-2:]
    foreground_masks = masks.new_zeros((batch_size, 1, h, w))
    batch_idx = batch_idx.long().flatten()

    for image_index in range(batch_size):
        instance_masks = masks[batch_idx == image_index]

        if instance_masks.numel() > 0:
            foreground_masks[image_index, 0] = (instance_masks.amax(dim=0) > 0).float()

    return foreground_masks


def extract_foreground_prototypes(
    feature: torch.Tensor,
    foreground_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract L2-normalized foreground prototypes and a valid foreground mask."""
    if feature.ndim != 4:
        raise ValueError(f"Unexpected feature shape: {tuple(feature.shape)}")

    soft_masks = F.interpolate(
        foreground_masks.float(),
        size=feature.shape[-2:],
        mode="area",
    )

    mask_area = soft_masks.sum(dim=(1, 2, 3))
    valid_mask = mask_area > 1e-6

    numerator = (feature * soft_masks).sum(dim=(2, 3))
    denominator = soft_masks.sum(dim=(2, 3)).clamp_min(1e-6)
    prototypes = numerator / denominator
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes, valid_mask


def proto_consistency_loss(
    prototypes: torch.Tensor,
    case_ids: Sequence[str],
    plane_ids: Sequence[str],
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Compare sag/cor/tra foreground prototypes within the same patient and sequence."""
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for index, case_id in enumerate(case_ids):
        if bool(valid_mask[index]):
            groups[str(case_id)][str(plane_ids[index]).lower()].append(index)

    case_losses = []

    for plane_to_indices in groups.values():
        if set(plane_to_indices) != VALID_PLANES:
            continue

        triplet_count = min(len(plane_to_indices[plane]) for plane in VALID_PLANES)

        for triplet_index in range(triplet_count):
            sag = prototypes[plane_to_indices["sag"][triplet_index]]
            cor = prototypes[plane_to_indices["cor"][triplet_index]]
            tra = prototypes[plane_to_indices["tra"][triplet_index]]

            pair_losses = [
                1.0 - F.cosine_similarity(sag.unsqueeze(0), cor.unsqueeze(0)).mean(),
                1.0 - F.cosine_similarity(sag.unsqueeze(0), tra.unsqueeze(0)).mean(),
                1.0 - F.cosine_similarity(cor.unsqueeze(0), tra.unsqueeze(0)).mean(),
            ]
            case_losses.append(torch.stack(pair_losses).mean())

    if not case_losses:
        return prototypes.sum() * 0.0, 0

    return torch.stack(case_losses).mean(), len(case_losses)


def build_image_foreground_masks(
    masks: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Backward-compatible wrapper for older call sites."""
    overlap_mask = masks.ndim == 3 and masks.shape[0] == batch_size
    return build_foreground_masks(masks, batch_idx, batch_size, overlap_mask)


def get_foreground_prototype(
    feature: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible wrapper for older call sites."""
    prototypes, _ = extract_foreground_prototypes(feature, foreground_mask)
    return prototypes
