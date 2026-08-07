from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plane_consistency import parse_case_info


PLANE_ORDER = ("sag", "cor", "tra")
PLANE_SET = set(PLANE_ORDER)


class TripletSampler(Sampler[int]):
    """Yield consecutive sag/cor/tra indices grouped by patient_id + sequence_name."""

    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        seed: int = 0,
        strict: bool = True,
    ):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.strict = bool(strict)
        self.epoch = 0

        if not hasattr(dataset, "im_files"):
            raise AttributeError(f"TripletSampler requires dataset.im_files, but got dataset type {type(dataset)}")

        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

        for index, image_path in enumerate(dataset.im_files):
            info = parse_case_info(image_path)
            case_id = info["case_id"]
            plane = info["plane"]

            if plane not in PLANE_SET:
                raise ValueError(f"Unknown plane '{plane}' in image: {image_path}")

            groups[case_id][plane].append(index)

        complete_triplets = []
        incomplete_groups = {}

        for case_id, plane_to_indices in groups.items():
            current_planes = set(plane_to_indices)

            if current_planes == PLANE_SET:
                for plane in PLANE_ORDER:
                    plane_to_indices[plane].sort(key=lambda i: str(dataset.im_files[i]))

                triplet_count = min(len(plane_to_indices[plane]) for plane in PLANE_ORDER)
                for triplet_index in range(triplet_count):
                    complete_triplets.append(
                        (f"{case_id}#{triplet_index:04d}", tuple(plane_to_indices[plane][triplet_index] for plane in PLANE_ORDER))
                    )

                if strict and any(len(plane_to_indices[plane]) != triplet_count for plane in PLANE_ORDER):
                    incomplete_groups[case_id] = {plane: len(plane_to_indices[plane]) for plane in PLANE_ORDER}
            else:
                incomplete_groups[case_id] = {plane: len(indices) for plane, indices in plane_to_indices.items()}

        complete_triplets.sort(key=lambda item: item[0])

        if strict and incomplete_groups:
            examples = list(incomplete_groups.items())[:10]
            LOGGER.warning(f"Found {len(incomplete_groups)} incomplete or uneven triplet groups. Examples: {examples}")

        self.triplets = complete_triplets
        self.num_samples = len(self.triplets) * 3

        if not self.triplets:
            raise ValueError("No complete sag/cor/tra triplets were found.")

        if strict and self.num_samples != len(dataset):
            used_indices = {index for _, indices in self.triplets for index in indices}
            unused_indices = sorted(set(range(len(dataset))) - used_indices)
            unused_files = [dataset.im_files[index] for index in unused_indices[:10]]

            LOGGER.warning(
                "Not all dataset images belong to a complete triplet.\n"
                f"dataset images: {len(dataset)}\n"
                f"triplet images: {self.num_samples}\n"
                f"unused examples: {unused_files}"
            )

        LOGGER.info(
            "TripletSampler initialized: "
            f"{len(self.triplets)} triplets, {self.num_samples} images, shuffle={self.shuffle}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        num_triplets = len(self.triplets)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            triplet_order = torch.randperm(num_triplets, generator=generator).tolist()
        else:
            triplet_order = list(range(num_triplets))

        self.epoch += 1

        for triplet_position in triplet_order:
            _, indices = self.triplets[triplet_position]
            for dataset_index in indices:
                yield dataset_index


def build_triplet_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    seed: int = 0,
    rank: int = -1,
    drop_last: bool = False,
):
    """Create a single-GPU dataloader that keeps sag/cor/tra triplets inside the same batch."""
    if rank != -1:
        raise NotImplementedError(
            "The current TripletSampler is implemented for single-GPU training only. Use device=0 before adding DDP."
        )

    batch = min(int(batch), len(dataset))
    if batch <= 0:
        raise ValueError(f"Invalid batch size: {batch}")
    if batch % 3 != 0:
        raise ValueError(f"Triplet training requires batch size divisible by 3, but got batch={batch}.")

    sampler = TripletSampler(dataset=dataset, shuffle=shuffle, seed=seed, strict=True)

    num_devices = torch.cuda.device_count()
    cpu_count = os.cpu_count() or 1
    num_workers = min(cpu_count // max(num_devices, 1), int(workers))

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)

    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )
