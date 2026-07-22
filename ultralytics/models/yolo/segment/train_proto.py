from copy import copy
from pathlib import Path
from typing import Dict, Optional, Union

from ultralytics.data import build_dataloader
from ultralytics.data.triplet_sampler import build_triplet_dataloader
from ultralytics.models import yolo
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import DeBiFormerProtoSegmentationModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class DeBiFormerProtoTrainer(SegmentationTrainer):

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        if mode not in {"train", "val"}:
            raise ValueError(f"Mode must be 'train' or 'val', got {mode}")

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        workers = self.args.workers if mode == "train" else self.args.workers * 2

        if mode == "train":
            if batch_size % 3 != 0:
                raise ValueError(
                    "Cross-plane prototype training requires batch_size % 3 == 0, "
                    f"but got batch_size={batch_size}"
                )

            LOGGER.info(
                "Using Triplet DataLoader for training: "
                f"batch={batch_size}, triplets_per_batch={batch_size // 3}"
            )

            return build_triplet_dataloader(
                dataset=dataset,
                batch=batch_size,
                workers=workers,
                shuffle=True,
                seed=self.args.seed,
                rank=rank,
                drop_last=False,
            )

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=workers,
            shuffle=False,
            rank=rank,
            drop_last=False,
        )

    def get_model(
        self,
        cfg: Optional[Union[Dict, str]] = None,
        weights: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        model = DeBiFormerProtoSegmentationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )

        model.lambda_proto = 0.02

        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "seg_loss",
            "cls_loss",
            "dfl_loss",
            "proto_loss",
        )

        return yolo.segment.SegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
