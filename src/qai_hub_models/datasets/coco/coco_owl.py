# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.datasets.coco.coco import (
    CocoDataset,
    CocoDatasetClass,
)
from qai_hub_models.datasets.common import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

TEXT_QUERY_TEMPLATE = "a photo of a {}"


class CocoOwlDataset(CocoDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        max_boxes: int = 100,
        processor: Any = None,
        norm_by_max: bool = False,
    ) -> None:
        if input_spec is None:
            raise ValueError("input_spec must be provided; got None.")
        shape = input_spec["pixel_values"][0]
        input_height, input_width = int(shape[2]), int(shape[3])
        owlvit_spec = {"image": ((1, 3, input_height, input_width), "float32")}

        self.norm_by_max = norm_by_max

        super().__init__(
            split=split,
            input_spec=owlvit_spec,  # type: ignore[arg-type]
            max_boxes=max_boxes,
            num_classes=CocoDatasetClass.SUBSET_CLASSES,
        )

        self.input_height = input_height
        self.input_width = input_width

        self.owlprocessor = processor
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda cat: cat["id"])
        text_queries = [TEXT_QUERY_TEMPLATE.format(cat["name"]) for cat in categories]
        text_inputs = self.owlprocessor(text=[text_queries], return_tensors="pt")
        # Full token table: [80, 16] - one row per class
        self._all_input_ids: torch.Tensor = text_inputs["input_ids"].to(torch.int32)
        self._all_attention_mask: torch.Tensor = text_inputs["attention_mask"].to(
            torch.int32
        )
        self._pixel_cache: dict[int, torch.Tensor] = {}

        # ------------------------------------------------------------------ #
        # Build the flat list of (img_idx, class_idx) pairs.
        # Only include pairs where the class actually appears in the image.
        # ------------------------------------------------------------------ #
        self._samples: list[tuple[int, int]] = []
        for img_idx in range(len(self.ids)):
            image_id = self.ids[img_idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            classes_present: set[int] = set()
            for ann in anns:
                cat_id = int(ann["category_id"])
                if cat_id in self.class80_label_map:
                    classes_present.add(self.class80_label_map[cat_id])
            for cls_idx in sorted(classes_present):
                self._samples.append((img_idx, cls_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(  # type: ignore[override]
        self,
        index: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        img_idx, class_idx = self._samples[index]
        image_id = self.ids[img_idx]

        # Load raw PIL image and COCO annotations
        image, target = CocoDetection.__getitem__(self, img_idx)
        src_w, src_h = image.size

        # Image pre-processing via OwlProcessor
        if img_idx not in self._pixel_cache:
            proc_out = self.owlprocessor(images=image, return_tensors="pt")
            self._pixel_cache[img_idx] = proc_out["pixel_values"].squeeze(0)
        pixel_values = self._pixel_cache[img_idx]

        # Single-class text tokens
        input_ids = self._all_input_ids[class_idx : class_idx + 1].squeeze(0)
        attention_mask = self._all_attention_mask[class_idx : class_idx + 1].squeeze(0)
        # Ground-truth boxes for this class only
        max_h_w = max(src_w, src_h)
        max_src_w, max_src_h = (
            (max_h_w, max_h_w) if self.norm_by_max else (src_w, src_h)
        )
        boxes_list: list[list[float]] = []
        for ann in target:
            cat_id = int(ann["category_id"])
            if (
                cat_id in self.class80_label_map
                and self.class80_label_map[cat_id] == class_idx
            ):
                bx, by, bw, bh = ann["bbox"]
                boxes_list.append(
                    [
                        bx / max_src_w,
                        by / max_src_h,
                        (bx + bw) / max_src_w,
                        (by + bh) / max_src_h,
                    ]
                )

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        num_boxes = len(boxes)

        # Pad / truncate to max_boxes
        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        elif num_boxes > self.max_boxes:
            boxes = boxes[: self.max_boxes]
            num_boxes = self.max_boxes
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0.0)

        # All GT labels for this sample are the same class
        labels = torch.full((self.max_boxes,), class_idx, dtype=torch.long)

        # Encode (image_id, class_idx) into a single unique sample ID so the
        # evaluator can match predictions to GT across batches.
        sample_id = image_id * 1000 + class_idx

        return (
            pixel_values,
            input_ids,
            attention_mask,
        ), (
            sample_id,
            self.input_height,
            self.input_width,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description=(
                "COCO val2017 - one (image, class) pair per sample, "
                "single text query per inference"
            ),
        )

    @staticmethod
    def default_samples_per_job() -> int:
        """Default number of samples per on-device inference job."""
        return 100
