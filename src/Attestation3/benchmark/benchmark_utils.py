import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from intervaltree import IntervalTree
from sklearn.metrics import adjusted_rand_score


class Segment(TypedDict):
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class BenchmarkVal:
    name: str
    micro_ari: float
    macro_ari: float


def run_benchmark_for(
    diarization_pipeline: Callable[[str], list[Segment] | Generator[Segment, Any, Any]], name: str
) -> BenchmarkVal:
    with Path("./benchmark_dataset/project-1-at-2025-11-30-21-28-28ca6ab2.json").open("rb") as json_file:
        dataset_json = json.load(json_file)
        results = []

        for dataset_file in dataset_json:
            dataset_file_path = f"./benchmark_dataset{dataset_file['audio']}"

            interval_tree = IntervalTree()
            for seg in dataset_file["label"]:
                interval_tree[seg["start"] : seg["end"]] = seg

            y_true = []
            y_pred = []
            for segment in diarization_pipeline(dataset_file_path):
                start = segment["start"]
                end = segment["end"]

                overlapping = interval_tree[start:end]
                for o in overlapping:
                    if o.data["labels"][0] != "overlap":
                        y_true.append(o.data["labels"][0])
                        y_pred.append(segment["label"])

            if len(y_true) == 0:
                continue

            results.append(
                (
                    adjusted_rand_score(y_true, y_pred),
                    len(y_true),
                )
            )

        weighted_ari = np.average(
            [r[0] for r in results],
            weights=[r[1] for r in results],
        )

        avg_ari = np.average(
            [r[0] for r in results],
        )

        return BenchmarkVal(
            name=name,
            macro_ari=weighted_ari,
            micro_ari=avg_ari,
        )
