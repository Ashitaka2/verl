# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert open-r1/DAPO-Math-17k-Processed to VERL single-turn parquet.

Source features (per dataset card):
- prompt: str  (already includes the "Solve... Answer:" instruction)
- solution: str
- data_source: str
- source_prompt: List[{content:str, role:str}]
- ability: str
- reward_model: {"ground_truth": str, "style": str}
- extra_info: {"index": str}  # uuid-like

Configs: "all", "en", "cn". Each has only a 'train' split.  (We default to 'en'.)
"""

import argparse
import os
import sys
from typing import Any, Dict

import datasets
from verl.utils.hdfs_io import copy, makedirs  # optional HDFS mirror


HF_DATASET = "open-r1/DAPO-Math-17k-Processed"


def make_map_fn(split: str, fallback_source: str):
    """Map processed DAPO rows -> VERL single-turn rows."""
    def _fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract with sensible fallbacks
        prompt_text = example.get("prompt", "")
        rm = example.get("reward_model") or {}
        gt = rm.get("ground_truth", example.get("solution", ""))
        # style = rm.get("style", "rule-lighteval/MATH_v2")
        style = rm.get("style", "rule")

        # Normalize ability to lower-case 'math' to match other helpers
        ability_raw = example.get("ability", "math")
        ability = ability_raw.lower() if isinstance(ability_raw, str) else "math"

        # Prefer the dataset-provided data_source; otherwise, use the HF path
        data_source = example.get("data_source") or fallback_source

        # Preserve breadcrumbs
        src_idx = None
        if isinstance(example.get("extra_info"), dict):
            src_idx = example["extra_info"].get("index")

        extra_info = {
            "split": split,
            "index": idx,
            "source_index": src_idx,           # original UUID from processed set
            "solution": example.get("solution"),
            "source_prompt": example.get("source_prompt"),
        }

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": ability,
            "reward_model": {"style": style, "ground_truth": gt},
            "extra_info": extra_info,
        }
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="~/data/dapo_math17k_proc_singleturn")
    ap.add_argument("--hdfs_dir", default=None)
    ap.add_argument("--config", default="en", choices=["en", "cn", "all"],
                    help='Dataset config to load from HF (default: "en").')
    args = ap.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading {HF_DATASET} (config={args.config}) ...", flush=True)
    ds = datasets.load_dataset(HF_DATASET, args.config)

    # Only 'train' exists for this dataset; keep the name for consistency
    split = "train"
    if split not in ds:
        raise RuntimeError(f"Expected split '{split}' not found; got keys={list(ds.keys())}")

    mapped = ds[split].map(
        function=make_map_fn(split=split, fallback_source="math_dapo"),
        with_indices=True,
        desc=f"to-verl[{args.config}/{split}]",
    )

    out_path = os.path.join(local_dir, f"{split}.parquet")
    print(f"Writing {out_path}", flush=True)
    mapped.to_parquet(out_path)

    if args.hdfs_dir is not None:
        print(f"Mirroring to HDFS: {args.hdfs_dir}", flush=True)
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[dapo_math17k] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
