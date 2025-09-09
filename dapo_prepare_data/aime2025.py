# -*- coding: utf-8 -*-
"""
Convert yentinglin/aime_2025 to VERL parquet in the
single-turn DAPO-style "Answer:" format you use for training.
"""

import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs  # optional HDFS mirror

INSTR_HEADER = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)
INSTR_FOOTER = 'Remember to put your answer on its own line after "Answer:".'


def make_map_fn(split, data_source, ability="math", reward_style="rule-lighteval/MATH_v2"):
    """Map rows -> VERL rows with DAPO-style 'Answer:' formatting."""
    def _fn(example, idx):
        # Columns on yentinglin/aime_2025: id, problem, answer, solution, url, year
        # (plus a pandas index column we ignore). 30 rows, 'train' split only.
        problem = (example.get("problem", "") or "").strip()
        # AIME labels are integers but may come as strings; keep as string ground truth
        answer = str(example.get("answer", "") or "").strip()

        user_msg = f"{INSTR_HEADER}\n\n{problem}\n\n{INSTR_FOOTER}"

        data = {
            "data_source": data_source,  # keep a concise source key for your pipeline
            "prompt": [{"role": "user", "content": user_msg}],
            "ability": ability,
            "reward_model": {
                "style": reward_style,    # e.g., "rule" or "rule-lighteval/MATH_v2"
                "ground_truth": answer,   # final numeric answer (as string)
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "raw_problem": problem,
                "yt_id": example.get("id", None),
                "url": example.get("url", None),
                "year": (str(example.get("year")) if example.get("year") is not None else None),
                "solution": example.get("solution", None),
                "answer_format": 'final line starts with "Answer:"',
            },
        }
        return data
    return _fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="~/data/aime_2025_dapo_style")
    ap.add_argument("--hdfs_dir", default=None)
    ap.add_argument(
        "--reward_style",
        default="rule",
        help='Default mimics common AIME scorers; set to "rule" for simple exact-match.',
    )
    args = ap.parse_args()

    hf_path = "yentinglin/aime_2025"
    print(f"Loading {hf_path} ...", flush=True)
    ds = datasets.load_dataset(hf_path)  # single 'train' split

    # Use a short source key that plays nicely with downstream scoring utils
    data_source = "aime_2025"

    train = ds["train"].map(
        function=make_map_fn(
            split="train",
            data_source=data_source,
            reward_style=args.reward_style,
        ),
        with_indices=True,
    )

    # NEW: harmonize the source 'year' column to string if it exists
    if "year" in train.column_names:
        train = train.cast_column("year", datasets.Value("string"))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    out_path = os.path.join(local_dir, "train.parquet")
    print(f"Writing {out_path}", flush=True)
    train.to_parquet(out_path)

    if args.hdfs_dir is not None:
        print(f"Mirroring to HDFS: {args.hdfs_dir}", flush=True)
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("Done.")
