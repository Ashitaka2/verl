# -*- coding: utf-8 -*-
"""
Convert AI-MO/aimo-validation-amc to VERL parquet in the
single-turn DAPO-style "Answer:" format.
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


def _normalize_answer(ans):
    # AMC labels are integers but can be stored as float or string; normalize to plain string int.
    if isinstance(ans, float) and ans.is_integer():
        ans = int(ans)
    return str(ans).strip()


def make_map_fn(split, data_source, ability="math", reward_style="rule-lighteval/MATH_v2"):
    """Map rows -> VERL rows with DAPO-style 'Answer:' formatting."""
    def _fn(example, idx):
        # Columns per dataset card: problem (string), answer (integer-ish), url (string)
        # (An 'id' may or may not exist; include if present.)
        problem = (example.get("problem", "") or "").strip()
        answer  = _normalize_answer(example.get("answer", ""))

        user_msg = f"{INSTR_HEADER}\n\n{problem}\n\n{INSTR_FOOTER}"

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": user_msg}],
            "ability": ability,
            "reward_model": {
                "style": reward_style,    # e.g., "rule" or "rule-lighteval/MATH_v2"
                "ground_truth": answer,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "raw_problem": problem,
                "url": example.get("url", None),
                "id": example.get("id", None),
                "answer_format": 'final line starts with "Answer:"',
            },
        }
        return data
    return _fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="~/data/amc_aimo_dapo_style")
    ap.add_argument("--hdfs_dir", default=None)
    ap.add_argument(
        "--reward_style",
        default="rule",
        help='Default mimics common AIME/AMC scorers; set to "rule" for exact-match.',
    )
    args = ap.parse_args()

    hf_path = "AI-MO/aimo-validation-amc"
    print(f"Loading {hf_path} ...", flush=True)
    ds = datasets.load_dataset(hf_path)  # single 'train' split (83 rows)

    # Keep a concise key; change if your scorer relies on specific names.
    data_source = "amc"

    train = ds["train"].map(
        function=make_map_fn(
            split="train",
            data_source=data_source,
            reward_style=args.reward_style,
        ),
        with_indices=True,
        remove_columns=ds["train"].column_names,  # <-- drop original {problem, answer, url, id, ...}
    )

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
