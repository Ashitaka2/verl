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
Preprocess the HuggingFaceH4/MATH-500 dataset to VERL parquet format
(DAPO-style instruction: final line must be "Answer: <final>").
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution_from_rationale(solution_str: str) -> str:
    """
    Match the MATH helper behavior: take the last \\boxed{...} from the rationale
    and strip the box. If parsing fails, return the raw string.
    """
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except Exception:
        return solution_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math500")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from Hugging Face...", flush=True)
    dataset = datasets.load_dataset(data_source)  # only provides 'test'

    # MATH-500 only has a 'test' split
    test_dataset = dataset["test"]

    # DAPO-style instruction requiring the final line "Answer: <final>"
    dapo_style_instruction = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer (without quotes) "
        "where $Answer is the answer to the problem.\n\n"
    )
    dapo_style_footer = 'Remember to put your answer on its own line after "Answer:".'

    def make_map_fn(split):
        def process_fn(example, idx):
            # Source columns seen in test.jsonl:
            #   problem, solution, answer, subject, level, unique_id
            question_raw = example.get("problem", "")

            # Build DAPO-style prompt: instruction header -> problem -> reminder footer
            question = f"{dapo_style_instruction}{question_raw}\n\n{dapo_style_footer}"

            # Prefer the provided 'answer' if present; otherwise parse from 'solution'
            provided_answer = example.get("answer", None)
            if provided_answer is not None and str(provided_answer).strip():
                solution = str(provided_answer).strip()
            else:
                solution = extract_solution_from_rationale(example.get("solution", ""))

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    # helpful breadcrumbs
                    "subject": example.get("subject", None),
                    "level": example.get("level", None),
                    "unique_id": example.get("unique_id", None),
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    out_test = os.path.join(local_dir, "test.parquet")
    print(f"Writing {out_test}", flush=True)
    test_dataset.to_parquet(out_test)

    if hdfs_dir is not None:
        print(f"Mirroring to HDFS: {hdfs_dir}", flush=True)
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print("Done.", flush=True)
