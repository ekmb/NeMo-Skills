# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_2step.json"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)
# 
# GSM8K validation split was used in the experiments


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    data_folder.mkdir(exist_ok=True)
    original_file = str(data_folder / f"original_GSM-IC_2step.jsonl")
    output_file_ic = str(data_folder / f"val_ic.jsonl")
    output_file_orig = str(data_folder / f"val.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    original_entries = set()
    with open(original_file, "rt") as fin, open(output_file_ic, "wt", encoding="utf-8") as fout_ic, open(output_file_orig, "wt", encoding="utf-8") as fout_orig:
        fin_data = json.loads(fin.read())
        for original_entry in fin_data:
            # original entries
            key = (original_entry["original_question"], original_entry["answer"])
            if key not in original_entries:
                original_entries.add(key)
                entry = dict(
                    question=original_entry["original_question"],
                    expected_answer=float(original_entry["answer"].replace(",", "")),
                )
                # converting to int if able to for cleaner text representation
                if int(entry["expected_answer"]) == entry["expected_answer"]:
                    entry["expected_answer"] = int(entry["expected_answer"])
                fout_orig.write(json.dumps(entry) + "\n")

            # entries with irrelevant context
            ic_entry = dict(
                question=original_entry["new_question"],
                expected_answer=float(original_entry["answer"].replace(",", "")),
            )
            # converting to int if able to for cleaner text representation
            if int(ic_entry["expected_answer"]) == ic_entry["expected_answer"]:
                ic_entry["expected_answer"] = int(ic_entry["expected_answer"])
            fout_ic.write(json.dumps(ic_entry) + "\n")
