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

import os
from tqdm import tqdm
import pandas as pd
import glob
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.prompt.utils import Prompt, PromptConfig, datasets, prompt_types
from nemo_skills.inference.server.model import ErrorRecoveryConfig, get_model, server_params
from nemo_skills.utils import get_fields_docstring, get_help_message, setup_logging
from nemo_skills.inference.compute_metrics import compute_metrics

LOG = logging.getLogger(__file__)


@dataclass
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 512
    repetition_penalty: float = 1.0


@dataclass
class GenerateSolutionsConfig:
    """Top-level parameters for the script"""
    # output_file: Optional[str] = None # Where to save the generations
    model_name: str  # for summary csv
    # Inference server configuration {server_params} {error_recovery_params}
    server: dict
    # Sandbox configuration {sandbox_params}
    sandbox: dict
    # Prompt configuration.
    # Available pre-configured prompts: {prompt_types}.
    # prompt: PromptConfig = field(default_factory=PromptConfig)
    save_dir: str = ''
    task: str = 'qqp'
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    # Choices: {datasets}.
    # dataset: Optional[str] = None
    # split_name: Optional[str] = None  # Can be train, validation, test or train_full (train + validation)
    data_file: Optional[str] = None  # Can directly specify a data file, if using a custom dataset

    batch_size: int = 16
    # max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    # skip_filled: bool = False  # If True, will skip the generations that are already in the output file
    # if > 0, will skip this many samples from the beginning of the data file.
    # Useful if need to run multiple slurm jobs on the same data file
    # offset: int = 0

    # def __post_init__(self):
    #     """Building data_file from dataset/split_name if not provided directly."""
    #     if self.data_file is not None:
    #         if self.dataset is not None or self.split_name is not None:
    #             raise ValueError("Either `data_fileRANDOM_SEEDname is None:
    #             raise ValueError("Either `data_file` or `dataset` and `split_name` should be provided")
    #         self.data_file = Path(__file__).parents[2] / "datasets" / self.dataset / f"{self.split_name}.jsonl"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


SEPARATORS = ["<extra_id_1>", "\n"]

# source: https://github.com/microsoft/promptbench/blob/main/promptbench/config.py
LABEL_TO_ID = {
    "mmlu": ["A", "B", "C", "D"],
    "sst2": ["negative", "positive"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "mnli_mismatched": ["entailment", "neutral", "contradiction"],
    "mnli_matched": ["entailment", "neutral", "contradiction"],
    "qqp": ["equivalent", "not_equivalent"],
    "qnli": ["entailment", "not_entailment"],
    "rte": ["entailment", "not_entailment"],
    "cola": ["unacceptable", "acceptable"],
    "mrpc": ["equivalent", "not_equivalent"],
    "wnli": ["entailment", "not_entailment"],
    "retrieve_kv": ["negative", "positive"],
    "boolq": ["true", "false"],
}

def postprocess_pred(predict_str: str, task_name: str):
    predict_str = predict_str.strip()

    # Truncate prediction based on Instruction/Dialog template
    for separator in SEPARATORS:
        if separator in predict_str:
            predict_str = predict_str.split(separator)[0].strip()

    predict_str = predict_str.lower()

    delimiters = [" ", ",", "."]
    quotes = ["'", '"', "'", "`", "`"]
    # if LABEL_TO_ID[task_name] doesn't contain any quotes, remove them from predict_str
    if not any([quote in "".join(LABEL_TO_ID[task_name]) for quote in quotes]):
        for quote in quotes:
            predict_str = predict_str.replace(quote, "")

    # remove repeated labels while making sure only the label is repeated
    for label in LABEL_TO_ID[task_name]:
        label_count = predict_str.count(label)
        if label_count > 1:
            for delimiter in delimiters:
                if delimiter in predict_str:
                    repeated_label = delimiter.join([label] * label_count)
                    if repeated_label == predict_str:
                        predict_str = predict_str.split(delimiter)[0]
                        break

    return predict_str


def add_version_to_file(file_name):
    file_folder = os.path.dirname(file_name)
    base_name = os.path.basename(file_name).split('.jsonl')[0]
    pred_files = os.listdir(file_folder)
    pred_files = [f for f in pred_files if f.endswith("_preds.jsonl")]
    pred_files = [f for f in pred_files if base_name in f]
    version = [int(f.split('_preds.jsonl')[0].split('__v')[-1]) for f in pred_files if '__v' in f]
    if len(version) > 0:
        version = max(version) + 1
    else:
        version = 0

    file_name = file_name.replace(".jsonl", f"__v{version}_preds.jsonl")
    return file_name


def collect_results(save_dir):
    summary_csvs = glob.glob(f"{save_dir}/**/summary*.csv", recursive=True)
    results = []
    for summary_file in summary_csvs:
        df = pd.read_csv(summary_file)
        results.append(df)
    df = pd.concat(results)
    df.to_csv(os.path.join(save_dir, "all_summary.csv"), index=False)
    print(f"Saved results to {os.path.join(save_dir, 'all_summary.csv')}")


@hydra.main(version_base=None, config_name='generation_config', config_path='.')
def generate_solutions(cfg: GenerateSolutionsConfig):
    cfg = OmegaConf.to_object(cfg)

    LOG.info("Config used: %s", cfg)
    sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
    llm = get_model(**cfg.server, sandbox=sandbox)

    data_file = list(open(cfg.data_file))
    data_file = [json.loads(line) for line in data_file]
    batch = []
    for i in tqdm(range(0, len(data_file), cfg.batch_size)):
        batch = data_file[i:min(i+cfg.batch_size, len(data_file))]
        prompts = [i['input'] for i in batch]
        outputs = llm(stop_phrases=SEPARATORS, prompts=prompts, **asdict(cfg.inference))
        for k, o_k in enumerate(outputs):
            data_file[i+k]['pred'] = postprocess_pred(o_k, 'qqp')

    if cfg.save_dir:
        save_dir = cfg.save_dir
    else:
        save_dir = os.path.join(os.path.dirname(cfg.data_file), 'preds')
    os.makedirs(save_dir, exist_ok=True)
    # make a folder with the name of the data file in save dir to keep predictions
    # useful for multiple prediction files
    output_dir = os.path.join(save_dir, os.path.basename(cfg.data_file).replace('.jsonl', ''))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(cfg.data_file))
    output_file = add_version_to_file(output_file)
    with open(output_file, "w") as f_out:
        for i in data_file:
            f_out.write(json.dumps(i) + "\n")

    prefix = os.path.basename(output_file).replace(".jsonl", "")
    subset=['Score', 'Nulls', 'template']
    eval_matadata = {'Task': cfg.task, 'batch_size': cfg.batch_size, 'model_name': cfg.model_name, **asdict(cfg.inference)}
    compute_metrics(os.path.dirname(output_file), 'qqp', prefix, 0, eval_metadata=eval_matadata)
    collect_results(save_dir)

error_recovery_params = '\n' + get_fields_docstring(
    ErrorRecoveryConfig,
    prefix='server.error_recovery.',
    level=2,
)


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    datasets=datasets,
    prompt_types=prompt_types,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
    error_recovery_params=error_recovery_params,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate_solutions()
