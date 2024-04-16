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
import numpy as np
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
# from nemo_skills.inference.compute_metrics import compute_metrics

LOG = logging.getLogger(__file__)


@dataclass
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 512
    repetition_penalty: float = 1.0
    add_special_tokens: bool = False
    process_prediction: bool = True


@dataclass
class GenerateSolutionsConfig:
    """Top-level parameters for the script"""
    # output_file: Optional[str] = None # Where to save the generations
    model_name: str  # for summary csv
    # Inference server configuration {server_params} {error_recovery_params}
    server: dict
    # Sandbox configuration {sandbox_params}
    sandbox: Optional[dict] = None
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
    data_file: Optional[list[str]] = None  # Can directly specify a data file, if using a custom dataset
    data_dir: Optional[str] = None
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


def add_version_to_file(file_name):
    file_folder = os.path.dirname(file_name)
    base_name = os.path.basename(file_name).split('.jsonl')[0]
    pred_files = os.listdir(file_folder)
    pred_files = [f for f in pred_files if f.endswith("_preds.jsonl")]
    pred_files = [f for f in pred_files if base_name in f]
    # version = [int(f.split('_preds.jsonl')[0].split('__v')[-1]) for f in pred_files if '__v' in f]
    # if len(version) > 0:
    #     version = max(version) + 1
    # else:
    #     version = 0
    version = np.random.randint(1, 1000000)
    file_name_tmp = file_name.replace(".jsonl", f"__v{version}_preds.jsonl")

    while os.path.isfile(file_name_tmp):
        version = np.random.randint(1, 10000000)
        file_name_tmp = file_name.replace(".jsonl", f"__v{version}_preds.jsonl")
    file_name = file_name_tmp

    return file_name


@hydra.main(version_base=None, config_name='generation_config', config_path='.')
def generate_solutions(cfg: GenerateSolutionsConfig):
    cfg = OmegaConf.to_object(cfg)

    LOG.info("Config used: %s", cfg)
    sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
    llm = get_model(**cfg.server, sandbox=sandbox)

    if cfg.data_dir:
        data_files = glob.glob(f"{cfg.data_dir}/**/*.jsonl", recursive=True)
        data_files = [df for df in data_files if not df.endswith('_preds.jsonl')]
    elif cfg.data_file:
        data_files = cfg.data_file
    else:
        raise ValueError('Either data_dir or data_files should be specified')

    for data_file_path in data_files:
        print('Doing', data_file_path)
        task = os.path.basename(os.path.dirname(os.path.dirname(data_file_path)))
        data_file = list(open(data_file_path))
        data_file = [json.loads(line) for line in data_file]
        batch = []
        for i in tqdm(range(0, len(data_file), cfg.batch_size)):
            batch = data_file[i:min(i+cfg.batch_size, len(data_file))]
            prompts = [i['input'] for i in batch]
            outputs = llm(stop_phrases=SEPARATORS, prompts=prompts, **asdict(cfg.inference))
            for k, o_k in enumerate(outputs):
                data_file[i+k]['pred'] = o_k
        if cfg.save_dir:
            save_dir = cfg.save_dir + '/'
            if cfg.data_dir:
                # prediction will be saved in a folder with original folder structure
                save_dir = os.path.dirname(data_file_path.replace(cfg.data_dir, cfg.save_dir, 1))
        else:
            save_dir = os.path.join(os.path.dirname(data_file_path), 'preds')
        # make a folder with the name of the data file in save dir to keep predictions
        # useful for multiple prediction files
        pred_folder = os.path.join(save_dir, os.path.basename(data_file_path).replace('.jsonl', ''))
        os.makedirs(pred_folder, exist_ok=True)
        pred_save_file = os.path.join(pred_folder, os.path.basename(data_file_path))
        pred_save_file = add_version_to_file(pred_save_file)
        eval_metadata = {'Task': task, 'batch_size': cfg.batch_size, 'model_name': cfg.model_name, **asdict(cfg.inference)}
        print('Saving Manifest to', pred_save_file)
        with open(pred_save_file, "w") as f_out:
            for i, data in enumerate(data_file):
                if i == 0:
                    data['eval_metadata'] = eval_metadata
                f_out.write(json.dumps(data) + "\n")
        # prefix = os.path.basename(pred_save_file).replace(".jsonl", "")
        # compute_metrics(os.path.dirname(pred_save_file), task, prefix, 0, eval_metadata=eval_matadata)
        # collect_results(save_dir)


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
