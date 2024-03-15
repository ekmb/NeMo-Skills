"""
Modified version of https://gitlab-master.nvidia.com/fjia/llm_long_context_eval/-/blob/main/scripts/eval/evaluate.py?ref_type=heads

Get summary.csv with score and null predictions amount.

Running example:
    python evaluate.py \
        --data_dir <path_to_folder_with_jsonl_predictions> \
        --task_name <task_name> \
        --verbose <number_of_lines_to_display>


data_dir should have the following structure:

    data_dir/
        retrieve_kv.jsonl
"""

import re
import os
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
from typing import List


def accuracy_score(prediction, ground_truth):
    return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_accuracy(predictions: List[str], references: List[str]):
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    accuracy = 0
    for prediction, ground_truths in zip(predictions, references):

        accuracy += metric_max_over_ground_truths(
            accuracy_score, prediction, ground_truths
        )
    return round(100.0 * accuracy / len(predictions), 2)


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

TASK_TO_METRICS = {
    "retrieve_kv": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "sst2": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "mrpc": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "qqp": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "mnli": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "qnli": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "rte": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "wnli": {"metric_fn": compute_accuracy, "metric_name": "accuracy",},
    "boolq": {"metric_fn": compute_accuracy, "metric_name": "accuracy",}
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Path to the prediction jsonl files"
)
parser.add_argument(
    "--task_name",
    type=str,
    default="retrieve_kv",
    choices=TASK_TO_METRICS.keys(),
    help="Task name. This is used to look up correct metric",
)
parser.add_argument(
    "--prefix",
    type=str,
    default="",
    help="Prefix of prediction field (usually derived from the input file)",
)
parser.add_argument(
    "--verbose", type=int, default=0, help="Number of lines to display."
)


def postprocess_pred(predict_str: str, task_name: str):
    predict_str = predict_str.strip()

    # Truncate prediction based on Instruction/Dialog template
    predict_str = predict_str.split("<extra_id_1>")[0].strip()

    while predict_str.startswith("`") or predict_str.startswith('"'):
        predict_str = predict_str[1:]

    while predict_str.endswith("`") or predict_str.endswith('"'):
        predict_str = predict_str[:-1]

    predict_str = predict_str.lower()

    delimiters = [" ", ",", "."]

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


def get_pred_and_ref(
    predictions_file: str,
    input_field: str = "input",
    references_field: str = "outputs",
    prediction_field: str = "pred",
    task_name: str = "retrieve_kv",
):
    with open(predictions_file, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    inputs = []
    predicts = []
    references = []

    for line in lines:
        input = line[input_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_name)
        reference = line.get(references_field, [line.get("output", "")])

        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
    return inputs, predicts, references


def run_evaluation_per_task(predictions_file: str, task_name: str, verbose: int = 0):
    inputs, predicts, references = get_pred_and_ref(predictions_file=predictions_file,
                                                    task_name=task_name)

    task_nulls = f"{sum([len(x)==0 for x in predicts])}/{len(predicts)}"
    task_score = TASK_TO_METRICS[task_name]["metric_fn"](predicts, references)

    if verbose != 0:
        print("=" * 40)
        for i, (input, reference, predict) in enumerate(
            zip(inputs, references, predicts)
        ):
            # print(f'Input     : {input}')
            print(
                "\n------------- CORRECT -------------"
                if predict in reference
                else "\n------------- WRONG -------------"
            )
            print(f"Reference : {reference}")
            print(f"Prediction: {predict}")
            # print('=' * 40)
            if i > verbose:
                break

    return task_score, task_nulls


def write_evaluation(results: dict, pred_file: str):
    task = results['Task']
    basename = os.path.basename(pred_file).replace("_preds.jsonl", "")
    try:
        version = int(basename.split("__v")[-1])
    except ValueError:
        version = 0
    basename = basename.split("__v")[0]
    pred_folder = os.path.dirname(pred_file)
    output_file = os.path.join(pred_folder, f"summary-{basename}-{task}.csv")
    df = pd.DataFrame(results, index=[version])
    if os.path.isfile(output_file):
        results_csv = pd.read_csv(output_file)
        df = pd.concat([results_csv, df], axis=0)
        df = df.drop_duplicates(subset=['Task', 'Score', 'Nulls', 'batch_size', 'temperature', 'top_k', 'top_p',
                                    'tokens_to_generate', 'greedy', 'template', 'model_name'])
    df.to_csv(output_file, index=False)
    print("\n=============================================\n")
    print(df)
    print(f"\nSaved results to {output_file}")


def aggregate_chunk(folder):
    jsonl_files = [file for file in os.listdir(folder) if Path(file).suffix == ".jsonl"]
    chunk_files = sorted(
        [file for file in jsonl_files if re.match(r".*-\d+\.jsonl", file)]
    )
    chunk_files_dict = defaultdict(list)
    for file in chunk_files:
        task = "-".join(file.split("-")[:-1])
        chunk_files_dict[task].append(file)

    for task, files in chunk_files_dict.items():
        lines = []
        for file in sorted(files):
            with open(os.path.join(folder, file), "r") as f:
                lines += [json.loads(line) for line in f.readlines()]

        with open(os.path.join(folder, f"{task}.jsonl"), "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")


def compute_metrics(data_dir: str, task_name: str = "retrieve_kv", prefix: str = "",
                    verbose: int = 0, eval_metadata: dict = {}):
    if not os.path.exists(data_dir):
        raise ValueError(f"Prediction folder {data_dir} not found.")

    # Aggregate all prediction files
    aggregate_chunk(data_dir)

    # Get scores and nulls
    pred_file = os.path.join(data_dir, f"{prefix}.jsonl")

    if not os.path.exists(pred_file):
        raise ValueError(f"Prediction file {pred_file} not found.")

    # check if pred_file is empty
    if os.stat(pred_file).st_size == 0:
        raise ValueError(f"Prediction file {pred_file} is empty.")

    task_score, task_nulls = run_evaluation_per_task(
        predictions_file=pred_file, task_name=task_name, verbose=verbose
    )

    for key in ['port', 'host', 'init_timeout', 'model_TRT', 'prompt', 'task']:
        if key in eval_metadata:
            eval_metadata.pop(key)
    eval_metadata['pred_file'] = pred_file
    eval_metadata['date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
    results = {'Task': task_name, 'Score': task_score, 'Nulls': task_nulls, **eval_metadata}
    # Write to csv
    write_evaluation(results, pred_file)


if __name__ == "__main__":
    args = parser.parse_args()
    compute_metrics(args.data_dir, args.task_name, args.prefix, args.verbose)
