# Instructions For Evaluation
First, install the repo by
```
pip install -e .
```

Configure trt_run_eval.sh and 00_launch_eval.sh and run. This will create jsonl manifests containting model predictions.
```
bash 00_launch_eval.sh
```
It can convert the .nemo models into TRT model and run evaluation. 00_launch_eval.sh can be used for evalualution of many models. It will submit separate jobs for each model.

To compute scores, run in NeMo-Skills/nemo_skills/inference
```
python compute_metrics.py --data_dir=SAVE_DIR_FROM_00.sh
```
where data_dir is the same as SAVE_DIR in 00_launch_eval.sh. This will create csv file with scores and prediction parameters next to each manifest's prediction files.

## How to configure arguments
```
NEMO_SKILLS_CODE - path to this repo OR put in "${HOME_DIR}/code/NeMo-Skills"
HF_MODEL_NAME - Model's HF name i.e. "mistralai/Mistral-7B-v0.1"
MODELS_DIR - directory where all models are saved
MODEL_PATH - model is expected to be in ${MODEL_DIR}/${MODEL_NAME}/checkpoints/megatron_gpt_sft_aligned-averaged.nemo"
TRT_PATH - where should TRT files be saved - by default "${PROJECT_DIR}/trt_models/${MODEL_NAME}"
LOGS_DIR - dir to save logs
```
Conversion is done by .nemo->HF->TRT
```
NEMO_HF - convert .nemo to HF format - 1 or 0
HF_TRT - convert HF to TRT - 1 or 0
RUN_EVAL - do evaluation or not - 1 or 0
```
Depending on which flag is 1 or 0, necessary dependency jobs will be created

## Conversion Params
```
PP - number of nodes, default 1 should be ok
TP - number of gpus, default 8 should be ok
MAX_INPUT_LEN - max number of tokens the model can have
MAX_OUTPUT_LEN - max output tokens of models
MAX_BATCH_SIZE - max batch size for model's input, can cause memory issues for bigger models, 512 works for 7B

We don't know how MAX_INPUT_LEN, MAX_OUTPUT_LEN and MAX_BATCH_SIZE affect inference speed of TRT model
```

## Evaluation Params
Currently only DATA_DIR argument works. All files with path like DATA_DIR/some_folders/task_name/one_folder/*.jsonl (e.g. DATA_DIR=/datasets/ and files are in /datasets/glue/qqp/clean/validation_0.jsonl) will be evaluated. task_name should be in generate_solutions.LABEL_TO_ID. Folder structre INSIDE DATA_DIR will be copied into SAVE_DIR and results will be saved in the respective directory of the jsonl in SAVE_DIR (e.g. SAVE_DIR=/preds, result of eval in /preds/glue/qqp/clean/validation_0/)
```
DATA_FILES - is not fully tested, leave ()
DATA_DIR - explained above
SAVE_DIR - explained above
# inference params
TEMPERATURE  # Temperature of 0 means greedy decoding
TOP_K
TOP_P
RANDOM_SEED
TOKENS_TO_GENERATE
REPETITION_PENALTY
```