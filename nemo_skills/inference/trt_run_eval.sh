# Function to generate the string to pass the exported variables
function generate_args() {
    local variables=("$@")
    local args_str=""

    for variable in "${variables[@]}"; do
        args_str+=",$variable=${!variable}"
    done

    echo "${args_str:1}"
}

ACCOUNT="llmservice_nemo_robustness"
TIME="00:20:00"
PARTITION="batch_block1,batch_block3,batch_block4"
NEMO_SKILLS_CODE="${HOME_DIR}/code/NeMo-Skills"
MODEL_PATH="${HOME_DIR}/checkpoints/Mistral_7B_3turns_baseline_backup/megatron_gpt_sft_aligned-averaged.nemo"
TRT_PATH="${HOME_DIR}/checkpoints/Mistral_7B_3turns_baseline_backup"
LOGS_DIR="${HOME_DIR}/results/eval/logs"
mkdir -p ${LOGS_DIR}
NEMO_HF=0
HF_TRT=0
RUN_EVAL=1

# Conversion Params
PP=1
TP=4
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=40
MAX_BATCH_SIZE=128
HF_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # Original model's HF name
CONVERSION_ARGS=$(generate_args TRT_PATH NEMO_SKILLS_CODE PP TP LOGS_DIR MAX_INPUT_LEN MAX_OUTPUT_LEN MAX_BATCH_SIZE HF_MODEL_NAME )

hf_trt_dependency=""
if [ "$NEMO_HF" -eq 1 ]; then
    nemo_hf_id=$(sbatch \
        --account=${ACCOUNT} \
        --export=${CONVERSION_ARGS} \
        --time=${TIME} \
        --job-name=${ACCOUNT}-nemo_hf_conv \
        --gres=gpu:${TP} \
        --nodes=${PP} \
        --partition="batch_block1,batch_block2,batch_block3,batch_block4" \
        "01_nemo_hf.sh"
        )
    hf_trt_dependency="--dependency=afterok:$nemo_hf_id"
fi

eval_dependency=""
if [ "$HF_TRT" -eq 1 ]; then
    hf_trt_id=$(sbatch $hf_trt_dependency \
        --account=${ACCOUNT} \
        --export=${CONVERSION_ARGS} \
        --time=${TIME} \
        --job-name=${ACCOUNT}-hf_trt_conv \
        --gres=gpu:${TP} \
        --nodes=${PP} \
        --partition="batch_block1,batch_block2,batch_block3,batch_block4" \
        "02_hf_trt.sh"
        )
    eval_dependency="--dependency=afterok:$hf_trt_id"
fi

# Evaluation Params
DATA_DIR="${HOME_DIR}/data/"
DATA_FILES=("/data/qqp_1k/clean_quot/validation_0.jsonl" )
MODEL_NAME='Mistral_7B_3turns_baseline_backup' # for summary csv
TASK='qqp'
TEMPERATURE=0  # Temperature of 0 means greedy decoding
TOP_K=0
TOP_P=0.95
RANDOM_SEED=0
TOKENS_TO_GENERATE=10
REPETITION_PENALTY=1.0
BATCH_SIZE=100

MOUNTS="${NEMO_SKILLS_CODE}:/code,${MODEL_PATH}:/model,${DATA_DIR}:/data"
EVAL_ARGS=$(generate_args MOUNTS DATA_DIR DATA_FILES  MODEL_NAME TEMPERATURE TOP_K TOP_P RANDOM_SEED TOKENS_TO_GENERATE REPETITION_PENALTY BATCH_SIZE TRT_PATH NEMO_SKILLS_CODE PP TP LOGS_DIR )

sbatch $eval_dependency \
    --account=${ACCOUNT} \
    --export=${EVAL_ARGS} \
    --time=${TIME} \
    --job-name=${ACCOUNT}-eval \
    -o=${LOGS_DIR}/out.log \
    --gres=gpu:${TP} \
    --nodes=${PP} \
    --partition=${PARTITION} \
    "03_run_eval.sh"

