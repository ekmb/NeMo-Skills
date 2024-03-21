# Function to generate the string to pass the exported variables
function generate_args() {
    local variables=("$@")
    local args_str=""

    for variable in "${variables[@]}"; do
        args_str+=",$variable=${!variable}"
    done

    echo "${args_str:1}"
}

# =============================== UPDATE =================================================
MODEL_NAME=${1:-""} #"Mistral_7B_3turns_emb_noise_1e-4" # for summary csv
echo "Running eval on ${MODEL_NAME}"
ACCOUNT="llmservice_nemo_robustness"
CONVERSION_TIME="00:15:00"
EVAL_TIME="01:00:00" 
PARTITION="batch_block1,batch_block3,batch_block4"
NEMO_SKILLS_CODE="${HOME_DIR}/code/NeMo-Skills"

HF_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # Original model's HF name
MODEL_PATH="/lustre/fsw/portfolios/llmservice/users/mnovikov/results/${MODEL_NAME}/checkpoints/megatron_gpt_sft_aligned-averaged.nemo"
TRT_PATH="${PROJECT_DIR}/trt_models/${MODEL_NAME}"
LOGS_DIR="${HOME_DIR}/results/eval/logs"
mkdir -p ${LOGS_DIR} ${TRT_PATH} 
touch ${TRT_PATH}/server_logs.txt
NEMO_HF=1
HF_TRT=1
RUN_EVAL=1
# Conversion Params
PP=1
TP=8
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=128
MAX_BATCH_SIZE=512
CONVERSION_ARGS=$(generate_args TRT_PATH MODEL_PATH PROJECT_DIR NEMO_SKILLS_CODE PP TP LOGS_DIR MAX_INPUT_LEN MAX_OUTPUT_LEN MAX_BATCH_SIZE HF_MODEL_NAME )

# ========================================================================================

hf_trt_dependency=""
if [ "$NEMO_HF" -eq 1 ]; then
    nemo_hf_id=$(sbatch \
        --account=${ACCOUNT} \
        --export=${CONVERSION_ARGS} \
        --time=${CONVERSION_TIME} \
        --job-name=${ACCOUNT}-nemo_hf_conv \
        --gres=gpu:${TP} \
        --nodes=${PP} \
        --partition="batch_block1,batch_block2,batch_block3,batch_block4" \
        "01_nemo_hf.sh"
        )
    hf_trt_dependency="--dependency=afterok:$nemo_hf_id"
fi
echo "TIME: ${CONVERSION_TIME} and ${EVAL_TIME}"
eval_dependency=""
if [ "$HF_TRT" -eq 1 ]; then
    hf_trt_id=$(sbatch $hf_trt_dependency \
        --account=${ACCOUNT} \
        --export=${CONVERSION_ARGS} \
        --time=${CONVERSION_TIME} \
        --job-name=${ACCOUNT}-hf_trt_conv \
        --gres=gpu:${TP} \
        --nodes=${PP} \
        --partition="batch_block1,batch_block2,batch_block3,batch_block4" \
        "02_hf_trt.sh"
        )
    eval_dependency="--dependency=afterok:$hf_trt_id"
fi

# Evaluation Params
DATA_FILES=() #("{$PROJECT_DIR}/datasets/glue_prompt/mnli/clean/validation_0.jsonl") # DONT TOUCH
DATA_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_robustness/datasets/benchmarks_v1" # /benchmark/task/clean/_.jsonl
SAVE_DIR="/lustre/fsw/portfolios/llmservice/users/ebakhturina/results/robustness_eval_v3"
TEMPERATURE=0  # Temperature of 0 means greedy decoding
TOP_K=0
TOP_P=0.95
RANDOM_SEED=0
TOKENS_TO_GENERATE=10
REPETITION_PENALTY=1.0
BATCH_SIZE=${MAX_BATCH_SIZE}
MOUNTS="${NEMO_SKILLS_CODE}:/code,${MODEL_PATH}:/model"
EVAL_ARGS=$(generate_args MOUNTS PROJECT_DIR SAVE_DIR DATA_DIR DATA_FILES MODEL_NAME TEMPERATURE TOP_K TOP_P RANDOM_SEED TOKENS_TO_GENERATE REPETITION_PENALTY BATCH_SIZE TRT_PATH NEMO_SKILLS_CODE PP TP LOGS_DIR )

sbatch $eval_dependency \
    --account=${ACCOUNT} \
    --export=${EVAL_ARGS} \
    --time=${EVAL_TIME} \
    --job-name=${ACCOUNT}-eval \
    -o=${LOGS_DIR}/out.log \
    --gres=gpu:${TP} \
    --nodes=${PP} \
    --partition=${PARTITION} \
    "03_run_eval.sh"

