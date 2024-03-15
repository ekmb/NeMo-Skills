ACCOUNT="llmservice_nemo_robustness"
TIME="00:20:00"
PARTITION="batch_block1,batch_block3,batch_block4"
JOB_NAME="convert_nm_hf"
NEMO_SKILLS_CODE="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/code/NeMo-Skills"
MODEL_PATH="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/checkpoints/Mistral_7B_3turns_baseline_backup"
MOUNTS="${NEMO_SKILLS_CODE}:/code,${MODEL_PATH}:/model"

PP=1
TP=4
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=40
MAX_BATCH_SIZE=128
HF_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # Original model's HF name
LOG_DIR="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/results/exp/"
MOUNTS="${NEMO_SKILLS_CODE}:/code,${MODEL_PATH}:/model"
CONVERSION_CMD="--gres=gpu:${TP} --nodes ${PP} --partition batch_block1,batch_block2,batch_block3,batch_block4 nemo_trt.sh"

sbatch \
    --account=${ACCOUNT} \
    --export=MODEL_PATH=${MODEL_PATH},MOUNTS=${MOUNTS},PP=${PP},TP=${TP},LOGS_DIR=${LOGS_DIR},MAX_INPUT_LEN=${MAX_INPUT_LEN},MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN},MAX_BATCH_SIZE=${MAX_BATCH_SIZE},HF_MODEL_NAME=${HF_MODEL_NAME} \
    --time=${TIME} \
    --job-name=${ACCOUNT}-conv \
    --output=${LOG_DIR}/out.log \
    ${CONVERSION_CMD}

