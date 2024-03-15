#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --overcommit

# ACCOUNT="llmservice_nemo_robustness"
# TIME="00:20:00"
# PARTITION="batch_block1,batch_block3,batch_block4"
# NEMO_SKILLS_CODE="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/code/NeMo-Skills"
# MODEL_PATH="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/checkpoints/Mistral_7B_3turns_baseline_backup"
# MOUNTS="${NEMO_SKILLS_CODE}:/code,${MODEL_PATH}:/model"

# NUM_GPUS=4
# MAX_INPUT_LEN=2048
# MAX_OUTPUT_LEN=40
# MAX_BATCH_SIZE=128
# HF_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # Original model's HF name
# LOGS_DIR="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/results/exp/"

set -x
read -r -d '' nemo_hf <<EOF
    nvidia-smi \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && echo "Converting .nemo to HF" \
    && python nemo_skills/conversion/nemo_to_hf.py \
        --in-path /model/megatron_gpt_sft_aligned-averaged.nemo \
        --out-path /model/mistral_hf \
        --hf-model-name ${HF_MODEL_NAME} \
        --precision bf16 \
        --max-shard-size 10GB
EOF

read -r -d '' hf_trt <<EOF
    nvidia-smi \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && echo "Converting HF to TRT " && \
    python nemo_skills/conversion/hf_to_trtllm.py \
        --model_dir /model/mistral_hf \
        --output_dir /model/trt_tmp \
        --dtype bfloat16 \
        --tp_size ${NUM_GPUS} \
    && trtllm-build \
        --checkpoint_dir /model/trt_tmp \
        --output_dir  /model/model_trt \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --context_fmha enable \
        --paged_kv_cache enable \
        --max_input_len ${MAX_INPUT_LEN} \
        --max_output_len ${MAX_OUTPUT_LEN} \
        --max_batch_size ${MAX_BATCH_SIZE} \
    && cp /model/mistral_hf/tokenizer.model /model/model_trt/tokenizer.model
EOF

srun   --mpi=pmix \
       --container-image=/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-sft-0.2.0.sqsh \
       --container-mounts=${MOUNTS} \
       --nodes ${PP} \
       --cpus-per-task=64 \
       --gpus-per-node ${TP} \
       --output=${LOGS_DIR}/nemo_trt.log \
       bash -c "${nemo_hf}" \
    &
wait $!
srun   --mpi=pmix \
       --container-image=/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-trtllm-0.2.0.sqsh \
       --container-mounts=${MOUNTS} \
       --nodes ${PP} \
       --gpus-per-node ${TP} \
       --cpus-per-task=64 \
       --output=${LOGS_DIR}/hf_trt.log \
       bash -c "${hf_trt}" \
    &
wait $!
