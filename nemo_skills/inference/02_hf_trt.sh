#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --overcommit

MOUNTS="${NEMO_SKILLS_CODE}:/code,${TRT_PATH}:/model"

set -x
read -r -d '' hf_trt <<EOF
    nvidia-smi \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && echo "Converting HF to TRT " && \
    python nemo_skills/conversion/hf_to_trtllm.py \
        --model_dir /model/model_hf \
        --output_dir /model/trt_tmp \
        --dtype bfloat16 \
        --tp_size ${TP} \
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
    && cp /model/model_hf/tokenizer.model /model/model_trt/tokenizer.model
EOF

srun   --mpi=pmix \
       --container-image=/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-trtllm-0.2.0.sqsh \
       --container-mounts=${MOUNTS} \
       --nodes ${PP} \
       --gpus-per-node ${TP} \
       -o "${LOGS_DIR}/hf_trt-slurm-%j-%n.out" \
       --cpus-per-task=64 \
       bash -c "${hf_trt}" \
    &
wait $!
