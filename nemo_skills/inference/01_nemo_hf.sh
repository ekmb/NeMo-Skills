#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --overcommit

MOUNTS="${NEMO_SKILLS_CODE}:/code,${TRT_PATH}:/model"

set -x
read -r -d '' nemo_hf <<EOF
    nvidia-smi \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && echo "Converting .nemo to HF" \
    && python nemo_skills/conversion/nemo_to_hf.py \
        --in-path /model/megatron_gpt_sft_aligned-averaged.nemo \
        --out-path /model/model_hf \
        --hf-model-name ${HF_MODEL_NAME} \
        --precision bf16 \
        --max-shard-size 10GB
EOF

srun   --mpi=pmix \
       --container-image=${PROJECT_DIR}/containers/nemo-skills-sft-0.2.0.sqsh \
       --container-mounts=${MOUNTS} \
       --nodes ${PP} \
        -o "${LOGS_DIR}/nemo_hf-slurm-%j-%n.out" \
       --cpus-per-task=64 \
       --gpus-per-node ${TP} \
       bash -c "${nemo_hf}" \
    &
wait $!
