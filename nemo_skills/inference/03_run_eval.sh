#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --overcommit

MOUNTS="${NEMO_SKILLS_CODE}:/code,${TRT_PATH}:/model,${DATA_DIR}:/data"

set -x

read -r -d '' up_eval <<EOF
    nvidia-smi \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && echo "Starting server" \
    && chmod -R 777 /model \
    && touch /model/server_logs.txt \
    && mpirun -n ${TP} --allow-run-as-root --oversubscribe python nemo_skills/inference/server/serve_trt.py --model_path /model/model_trt/ > /model/server_logs.txt \
    & echo "Waiting for the server to start" \
    && tail -n0 -f /model/server_logs.txt | sed '/Running on all addresses/ q' \
    && echo "Server Started" \
    && cd /code \
    && export PYTHONPATH=\$PYTHONPATH:/code \
    && for DATA_FILE in "${DATA_FILES[@]}"
    do
       python nemo_skills/inference/generate_solutions.py \
       data_file="\${DATA_FILE}" \
       server.host=127.0.0.1 \
       model_name=${MODEL_NAME} \
       batch_size=${BATCH_SIZE} \
       task=${TASK} \
       inference.temperature=${TEMPERATURE} \
       inference.top_k=${TOP_K} \
       inference.top_p=${TOP_P} \
       inference.random_seed=${RANDOM_SEED} \
       inference.tokens_to_generate=${TOKENS_TO_GENERATE} \
       inference.repetition_penalty=${REPETITION_PENALTY}
    done \
    && echo "Killing server" \
    && kill %1 \
    && rm /model/server_logs.txt;
EOF

srun   --mpi=pmix \
       --container-image=/lustre/fsw/portfolios/llmservice/users/igitman/llm/images/nemo-skills-trtllm-0.2.0.sqsh \
       --container-mounts=${MOUNTS} \
       --nodes ${PP} \
       --gpus-per-node ${TP} \
       -o "${LOGS_DIR}/eval-slurm-%j-%n.out" \
       --cpus-per-task=64 \
       bash -c "${up_eval}" \
    &
wait $!
