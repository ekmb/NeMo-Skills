MODELS=(
    # Mistral_7B_3turns_neft_alpha_10
    # Mistral_7B_3turns_neft_alpha_5
    # Mistral_7B_3turns_emb_noise_1e-5
    Mistral_7B_3turns_baseline
    Mistral_7B_3turns_neft_alpha_5_reimplement
    Mistral_7B_3turns_emb_noise_5e-5
    # Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.9_gamma10
    # Mistral_7B_3turns_baseline_b
    # Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.75_gamma10
    # Mistral_7B_3turns_adv_training_1e-4
    # Mistral_7B_3turns_adv_training_1e-5
    # Mistral_7B_3turns_neft_alpha_1
    # Mistral_7B_3turns_emb_noise_1e-4
)


SAVE_DIR="/lustre/fsw/portfolios/llmservice/users/${USER}/results/robustness_eval/benchmarks_v1_adv_glue_rerun_mistral_mixtral_llama"
NEMO_HF=0 # set to 1 for new models
HF_TRT=0 # set to 1 for new models
RUN_EVAL=1
TEMPERATURE=0
RANDOM_SEED=0

# DATA_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_robustness/datasets/benchmarks_v1/adv_glue" 
# for MODEL_NAME in "${MODELS[@]}"; do
#     # for RANDOM_SEED in {140..200}; do
#     bash trt_run_eval.sh $MODEL_NAME $DATA_DIR $SAVE_DIR $NEMO_HF $HF_TRT $RUN_EVAL $TEMPERATURE $RANDOM_SEED
#     # done
# done


DATA_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_robustness/datasets/benchmarks_v1_hf/adv_glue" 
for MODEL_NAME in "mistral_7b" "mixtral"; do
    # for RANDOM_SEED in {140..200}; do
    TRT_PATH="/lustre/fs8/portfolios/llmservice/users/gnalbandyan/checkpoints/${MODEL_NAME}"
    bash trt_run_eval.sh $MODEL_NAME $DATA_DIR $SAVE_DIR $NEMO_HF $HF_TRT $RUN_EVAL $TEMPERATURE $RANDOM_SEED $TRT_PATH
    # done
done
