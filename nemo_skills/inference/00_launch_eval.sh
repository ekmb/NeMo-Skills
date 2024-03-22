MODELS=(
    Mistral_7B_3turns_neft_alpha_10
    Mistral_7B_3turns_neft_alpha_5
    Mistral_7B_3turns_emb_noise_1e-5
    Mistral_7B_3turns_baseline
    Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.9_gamma10
    Mistral_7B_3turns_baseline_b
    Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.75_gamma10
    Mistral_7B_3turns_adv_training_1e-4
    Mistral_7B_3turns_adv_training_1e-5
    Mistral_7B_3turns_neft_alpha_1
    Mistral_7B_3turns_emb_noise_1e-4
)


DATA_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_robustness/datasets/benchmarks_v1/glue" 
SAVE_DIR="/lustre/fsw/portfolios/llmservice/users/ebakhturina/results/robustness_eval_v3/glue"
NEMO_HF=0
HF_TRT=0
RUN_EVAL=1

for MODEL_NAME in "${MODELS[@]}"; do
    bash trt_run_eval.sh $MODEL_NAME $DATA_DIR $SAVE_DIR $NEMO_HF $HF_TRT $RUN_EVAL 
done
