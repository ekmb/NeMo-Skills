MODELS=(Mistral_7B_3turns_neft_alpha_10
Mistral_7B_3turns_neft_alpha_5
Mistral_7B_3turns_emb_noise_1e-5
Mistral_7B_3turns_baseline
Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.9_gamma10
Mistral_7B_3turns_baseline_b
Mistral_7B_3turns_emb_noise_1e-5_steps0.5_0.75_gamma10
Mistral_7B_3turns_adv_training_1e-4
Mistral_7B_3turns_adv_training_1e-5
Mistral_7B_3turns_neft_alpha_1
Mistral_7B_3turns_emb_noise_1e-4)

for MODEL_NAME in "${MODELS[@]}"; do
    bash trt_run_eval.sh $MODEL_NAME
done
