
MODEL_DIR="/lustre/fsw/portfolios/llmservice/users/igitman/llm/trt_models/openmath"
MODEL_NAME="codellama-70b"

for DATASET in "gsm-ic-mstep" "gsm-ic-2step"
do
  for SPLIT in "val" "val_ic"
  do
  echo "Running evaluation for ${DATASET} ${SPLIT}"
  python pipeline/run_eval.py \
    --model_path ${MODEL_DIR}/${MODEL_NAME} \
    --server_type tensorrt_llm \
    --output_dir /lustre/fs3/portfolios/llmservice/users/ebakhturina/results/openmath-${MODEL_NAME}-eval-results/${DATASET}_${SPLIT} \
    --benchmarks ${DATASET}:0 \
    --num_gpus 8 \
    --num_nodes 1 \
    +prompt=code_sfted \
    ++prompt.num_few_shots=0 \
    ++split_name=${SPLIT} \
    ++server.max_code_executions=6 \
    ++server.stop_on_code_error=False \
    ++batch_size=64
    
  done
done

# # get metrics
# for DATASET in "gsm-ic-mstep" "gsm-ic-2step"
# do
#   for SPLIT in "test_ic" #"test_ic"
#   do
#   echo "===================="
#   echo "Collecting results for ${DATASET} ${SPLIT}"
#   python pipeline/summarize_results.py /lustre/fs3/portfolios/llmservice/users/ebakhturina/openmath-mistral-7b-eval-results/${DATASET}_${SPLIT}
#   echo "===================="     
#   done
# done