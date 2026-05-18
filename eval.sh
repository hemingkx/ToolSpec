export HF_ENDPOINT='https://hf-mirror.com'
MODEL_PATH=/YOUR_MODEL_PATH/Llama-3.1-8B-Instruct  # Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct
MODEL_NAME=Llama-3.1-8B-Instruct
Eagle3_PATH=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
TEMP=0.0
GPU_DEVICES=0
QUESTION_NUM=-1
BENCHMARK="APIBank"

torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --temperature $TEMP --dtype $torch_dtype --question-num $QUESTION_NUM --benchmark $BENCHMARK
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-pld-${torch_dtype} --dtype $torch_dtype --question-num $QUESTION_NUM --benchmark $BENCHMARK
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --dtype $torch_dtype --question-num $QUESTION_NUM --benchmark $BENCHMARK
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-samd --benchmark $BENCHMARK --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method token_recycle --attn_implementation sdpa --question-num $QUESTION_NUM
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_toolspec --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-toolspec-${torch_dtype} --dtype $torch_dtype --question-num $QUESTION_NUM --benchmark $BENCHMARK
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --model-path $MODEL_PATH --model-name ${MODEL_NAME} --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --benchmark $BENCHMARK --temperature $TEMP --dtype $torch_dtype --question-num $QUESTION_NUM