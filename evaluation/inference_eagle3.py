import argparse
import torch
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file
from model.eagle3.ea_model import EaModel
from model.eagle3.kv_cache import initialize_past_key_values
from model.eagle3.utils import *


def ea_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    output_ids, new_token, step, accept_length_list = model.eagenerate(
        torch.as_tensor(input_ids).cuda(),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        log=True,
    )
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Base model path.")
    parser.add_argument("--ea-model-path", type=str, required=True, help="Eagle model path.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--question-num", type=int, default=-1, help="The number of questions to evaluate.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The number of draft tokens used by Eagle.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The tree depth used by Eagle.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k branches used by Eagle.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Eagle generation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument("--benchmark", type=str, default="APIBank", choices=["toolalpaca", "APIBank", "bfcl"], help="The benchmark to evaluate.")
    args = parser.parse_args()

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"output/{args.benchmark}/{args.model_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = EaModel.from_pretrained(
        base_model_path=args.model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = model.get_tokenizer()

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ea_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        question_num=args.question_num,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        benchmark=args.benchmark,
        temperature=args.temperature,
    )

    reorg_answer_file(answer_file)
