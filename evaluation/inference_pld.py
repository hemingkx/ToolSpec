import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.pld.pld import greedy_search_pld


def pld_forward(inputs, model, tokenizer, max_new_tokens):
    input_ids = inputs.input_ids
    output_ids, idx, accept_length_list = model.greedy_search_pld(
              inputs.input_ids,
              attention_mask=inputs.attention_mask,
              stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)]),
              draft_matching_window_size=3,
              draft_num_candidate_tokens=10,
              use_cache=True,
              pad_token_id=tokenizer.pad_token_id,
              eos_token_id=tokenizer.eos_token_id,
              return_dict_in_generate=False)
    input_len = len(input_ids[0])
    new_token = len(output_ids[0][input_len:])
    if tokenizer.eos_token_id in output_ids[0, input_len:].tolist():
        for i, id in enumerate(output_ids[0, input_len:]):
            if id == tokenizer.eos_token_id:
                eos_token_ids_index = i
        invalid_len = len(output_ids[0, input_len:]) - eos_token_ids_index - 1
        if invalid_len > 0:
            accept_length_list[-1] -= invalid_len
            new_token -= invalid_len
    return output_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
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
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
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


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.greedy_search_pld = greedy_search_pld.__get__(model, type(model))

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=pld_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        question_num=args.question_num,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        benchmark=args.benchmark,
    )

    reorg_answer_file(answer_file)