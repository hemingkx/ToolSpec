import json
import argparse
from transformers import AutoTokenizer
import numpy as np


def speed(jsonl_file, jsonl_file_base, tokenizer, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"]['new_tokens'])
        times = sum(datapoint["choices"]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"]['accept_lengths'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        tokens=sum(datapoint["choices"]['new_tokens'])
        times = sum(datapoint["choices"]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    if report:
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        default='./output/APIBank/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct-toolspec-float16.jsonl', # Qwen2.5-7B-Instruct-sps-0.5B-float16-temp-0.0
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default='./output/APIBank/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct-vanilla-float16-temp-0.0.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default='/root/autodl-tmp/Llama-3.2-3B-Instruct',
        type=str,
        help="The file path of evaluated baseline.",
    )

    args = parser.parse_args()

    speed(jsonl_file=args.file_path, jsonl_file_base=args.base_path, tokenizer=args.tokenizer_path)