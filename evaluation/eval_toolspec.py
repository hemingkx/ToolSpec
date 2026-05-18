import json
import os
import time
import torch
import numpy as np
import shortuuid
from model.toolspec.schema_fsm import SchemaFSM
from tqdm import tqdm

def load_questions(question_num=-1, benchmark="APIBank"):
    questions = []
    count = 0

    if benchmark == "APIBank":
        data_paths = [
            f"./data/API-Bank/level-{level}-api_processed.json"
            for level in ["1", "2", "3"]
        ]
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    for data_path in data_paths:
        datas = json.load(open(data_path, "r", encoding="utf-8"))
        for data in datas:
            question = {
                "question_id": count,
                "messages": [
                    {"role": "system", "content": data["system"]},
                    {"role": "user", "content": data["user"]},
                ],
            }
            questions.append(question)
            count += 1
            if question_num > 0 and count >= question_num:
                return questions

    return questions


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        answer_file,
        question_num,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        benchmark="API-Bank",
        **kwargs,
):
    questions = load_questions(question_num=question_num, benchmark=benchmark)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    warmup_output_memory = []
    for _ in range(3):
        torch.manual_seed(0)
        messages = question["messages"]
        schema_fsm = SchemaFSM(question["messages"][0]["content"], tokenizer)
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, step, accept_length_tree, question_hidden_state = forward_func(
                inputs,
                warmup_output_memory,
                schema_fsm,
                model,
                tokenizer,
                max_new_tokens,
                **kwargs,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            warmup_output_memory.append(
                {
                    "question_id": question["question_id"],
                    "question_hidden_state": question_hidden_state,
                    "output_ids": output_ids.tolist(),
                }
            )

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

        except RuntimeError as e:
            print("ERROR question ID: ", question["question_id"])
            output = "ERROR"

    print('Warmup done')

    accept_lengths_tree = []
    output_memory = []
    for i, question in tqdm(enumerate(questions)):
        schema_fsm = SchemaFSM(question["messages"][0]["content"], tokenizer) # initialize schema fsm with system prompt

        cur_accept_lengths_tree = []
        torch.manual_seed(i)
        steps = []
        new_tokens = []
        wall_time = []
        messages = question["messages"]
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, step, accept_length_tree, question_hidden_state = forward_func(
                inputs,
                output_memory,
                schema_fsm,
                model,
                tokenizer,
                max_new_tokens,
                **kwargs,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            accept_lengths_tree.extend(accept_length_tree)
            output_ids = output_ids[0][len(input_ids[0]):]
            output_memory.append(
                {
                    "question_id": question["question_id"],
                    "question_hidden_state": question_hidden_state,
                    "output_ids": output_ids.tolist(),
                }
            )
            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

        except RuntimeError as e:
            print("ERROR question ID: ", question["question_id"])
            output = "ERROR"
            output_ids = input_ids
            new_token = 0
            step = 0
            accept_length_tree = []
            total_time = 0.0
            question_hidden_state = None

        steps.append(int(step))
        new_tokens.append(int(new_token))
        wall_time.append(total_time)
        cur_accept_lengths_tree.extend(accept_length_tree)
        choices = {"index": i, "output": output, "decoding_steps": steps, "new_tokens": new_tokens,
                        "wall_time": wall_time, "accept_lengths": cur_accept_lengths_tree}

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))



def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
