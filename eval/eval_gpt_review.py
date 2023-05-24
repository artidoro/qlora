# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/eval_gpt_review.py
import argparse
import json
import os
import time

import openai
from tqdm import tqdm
import ray

import shortuuid
import logging
import numpy as np
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 1000
REQ_TIME_GAP = 2


@ray.remote(num_cpus=4)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(min(5*(i+1), 100))
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def parse_three_class_score(review):
    try:
        score = int(review.strip().split("\n")[-1].strip())
        return score
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1
    
def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-m", "--model", default='gpt-4')
    parser.add_argument("-id", "--id-key", default='question_id')
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    else:
        threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
        dest = os.path.join(
            args.output_review_file,
            '_vs_'.join([elt.split('/')[-1].replace('.jsonl', '') for elt in args.answer_file_list]) + f'_{args.model}_reviewer{threeclass_suff}' + '.jsonl'
        )

    ray.init()

    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    question_ids = set(question[args.id_key] for question in question_jsons)
    question_jsons = sorted(question_jsons, key=lambda x: x[args.id_key])
    answer1_jsons = sorted(
        [answer for answer in answer1_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )
    answer2_jsons = sorted(
        [answer for answer in answer2_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )

    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    handles = []
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))

    for i in tqdm(question_idx_list):
        assert (
            answer1_jsons[i][args.id_key]
            == question_jsons[i][args.id_key]
            == answer2_jsons[i][args.id_key]
        )

        ques = question_jsons[i]["text"]
        cat = question_jsons[i]["category"]
        if 'generation_truncated' in answer1_jsons[i]:
            ans1 = answer1_jsons[i]["generation_truncated"].split(
                'A chat between a curious human and an artificial intelligence')[0]
        elif 'generation' in answer1_jsons[i]:
            ans1 = answer1_jsons[i]["generation"].split(
                'A chat between a curious human and an artificial intelligence')[0]
        else:
            ans1 = answer1_jsons[i]["text"]
        # ans1 = answer1_jsons[i]["text"]
        if 'generation_truncated' in answer2_jsons[i]:
            ans2 = answer2_jsons[i]["generation_truncated"].split(
                'A chat between a curious human and an artificial intelligence')[0]
        elif 'generation' in answer2_jsons[i]:
            ans2 = answer2_jsons[i]["generation"].split(
                'A chat between a curious human and an artificial intelligence')[0]
        else:
            ans2 = answer2_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                args.id_key: question_jsons[i][args.id_key],
                "answer1_id": answer1_jsons[i]["answer_id"] if 'answer_id' in answer1_jsons[i] else shortuuid.uuid(ans1),
                "answer2_id": answer2_jsons[i]["answer_id"] if 'answer_id' in answer2_jsons[i] else shortuuid.uuid(ans2),
                "reviewer_id": reviewer_id,
                "metadata": {},
            }
        )
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval.remote(sys_prompt, prompt, args.max_tokens, args.model))
        logger.info(
            f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
        )
        time.sleep(REQ_TIME_GAP)

    reviews = ray.get(handles)
    with open(dest, "w") as output_review_file:
        for idx, review in enumerate(reviews):
            if 'threeclass' in args.prompt_file:
                scores = parse_three_class_score(review)
            else:
                scores = parse_score(review)
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
            output_review_file.flush()
