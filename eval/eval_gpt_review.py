# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/eval_gpt_review.py
import argparse
import json
import os
import time

import openai
import ray
from tqdm import tqdm

import shortuuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 1000
REQ_TIME_GAP = 2

openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptGenerator:
    def __init__(self, reviewer_jsons, prompt_jsons):
        self.reviewer_jsons = reviewer_jsons
        self.prompt_jsons = prompt_jsons

    def gen_prompt(self, cat, ques, ans1, ans2):
        reviewer_idx = next((idx for idx, reviewer in enumerate(self.reviewer_jsons) if reviewer["category"] == cat), 0)
        prompt_id = self.reviewer_jsons[reviewer_idx]["prompt_id"]
        prompt_json = self.prompt_jsons[prompt_id - 1]
        assert prompt_json["prompt_id"] == prompt_id

        sys_prompt = prompt_json["system_prompt"]
        prompt_template = prompt_json["prompt_template"]
        defaults = prompt_json["defaults"]
        prompt = prompt_template.format(question=ques, answer_1=ans1, answer_2=ans2, **defaults)

        return sys_prompt, prompt, reviewer_idx + 1


@ray.remote(num_cpus=4)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(min(5 * (i + 1), 100))
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


def parse_three_class_score(review):
    try:
        score = int(review.strip().split("\n")[-1].strip())
        return score
    except Exception as e:
        logger.error(f"{e}\nContent: {review}\nYou must manually fix the score pair.")
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
        logger.error(f"{e}\nContent: {review}\nYou must manually fix the score pair.")
        return [-1, -1]


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = [json.loads(line) for line in f]
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-m", "--model", default='gpt-4')
    parser.add_argument("-id", "--id-key", default='question_id')
    parser.add_argument("--max-tokens", type=int, default=1024, help="maximum number of tokens produced in the output")
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

    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    prompt_generator = PromptGenerator(reviewer_jsons, prompt_jsons)
    handles = []
    review_jsons = []

    for i, question in enumerate(tqdm(question_jsons)):
        assert (
            answer1_jsons[i][args.id_key]
            == question[args.id_key]
            == answer2_jsons[i][args.id_key]
        )

        ques = question["text"]
        cat = question["category"]
        if 'generation_truncated' in answer1_jsons[i]:
            ans1 = answer1_jsons[i]["generation_truncated"].split('A chat between a curious human and an artificial intelligence')[0]
        elif 'generation' in answer1_jsons[i]:
            ans1 = answer1_jsons[i]["generation"].split('A chat between a curious human and an artificial intelligence')[0]
        else:
            ans1 = answer1_jsons[i]["text"]
        if 'generation_truncated' in answer2_jsons[i]:
            ans2 = answer2_jsons[i]["generation_truncated"].split('A chat between a curious human and an artificial intelligence')[0]
        elif 'generation' in answer2_jsons[i]:
            ans2 = answer2_jsons[i]["generation"].split('A chat between a curious human and an artificial intelligence')[0]
        else:
            ans2 = answer2_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = prompt_generator.gen_prompt(cat, ques, ans1, ans2)
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                args.id_key: question[args.id_key],
                "answer1_id": answer1_jsons[i].get("answer_id", shortuuid.uuid(ans1)),
                "answer2_id": answer2_jsons[i].get("answer_id", shortuuid.uuid(ans2)),
                "reviewer_id": reviewer_id,
                "score": None,
                "review": None,
            }
        )
        handles.append(
            get_eval.remote(sys_prompt, prompt, args.max_tokens, args.model)
        )
        time.sleep(REQ_TIME_GAP)

    review_texts = ray.get(handles)

    assert len(review_jsons) == len(review_texts)

    for i, review_text in enumerate(review_texts):
        review_jsons[i]["review"] = review_text.strip()
        if 'threeclass' in args.prompt_file:
            review_jsons[i]["score"] = parse_three_class_score(review_text)
        else:
            review_jsons[i]["score"] = parse_score(review_text)

    with open(dest, "w") as f:
        for review_json in review_jsons:
            f.write(json.dumps(review_json) + "\n")

    logger.info(f"Review written to {dest}")

    ray.shutdown()
