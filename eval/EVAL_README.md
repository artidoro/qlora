# Evaluation

We provide scripts to evaluate model responses with GPT-4. We also provide a script to generate new responses using the OpenAI API. These scripts are based on the evaluation scripts of the [FastChat repo](https://github.com/lm-sys/FastChat/tree/main/fastchat/eval).

## Benchmark Queries
In our paper we use both Vicuna and Open Assistant data. The exact queries for both benchmarks are found at `prompts/vicuna_questions.jsonl` and `prompts/oa_questions.jsonl`. Note that the Vicuna questions are the same set of 80 prompts found in the FastChat repo. For the OpenAssistant benchmark, we select all the user queries in the validation dataset. We also include the previous turns in the dialog for context when applicable.

## Sample Generations and Model Ratings
The `generation` folder has outputs from the models studied in our paper. This includes generations for all model sizes (from 7B to 65B) and instruction following datasets.

The `ratings-gpt4` and `ratings-human` have the automated and human ratings described in our paper.

We use the following naming conventions files with generations from the models we train: 
```
<model size>-<model name>-[oa or vicuna]-generations-topp0.9-temp0.7.jsonl
```

For automated comparisons, we use the convention:
```
<generation file name 1>-vs-<generation file name 2>-<reviewer>-reviewer.jsonl
```
We note that pairwise comparisons also have the additional `-threeclass` term in their name. Finally OpenAI models and baselines from other papers might have slightly different naming styles.


## Generating Responses
After adding the OpenAI key as an evnironment variable, you can generate responses for both Vicuna and OA benchmarks using OpenAI models with the command below. Note that you can use `MODEL` to change between `gpt-4` and `gpt-3.5-turbo`.

```bash
python qa_baseline_gpt35.py --question prompts/oa_questions.jsonl --output generations/answer_gpt35.jsonl
```

## GPT-4 Automatic Ratings
In our paper we explore using GPT-4 to rate generations automatically. Note that we use different prompts for the Vicuna and OA benchmarks. We also have different prompts for comparisons relative to ChatGPT-3.5 with a 10 point scale and for pairwise three class comparisons (win, loose, tie). Depending on which type of evaluation you want to perform you should choose `prompts/vicuna_prompt_threeclass.jsonl`, `prompts/vicuna_prompt_relative.jsonl` or `prompts/oa_prompt_threeclass.jsonl`. 

The ratings for the Vicuna benchmark can be obtained with the following command:

```bash
python eval_gpt_review.py \
    -a generations/vicuna/answer_gpt35.jsonl generations/vicuna/65b-guanaco-vicuna-generations-topp0.9-beam1.jsonl \
    -q prompts/vicuna_questions.jsonl \
    -p prompts/vicuna_prompt_threeclass.jsonl \
    -r prompts/reviewer.jsonl \
    -o ratings/ \
    -m gpt-4
```

The ratings for the OA benchmark can be obtained with the following command:

```bash
python eval_gpt_review.py \
    -a generations/oa/gpt-3.5-oa-generations.jsonl generations/oa/65b-guanaco-oa-generations-topp0.9-beam1.jsonl \
    -q prompts/oa_questions.jsonl \
    -p prompts/oa_prompt_threeclass.jsonl \
    -r prompts/reviewer.jsonl \
    -o ratings/ \
    -m gpt-4
```

## Recommendations

We describe various considerations and biases of this sort of automated evaluation in our paper and recommend following best-practices described there when assessing new models.
