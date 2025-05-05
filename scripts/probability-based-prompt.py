"""
Prototype of JennersdorfGPT
"""

import json
import hashlib
import time

import jsonschema
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from litellm import batch_completion
from time import sleep
from typing import Dict, List
from collections import defaultdict

SEED = 42

LLM_SERVICE = "OpenAI"  # "OpenAI" or "Groq"
LLM_GENERATED_SAMPLE_SIZE = 10_000  # The size of the generated sample
LLM_TEMPERATURE = 1.0
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 100

NUM_NUMERIC_BINS = 10 # Number of bins for numeric features

# JSON schema for output validation (example schema, adjust as needed)
output_schema = {
    "type": "object",
    "patternProperties": {
        ".*": {"type": "number"}
    },
    "additionalProperties": False
}


def hash_key(key):
    """
    Generates a SHA-256 hash for a given key.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def validate_output(output, schema):
    """
    Validates the LLM output against the provided JSON schema.
    """
    jsonschema.validate(instance=output, schema=schema)


def get_probabilities_prompt(feature: str, categories: list, context: str, data_description: str) -> str:
    """
    Constructs a prompt for the LLM based on feature, categories, and additional context.
    """

    user_prompt = f"""Based on the provided context and categories, estimate the probability distribution for {feature}. Calculate the probabilities such that they are normalized, ensuring that the sum of all probabilities equals 1.0.

Context: {context.rstrip()}

Categories: {", ".join(categories)}

Calculate each probability, then normalize these values to ensure their sum equals 1.0000. The probabilities should be real numbers. Return the results in a JSON format where each category is represented as a key and its corresponding normalized probability as a value. The categories in the response must match exactly as they are given. The output should be limited strictly to the JSON structure without any additional explanations or formatting."""


    return user_prompt


def get_llm_output(user_prompts: List[str],
                   model: str,
                   LLM_settigns: Dict,
                   json_response_format=True) -> (List[Dict], Dict):
    """
    Sends a prompt to an LLM and expects a JSON-formatted response.
    """
    system_prompt = "You are an expert in the population statistics. Respond only with the needed information."
    messages = [[{"role": "system", "content": system_prompt}]+[{"role": "user", "content": user_prompt}] for user_prompt in user_prompts]
    output = batch_completion(model,
                              messages=messages,
                              response_format={"type": "json_object"} if json_response_format else None,
                              temperature=LLM_settigns["temperature"],
                              top_p=LLM_settigns["top_p"],
                              max_tokens=LLM_settigns["max_tokens"],
                              )

    return [json.loads(o.choices[0].message.content) for o in output] if json_response_format else [o.choices[0].message.content for o in output]


def process_categories(target_categories: Dict, seed_df: pd.DataFrame, data_description: str, sample_size: int, model: str, LLM_settigns: Dict):
    """
    Process categories to generate samples based on the LLM output.

    Potentially the dictionary containing probabilities for each category can be global for all features without
    re-seting it. This will allow to reuse the dictionary for generating more records if needed without re-prompting LLMs.
    """
    generated_results = {}
    contexts = [""] * sample_size if seed_df is None else seed_df.apply(
        lambda row: ". ".join(f"The {col} is {val}" for col, val in row.items()) + ".", axis=1).tolist()
    if seed_df is not None:
        print(f"Seed provided with {len(seed_df.columns)} column(s)")
        # Convert the DataFrame to a dictionary with list orientation
        generated_results = seed_df.to_dict(orient="list")

    for feature_name, feature_prop in target_categories.items():
        user_prompt = feature_prop.get("prompt")
        if user_prompt is None:
            user_prompt = feature_name
        samples = []
        llm_outputs_storage = {}
        # if we need to create only one category then skip the LLM
        feature_categories = feature_prop.get("categories","")

        print(f"{feature_name}: {len(feature_categories)} categories")

        # unique hash for each context
        unique_contexts = list(set(contexts))
        hashed_unique_contexts = {hash_key(ctx): ctx.lstrip(" ") for ctx in unique_contexts}

        if len(feature_categories) > 1:

            # now getting LLM output for each unique context; it should be done in batches later
            # and store the results in the dictionary with hashed context as a key and the output as a value
            full_prompts = [get_probabilities_prompt(user_prompt, feature_categories, ctx_value, data_description) for ctx_value in unique_contexts]

            # implement repeated tries if the LLM fails
            maxRetries = 3
            retryCount = 0
            while retryCount < maxRetries:
                try:
                    raw_outputs = get_llm_output(full_prompts, model, LLM_settigns, json_response_format=True)
                    for ctx_key, raw_output in zip(hashed_unique_contexts.keys(), raw_outputs):
                        validate_output(raw_output, output_schema)
                        # below we re-normalize the probabilities because LLM does not guarantee that they sum to 1
                        cats, probs = zip(*raw_output.items())
                        probs = np.array(probs)
                        normalized_probs = probs / probs.sum()
                        llm_outputs_storage[ctx_key] = dict(zip(cats, normalized_probs))

                    break
                except Exception as e:
                    retryCount += 1
                    print(f"Error in generating output: {e}")
                    sleep(1)
                    continue

            print(f"{feature_name}: {len(llm_outputs_storage)} unique contexts processed")

            # perhaps it can be vectorized if we count frequencies for unique contexts
            # and then sample not one element but all for the unique context
            cats_probs = [(list(llm_outputs_storage[ctx_key].keys()), list(llm_outputs_storage[ctx_key].values()))
                          for ctx_key in map(hash_key, contexts)]

            for cats, probs in tqdm(cats_probs, desc=f"Generating samples for {feature_name}"):
                sample = np.random.choice(a=cats, size=1, p=probs)[0]
                samples.append(sample)

            samples = [sample.tolist() if hasattr(sample, 'tolist') else sample for sample in samples]

        elif len(feature_categories) == 1:
            # Directly choose if only one category should be generated
            samples = feature_categories * sample_size

        contexts = [context
                    + " "
                    + f"The {feature_name} is {sample}." for context, sample in zip(contexts, samples)]

        generated_results[feature_name] = samples

    return generated_results

if __name__ == "__main__":

    for run in range(1,6):
        print(f"+++++++ Run {run} +++++++")
        seed_df = pd.DataFrame({"State":['California']*LLM_GENERATED_SAMPLE_SIZE})
        data_description = 'The data contains the information on US population.'
        target_categories = {
            "Race":{
                "categories":["Latino", "White", "Asian/Pacific Islander", "Black", "Native American", "Multiracial/Other"],
                "dtype":"category"
                },
            "Age Group":{
                "categories":["Children (0-17)","College-going age (18-24)","Prime-working age (25-54)","Adults (55-64)","65 and older"],
                "dtype":"category"
            }
        }
        MODEL = "gpt-4o"  # "gpt-4o" or "gpt-4o-turbo" or gpt-4o-mini
        LLM_settigns = dict(temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, max_tokens=LLM_MAX_TOKENS)
        start_time: float = time.time()
        llm_generated_results = process_categories(target_categories,
                                                    seed_df,
                                                    data_description,
                                                    LLM_GENERATED_SAMPLE_SIZE,
                                                    MODEL,
                                                    LLM_settigns)
        stop_time: float = time.time()
        print(f'Time elapsed: {stop_time - start_time:.2f} seconds')

        llm_final_result_as_df = pd.DataFrame.from_records(llm_generated_results)[list(target_categories.keys())]
        llm_final_result_as_df['State'] = 'California'
        llm_final_result_as_df[['State','Age Group','Race']].to_parquet(f'jennersdorfgpt_model_{MODEL}_run_{run}.parquet')
