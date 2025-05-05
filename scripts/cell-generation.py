import pandas as pd
from pydantic import BaseModel
import json
import time
from tqdm.auto import tqdm
from litellm import batch_completion

OPENAI_MODEL = 'gpt-4o'
SYSTEM_PROMPT = "You are an expert data generator with knowledge in the population statistics. The output should be limited strictly to the JSON structure without any additional explanations or formatting."
TEMPERATURE = 1.0
TOP_P = 0.95
SAMPLE_SIZE = 10_000
BATCH_SIZE = 64

def get_llm_output(user_prompts: str,max_tokens: int):
    """
    Sends a prompt to an LLM and expects a JSON-formatted response.
    """

    messages = [[{"role": "system", "content": SYSTEM_PROMPT}]+[{"role": "user", "content": p}] for p in user_prompts]
    output = batch_completion(OPENAI_MODEL,
                              messages=messages,
                              response_format={"type": "json_object"},
                              temperature=TEMPERATURE,
                              top_p=TOP_P,
                              max_tokens=max_tokens,
                              )
    return  [json.loads(o.choices[0].message.content) for o in output]

def create_prompt(sample: dict) -> str:
    categories = {", ".join(sample['categories'])}
    description = sample['data_description']
    user_prompt = sample['user_prompt']
    context = ". ".join([f"The {k} is {v}" for k, v in sample['features'].items()])
    prompt = f"""Based on the provided context and data description, generate one random sample for the column {user_prompt}.
Sample from the following categories: {categories}
# Context: {context}
# Data Description: {description}
The output should be limited strictly to the chosen category without any additional explanations or formatting.

# Response:"""
    return prompt


def create_prompts(samples: pd.DataFrame, batch_size: int):

  n_samples = samples.shape[0]
  indices = range(0,n_samples)

  for start in range(0, n_samples, batch_size):
    stop = min(start + batch_size, n_samples)
    batch_idx = indices[start:stop]

    prompts = []
    for idx in batch_idx:
      prompts += [create_prompt(samples.iloc[idx].to_dict())]

    yield prompts



if __name__ == "__main__":

    for run in range(1,6):
        # Feature to Generate
        target_features = {
            "Ethnicity Group":{
                "categories":["Latino", "White", "Asian/Pacific Islander", "Black", "Native American", "Multiracial/Other"],
                "prompt": "Ethnicity of individuals in California",
                "dtype":"category"
                },
            "Age Group":{
                "categories":["Children (0-17)","College-going age (18–24)","Prime-working age (25–54)","Adults (55–64)","65 and older"],
                "prompt": "Age group of individuals in California",
                "dtype":"category"
            }
        }

        data_description = 'The data contains the information on US population.'
        print(f'++++++ Run {run} ++++++')

        start_time = time.time()
        enriched_data = pd.DataFrame({'State':SAMPLE_SIZE*['California']})
        response = []
        for feature, properties in target_features.items():
             print(f"Feature: {feature}")

             instructions = pd.DataFrame({
                 "features":enriched_data.to_dict(orient='records'),
                 "user_prompt":feature,
                 "categories":SAMPLE_SIZE*[properties.get("categories")],
                 "data_description":data_description,
             })

             prompts = create_prompts(instructions, BATCH_SIZE)
             response = [get_llm_output(p, max_tokens=1000) for p in tqdm(prompts,total=SAMPLE_SIZE//BATCH_SIZE)]
             enriched_data[feature] = [j[feature] for i in response for j in i]
             stop_time = time.time()
             print(f'Total Generation Time: {stop_time-start_time:.2f}s')

             enriched_data.to_parquet(f'cell-generation_model-{OPENAI_MODEL}_run-{run}.parquet')
