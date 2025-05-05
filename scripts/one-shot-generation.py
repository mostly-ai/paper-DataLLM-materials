import pandas as pd
from pydantic import BaseModel
import json
import time
from litellm import batch_completion

OPENAI_MODEL = 'gpt-4o'
SYSTEM_PROMPT = "You are an expert in statistics. The output should be limited strictly to the JSON structure without any additional explanations or formatting."
TEMPERATURE = 1.0
TOP_P = 0.95
SAMPLE_SIZE = 100

def get_llm_output(user_prompt: str,max_tokens: int):
    """
    Sends a prompt to an LLM and expects a JSON-formatted response.
    """

    messages = [[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": user_prompt}] for i in range(100)]
    output = batch_completion(OPENAI_MODEL,
                              messages=messages,
                              response_format={"type": "json_object"},
                              temperature=TEMPERATURE,
                              top_p=TOP_P,
                              max_tokens=max_tokens,
                              )

    return [r for o in output for r in json.loads(o.choices[0].message.content)['records']]


if __name__ == "__main__":

    for run in range(1,6):
        # Feature to Generate
        target_features = {
            "State":{
                "categories":['California'],
                "prompt": "State of residence",
                "dtype":"category"
                },
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

        prompt = f'''
Generate a table with {SAMPLE_SIZE} records and with columns 'State', 'Age Group', and 'Ethnicity Group'.
'State' contains identical values, all set to 'California/CA',
'Age Group' should be sampled from the categories {target_features['Age Group']['categories']},
and 'Ethnicity Group' should be sampled from the categories {target_features['Ethnicity Group']['categories']}
reflecting population in 'State' of California/CA.

Strictly follow the output format:

{{
'records':
{{
'State':'California',
'Age Group':'Prime-working age (25–54)',
'Ethnicity Group':'Black',
}}
}}
'''

        print(f'++++++ Run {run} ++++++')
        start_time = time.time()
        try:
            results = get_llm_output(user_prompt=prompt, max_tokens=5_000)
            stop_time = time.time()
            print(f'Total Generation Time: {stop_time-start_time:.2f}s')
            # convert to Pandas dataframe
            results_df = pd.DataFrame(results)
            results_df.to_parquet(f'one-shot-generation_model-{OPENAI_MODEL}_run-{run}.parquet')
        except Exception as e:
            print(f"++++++ FAIL ++++++:\n{e}")
