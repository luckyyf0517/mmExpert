import os
import json
import httpx
from tqdm import tqdm
from openai import OpenAI
from copy import deepcopy
import multiprocessing as mp
import re


def clean_unicode_chars(text):
    """Clean problematic Unicode characters that can cause JSON parsing issues"""
    # Replace smart quotes and other problematic characters
    replacements = {
        ''': "'",   # Left single quotation mark
        ''': "'",   # Right single quotation mark
        '"': '"',   # Left double quotation mark
        '"': '"',   # Right double quotation mark
        '–': '-',   # En dash
        '—': '-',   # Em dash
        '…': '...', # Horizontal ellipsis
        '•': '*',   # Bullet
        '°': ' deg', # Degree symbol
        '×': 'x',   # Multiplication sign
        '÷': '/',   # Division sign
        '≤': '<=',  # Less than or equal
        '≥': '>=',  # Greater than or equal
        '≠': '!=',  # Not equal
        '∞': 'infinity', # Infinity
        'α': 'alpha',    # Greek letters
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'θ': 'theta',
        'λ': 'lambda',
        'μ': 'mu',
        'π': 'pi',
        'σ': 'sigma',
        'τ': 'tau',
        'φ': 'phi',
        'ω': 'omega',
    }
    
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    return text


def make_QAs(source_json, local_rank=0, world_size=1):
    
    # Initialize the client with your base URL and API key from environment variables
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
        ),
    )
    
    # Load prompt template from file
    prompt_template_path = os.path.join(os.path.dirname(__file__), 'make_QAs_prompt.txt')
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    data_new = {}
    with open(source_json, 'r') as f:
        data = json.load(f)
    len_data = len(data)
    data_items = list(data.items())[local_rank * len_data // world_size: (local_rank + 1) * len_data // world_size]
    for idx, (key, item) in enumerate(data_items):
        item_new = deepcopy(item)
        captions = '\n'.join(item['captions'])
        print('%02d:generating QAs (%04d/%04d)' % (local_rank, idx, len(data_items)))
        
        # Format prompt template with captions
        prompt_for_qa_generation = prompt_template.format(captions=captions)
        while True: 
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_for_qa_generation}],
                    response_format={"type": "json_object"}
                )
                response = response.choices[0].message.content
                response = response.replace('```json', '')
                response = response.replace('```', '')
                
                # Clean problematic Unicode characters
                response = clean_unicode_chars(response)
                
                # Use json.loads instead of eval for safer parsing
                response_json = json.loads(response)
                break
            except httpx.TimeoutException:
                print("Request timed out. Retrying...")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}. Retrying...")
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
        item_new['questions'] = response_json
        data_new[key] = item_new
        
        # save the data_new to a json file
        split = source_json.split('/')[-1].split('.')[0]
        with open(f'dataset/HumanML3D/_split/{split}_QAs/part_{local_rank}.json', 'w') as f:
            json.dump(data_new, f, indent=2)
    return data_new

def worker(local_rank, world_size):
    make_QAs(f'dataset/HumanML3D/_split/train.json', local_rank=local_rank, world_size=world_size)
    make_QAs(f'dataset/HumanML3D/_split/test.json', local_rank=local_rank, world_size=world_size)

if __name__ == "__main__":
    world_size = 10
    processes = []
    for local_rank in range(world_size):
        p = mp.Process(target=worker, args=(local_rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        