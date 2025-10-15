import os
import json
import httpx
import re
import random
from tqdm import tqdm
from openai import OpenAI


prompt = (
    "You are provided with 2 sets of QA about the same action. "
    "Your task is to evaluate the quality of the QA of the model. "
    "Identify the key aspects (e.g. categories, temporal orders) of the action. "
    "The captions will also be provided, to help you judge the correctness. "
    "Assign correctness points for each distinct correct attribute. Each correct attribute contributes 1 point. If two actions are the same, the score is 2. "
    "And calculate the percentage of these aspects that are correctly mentioned or partially matched in the model-generated caption." 
    "Partial correctness should be graded on a scale of 0 to 1 depending on accuracy, with similar concepts considered for partial scores."
    "Each aspect contributes equally to the final score, with no extra penalties for repeated inaccuracies of the same attribute. "
    "Also, assign hallucination points for incorrect details in the model output, with 1 point per incorrect attribute. "
    "Repetitive inaccuracies based on one attribute should incur only a single hallucination point. "
    "However, if a detail is described uncertainly or speculatively (e.g., using phrases like 'probably for', 'possibly', 'resemble' or 'indicating'), assign fewer or no hallucination points."
    "Provide your score and a short justification (less than 25 words) in the format of example."
    "If there are redundant actions, please analyze whether it is reasonable. If it's reasonable or normal, it's not an error."
    "Note: there is no need to judge left-right. For example, the action is about left, but the caption says right is OK."
    "You should response in the following format:"
    "Example1: "
    "<Input>"
    "- question: Describe the human motion based on the provided wave signal representing joint movements."
    "- pred: [a person does a golf swing]"
    "- gt: [a person is golf putting a ball; a person holds their hands together below their waist and swings their arm; a person grasps something and then does a hitting motion]"
    "<Output>"
    "- Correctness: 1 # Correct identification of the action category"
    "- Hallucination: 0 # No incorrect details"
    # "Example2: "
    # "<Input>"
    # "- question: Describe the human motion based on the provided wave signal representing joint movements."
    # "- pred: [a person picks something up with their right hand, then bends down and picks something up with their left hand]"
    # "- gt: [the person is bowing over, a person bows twice, step to the side and back, person is making bowing gestures]"
    # "<Output>"
    # "- Correctness: 2 # Correct identification of the action category and temporal order (twice)"
    # "- Hallucination: 0 # No incorrect details"
)



# if __name__ == "__main__":
def evaluate(output_path, local_rank=0, world_size=1, caption_only=False):
    
    # Initialize the client with your base URL and API key
    client = OpenAI(
        base_url="",
        api_key="",
        http_client=httpx.Client(
            base_url="",
            follow_redirects=True,
        ),
    )
    
    savedict = {}
    total_correctness = 0
    total_hallucination = 0
    total_correctness_cap = 0
    total_hallucination_cap = 0
    json_to_evaluate = json.load(open(output_path))
    json_to_evaluate = list(json_to_evaluate.items())[local_rank * len(json_to_evaluate) // world_size: (local_rank + 1) * len(json_to_evaluate) // world_size]
    save_path = output_path.replace('.json', f'_gpt_eval_{local_rank}.json')
    # for item in tqdm(json_to_evaluate):
    index = 0
    for key, item in tqdm(json_to_evaluate) if local_rank == 0 else json_to_evaluate:
        outdict = item
        if not caption_only:    
            question = outdict['question'].replace('left', 'right')
            pred = outdict['pred'][0].replace('left', 'right')
            pred_caption = outdict['pred_caption'][0].replace('left', 'right')
            caption = '; '.join(outdict['caption']).replace('left', 'right')
            gt = outdict['gt'][0].replace('left', 'right')
        else: 
            # question = 'Describe the human motion based on the provided wave signal representing joint movements.'
            question = outdict['question'].replace('left', 'right')
            pred = outdict['pred_caption'][0].replace('left', 'right')
            gt = '; '.join(outdict['caption']).replace('left', 'right')
        # evaluate captions
        while True:
            caption_ = random.choice(caption.split('; '))
            response_cap = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": 
                    "<Input>\n"
                    f"- question: {question}\n"
                    f"- pred: {pred_caption}\n"
                    f"- gt: {caption_}</Input>"
                    f"- caption: {caption}</Input>"}],
            )
            response_cap = response_cap.choices[0].message.content
            match_correctness = re.search(r'Correctness: (\d+) #', response_cap)
            match_hallucination = re.search(r'Hallucination: (\d+) #', response_cap)
            if match_correctness and match_hallucination:
                correctness_cap = int(match_correctness.group(1))
                hallucination_cap = int(match_hallucination.group(1))
                break
        total_correctness_cap += correctness_cap
        total_hallucination_cap += hallucination_cap
        # evaluate QAs
        while True:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": 
                    "<Input>\n"
                    f"- question: {question}\n"
                    f"- answer: {pred}\n"
                    f"- gt: {gt}\n"
                    f"- caption: {caption}</Input>"}],
            )
            response = response.choices[0].message.content
            match_correctness = re.search(r'Correctness: (\d+) #', response)
            match_hallucination = re.search(r'Hallucination: (\d+) #', response)
            if match_correctness and match_hallucination:
                correctness = int(match_correctness.group(1))
                hallucination = int(match_hallucination.group(1))
                break
        total_correctness += correctness
        total_hallucination += hallucination
        
        savedict['%06d' % index] = {'response': response, 'response_cap': response_cap, 'correctness': correctness, 'hallucination': hallucination, 'correctness_cap': correctness_cap, 'hallucination_cap': hallucination_cap}
        json.dump(savedict, open(save_path, 'w'), indent=2)
        index += 1
    # print(f'Total correctness: {total_correctness}; Total hallucination: {total_hallucination}')
    # print(f'Total correctness_cap: {total_correctness_cap}; Total hallucination_cap: {total_hallucination_cap}')

import multiprocessing as mp
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    world_size = 10
    
    if not args.summarize:
        
        processes = []
        for local_rank in range(world_size):
            p = mp.Process(target=evaluate, args=(args.output_path, local_rank, world_size))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
    else: 
        files = glob.glob(args.output_path.replace('.json', '_gpt_eval_*.json'))
        correctness = []
        hallucination = []
        correctness_cap = []
        hallucination_cap = []
        for file in files:
            savedict = json.load(open(file))
            for key, item in savedict.items():
                correctness.append(item['correctness'])
                hallucination.append(item['hallucination'])
                correctness_cap.append(item['correctness_cap'])
                hallucination_cap.append(item['hallucination_cap'])
        # sum
        correctness = sum(correctness) 
        hallucination = sum(hallucination)
        correctness_cap = sum(correctness_cap)
        hallucination_cap = sum(hallucination_cap)
        print(f'Correctness: {correctness}; Hallucination: {hallucination}', 'Precision:', correctness / (correctness + hallucination))
        print(f'Correctness_cap: {correctness_cap}; Hallucination_cap: {hallucination_cap}', 'Precision:', correctness_cap / (correctness_cap + hallucination_cap))
        