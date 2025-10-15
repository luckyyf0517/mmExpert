from transformers import AutoTokenizer,AutoConfig
import torch
from torch.utils.data import DataLoader
import argparse
from src.wavellm import *
from src.datasets.wavellm_dataset import WaveCaptionDataset
from src.wavellm.conversation import conv_templates
from src.wavellm import conversation as conversation_lib
import random
from tqdm import tqdm
from copy import deepcopy
import json
import os.path as osp
from termcolor import colored


import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def config_model_and_tokenizer(model_path,torch_dtype=torch.float16,lora=False,adapters_name=None,encoder_path=None):
    # Set cache directory for offline mode
    cache_dir = "/root/autodl-tmp/mmExpert/huggingface"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=True  # Force offline mode
    )
    model = WaveLLMForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=True,  # Force offline mode
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).cuda()

    model.initialize_tokenizer_wave_backbone_config(tokenizer,'cuda')

    model.get_model().load_wave_encoder(encoder_path)
    model.get_model().wave_encoder.bfloat16().to('cuda')
    if lora:
        from peft import PeftModel  
        model = PeftModel.from_pretrained(model, adapters_name)
        embed_and_proj = torch.load(osp.join(adapters_name, "non_lora_trainables.bin"))
        assert set(embed_and_proj.keys()).issubset(set(model.state_dict().keys())), "embed and projection keys are missing in the model state_dict"
        model.load_state_dict(embed_and_proj, strict=False)
    return model, tokenizer


def init_model(args):
    # Model
    disable_torch_init()
    model_name = args.model_name

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    model, tokenizer = config_model_and_tokenizer(model_name,
                                                  adapters_name=args.adapter_name,
                                                  lora=True,
                                                  encoder_path=args.data_root)

    conversation_lib.default_conversation = conversation_lib.conv_templates["conv_phi3"]
    conv = conv_templates["conv_phi3"].copy()

    return model, tokenizer, conv


def collate_fn(instances):
    batch = {}
    if "wave_embed" in instances[0]:
        batch['input_wave_embeds'] = torch.stack([instance['wave_embed'] for instance in instances])
    if "caption" in instances[0]:
        batch['captions'] = [instance['caption'] for instance in instances]
    if "question" in instances[0]:
        batch['questions'] = [instance['question'] for instance in instances]
    if "answer" in instances[0]:
        batch['answers'] = [instance['answer'] for instance in instances]
    return batch


def generate_outputs(model, tokenizer, input_ids, input_wave_tokens=None, input_wave_embeds=None, do_sample=True,beam_size=4, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                input_wave_tokens=input_wave_tokens,
                input_wave_embeds=input_wave_embeds,
                attention_mask = torch.ones(input_ids.shape,
                                            dtype=torch.long,
                                            device=input_ids.device),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                num_beams=beam_size,
                max_new_tokens=30,
                top_p=top_p) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    return outputs


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader


def get_input_ids(questions, model, tokenizer, conv):
    wave_token_len = model.config.wave_token_len
    default_wave_patch_token = model.config.default_wave_patch_token
    default_wave_start_token = model.config.default_wave_start_token
    default_wave_end_token = model.config.default_wave_end_token
    mm_use_wave_start_end = model.config.mm_use_wave_start_end
    prompt_list = []
    for question in questions:
        conv_ = deepcopy(conv)
        if mm_use_wave_start_end:
            qs = default_wave_start_token + default_wave_patch_token * wave_token_len + default_wave_end_token + '\n' + question
        else:
            qs = default_wave_patch_token * wave_token_len + '\n' + question    
        conv_.append_message(conv_.roles[0], qs)
        conv_.append_message(conv_.roles[1], None)
        prompt = conv_.get_prompt()
        prompt_list.append(prompt)
    inputs = tokenizer(prompt_list, padding=True)
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of B, L
    return input_ids_


def start_generation(model, tokenizer, conv, dataloader, output_dir, output_file=None):
    responses = {}

    i = 0
    for batch in tqdm(dataloader):
        input_wave_embeds = batch["input_wave_embeds"].cuda().to(model.dtype) # * tensor of B, N, C(3)

        batchsize = len(input_wave_embeds)
        captions = batch['captions']

        questions = batch['questions']
        answers = batch['answers']
        input_ids = get_input_ids(questions, model, tokenizer, conv)
        outputs = generate_outputs(model, tokenizer, input_ids, input_wave_embeds=input_wave_embeds) # List of str, length is B
        
        questions_caption = ['Describe the human motion based on the provided wave signal representing joint movements.'] * batchsize
        captions = batch['captions']
        input_ids_caption = get_input_ids(questions_caption, model, tokenizer, conv)
        outputs_caption = generate_outputs(model, tokenizer, input_ids_caption, input_wave_embeds=input_wave_embeds) # List of str, length is B
        
        print(colored("Captions and Outputs:", "blue"))
        for idx, (caption, question, output, ground_truth, output_caption) in enumerate(zip(captions, questions, outputs, answers, outputs_caption)):
            print("---")
            print(f"{idx} -", colored("Question:", "blue"), colored(question, "blue"))
            print(f"{idx} -", colored("Output:", "green"), colored(output, "green"))
            print(f"{idx} -", colored("Ground Truth:", "yellow"))
            for gt in ground_truth.split('#'):
                print(f"  -", colored(gt, "white"))
            
        # saving results
        for cap, question, output, ground_truth, output_caption in zip(captions, questions, outputs, answers, outputs_caption):
            responses[f"QA_Single_EQ_{i}"] = {
                # "caption": cap.split('\n'), 
                # "pred_caption": output_caption.split('\n'),
                "question": question,
                "pred": [output],
                "gt": ground_truth.split('#')
            }
            i += 1
    
        results = responses
        os.makedirs(output_dir, exist_ok=True)
        # save the results to a JSON file
        with open(os.path.join(output_dir, output_file), 'w') as fp:
            json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def evaluate(args):
    args.output_dir = os.path.join(args.model_name, "evaluation")

    args.output_file = f"output.json" if not args.test_on_real else f"output_real.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    model, tokenizer, conv = init_model(args)

    if args.test_on_real: 
        dataset = WaveCaptionDataset(args.data_root, split="real", tokenizer=tokenizer)
    else: 
        # todo: change code of test or test_QAs
        dataset = WaveCaptionDataset(args.data_root, split="test_QAs", tokenizer=tokenizer)
    loader = get_dataloader(dataset,args.batch_size)

    results = start_generation(model, tokenizer, conv, loader, args.adapter_name, args.output_file)

# from src.datasets.wavellm_dataset import mini_dataset
# def evaluate_interact(args):
#     model, tokenizer, conv = init_model(args)
#     filename = 'dataset/DEMO/udoppler/1001.npy'
#     question = 'What objects might he have picked up?'
#     dataset = [mini_dataset(filename, tokenizer, question, answer='None', caption='None')]
#     batch = collate_fn(dataset)
#     input_wave_embeds = batch["input_wave_embeds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
#     batchsize = len(input_wave_embeds)
#     assert batchsize == 1
#     questions = batch['questions']
#     input_ids = get_input_ids(questions, model, tokenizer, conv)
#     outputs = generate_outputs(model, tokenizer, input_ids, input_wave_embeds=input_wave_embeds) # List of str, length is B
#     print(filename)
#     print(question)
#     print(outputs[0])
#     exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./ckpt/Phi-3-mini-4k-instruct") 
    parser.add_argument("--adapter_name", type=str, default="outputs/train")
    parser.add_argument("--test_on_real", action="store_true")

    # * dataset type
    parser.add_argument("--data_root", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--model_path")

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--interact", action="store_true")

    args = parser.parse_args()
    if args.interact:
        evaluate_interact(args)
    else:
        evaluate(args)