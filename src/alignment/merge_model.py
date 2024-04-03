import sys
import shutil
import os
import torch
from peft import (
    PeftModel,
    PeftConfig
)

from transformers import AutoModelForCausalLM, AutoTokenizer

from argparse import ArgumentParser
def argparser():
    ap = ArgumentParser()
    ap.add_argument('--lora_adapter', type=str)
    ap.add_argument('--output_dir', type=str)
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    # Load PeFT LoRA model
    print("LoRA adatper:", args.lora_adapter)
    config = PeftConfig.from_pretrained(args.lora_adapter)
    print("Base model:", config.base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 cache_dir=args.transformers_cache)
    base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    print("Loaded base model")
    print("Merging model")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter, torch_dtype=torch.bfloat16)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    base_tokenizer.save_pretrained(args.output_dir)
    # copy tokenizer files to merged model
    # print("Copying tokenizer")
    # tokenizer_files = os.listdir(args.lora_adapter)
    # tokenizer_files = [f for f in tokenizer_files if "token" in f]
    # for f in tokenizer_files:
    #     shutil.copyfile(os.path.join(args.lora_adapter, f), args.output_dir)
    print("Done merging model! Saved merged model to", args.output_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv))