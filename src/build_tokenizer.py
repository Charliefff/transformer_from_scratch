from dataset import DefDataset, causal_mask
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace

from typing import Any
from pathlib import Path

def get_all_sen(dataset, 
                lang: str):
    
    for item in dataset:
        yield item['translation'][lang]

def build_tokenizer(dataset: Any, lang: str) -> Tokenizer:
    tokenizer_path = Path(f"./{lang}_tokenizer.json")
    
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sen(dataset, lang), trainer=trainer)
        if lang == 'yue':
            tokenizer.save('./yue_tokenizer.json')
        elif lang == 'zh':
            tokenizer.save('./zh_tokenizer.json')
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer for {lang} loaded")
    return tokenizer

def get_dataset(config: dict):
    
    dataset_raw = load_dataset("raptorkwok/cantonese-traditional-chinese-parallel-corpus", split="train")
    dataset_val = load_dataset("raptorkwok/cantonese-traditional-chinese-parallel-corpus", split="validation")
    
    tokenizer_src = build_tokenizer(dataset_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(dataset_raw, config['lang_tgt'])
    
    train_datset = DefDataset(dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = DefDataset(dataset_val, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src, max_len_tgt = 0, 0
    
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_datset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

    