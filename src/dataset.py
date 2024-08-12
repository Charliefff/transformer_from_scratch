import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any


class DefDataset(Dataset):

    def __init__(self, 
                 dataset: t.tensor,
                 tokenizer_src: object,
                 tokenizer_tgt: object,
                 src_lang: str,
                 tar_lang: str,
                 seq_len: int = 128):

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.seq_len = seq_len 
        
        self.sos_token = t.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=t.int64)
        self.eos_token = t.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=t.int64)
        self.pad_token = t.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=t.int64)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, 
                    index: Any) -> Any:
        
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tar_text = src_target_pair['translation'][self.tar_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tar_text).ids
        
        # Truncate sequences if they are too long
        enc_input_tokens = enc_input_tokens[:self.seq_len-2]
        dec_input_tokens = dec_input_tokens[:self.seq_len-1]
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        
        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError('Too long sequence !')
        
        encoder_input = t.cat(
            [
                self.sos_token, # start
                t.tensor(enc_input_tokens, dtype=t.int64), # input tokens
                self.eos_token, # end
                t.tensor([self.pad_token] * enc_num_padding_tokens, dtype=t.int64) # padding
            ],
            dim=0,
        )
        
        decoder_input = t.cat(
            [
                self.sos_token, # start
                t.tensor(dec_input_tokens, dtype=t.int64), # input tokens
                t.tensor([self.pad_token] * dec_num_padding_tokens, dtype=t.int64) # padding
                    
            ],
            dim=0,
        )
        
        label = t.cat(
            [
                t.tensor(dec_input_tokens, dtype=t.int64), # target tokens right translation 1
                self.eos_token, # end
                t.tensor([self.pad_token] * dec_num_padding_tokens, dtype=t.int64) # padding
            ],
            dim=0,
        )
        
        # check seq length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tar_text": tar_text
        }

            
def causal_mask(size):
    mask = t.triu(t.ones((1, size, size)), diagonal=1).type(t.int)
    return mask == 0