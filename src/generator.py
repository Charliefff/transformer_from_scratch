import torch as t
import torch.nn as nn
from tokenizers import Tokenizer
from model import make_model
from config import get_config
from validation import greedy_decode

def load_model(config: dict,
               src_vocab_size:int,
               tar_vocab_size:int) -> nn.Module:

    model = make_model(src_vocab_size, 
                    tar_vocab_size, 
                    config['seq_len'], 
                    config['seq_len'], 
                    config['d_model'], 
                    config['layers'], 
                    config['heads'], 
                    config['dropout'], 
                    config['feedforward_dim'])
    
    if config['checkpoint']:
        checkpoint = t.load(config['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from checkpoint")
        
    else:
        print("No model loaded")
        
    model.eval()
    return model

def predict(model: nn.Module, 
            src_sentence: str, 
            tokenizer_src: Tokenizer, 
            tokenizer_tgt: Tokenizer, 
            max_len: int, 
            device: t.device) -> str:

    src_tokens = tokenizer_src.encode(src_sentence).ids
    src_tensor = t.tensor(src_tokens).unsqueeze(0).to(device) 
    src_mask = t.ones((1, 1, len(src_tokens))).bool().to(device)
    
    predicted_ids = greedy_decode(model, src_tensor, src_mask, tokenizer_src, tokenizer_tgt, max_len, device)
    
    predicted_sentence = tokenizer_tgt.decode(predicted_ids.detach().cpu().numpy())
    
    return predicted_sentence

if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
    config = get_config()
    tokenizer_src = Tokenizer.from_file(config['src_tokenizer'])
    tokenizer_tgt = Tokenizer.from_file(config['tgt_tokenizer'])

    # load pre-trained model
    model = load_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # enter your input
    src_sentence = "食咗飯未呀？"
    max_len = 60
    
    # predict
    predicted_sentence = predict(model, src_sentence, tokenizer_src, tokenizer_tgt, max_len, device)
    
    print()
    print("Translation:")
    print(f"Source: {src_sentence}")
    print(f"Predicted: {predicted_sentence}")
    
