from model import make_model
from config import get_config, get_weights_file_path, latest_weights_file_path
from validation import run_validation
from build_tokenizer import get_dataset

import torch as t
import torch.nn as nn


from pathlib import Path
from typing import Any
import warnings
from tqdm import tqdm

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def get_model(config, 
              vocab_size_src: int, 
              vocab_size_tgt: int) -> nn.Module:
    
    model = make_model(vocab_size_src, 
                    vocab_size_tgt, 
                    config['seq_len'], 
                    config['seq_len'], 
                    config['d_model'], 
                    config['layers'], 
                    config['heads'], 
                    config['dropout'], 
                    config['feedforward_dim'])
    return model
    
    
def train_model(config: dict):
    
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')
        raise ValueError('No GPU available !')
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # tensorboard
    writer = SummaryWriter(log_dir=config['exp_name'])
    
    optimizer = t.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    checkpoint = config['checkpoint']
    model_filename = latest_weights_file_path(config) if checkpoint is None else get_weights_file_path(config, checkpoint)
    
    if model_filename:
        checkpoint = t.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    else:
        print("No model loaded")
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    
    # start training 
    for epoch in range(initial_epoch, config['num_epochs']):
        t.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            
            # TODO : trainer
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        print("Running validation")
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        t.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)