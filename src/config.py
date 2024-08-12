from pathlib import Path
def get_config():
    return {
        "batch_size": 100,
        "num_epochs": 20,
        "lr": 5e-4,
        "seq_len": 300,
        "d_model": 512,
        "layers": 6,
        "heads": 8,
        "dropout": 0.1,
        "feedforward_dim": 1024,
        "datasource": 'raptorkwok/cantonese-traditional-chinese-parallel-corpus',
        "lang_src": 'yue',
        "lang_tgt": 'zh',
        "model_folder": "weights",
        "model_basename": "transformermodel_",
        "checkpoint": "/data/tzeshinchen/AI_example/transformer/src/weights/train2/transformermodel_03.pt",
        "tokenizer_file": "tokenizer_{0}.json",
        "exp_name": "logs/transformer",
        "src_tokenizer": "./yue_tokenizer.json",
        "tgt_tokenizer": "./zh_tokenizer.json",
        
        
    }
    
    
def get_weights_file_path(config, 
                         epoch:str):
    
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])