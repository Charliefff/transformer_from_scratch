import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from pathlib import Path

# size(b, seq_len, d_model)
class InputEmbedding(nn.Module):
    
    def __init__(self, 
                 d_model: int, 
                 vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x) -> t.tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
            
class PositionalEncoding(nn.Module):
    
    def __init__(self, 
               d_model: int,
               seq_len: int,
               dropout: float):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # built a pre def dictionary of positional encoding
        # size seq_len x d_model
        pe = t.zeros(seq_len, d_model)
        pos = t.arange(0, seq_len, dtype=t.float).unsqueeze(1) 
        
        # T = 1/10000^(2i/d_model)
        # lnT = -2i/d_model(ln10000)
        # T = exp^(-2i/d_model(ln10000))
        div = t.exp(t.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(pos * div)
        pe[:, 1::2] = t.cos(pos * div)
        pe = pe.unsqueeze(0)# batch size space
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: t.tensor) -> t.tensor:
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        return x
        
# https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
class LayerNormalization(nn.Module):
    
    def __init__(self, 
                 features: int,
                 eps: float = 10e-6) -> None:
        super().__init__()
        
        self.alpha = nn.Parameter(t.ones(features))
        self.beta = nn.Parameter(t.zeros(features))
        self.eps = eps
        
    def forward(self, x: t.tensor) -> t.tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.alpha + self.beta
    
    
class FeedForward(nn.Module):
    
    def __init__(self, 
                 d_model: int, 
                 ff_dim: int, 
                 dropout: float) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: t.tensor) -> t.tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
# @abstractmethod
class MultiHeadAttention(nn.Module):
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(q: t.tensor, 
                  k: t.tensor, 
                  v: t.tensor, 
                  mask: Optional[t.tensor] = None, 
                  dropout: Optional[nn.Dropout] = None) -> t.tensor:
        
        # TODO: attention
        d_k = q.size(-1)
        
        
        # (b, seq_len, d_model) x (b, d_model, seq_len) -> (b, seq_len, seq_len)
        qk = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.tensor.masked_fill_.html#torch.tensor.masked_fill_
            qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        if dropout is not None:
            qk = dropout(qk)

        return qk @ v, qk
    
    def forward(self,
                q: t.tensor,
                k: t.tensor,
                v: t.tensor,
                mask: Optional[t.tensor] = None) -> t.tensor:
        
        # TODO: multi head attention
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # split the d_model into num_heads
        # (b, seq_len, d_model)     
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        
        x, _ = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        # (b, h, seq_len, d_k) -> (b, seq_len, h, d_k) -> (b, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        
        return self.w_o(x) #merge the heads back to d_model
    
class Residual(nn.Module):
    
    def __init__(self,
                 features: int,
                 dropout: float) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self,
                x: t.tensor,
                sublayer: nn.Module) -> t.tensor:
        
        # TODO
        return x + self.dropout(sublayer(self.norm(x)))  
    
class EncoderBlock(nn.Module):
    
    def __init__(self, 
                 features:int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout: float) -> None:
        super().__init__()
        
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList([Residual(features, dropout) for _ in range(2)])
        
    def forward(self,
                x: t.tensor,
                mask: Optional[t.tensor] = None) -> t.tensor:
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    
    def __init__(self, 
                 features: int,
                 layers: nn.ModuleList) -> None:
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self,
                x: t.tensor,
                mask: Optional[t.tensor] = None) -> t.tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, 
                 features:int,
                 self_attn: MultiHeadAttention,
                 cross_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout: float) -> None:
        super().__init__()
        
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.ModuleList([Residual(features, dropout) for _ in range(3)])
        
    def forward(self,
                x: t.tensor,
                enc: t.tensor,
                src_mask: Optional[t.tensor] = None,
                tar_mask: Optional[t.tensor] = None) -> t.tensor:
        
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, tar_mask))
        x = self.residual[1](x, lambda x: self.cross_attn(x, enc, enc, src_mask))
        x = self.residual[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self,
                 features: int,
                 layers: nn.ModuleList) -> None:
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, 
                x: t.tensor,
                enc: t.tensor,
                self_mask: Optional[t.tensor] = None,
                cross_mask: Optional[t.tensor] = None) -> t.tensor:
        for layer in self.layers:
            x = layer(x, enc, self_mask, cross_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    
    def __init__(self, 
                 d_model: int,
                 vocab_size: int) -> None:
        super().__init__()
        
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, 
                x: t.tensor) -> t.tensor:
        return F.log_softmax(self.linear(x), dim=-1)
        
                
class Transformer(nn.Module):
    
    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embedding: InputEmbedding,
                 tar_embedding: InputEmbedding,
                 src_pos: PositionalEncoding,
                 tar_pos: PositionalEncoding,
                 proj: ProjectionLayer) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tar_embedding = tar_embedding
        self.src_pos = src_pos
        self.tar_pos = tar_pos
        self.proj = proj
        
    def encode(self, 
               src: t.tensor,
               src_mask: t.tensor) -> t.tensor:
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)# (b, seq_len, d_model)
    
    def decode(self, 
               enc: t.tensor,
               self_mask: t.tensor,
               tar: t.tensor,
               cross_mask: t.tensor) -> t.tensor:
               
        tar = self.tar_embedding(tar)
        tar = self.tar_pos(tar)
        return self.decoder(tar, enc, self_mask, cross_mask) # (b, seq_len, d_model)
    
    def project(self, 
                x: t.tensor) -> t.tensor:
        
        # (b, seq_len, d_model) -> (b, seq_len, vocab_size)
        return self.proj(x)

def make_model(src_vocab_size: int,
                tar_vocab_size: int,
                src_seq_len: int,
                tar_seq_len: int,
                d_model: int=512,
                N: int=3,
                heads: int=6,
                dropout: float=0.1,
                ff_dim: int=1024) -> Transformer:
    
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tar_embedding = InputEmbedding(d_model, tar_vocab_size)
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tar_pos = PositionalEncoding(d_model, tar_seq_len, dropout)
    
    encoder_blocks = list()
    decoder_blocks = list()
    
    for _ in range(N):
        # init encoder
        encoder_att_block = MultiHeadAttention(d_model, heads, dropout)
        ff_block = FeedForward(d_model, ff_dim, dropout)
        encoder_block = EncoderBlock(d_model, encoder_att_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)
        
        # init decoder
        decoder_self_att_block = MultiHeadAttention(d_model, heads, dropout)
        decoder_cross_att_block = MultiHeadAttention(d_model, heads, dropout)
        decoder_ff_block = FeedForward(d_model, ff_dim, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_att_block, decoder_cross_att_block, decoder_ff_block, dropout)
        decoder_blocks.append(decoder_block)
        
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tar_vocab_size)
    
    model = Transformer(encoder, decoder, src_embedding, tar_embedding, src_pos, tar_pos, projection_layer)
    
 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print("Initialized model with random weights.")
        
    return model




def test_model():
    src_vocab_size = 10
    tar_vocab_size = 10
    src_seq_len = 5
    tar_seq_len = 5
    d_model = 8
    N = 2
    heads = 2
    dropout = 0.1
    ff_dim = 32

    model = make_model(src_vocab_size, tar_vocab_size, src_seq_len, tar_seq_len, d_model, N, heads, dropout, ff_dim)
    src = t.randint(0, src_vocab_size, (2, src_seq_len))
    tar = t.randint(0, tar_vocab_size, (2, tar_seq_len))
    src_mask = t.ones((2, 1, src_seq_len)).bool()
    tar_mask = t.ones((2, 1, tar_seq_len)).bool()

    enc = model.encode(src, src_mask)
    dec = model.decode(enc, src_mask, tar, tar_mask)
    out = model.project(dec)
    
    print()
    print("Final output size: ", out.shape)  # 應該輸出 (2, tar_seq_len, tar_vocab_size)
    if out.shape == (2, tar_seq_len, tar_vocab_size):
        
        print("恭喜你的模型通過測試！")
    else:
        print("繼續加油吧")        
    print()

if __name__ == "__main__":
    test_model()
        