import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import tiktoken

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, dropout=0.0):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout  # 0.0, Karpathy doesn't use dropout in the video



class CausalSelfAttentionMarcin(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('bias', torch.tril(torch.ones((1, 1, config.block_size, config.block_size))))
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)  # B, T, nh*hs
        q = q.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        k = k.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        v = v.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        q = q.transpose(1, 2)  # B,nh,T,hs
        k = k.transpose(1, 2)  # B,nh,T,hs
        v = v.transpose(1, 2)  # B,nh,T,hs

        H = k.shape[-1]
        W_affin = q @ k.mT / H**0.5  # B,nh,T,hs @ B,nh,hs,T -> B,nh,T,T
        W_affin = W_affin.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        W_affin = torch.softmax(W_affin, dim=-1)  # B,nh,T,T
        W_affin = self.drop(W_affin)

        y = W_affin @ v    # B,nh,T,T @ B,nh,T,hs -> B,nh,T,hs
        y = y.transpose(1, 2)  # B,T,nh,hs
        y = y.contiguous()
        y = y.view(B,T,C)

        out = self.c_proj(y)
        return out

class MLP(nn.Module):
    """Linear transform and activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionMarcin(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))        # B,T,E pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        assert idx.dtype == torch.long
        assert targets is None or targets.dtype == torch.long
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.transformer.wte(idx)    #  B,T,E <- B,T
        pos_emb = self.transformer.wpe(torch.arange(T, device=device))    #  B,T,E <- B,T
        x = tok_emb + pos_emb  #  B,T,E
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)   # B,T,V <- B,T,E

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_)
            return logits, loss
        
    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name == 'gpt2'  # for now
        from transformers import GPT2LMHeadModel
       
        # HF Model
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

        # Our Model
        model = GPTModel(GPTConfig(
            block_size=1024,     # max context length, max len feed into the model,
            vocab_size=50257,    # 256 original, 50_000 merges, 1 <|end_of_doc|> token,
            n_layer=12,
            n_head=12,           # head size 384/6=64,
            n_embd=768,          # size of embeddings, i.e. 'first layer',
        ))
        model.eval()

        # Load weights from HF model
        our_sd = model.state_dict()
        our_keys = set(k for k in our_sd.keys() if not k.endswith('.attn.bias'))
        hf_sd = model_hf.state_dict()
        hf_keys = set(hf_sd.keys())
        assert our_keys == hf_keys
        for key, hf_val in hf_sd.items():
            hf_shape = hf_val.shape
            our_shape = our_sd[key].shape
            transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            if any(key.endswith(suffix) for suffix in transpose):
                assert hf_shape[::-1] == our_shape
                our_sd[key].copy_(hf_val.t().clone())
            else:
                assert hf_shape == our_shape
                our_sd[key].copy_(hf_val.clone())

        return model



    def generate(self, idx, n_seq, max_tokens):
        """Generate max_tokens starting from idx[B,T]"""
        # assert idx.shape == (n_batch, n_seq)
        assert idx.dtype == torch.long
        assert isinstance(max_tokens, int)

        for _ in range(max_tokens):

            # Sliding window over idx
            idx_tail = idx[:, -n_seq:]

            # Model Output
            logits, _ = self(idx_tail)      # B,T,C <- B,T

            # Discard all but last step
            logits = logits[:, -1, :]  # B,C <- B,T,C

            probs = F.softmax(logits, dim=-1)  # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1

            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1

        return idx




@torch.no_grad()
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = GPTConfig(
        block_size=1024,     # max context length, max len feed into the model,
        vocab_size=50257,    # 256 original, 50_000 merges, 1 <|end_of_doc|> token,
        n_layer=12,
        n_head=12,           # head size 384/6=64,
        n_embd=768,          # size of embeddings, i.e. 'first layer',
    )
    n_batch = 4

    # Random input, long tensor with values in [0, n_vocab)
    x = torch.randint(0, config.vocab_size, (n_batch, config.block_size))
    x = x.to(device)
    
    # HF Model
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    model_hf.to(device)
    model_hf.eval()
    gpt2_logits = model_hf(input_ids=x).logits
    print("GPT2 logits shape:", gpt2_logits.shape)   # B,T,C
    print("GPT2 logits dtype:", gpt2_logits.dtype)   # B,T,C
    print(gpt2_logits[0,0,0:10])

    # Our Model
    model = GPTModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    logits, loss = model(x)
    print("logits shape:", logits[0].shape)   # B,T,C
    print("logits dtype:", logits[0].dtype)   # B,T,C
    print(logits[0,0,0:10])

if __name__ == "__main__":
    main()
