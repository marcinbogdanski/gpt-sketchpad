import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken

class CausalSelfAttentionMarcin(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, block_size, n_head, n_embd, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('bias', torch.tril(torch.ones((1, 1, block_size, block_size))))
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

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
        W_affin = self.drop_1(W_affin)

        y = W_affin @ v    # B,nh,T,T @ B,nh,T,hs -> B,nh,T,hs
        y = y.transpose(1, 2)  # B,T,nh,hs
        y = y.contiguous()
        y = y.view(B,T,C)
        y = self.drop_2(y)

        out = self.c_proj(y)
        return out

class FeedForward(nn.Module):
    """Linear transform and activation"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4*n_embd)
        self.act = nn.ReLU()
        self.c_proj = nn.Linear(4*n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, dropout):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionMarcin(
            block_size=block_size, n_head=num_heads, n_embd=n_embd, dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))        # B,T,E pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, block_size, n_vocab, n_embd, num_heads, n_layer, dropout):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.Sequential(
            *[Block(block_size=block_size, n_embd=n_embd, num_heads=num_heads, dropout=dropout) for _ in range(n_layer)]            
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, idx, targets=None):
        assert idx.dtype == torch.long
        assert targets is None or targets.dtype == torch.long
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.wte(idx)    #  B,T,E <- B,T
        pos_emb = self.wpe(torch.arange(T, device=device))    #  B,T,E <- B,T
        x = tok_emb + pos_emb  #  B,T,E
        x = self.h(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # B,T,V <- B,T,E

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_)
            return logits, loss
    
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

    n_vocab = 50257    # 256 original, 50_000 merges, 1 <|end_of_doc|> token
    n_batch = 4        # mini-bach, how many in parallel
    block_size = 1024       # max context length, max len feed into the model
    n_embd = 768         # size of embeddings, i.e. 'first layer'
    num_heads = 12     # head size 384/6=64
    num_layer = 12
    dropout = 0.0



    # Random input, long tensor with values in [0, n_vocab)
    x = torch.randint(0, n_vocab, (n_batch, block_size))
    x = x.to(device)
    
    
    # HF Model
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=n_vocab,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=num_layer,
        n_head=num_heads,
    )
    model_hf = GPT2LMHeadModel(config)
    model_hf.to(device)
    model_hf.eval()

    # Inference
    gpt2_logits = model_hf(input_ids=x).logits
    print("GPT2 logits shape:", gpt2_logits.shape)   # B,T,C
    print("GPT2 logits dtype:", gpt2_logits.dtype)   # B,T,C
    print(gpt2_logits[0,0,0:10])

    # Our Model
    model = TransformerModel(
        block_size=block_size,
        n_vocab=n_vocab,
        n_embd=n_embd,
        num_heads=num_heads,
        n_layer=num_layer,
        dropout=dropout
    )
    model.to(device)
    model.eval()

    # Load weights from HF model
    our_sd = model.state_dict()
    our_keys = set(our_sd.keys())
    hf_sd = model_hf.state_dict()
    hf_keys = set(k.replace('transformer.', '') for k in hf_sd.keys())

    # Keys in ours but not in HF, and vice versa
    print("Keys in our model not found in HF model:", our_keys - hf_keys)
    print("Keys in HF model not found in our model:", hf_keys - our_keys)
    print('---')
    for hf_key, hf_val in hf_sd.items():
        our_key = hf_key.replace('transformer.', '')
        if our_key not in our_sd:
            print(f"Skipping key {our_key} not found in our model")
            continue
        hf_shape = hf_val.shape
        our_shape = our_sd[our_key].shape
        if hf_shape == our_shape:
            our_sd[our_key] = hf_val.clone()
        elif hf_shape == (our_shape[1], our_shape[0]):
            our_sd[our_key] = hf_val.t().clone()
        else:
            print(f"Shape mismatch for key {our_key}: HF shape {hf_shape}, our shape {our_shape}")
    model.load_state_dict(our_sd)
    print('---')

    logits, loss = model(x)
    print("logits shape:", logits[0].shape)   # B,T,C
    print("logits dtype:", logits[0].dtype)   # B,T,C
    print(logits[0,0,0:10])

if __name__ == "__main__":
    main()
