import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP



class MLP(nn.Module):
    """Linear transform and activation"""
    def __init__(self, n_in, n_hid, n_out):
        super().__init__()
        self.ln1 = nn.Linear(n_in, n_hid)
        self.act1 = nn.Tanh()
        self.ln2 = nn.Linear(n_hid, n_out)
    
    def forward(self, x, targets=None):
        x = self.ln1(x)
        x = self.act1(x)
        logits = self.ln2(x)
        if targets is None:
            return logits, None
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            return logits, loss


def make_moons(n_samples=1000, noise=0.1):
    n_samples_per_moon = n_samples // 2
    theta1 = torch.linspace(0, torch.pi, n_samples_per_moon)
    x1, y1 = torch.cos(theta1), torch.sin(theta1)
    theta2 = torch.linspace(0, torch.pi, n_samples - n_samples_per_moon)
    x2, y2 = 1 - torch.cos(theta2), 0.5 - torch.sin(theta2)
    X = torch.vstack([torch.column_stack([x1, y1]), torch.column_stack([x2, y2])])
    y = torch.cat([torch.zeros(n_samples_per_moon), torch.ones(n_samples - n_samples_per_moon)])
    return X, y.long()

class DataLoader():
    def __init__(self, batch_size, dataset_size, local_rank, world_size):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.X, y = make_moons(dataset_size, noise=0.05)
        self.Y = y.float().unsqueeze(1)
        self.pos = self.batch_size * local_rank
    
    def get_batch(self):
        x = self.X[self.pos:self.pos+self.batch_size]
        y = self.Y[self.pos:self.pos+self.batch_size]

        self.pos += self.batch_size * self.world_size
        if self.pos + (self.batch_size * self.world_size) > self.dataset_size:
            self.pos = 0

        return x, y


from contextlib import nullcontext

def main():
    # DDP Init
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        ddp_master = ddp_rank == 0  # is this a master?
        device = f'cuda:{ddp_local_rank}'
        assert torch.cuda.is_available()
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_master = ddp_rank == 0  # is this a master?
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{ddp=} {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")

    # Random Seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Batching is affected
    dataset_size = 1024
    total_batch_size = 64
    micro_batch = 8     # micro batch
    block_size = 2     # just (x1,x2) pair per example
    grad_accum = total_batch_size // (micro_batch*block_size*ddp_world_size)
    print(f"{total_batch_size=}, {block_size=}, {micro_batch=}, {grad_accum=}")

    # Data loader
    data_loader = DataLoader(
        batch_size=micro_batch,
        dataset_size=dataset_size,
        local_rank=ddp_local_rank,
        world_size=ddp_world_size
    )

    # Model needs to be wrapped
    model = MLP(n_in=2, n_hid=16, n_out=1)
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    model_raw = model.module if ddp else model

    # Learning Rate
    lr = 0.1
    # max_steps = 10*dataset_size // total_batch_size
    # max_steps = 10_000
    max_steps = 1000

    print('---')
    #print(data_loader.X[:5])
    print(ddp_local_rank, model_raw.ln1.weight[:5].detach().cpu())
    print('---')

    # Forward pass stays transparent

    # Backward pass by default auto-distributes gradients in loss.backwards
    # because we will use grad accumulation, we don't want to sync grads in each loss.backward,
    # instead we want to average gradients on the last step.
    # Officially do this with ddp.no_sync() context manager

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for i in range(max_steps):
        loss_accum = 0.0
        optimizer.zero_grad()
        for ii in range(grad_accum):
            x, y = data_loader.get_batch()
            x, y = x.to(device), y.to(device)
            
            # Sync only if DDP and last backward step
            context = model.no_sync() if (ddp and ii < grad_accum - 1) else nullcontext()
            with context:
                logits, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
            loss_accum += loss.detach()
            
        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        
        optimizer.step()        
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()

        tok_processed = micro_batch * block_size * grad_accum * ddp_world_size
        if ddp_master:
            if i % 1000 == 0 or i == max_steps-1:
                print(f"{i:4d}:, L={loss_accum.item():.6f}, tok={tok_processed}")

    print('---')
    #print(data_loader.X[:5])
    print(ddp_local_rank, model_raw.ln1.weight[:5].detach().cpu())
    print('---')

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
