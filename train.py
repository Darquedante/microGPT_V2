from gpt import GPT
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from transformers import GPT2TokenizerFast
from config import GPTConfig
import numpy as np
import math
import torch
import torch.distributed as dist
import os
import wandb
import json
import argparse

# DDP: Parse command line arguments for DDP
parser = argparse.ArgumentParser(description='Distributed GPT Training')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank of the process. Should be provided by the launch utility.')
args = parser.parse_args()

# DDP: Initialize process group
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

# DDP: Setup device for the current process
device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

max_length = 512
tokenizer_path="tokenizer/tokenizer.json"
tokenizer = GPT2TokenizerFast(
    tokenizer_file=tokenizer_path,
    pad_token="[PAD]",
    padding_side="right",
    model_max_length=max_length,
)

# hyperparameters
config_file="config/config.json"

# model config
max_length = 512
batch_size = 12
num_accumulation_steps = 5 * 8
n_steps = 600000
epochs = 1
grad_norm_clip = 1.0
checkpoint_interval = 10000
eval_interval = 2000
eval_iters = 100
vocab_size = len(tokenizer.get_vocab())
save_directory = "models"
# adamw optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True
config_lr = {
    'warmup_steps': 2000,
    'lr_decay_steps': n_steps,
    'min_lr': 6e-5,
    'lr': learning_rate
}

# checkpoints
checkpoint=False
model_path=""

# wandb
wandb_log=True
project="GPT Training"
name=""
resume=False
id=None

# dataset
dataset = "memmap"
data_dir = 'datasets'
if dataset == "memmap":
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'validation.bin'), dtype=np.uint16, mode='r')
    # For 'memmap', ensure custom logic later to handle data distribution among processes
elif dataset == "huggingface":
    from datasets import load_from_disk
    from torch.utils.data.distributed import DistributedSampler

    num_workers = 4
    train_data = load_from_disk("datasets/train")
    val_data = load_from_disk("datasets/validation")
    
    train_data.set_format(type="torch", output_all_columns=True)
    val_data.set_format(type="torch", output_all_columns=True)

    # Use DistributedSampler for distributed training
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)  # Typically, validation data isn't shuffled

    dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    dataloader_val = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
    
    # Remove 'shuffle=True' from DataLoader as it's handled by DistributedSampler
    # Convert iterators to DataLoader directly
elif dataset == "torch_dataset":
    from dataset import CorpusDataset
    from utils import pad_collate
    from sklearn.model_selection import train_test_split
    # [The 'torch_dataset' section will need adjustments similar to the 'huggingface' section if applicable]
else:
    print("Invalid dataset.")


def get_batch(split):
    if dataset == "memmap":
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - max_length, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i:i+max_length]).astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy((data[i+1:i+1+max_length]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
    else:
        inpt = next(train_data) if split == 'train' else next(val_data)
        x = inpt[:, :max_length]
        y = inpt[:, 1:max_length+1]
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters):
    model.eval()
    out = {}
    with tqdm(total=eval_iters * 2, unit="batch", position=1, leave=True) as eval_steps:
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            k = 0
            while k < eval_iters:
                xBatch, yBatch = get_batch(split)
                _, loss = model(xBatch, target=yBatch)
                losses[k] = loss.item()
                k += 1
                eval_steps.set_description('Loss Estimation')
                eval_steps.update(1)
                eval_steps.set_postfix(
                    step_loss=loss.item(), learning_rate=lr, target=split)
            out[split] = losses.mean()
        eval_steps.clear()
    model.train()
    return out


def get_lr_cosine_warmup(config_lr, steps):
    if steps < config_lr['warmup_steps']:
        return config_lr['lr'] * steps / config_lr['warmup_steps']
    if steps > config_lr['lr_decay_steps']:
        return config_lr['min_lr']
    decay_ratio = (steps - config_lr['warmup_steps']) / \
        (config_lr['lr_decay_steps'] - config_lr['warmup_steps'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config_lr['min_lr'] + coeff * (config_lr['lr'] - config_lr['min_lr'])


if __name__ == '__main__':
    # Initialize DDP environment
    torch.cuda.set_device(args.local_rank)  # Set current device to local rank
    dist.init_process_group(backend='nccl')  # Initialize the process group

    with open(config_file, 'r') as f:
        config = json.load(f)

    config = GPTConfig(**config)
    if args.local_rank == 0:
        print(f'Max length: {config.max_length} Vocab Size: {config.vocab_size}')
    model = GPT(config)
    
    # Move and wrap model for DDP
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    step = 0
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Load checkpoint only on the master process
    if checkpoint and args.local_rank == 0:
        model_stat = torch.load(model_path, map_location=device)
        model.load_state_dict(model_stat["model_state_dict"])
        step = model_stat['step']
        optimizer.load_state_dict(model_stat["optimizer_state_dict"])
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # Initialize wandb only on the master process
    if wandb_log and args.local_rank == 0:
        wandb.init(project=project, resume=resume, id=id, name=name, config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_fn": "crossentropyloss",
        })

    if args.local_rank == 0:
        print('Device: ', device)
        # Example tokenizer usage omitted for brevity

        "---------------GPT Model----------------")))
# AMP & Gradient Accumulation to prevent CUDA Out of Memory Error
scaler = torch.cuda.amp.GradScaler()

with tqdm(total=n_steps, unit="batch", position=0, leave=True) as t_steps:
    t_steps.update(step)
    while step < n_steps:
        xBatch, yBatch = get_batch('train')

        lr = get_lr_cosine_warmup(config_lr, step + 1) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.cuda.amp.autocast(device_type=device_type):
            logits, loss = model(xBatch, target=yBatch)

        scaler.scale(loss / num_accumulation_steps).backward()
        t_steps.update(1)
        t_steps.set_postfix(step_loss=loss.item(), learning_rate=lr)
        if ((step + 1) % num_accumulation_steps == 0):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Checkpoint saving only by master process
        if args.local_rank == 0 and (step % checkpoint_interval == 0):
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{save_directory}/microGPT-step-{step}.pth")

        # Evaluation and logging only by master process
        if args.local_rank == 0 and step % eval_interval == 0:
            losses = estimate_loss(model, eval_iters=eval_iters)
            print(f"Current step {step}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, lr: {lr}")
            if wandb_log:
                metrics = {
                    "train/train_loss": losses["train"],
                    "train/steps": step,
                    "validation/val_loss": losses["val"],
                    "validation/steps": step,
                    "models/lr": lr
                }
                wandb.log(metrics)
        step += 1

# Final evaluation and logging only by master process
if args.local_rank == 0:
    losses = estimate_loss(model, eval_iters=eval_iters)
    print(f"Final train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    # Generation example omitted for brevity
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{save_directory}/microGPT.pth")

