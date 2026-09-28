from model.config import ModelConfig
from tokenizer.basic_tokenizer import BaseTokenizer
from tokenizer.regex_tokenizer import RegexTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import LanguageModel
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import trackio
from accelerate import Accelerator

def calculate_loss(target, logits):
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    target = target.view(B*T)
    loss = F.cross_entropy(logits, target)
    return loss

class PretrainDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        super().__init__()
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.data = np.load(self.file_path, mmap_mode="r")
    def __len__(self):
        return len(self.data) - self.max_seq_len
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def run(args):
    trackio.init(
        project="nanoGPT",
        config={"steps": args.train_steps, "learning_rate":args.learning_rate, "batch_size": args.batch_size}
    )
    accelerator  = Accelerator()
    model_config = ModelConfig(
                vocab_size=args.vocab_size if args.vocab_size else 512, 
                max_seq_len=args.max_seq_len if args.max_seq_len else 512, 
                d_model=args.d_model if args.d_model else 512,
                n_layers=args.n_layers if args.n_layers else 8,
                n_heads=args.n_heads if args.n_heads else 4,
                n_kv_heads=args.n_kv_heads if args.n_kv_heads else 4,
                mlp_ratio=args.mlp_ratio if args.mlp_ratio else 4.0,
                mlp_type=args.mlp_type if args.mlp_type else 'ffn',
                use_rope=False,
                norm_type='layernorm',)



    model = LanguageModel(args=model_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    accelerator.print(f"Total params: {total_params}M")

    train_dataset = PretrainDataset(args.train_data_path, args.max_seq_len)
    val_dataset = PretrainDataset(args.val_data_path, args.max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer
    )

    progress = tqdm(range(args.train_steps))
    for step in progress:
        for input, target in train_dataloader:
            input, target = input.to(accelerator.device), target.to(accelerator.device)
            logits = model(input)
            optimizer.zero_grad()
            loss = calculate_loss(target=target, logits=logits)
            accelerator.backward(loss)
            optimizer.step()

            metrics = {
                "train_loss": loss.item(),
            }

            if step%args.validation_step==0:
                accelerator.print("Running evaluation")
                model.eval()
                with torch.no_grad():
                    accelerator.save_state("model2")
                    total_val_loss = 0
                    for val_input, val_target in tqdm(val_dataloader):
                        val_input, val_target = val_input.to(accelerator.device), val_target.to(accelerator.device)
                        logits = model(val_input)
                        val_loss = calculate_loss(target=val_target, logits=logits)
                        total_val_loss+=val_loss.item()
                model.train()
                metrics.update({
                    "val_loss": total_val_loss})
            trackio.log(metrics, step=step)
            progress.set_postfix(
                train_loss=f"{loss.item():.4f}"
            )
    trackio.finish()


def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--mlp_type", type=str, default='ffn')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--validation_step", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_data_path", type=str, default='data/regex_pretrain_train.npy')
    parser.add_argument("--val_data_path", type=str, default='data/regex_pretrain_val.npy')

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()
