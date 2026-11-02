import os
from src.model.config import ModelConfig
from src.tokenizer.regex_tokenizer import RegexTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.transformer import LanguageModel
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import bitsandbytes as bnb
from src.utils.utils import get_latest_checkpoint, cleanup_checkpoints
import math

def calculate_loss(target, logits):
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    target = target.view(B*T)
    loss = F.cross_entropy(logits, target)
    return loss


def load_state(model, optimizer, lr_scheduler, path):
    state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler.load_state_dict(state_dict['lr_scheduler'])


class PretrainDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        super().__init__()
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.data = np.load(self.file_path, mmap_mode="r")
    def __len__(self):
        return (len(self.data) - self.max_seq_len)//self.max_seq_len
    def __getitem__(self, idx):
        start = idx*self.max_seq_len
        end = start + self.max_seq_len + 1
        chunk = self.data[start:end]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def run(args):
    os.environ["TRACKIO_DIR"] = args.log_dir
    import trackio
    run_name = "pretraining_baseline"
    trackio.init(
        project=args.project_name,
        name=run_name,
        group="pretraining", 
        resume='allow',
        config={"steps": args.max_train_steps, "learning_rate":args.learning_rate, "batch_size": args.batch_size}
    )

    device = args.device 
    model_config = ModelConfig(
                vocab_size=args.vocab_size if args.vocab_size else 512, 
                max_seq_len=args.max_seq_len if args.max_seq_len else 512, 
                d_model=args.d_model if args.d_model else 512,
                n_layers=args.n_layers if args.n_layers else 8,
                n_heads=args.n_heads if args.n_heads else 4,
                n_kv_heads=args.n_kv_heads if args.n_kv_heads else 4,
                mlp_ratio=args.mlp_ratio if args.mlp_ratio else 4.0,
                mlp_type=args.mlp_type if args.mlp_type else 'ffn',
                pe_type=args.pe_type if args.pe_type else 'sincos',
                use_rope=False,
                norm_type='layernorm',)



    model = LanguageModel.from_config(model_config).to(device)
    tokenizer = RegexTokenizer()
    tokenizer.load(args.tokenizer_path)
    test_ids = torch.tensor([tokenizer.encode(args.test_text)]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f"Total params: {total_params}M")

    train_dataset = PretrainDataset(args.train_data_path, args.max_seq_len)
    val_dataset = PretrainDataset(args.val_data_path, args.max_seq_len)

    print(f"Total Training tokens: {len(train_dataset.data):,}")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Tokens per sequence: {train_dataset.max_seq_len}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)


    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_train_steps)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest" and args.resume_from_checkpoint != "best":
            path, global_step = args.resume_from_checkpoint, int(args.resume_from_checkpoint.split('-')[-1])

        elif args.resume_from_checkpoint == "latest":
            path, global_step = get_latest_checkpoint(args.ckpt_dir, 'ckpt-')

        elif args.resume_from_checkpoint == "best":
            path, global_step = get_latest_checkpoint(args.ckpt_dir, 'best-ckpt-')

        if path is None:
            print("Starting a new training run.")
            initial_global_step = 0
            first_epoch = 0
        else:
            print(f"Resuming from checkpoint {path}")
            load_state(model, optimizer, lr_scheduler, path)
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
        first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    curr_best=10
    global_step=initial_global_step
    for epoch in range(first_epoch, args.num_train_epochs):
        for input, target in train_dataloader:
            input, target = input.to(device), target.to(device)
            logits = model(input)
            loss = calculate_loss(target=target, logits=logits)
            train_loss = loss.item()
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            if global_step%args.gradient_accumulation_steps==0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            metrics = {
                "train_loss": train_loss,
            }
            
            if global_step%args.checkpoint_step==0:
                state_dict = {'model':model.state_dict(),
                            'optimizer':optimizer.state_dict(), 
                            'lr_scheduler':lr_scheduler.state_dict()}
                ckpt_path = os.path.join(args.ckpt_dir, f'ckpt-{global_step}')
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(state_dict, os.path.join(ckpt_path, 'state_dict.pt'))
                model.save_pretrained(os.path.join(ckpt_path, 'transformer'))
                cleanup_checkpoints(args.ckpt_dir, 'ckpt')
            progress_bar.update(1)
            global_step+=1
            if global_step%args.validation_step==0:
                print("Running evaluation")
                model.eval()
                with torch.no_grad():
                    total_val_loss = 0
                    for val_input, val_target in tqdm(val_dataloader):
                        val_input, val_target = val_input.to(device), val_target.to(device)
                        logits = model(val_input)
                        val_loss = calculate_loss(target=val_target, logits=logits)
                        total_val_loss+=(val_loss.item() /len(val_dataloader))
                    
                    generated_ids = model.generate(test_ids).squeeze(0).tolist()
                    generated_text = tokenizer.decode(generated_ids)
                    metrics.update({
                                "generations": trackio.Table(
                                    columns=["Prompt", "Generated Text"],
                                    data=[[args.test_text, generated_text]]
                                )
                            })

                    if total_val_loss<curr_best:
                        curr_best = total_val_loss
                        ckpt_path = os.path.join(args.ckpt_dir, f'best-ckpt-{global_step}')
                        os.makedirs(ckpt_path, exist_ok=True)
                        state_dict = {'model':model.state_dict(),
                                    'optimizer':optimizer.state_dict(), 
                                    'lr_scheduler':lr_scheduler.state_dict()}
                        torch.save(state_dict, os.path.join(ckpt_path, 'state_dict.pt'))
                        model.save_pretrained(os.path.join(ckpt_path, 'transformer'))
                        cleanup_checkpoints(args.ckpt_dir, 'best-ckpt')

                model.train()
                metrics.update({
                    "val_loss": total_val_loss})
            trackio.log(metrics, step=global_step)
    
            progress_bar.set_postfix(
                train_loss=f"{loss.item():.4f}"
            )
            if global_step >= args.max_train_steps:       
                break
    trackio.finish()


def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--vocab_size", type=int, default=530)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--mlp_type", type=str, default='ffn')
    parser.add_argument("--pe_type", type=str, default='sincos')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--validation_step", type=int, default=5000)
    parser.add_argument("--checkpoint_step", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_data_path", type=str, default='data/regex_pretrain_train.npy')
    parser.add_argument("--val_data_path", type=str, default='data/regex_pretrain_val.npy')
    parser.add_argument("--log_dir", type=str, default='pretrain_logs')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--project_name", type=str, default='nanoLM')
    parser.add_argument("--experiment_name", type=str, default='pretrain')
    parser.add_argument("--ckpt_dir", type=str, default='pretrain_checkpoints')
    parser.add_argument("--resume_from_checkpoint", type=str, default='latest')
    parser.add_argument("--test_text", type=str, default='Hi there, How are you?')
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_model/regex", help="Path to load the tokenizer.")

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()
