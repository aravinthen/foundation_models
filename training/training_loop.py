#
# Program name: training_loop.py
# Description:  A common training loop for a GPT style transformer.
#

import torch
from torch.distributed.checkpoint import optimizer
from torch.utils.data import DataLoader
import tiktoken

import matplotlib.pyplot as plt

from llms.BabyGPT import (BabyGPTModel, generate_text, text_to_token_ids, token_ids_to_text)
from encoding.simple_data import SimpleDataset

def create_dataloader(txt, tokenizer, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    """
    Admittedly I got this one straight from Raschka - minor tweaks
    :param txt:
    :param tokenizer:
    :param batch_size:
    :param max_length:
    :param stride:
    :param shuffle:
    :param drop_last:
    :param num_workers:
    :return:
    """

    # Create dataset
    dataset = SimpleDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def batch_loss(input_batch, target_batch, model, device):
    total_loss = 0
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    # flatten combines the first and second dimension, effectively generating a single vector
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(0, 1))

    return loss

def calc_loss_loader(data_loader, model, device, num_batches):
    """
    iterates over all batches n a given data loader and accumulates the loss
    :param data_loader:
    :param model:
    :param device:
    :param num_batches:
    :return:
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')

    # case handling when the number of batches isn't the same as the batches produced by the
    # data loader
    num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break

    # return the average loss
    return total_loss/num_batches

def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs): # the number of times the network is trained: simpler than rl!
        model.train() # train mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset the gradients

            # calculate the loss
            loss = batch_loss(input_batch, target_batch, model, device)

            # calculate loss gradients
            loss.backward()

            optimizer.step() # update model weights

            tokens_seen += input_batch.numel()
            global_step +=1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)

                print(f"Episode {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Carry out loss calculation with frozen weights.
    :param model:
    :param train_loader:
    :param val_loader:
    :param device:
    :param eval_iter:
    :return:
    """
    model.eval() # disables dropout

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

        model.train() # reset training mode

    return train_loss, val_loss

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plot the losses against the epochs.
    :param epochs_seen:
    :param token_seen:
    :param train_losses:
    :param val_losses:
    :return:
    """
    fig, ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(epochs_seen, train_losses, val_losses)

    ax1.legend(loc='upper right')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twiny()
    ax2.plot(epochs_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    my_config = {
        "vocab_size": 50257,  # the vocabulary size
        "context_length": 256,  # context length for maximum sequence
        "emb_dim": 768,  # the embedding dimension
        "n_heads": 12,  # number of attention heads
        "n_layers": 12,  # number of transformer blocks
        "drop_rate": 0.1,  # rate of dropout
    }

    torch.manual_seed(42)
    model = BabyGPTModel(my_config)
    my_tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()

    # proper test
    initial = "You got a bee on your"

    encoded_tensor = text_to_token_ids(initial, tokenizer_f=my_tokenizer)

    out = generate_text(model=model,
                        idx=encoded_tensor,
                        max_new_tokens=1,
                        context_size=my_config["context_length"])

    decoded_tensor = token_ids_to_text(out, tokenizer_f=my_tokenizer)

    print("Output:", decoded_tensor)
    print("Output length:", len(decoded_tensor[0]))

    file_path = "../the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]

    train_loader = create_dataloader(train_data,
                                     my_tokenizer,
                                     2,
                                     max_length=int(my_config["context_length"]),
                                     stride=int(my_config["context_length"]),
                                     drop_last=True,
                                     shuffle=True,
                                     num_workers=0)

    val_loader = create_dataloader(val_data,
                                   my_tokenizer,
                                   2,
                                   max_length=int(my_config["context_length"]),
                                   stride=int(my_config["context_length"]),
                                   drop_last=True,
                                   shuffle=True,
                                   num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Training start!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model(model,
                                                        train_loader,
                                                        val_loader,
                                                        optimizer,
                                                        device,
                                                        num_epochs=num_epochs,
                                                        eval_freq=5,
                                                        eval_iter=5)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    model.eval()

    out = generate_text(model=model,
                        idx=encoded_tensor,
                        max_new_tokens=1,
                        context_size=my_config["context_length"])

    decoded_tensor = token_ids_to_text(out, tokenizer_f=my_tokenizer)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()},
        "model_and_optimizer.pth"
    )

