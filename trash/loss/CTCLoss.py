import torch
import torch.nn.functional as f


def text_trans(output, target, pad_idx):
    # Flatten the output and target tensors to compute the loss
    output_flat = output.view(-1, output.size(-1))  # shape: (batch_size * seq_len, vocab_size)
    target_flat = target.view(-1)  # shape: (batch_size * seq_len)

    # Compute the cross-entropy loss
    loss = f.cross_entropy(output_flat, target_flat, ignore_index=pad_idx)

    return loss
