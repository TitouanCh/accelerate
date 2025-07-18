import random

import torch
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator


# ---- Step 1: Synthetic Dataset (y is a string) ---- #
class SyntheticIterableDataset(IterableDataset):
    def __init__(self, num_samples, input_dim, vocab):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.vocab = vocab

    def __iter__(self):
        for _ in range(self.num_samples):
            x = torch.randint(0, 256, (self.input_dim,), dtype=torch.float32)
            y_str = random.choice(self.vocab)
            yield {"x": x, "y": y_str}


def custom_collate(batch):
    collated = {}
    for sample in batch:
        for k, v in sample.items():
            if k not in collated:
                collated[k] = []
            collated[k].append(v)
    for k in collated:
        if torch.is_tensor(collated[k][0]):
            collated[k] = torch.stack(collated[k])
    return collated


# ---- Step 3: Setup ---- #
vocab = ["cat", "dog", "snake"]
vocab_to_idx = {label: i for i, label in enumerate(vocab)}
num_classes = len(vocab)
input_dim = 1000 + num_classes

dataset = SyntheticIterableDataset(1000, 1000, vocab=vocab)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)

model = torch.nn.Linear(input_dim, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
device = accelerator.device

# ---- Step 4: Training Loop ---- #
for epoch in range(10):
    for batch in dataloader:
        print(batch)
        x = batch["x"].to(device)

        # y to one-hot
        y_str_list = batch["y"]
        y_idx = torch.tensor([vocab_to_idx[y] for y in y_str_list], device=device)
        y_onehot = torch.nn.functional.one_hot(y_idx, num_classes=num_classes).float()

        model_input = torch.cat([x, y_onehot], dim=1)

        output = model(model_input)
        loss = output.sum()

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    print(f"Epoch {epoch} complete.")
