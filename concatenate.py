
import numpy as np
import torch

from accelerate.utils import concatenate


def test_batches():
    print("=== Test 1: Tensor batches ===")
    batch1 = {
        "x": torch.rand(4, 1),
        "y": torch.from_numpy(
            np.array(
                [[1.0, 2.0, 3.0]] * 4,
                dtype=np.float32,
            )
        ),
    }

    batch2 = {
        "x": torch.rand(4, 1),
        "y": torch.from_numpy(
            np.array(
                [[1.0, 2.0, 3.0]] * 4,
                dtype=np.float32,
            )
        ),
    }

    batch = concatenate([batch1, batch2], dim=0)

    print(batch)
    print("x shape:", batch["x"].shape)  # Should be (8, 1)
    print("y shape:", batch["y"].shape)  # Should be (8, 3)

    print("\n=== Test 2: Mixed types (with lists) ===")
    batch1 = {"x": torch.rand(4, 1), "animals": ["dog", "cat", "baby", "penguin"]}
    batch2 = {
        "x": torch.rand(4, 1),
        "animals": ["koala", "samurai", "iguana", "rabbit"],
    }

    batch = concatenate([batch1, batch2], dim=0)

    print(batch)
    print("x shape:", batch["x"].shape)  # Should be (8, 1)
    print("animals:", batch["animals"])  # Should fallback to batch1["animals"]


if __name__ == "__main__":
    test_batches()
