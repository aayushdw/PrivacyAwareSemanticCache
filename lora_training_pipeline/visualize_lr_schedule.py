"""
Visualize the learning rate schedule to understand warmup + cosine annealing.
"""

import matplotlib.pyplot as plt
import numpy as np
from transformers import get_cosine_schedule_with_warmup
import torch


def visualize_lr_schedule(
    initial_lr: float = 2e-4,
    num_epochs: int = 2,
    steps_per_epoch: int = 31,  # Approximate for medium dataset
    warmup_ratio: float = 0.1
):
    """
    Visualize the learning rate schedule.

    Args:
        initial_lr: Initial learning rate
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch (batches)
        warmup_ratio: Fraction of steps for warmup
    """
    # Create dummy optimizer
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy_param], lr=initial_lr)

    # Calculate steps
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Collect learning rates
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Create x-axis (convert steps to epochs for readability)
    steps = np.arange(total_steps)
    epochs = steps / steps_per_epoch

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, lrs, linewidth=2, label='Learning Rate')
    plt.axvline(x=warmup_steps/steps_per_epoch, color='red', linestyle='--',
                label=f'Warmup End (step {warmup_steps})', alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'Cosine Annealing LR Schedule with Warmup\n'
              f'Initial LR: {initial_lr:.2e}, Warmup: {warmup_ratio:.0%}, '
              f'Total Steps: {total_steps}',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save
    plt.savefig('lr_schedule.png', dpi=300, bbox_inches='tight')
    print(f"Learning rate schedule saved to: lr_schedule.png")

    # Print statistics
    print(f"\nLearning Rate Schedule Statistics:")
    print(f"  Initial LR: {initial_lr:.2e}")
    print(f"  Peak LR (after warmup): {max(lrs):.2e}")
    print(f"  Final LR: {lrs[-1]:.2e}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_ratio:.0%})")
    print(f"  Total steps: {total_steps}")
    print(f"  LR reduction ratio: {lrs[-1]/max(lrs):.4f}")

    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Learning Rate Schedule Visualization")
    print("="*60)

    # Visualize with current training settings
    visualize_lr_schedule(
        initial_lr=2e-4,
        num_epochs=2,
        steps_per_epoch=31,  # ~1000 samples / batch_size 32
        warmup_ratio=0.1
    )
