import math


def cosine_lr_with_warmup(initial_lr: float, warmup_epochs: int,
                           total_epochs: int, min_lr: float = 1e-7):

    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

    return schedule
