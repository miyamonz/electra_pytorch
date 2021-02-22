from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    # general one
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup_same_time(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.
    def lr_lambda(current_step: int):
        lr =  max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps))
        )
        lr *= min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)