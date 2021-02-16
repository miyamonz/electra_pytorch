from functools import partial
from fastai.text.all import ParamScheduler

# こっちがoriginal
def linear_warmup_and_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct:
        warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.

    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
    decayed_lr = (lr_max-end_lr) * (1 - step_i/total_steps) ** decay_power + end_lr
    # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
    warmed_lr = decayed_lr * min(1.0, step_i/warmup_steps)
    return warmed_lr

def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct:
        warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    if step_i <= warmup_steps: # warm up
        return lr_max * min(1.0, step_i/warmup_steps)
    else:  # decay
        return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr


def get_scheduler(c):
    # c.scheduler == 'original_linear'

    # Learning rate shedule
    if c.schedule.endswith('linear'):
        # 何故か使われてなかった　fine-tuningには使われてた
        lr_shed_func = linear_warmup_and_then_decay if c.schedule=='separate_linear' else linear_warmup_and_decay
        return ParamScheduler({'lr': partial(linear_warmup_and_decay,
                                                 lr_max=c.lr,
                                                 warmup_steps=10000,
                                                 total_steps=c.steps,)})
