from functools import partial
from fastai.text.all import Adam


from fastai.text.all import weight_decay, average_grad, average_sqr_grad, step_stat, l2_reg
from fastai.text.all import Optimizer


def adam_no_correction_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    p.data.addcdiv_(grad_avg, (sqr_avg).sqrt() + eps, value=-lr)
    return p

# これはAdamの一部を変えたもの


def Adam_no_bias_correction(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True),
            average_sqr_grad, step_stat, adam_no_correction_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)


def get_optim(c):
    if c.adam_bias_correction:
        opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
    else:
        # いまはこっち使ってる
        opt_func = partial(Adam_no_bias_correction, eps=1e-6,
                           mom=0.9, sqr_mom=0.999, wd=0.01)
    return opt_func
