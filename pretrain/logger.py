import wandb
from fastai.callback.wandb import WandbCallback

def get_wandb_callback(c):
    wandb.init(name=c.run_name, project='electra_pretrain', config=c)
    return WandbCallback(log_preds=False, log_model=False)


def get_neptune_callback(c):
    import neptune
    from fastai.callback.neptune import NeptuneCallback
    neptune.init(project_qualified_name='richard-wang/electra-pretrain')
    neptune.create_experiment(name=c.run_name, params=c)
    return NeptuneCallback(log_model_weights=False)
