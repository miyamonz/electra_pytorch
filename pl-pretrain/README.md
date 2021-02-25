this directry is for rewriting [richarddwang's electra implementation](https://github.com/richarddwang/electra_pytorch) with pytorch-lightning.

the reason is that I don't want to depend fastai and hugdatafast, which is integration library between huggingface/datasets and fastai.
this repo is WIP.

what I did

- remove hugdatafast dependency
  - I read HF_Dataset, HF_Datasets, MySortedDL and so on. and copied minimal code.
- split make_tokens function, and write MaskedLM thing directly into pytorch-lightning's `training_step` hook
- prepare same Optimizer and Scheduler


todo
- some util like functions from fastai are stil imported.
