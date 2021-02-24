import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial

from scheduler import get_linear_schedule_with_warmup_same_time
from mask_tokens import mask_tokens

from torch.optim import Adam
from adam_wo_bc import AdamWoBC

from fastai.text.all import CrossEntropyLossFlat
from fastai.text.all import LabelSmoothingCrossEntropyFlat


class LitElectra(pl.LightningModule):
    def __init__(self, electra_model, electra_loss_func, hf_tokenizer, config):
        super().__init__()

        self.hf_tokenizer = hf_tokenizer
        self.config = config
        self.mask_tokens = partial(mask_tokens,
                                   mask_token_index=hf_tokenizer.mask_token_id,
                                   special_token_indices=hf_tokenizer.all_special_ids,
                                   vocab_size=hf_tokenizer.vocab_size,
                                   ignore_index=-100,
                                   replace_prob=0.0 if config.electra_mask_style else 0.1,
                                   original_prob=0.0 if config.electra_mask_style else 0.1,
                                   )
        self.model = electra_model
        self.loss_fn = electra_loss_func

    def on_train_start(self):
        self.model.to(self.config.device)
        
    def training_step(self, batch, batch_idx):
        # maskedLM
        input_ids, sentA_lenths = batch
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
        xb = (masked_inputs, sentA_lenths, is_mlm_applied, labels)
        yb = (labels,)

        ret = self.model(*xb)
        loss = self.loss_fn(ret, labels)
        self.log('raw_loss', loss)
        return loss

    def configure_optimizers(self):
        # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
        # momentum, squared momentumはpytorchではbetasにあたる。で↑は初期値と一緒
        optimizer = AdamWoBC(
        #optimizer = Adam(
            self.parameters(), lr=self.config.lr, eps=1e-6, weight_decay=0.01)

        scheduler = {
            'scheduler': get_linear_schedule_with_warmup_same_time(
                optimizer,
                num_warmup_steps=10000,
                num_training_steps=self.config.steps,
                ),    
            'interval':'step',
            }
        
        return [optimizer], [scheduler]
    
