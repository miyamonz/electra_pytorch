import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial

from transformers import get_linear_schedule_with_warmup
from mask_tokens import mask_tokens

from fastai.text.all import CrossEntropyLossFlat
from fastai.text.all import LabelSmoothingCrossEntropyFlat


class LitElectra(pl.LightningModule):
    def __init__(self, generator, discriminator, hf_tokenizer, sampling, config):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.hf_tokenizer = hf_tokenizer
        self.sampling = sampling

        self.config = config
        self.mask_tokens = partial(mask_tokens,
                                   mask_token_index=hf_tokenizer.mask_token_id,
                                   special_token_indices=hf_tokenizer.all_special_ids,
                                   vocab_size=hf_tokenizer.vocab_size,
                                   ignore_index=-100,
                                   replace_prob=0.0 if config.electra_mask_style else 0.1,
                                   original_prob=0.0 if config.electra_mask_style else 0.1,
                                   )

        self.loss_fn = ELECTRALoss(
            gen_label_smooth=config.gen_smooth_label, disc_label_smooth=config.disc_smooth_label)

    def training_step(self, batch, batch_idx):
        # maskedLM
        input_ids, sentA_lenths = batch
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
        xb = (masked_inputs, sentA_lenths, is_mlm_applied, labels)
        yb = (labels,)

        ret = self(*xb)
        mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied = ret
        loss = self.loss_fn(ret, labels)

        return loss

    def configure_optimizers(self):
        # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
        # momentum, squared momentumはpytorchではbetasにあたる。で↑は初期値と一緒
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, eps=1e-6, weight_decay=0.01)

        # 一旦、他のpytorch実装に習う
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10000,
            num_training_steps=self.config.steps,
        )
        return [optimizer], [scheduler]

    def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        sentA_lenths (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(
            masked_inputs, sentA_lenths)
        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[
            0]  # (B, L, vocab size)
        # reduce size to save space and speed
        # ( #mlm_positions, vocab_size)
        mlm_gen_logits = gen_logits[is_mlm_applied, :]

        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = masked_inputs.clone()  # (B,L)
            generated[is_mlm_applied] = pred_toks  # (B,L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B,L)
            is_replaced[is_mlm_applied] = (
                pred_toks != labels[is_mlm_applied])  # (B,L)

        disc_logits = self.discriminator(
            generated, attention_mask, token_type_ids)[0]  # (B, L)
        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor(
            [([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],
            device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
        if self.sampling == 'fp32_gumbel':
            return (logits.float() + self.gumbel_dist.sample(logits.shape).to(logits.device)).argmax(dim=-1)
        elif self.sampling == 'fp16_gumbel':  # 5.06 ms
            return (logits + self.gumbel_dist.sample(logits.shape).to(logits.device)).argmax(dim=-1)
        elif self.sampling == 'multinomial':  # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class ELECTRALoss():
    def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
        self.loss_weights = loss_weights
        self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(
            eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
        self.disc_loss_fc = nn.BCEWithLogitsLoss()
        self.disc_label_smooth = disc_label_smooth

    def __call__(self, pred, targ_ids):
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        gen_loss = self.gen_loss_fc(
            mlm_gen_logits.float(), targ_ids[is_mlm_applied])
        disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
        is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor
        if self.disc_label_smooth:
            is_replaced = is_replaced.float().masked_fill(
                ~is_replaced, self.disc_label_smooth)
        disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
        return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]
