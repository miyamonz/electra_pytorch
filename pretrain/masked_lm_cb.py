from functools import partial
import torch

from fastai.text.all import Callback, delegates, TfmdDL

from mask_tokens import mask_tokens


class MaskedLMCallback(Callback):
    @delegates(mask_tokens)
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, for_electra=False, **kwargs):
        self.ignore_index = ignore_index
        self.for_electra = for_electra
        self.mask_tokens = partial(mask_tokens,
                                   mask_token_index=mask_tok_id,
                                   special_token_indices=special_tok_ids,
                                   vocab_size=vocab_size,
                                   ignore_index=-100,
                                   **kwargs)

    def before_batch(self):
        input_ids, sentA_lenths = self.xb
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
        if self.for_electra:
            self.learn.xb, self.learn.yb = (
                masked_inputs, sentA_lenths, is_mlm_applied, labels), (labels,)
        else:
            self.learn.xb, self.learn.yb = (
                masked_inputs, sentA_lenths), (labels,)

    @delegates(TfmdDL.show_batch)
    def show_batch(self, dl, idx_show_ignored, verbose=True, **kwargs):
        b = dl.one_batch()
        input_ids, sentA_lenths = b
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(
            input_ids.clone())
        # check
        assert torch.equal(is_mlm_applied, labels != self.ignore_index)
        assert torch.equal((~is_mlm_applied * masked_inputs +
                            is_mlm_applied * labels), input_ids)
        # change symbol to show the ignored position
        labels[labels == self.ignore_index] = idx_show_ignored
        # some notice to help understand the masking mechanism
        if verbose:
            print("We won't count loss from position where y is ignore index")
            print(
                "Notice 1. Positions have label token in y will be either [Mask]/other token/orginal token in x")
            print("Notice 2. Special tokens (CLS, SEP) won't be masked.")
            print(
                "Notice 3. Dynamic masking: every time you run gives you different results.")
        # show
        tfm_b = (masked_inputs, sentA_lenths, is_mlm_applied,
                 labels) if self.for_electra else (masked_inputs, sentA_lenths, labels)
        dl.show_batch(b=tfm_b, **kwargs)
