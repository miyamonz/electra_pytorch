import torch
from torch.nn.utils.rnn import pad_sequence
from fastai.text.all import TfmdDL


class MySortedDL(TfmdDL):
    def __init__(self, dataset, pad_idx, srtkey_fc=None, filter_fc=False, cache_file=None, tfmd_args=None):
        """
        dataset: HF_Dataset Actually any object implements __len__ and __getitem__ that return a tuple as a sample.
        """
        print('MySortedDL')

        print('pad_idx', pad_idx)  # 0
        pad_idxs = [pad_idx] * len(dataset[0])
        print('pad_idxs', pad_idxs)

        # Save attributes
        super().__init__(dataset, **tfmd_args)
        self.pad_idxs = pad_idxs

    def create_item(self, i):
        return self.dataset[i]

    def create_batch(self, samples):
        # if self.pad_idx is False: return super().create_batch(samples)
        pad_idxs = self.pad_idxs
        return tuple(
            pad_sequence(attr, batch_first=True,
                         padding_value=pad_idxs[i]) if attr[0].shape else torch.stack(attr)
            for i, attr in enumerate(zip(*samples)))

#     @delegates(TfmdDL.new)
#     def new(self, dataset=None, **kwargs):
#         if 'get_idxs' in kwargs: # when Learner.get_preds, dataload has `get_idxs` will be cloned. So we need to prevent sorting again
#           kwargs['cache_file'] = self.cache_file
#         # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set.
#         return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)
