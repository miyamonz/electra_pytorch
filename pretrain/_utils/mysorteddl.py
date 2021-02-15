from torch.nn.utils.rnn import pad_sequence
from fastai.text.all import *

class MySortedDL(TfmdDL):
    def __init__(self, dataset, srtkey_fc=None, filter_fc=False, pad_idx=None, cache_file=None, **kwargs):
        print('MySortedDl')
        # Defaults
        print('srtkey_fc', srtkey_fc)
        print('filter_fc', filter_fc)
        print('pad_idx', pad_idx)
        print('cache_file', cache_file) # is only used when srtkey or filter fc exists
        print('kwargs', kwargs)

        if pad_idx is None: pad_idx = getattr(dataset, 'pad_idx', False)
        print('pad_idx', pad_idx)  # 0
        pad_idxs = [pad_idx] * len(dataset[0])
        print('pad_idxs', pad_idxs)


        # Save attributes
        super().__init__(dataset, **kwargs)
        store_attr('pad_idxs,srtkey_fc,cache_file', self)

    def create_item(self, i): return self.dataset[i]

    def create_batch(self, samples):
        # if self.pad_idx is False: return super().create_batch(samples)
        pad_idxs = self.pad_idxs
        return tuple( pad_sequence(attr, batch_first=True, padding_value=pad_idxs[i]) if attr[0].shape else torch.stack(attr) for i, attr in enumerate(zip(*samples)))

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        if 'get_idxs' in kwargs: # when Learner.get_preds, dataload has `get_idxs` will be cloned. So we need to prevent sorting again
          kwargs['cache_file'] = self.cache_file
        # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set. 
        return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)
