from functools import partial
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from fastai.text.all import *

@delegates()
class MySortedDL(TfmdDL):

    def __init__(self, dataset, srtkey_fc=None, filter_fc=False, pad_idx=None, cache_file=None, **kwargs):
        print('MySortedDl')
        # Defaults
        if srtkey_fc is not False: srtkey_fc = lambda *x: len(x[0])
        if pad_idx is None: pad_idx = getattr(dataset, 'pad_idx', False)
        if isinstance(pad_idx, int): pad_idxs = [pad_idx] * len(dataset[0])
        elif isinstance(pad_idx, (list, tuple)): pad_idxs = pad_idx
        cache_file = Path(cache_file) if cache_file else None
        idmap = list(range(len(dataset)))

        # Save attributes
        super().__init__(dataset, **kwargs)
        store_attr('pad_idxs,srtkey_fc,filter_fc,cache_file,idmap', self)

        # Prepare records for sorting / filtered samples
        if srtkey_fc or filter_fc:
          if cache_file and cache_file.exists():
            # load cache and check
            with cache_file.open(mode='r') as f: cache = json.load(f)
            idmap, srtkeys = cache['idmap'], cache['srtkeys']
            if srtkey_fc: 
              assert srtkeys, "srtkey_fc is passed, but it seems you didn't sort samples when creating cache."
              self.srtkeys = srtkeys
            if filter_fc:
              assert idmap, "filter_fc is passed, but it seems you didn't filter samples when creating cache."
              self.idmap = idmap
          else:
            # overwrite idmap if filter, get sorting keys if sort
            idmap = []; srtkeys = []
            for i in tqdm(range_of(dataset), leave=False):
                sample = self.do_item(i)
                if filter_fc and not filter_fc(*sample): continue
                if filter_fc: idmap.append(i)
                if srtkey_fc: srtkeys.append(srtkey_fc(*sample))
            if filter_fc: self.idmap = idmap
            if srtkey_fc: self.srtkeys = srtkeys
            # save to cache
            if cache_file:
              try: 
                with cache_file.open(mode='w+') as f: json.dump({'idmap':idmap,'srtkeys':srtkeys}, f)
              except: os.remove(str(cache_file))
          # an info for sorting
          if srtkey_fc: self.idx_max = np.argmax(self.srtkeys)
          # update number of samples
          if filter_fc: self.n = self.n = len(self.idmap)

    def create_item(self, i): return self.dataset[self.idmap[i]]

    def create_batch(self, samples):
        if self.pad_idx is False: return super().create_batch(samples)
        return tuple( pad_sequence(attr, batch_first=True, padding_value=self.pad_idxs[i]) if attr[0].shape and isinstance(self.pad_idxs[i], int) else torch.stack(attr) for i, attr in enumerate(zip(*samples)))

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle: return idxs
        if self.srtkey_fc: return sorted(idxs, key=lambda i: self.srtkeys[i], reverse=True)
        return idxs

    def shuffle_fn(self,idxs):
        if not self.srtkey_fc: return super().shuffle_fn(idxs)
        idxs = np.random.permutation(self.n)
        idx_max = np.where(idxs==self.idx_max)[0][0]
        idxs[0],idxs[idx_max] = idxs[idx_max],idxs[0]
        sz = self.bs*50
        chunks = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.srtkeys[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = np.concatenate(np.random.permutation(batches[1:-1])) if len(batches) > 2 else np.array([],dtype=np.int)
        sort_idx = np.concatenate((batches[0], sort_idx) if len(batches)==1 else (batches[0], sort_idx, batches[-1]))
        return iter(sort_idx)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        if 'get_idxs' in kwargs: # when Learner.get_preds, dataload has `get_idxs` will be cloned. So we need to prevent sorting again
          kwargs['cache_file'] = self.cache_file
        # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set. 
        return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)
