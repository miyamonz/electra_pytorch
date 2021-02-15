from fastai.text.all import *

class HF_Dataset():
  
  def __init__(self, hf_dset, cols=None, hf_toker=None, neat_show=False, n_inp=1):
    print('HF_Dataset')
    # some default setting for tensor type used in decoding
    print('cols', cols)
    print('neat_show', neat_show)
    print('n_inp', n_inp)

#     if cols is None: cols = hf_dset.column_names
#     if isinstance(cols, list): 
#       if n_inp==1: 
#         if len(cols)==1: cols = {cols[0]: TensorText}
#         elif len(cols)==2: cols = {cols[0]: TensorText, cols[1]: TensorCategory}
#       else: cols = { c: noop for c in cols }

    assert isinstance(cols, dict)
    
    # make dataset output pytorch tensor
    hf_dset.set_format( type='torch', columns=list(cols.keys()))

    # store attributes
    self.pad_idx = hf_toker.pad_token_id
    self.hf_dset = hf_dset
    store_attr("cols,n_inp,hf_toker,neat_show", self)

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  # __getitem__, __len__以外はいらないはず

