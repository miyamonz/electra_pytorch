from fastai.text.all import *

class HF_Dataset():
  
  def __init__(self, hf_dset, cols=None, hf_toker=None, neat_show=False, n_inp=1):
    print('HF_Dataset')
    # some default setting for tensor type used in decoding
    if cols is None: cols = hf_dset.column_names
    if isinstance(cols, list): 
      if n_inp==1: 
        if len(cols)==1: cols = {cols[0]: TensorText}
        elif len(cols)==2: cols = {cols[0]: TensorText, cols[1]: TensorCategory}
      else: cols = { c: noop for c in cols }
    assert isinstance(cols, dict)
    
    # make dataset output pytorch tensor
    hf_dset.set_format( type='torch', columns=list(cols.keys()) )

    # store attributes
    self.pad_idx = hf_toker.pad_token_id
    self.hf_dset = hf_dset
    store_attr("cols,n_inp,hf_toker,neat_show", self)

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  @property
  def col_names(self): return list(self.cols.keys())

  def decode(self, o, full=True): # `full` is for micmic `Dataset.decode` 
    if len(self.col_names) != len(o): return tuple( self._decode(o_) for o_ in o )
    return tuple( self._decode(o_, self.col_names[i]) for i, o_ in enumerate(o) )

  def _decode_title(self, d, title_cls, title): 
    if title: return title_cls(d, title=title)
    else: return title_cls(d)

  @typedispatch
  def _decode(self, t:torch.Tensor, title):
    if t.shape: title_cls = _TitledTuple
    elif isinstance(t.item(),bool): title_cls = _TitledBool # bool is also int, so check whether is bool first
    elif isinstance(t.item(),float): title_cls = _TitledFloat
    elif isinstance(t.item(),int): title_cls = _TitledInt
    return self._decode_title(t.tolist(), title_cls , title)

  @typedispatch
  def _decode(self, t:TensorText, title): 
    assert self.hf_toker, "You should give a huggingface tokenizer if you want to show batch."
    if self.neat_show: text = self.hf_toker.decode([idx for idx in t if idx != self.hf_toker.pad_token_id])
    else: text = ' '.join(self.hf_toker.convert_ids_to_tokens(t))
    return self._decode_title(text, _TitledStr, title)

  @typedispatch
  def _decode(self, t:LMTensorText, title): return self._decode[TensorText](self, t, title)

  @typedispatch
  def _decode(self, t:TensorCategory, title): return self._decode_title(t.item(), _Category, title)

  @typedispatch
  def _decode(self, t:TensorMultiCategory, title): return self._decode_title(t.tolist(), _MultiCategory, title)

  def __getattr__(self, name):
    "If not defined, let the datasets.Dataset in it act for us."
    if name in HF_Dataset.__dict__: return HF_Dataset.__dict__[name]
    elif name in self.__dict__: return self.__dict__[name]
    elif hasattr(self.hf_dset, name): return getattr(self.hf_dset, name)
    raise AttributeError(f"Both 'HF_Dataset' object and 'datasets.Dataset' object have no '{name}' attribute ")
