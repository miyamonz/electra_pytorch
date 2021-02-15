#from hugdatafast import HF_Datasets
from _utils.hugdatafast import HF_Datasets
from _utils.hugdatafast import MySortedDL
from pathlib import Path
from fastai.text.all import DataLoaders
from fastai.text.all import TensorText, noop

def get_dataloaders(c, hf_tokenizer, train_dset):
    ds = get_dataset(train_dset, hf_tokenizer)

    dl_args = {
            'bs': c.bs,
            'num_workers': c.num_workers,
            'pin_memory': False,
            'srtkey_fc': False,
            }
    shuffle_train = True
    cache_file = get_cache_file('./datasets/electra_dataloader', 'dl_{split}.json')

    # MySortedDL, TfmdDLがkwargsをどう使うか見る必要があるが、shuffle_trainはなかった
    device='cpu'
    dl = MySortedDL(ds, shuffle=shuffle_train, drop_last=False, device=device,
            cache_file=cache_file, **dl_args)
    return DataLoaders(dl, path='.', device=device)

def get_cache_file(cache_dir, cache_name, split='train'):
    assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if not cache_name.endswith('.json'): cache_name += '.json'
    cache_file = cache_dir / cache_name.format(split=split)
    return cache_file

def get_dataset(train_dset, hf_tokenizer):
    merged_dsets = {'train': train_dset}
    hf_dsets = HF_Datasets(
            merged_dsets,
            cols={'input_ids': TensorText, 'sentA_length': noop},
            hf_toker=hf_tokenizer, n_inp=2)
    hf_dset_dict = hf_dsets.hf_dsets
    ds = list(hf_dset_dict.values())[0]
    return ds

