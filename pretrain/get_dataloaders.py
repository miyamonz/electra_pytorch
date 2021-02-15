from _utils.hf_dataset import HF_Dataset
from _utils.mysorteddl import MySortedDL
from pathlib import Path
from fastai.text.all import DataLoaders
from fastai.text.all import TensorText, noop

def get_dataloaders(c, hf_tokenizer, train_dset):
    print('train_dset', train_dset)
    ds = get_dataset(train_dset, hf_tokenizer)

    dl_args = {
            'bs': c.bs,
            'num_workers': c.num_workers,
            'pin_memory': False,
            'srtkey_fc': False,
            }
    shuffle_train = True
    device = 'cpu'
    # MySortedDL, TfmdDLがkwargsをどう使うか見る必要があるが、shuffle_trainはなかった
    dl = MySortedDL(ds, shuffle=shuffle_train, drop_last=False, device=device, **dl_args)
    return DataLoaders(dl, path='.', device=device)

# currently not used because MySortedDL doesn't have cache target when srtkey_fc = False
# cache_file = get_cache_file('./datasets/electra_dataloader', 'dl_{split}.json')
def get_cache_file(cache_dir, cache_name, split='train'):
    assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if not cache_name.endswith('.json'):
        cache_name += '.json'
    cache_file = cache_dir / cache_name.format(split=split)
    return cache_file

def get_dataset(train_dset, hf_tokenizer):
    args = {
            'cols': {'input_ids': TensorText, 'sentA_length': noop},
            'hf_toker': hf_tokenizer,
            'n_inp': 2,
            }
    ds = HF_Dataset(train_dset, **args)
    return ds
