from functools import partial
import datasets
from _utils.electra_dataprocessor import ELECTRADataProcessor
#from hugdatafast import HF_Datasets
from _utils.hugdatafast import HF_Datasets
from _utils.hugdatafast import MySortedDL
from pathlib import Path
from fastai.text.all import DataLoaders
from fastai.text.all import TensorText, noop


def download_and_dls(c, hf_tokenizer, num_proc=1):
    dsets = []
    ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)
    # Wikipedia
    if 'wikipedia' in c.datas:
        print('load/download wiki dataset')
        wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='./datasets')['train']
        print('load/create data from wiki dataset for ELECTRA')
        e_wiki = ELECTRAProcessor(wiki).map(cache_file_name=f"1000_electra_wiki_{c.max_length}.arrow", num_proc=num_proc)
        dsets.append(e_wiki)

    # OpenWebText
    if 'openwebtext' in c.datas:
        print('load/download OpenWebText Corpus')
        owt = datasets.load_dataset('openwebtext', cache_dir='./datasets')['train']
        print('load/create data from OpenWebText Corpus for ELECTRA')
        e_owt = ELECTRAProcessor(owt, apply_cleaning=False).map(cache_file_name=f"electra_owt_{c.max_length}.arrow", num_proc=num_proc)
        dsets.append(e_owt)

    assert len(dsets) == len(c.datas)

    merged_dsets = {'train': datasets.concatenate_datasets(dsets)}
    hf_dsets = HF_Datasets(
            merged_dsets,
            cols={'input_ids': TensorText, 'sentA_length': noop},
            hf_toker=hf_tokenizer, n_inp=2)
    hf_dset_dict = hf_dsets.hf_dsets
    dls = dataloaders(
            hf_dset_dict,
            './datasets/electra_dataloader',  # cache_dir
            'dl_{split}.json',  # cache_name
            #  kwargs
            bs=c.bs,
            num_workers=c.num_workers, pin_memory=False,
            shuffle_train=True,
            srtkey_fc=False,
    )
    return dls


def dataloaders(hf_dset_dict, cache_dir, cache_name, device='cpu', **kwargs):
    print('device', device)  # cpu
    print('kwargs', kwargs)  # {'bs': 128, 'num_workers': 3, 'pin_memory': False, 'shuffle_train': True, 'srtkey_fc': False}

#     dl_kwargs = [{} for _ in range(len(hf_dset_dict))]
#     print('dl_kwargs', dl_kwargs)
    # infer cache file names for each dataloader if needed

    assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if not cache_name.endswith('.json'): cache_name += '.json'
#     for i, split in enumerate(hf_dset_dict):
#         cache_file = cache_dir / cache_name.format(split=split)
#         assert('cache_file' not in dl_kwargs[i])
#         dl_kwargs[i]['cache_file'] = cache_file
    cache_file = cache_dir / cache_name.format(split='train')

    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)

    # when corpus like glue/ax has only testset, set it to non-train setting
#     if list(hf_dset_dict.keys())[0].startswith('test'):
#         kwargs['shuffle_train'] = False
#         kwargs['drop_last'] = False
    
    print('kwargs', kwargs)  # drop_last: false だけふえた
    ds = list(hf_dset_dict.values())[0]
    return FilteredBase_dataloaders(ds, cache_file, device, **kwargs)


def FilteredBase_dataloaders(ds, cache_file,device, bs=64, shuffle_train=True, path='.', **kwargs):
    print('kwargs', kwargs)  # {'num_workers': 3, 'pin_memory': False, 'srtkey_fc': False, 'drop_last': False} 
    print('path', path)

    drop_last = kwargs.pop('drop_last', shuffle_train)
    print('drop_last', drop_last)  # false
    dl = MySortedDL(ds, bs=bs, shuffle=shuffle_train, drop_last=drop_last, n=None, device=device,
            cache_file=cache_file, **kwargs)
    return DataLoaders(dl, path=path, device=device)
