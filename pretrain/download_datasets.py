from functools import partial
import datasets
from _utils.electra_dataprocessor import ELECTRADataProcessor
#from hugdatafast import HF_Datasets
from _utils.hugdatafast import HF_Datasets
from _utils.hugdatafast import MySortedDL
from pathlib import Path
from fastai.text.all import FilteredBase, merge, DataLoaders
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
            cache_dir='./datasets/electra_dataloader',
            cache_name='dl_{split}.json',
            bs=c.bs,
            num_workers=c.num_workers, pin_memory=False,
            shuffle_train=True,
            srtkey_fc=False,
    )
    return dls


def dataloaders(hf_dset_dict, cache_dir, cache_name, device='cpu', dl_kwargs=None, **kwargs):
    print('device', device)
    print('kwargs', kwargs) #{'bs': 128, 'num_workers': 3, 'pin_memory': False, 'shuffle_train': True, 'srtkey_fc': False}
    print('dl_kwargs', dl_kwargs)

    if dl_kwargs is None: dl_kwargs = [{} for _ in range(len(hf_dset_dict))]
    elif isinstance(dl_kwargs, dict):
        dl_kwargs = [ dl_kwargs[split] if split in dl_kwargs else {} for split in hf_dset_dict]
    # infer cache file names for each dataloader if needed
    dl_type = MySortedDL
    print('dl_type', dl_type)
    if cache_name:
        assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        if not cache_name.endswith('.json'): cache_name += '.json'
        for i, split in enumerate(hf_dset_dict):
            filled_cache_name = dl_kwargs[i].pop('cache_name', cache_name.format(split=split))
            if 'cache_file' not in dl_kwargs[i]:
                dl_kwargs[i]['cache_file'] = cache_dir/filled_cache_name
    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)

    # when corpus like glue/ax has only testset, set it to non-train setting
    if list(hf_dset_dict.keys())[0].startswith('test'):
        kwargs['shuffle_train'] = False
        kwargs['drop_last'] = False
    
    print('kwargs', kwargs) #drop_last: false だけふえた
    return FilteredBase_dataloaders(hf_dset_dict, dl_kwargs=dl_kwargs, device=device, **kwargs)


def FilteredBase_dataloaders(hf_dset_dict, dl_kwargs, bs=64, val_bs=None, shuffle_train=True, n=None, path='.', device=None, **kwargs):
    n_subsets = len(hf_dset_dict)  # HF_Dataset.n_subsets
    print('n_subsets', n_subsets)  # = 1
    print('dl_kwargs', dl_kwargs)  # [{cache_file}]
    # if device is None: device=default_device()
    drop_last = kwargs.pop('drop_last', shuffle_train)
    print('drop_last', drop_last)
    ds = list(hf_dset_dict.values())[0]
    dl = MySortedDL(ds, bs=bs, shuffle=shuffle_train, drop_last=drop_last, n=n, device=device,
        **merge(kwargs, dl_kwargs[0]))
    # dls = [dl] + [dl.new(self.subset(i), bs=(bs if val_bs is None else val_bs), shuffle=False, drop_last=False, n=None, **dl_kwargs[i]) for i in range(1, n_subsets)]
    return DataLoaders(dl, path=path, device=device)
