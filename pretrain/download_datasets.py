from functools import partial
import datasets
from _utils.electra_dataprocessor import ELECTRADataProcessor
#from hugdatafast import HF_Datasets
from _utils.hugdatafast import HF_Datasets
from _utils.hugdatafast import MySortedDL
from pathlib import Path
from fastai.text.all import FilteredBase
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
    dls = dataloaders(
            hf_dsets,
            cache_dir='./datasets/electra_dataloader',
            cache_name='dl_{split}.json',
            bs=c.bs,
            num_workers=c.num_workers, pin_memory=False,
            shuffle_train=True,
            srtkey_fc=False,
    )
    return dls


def dataloaders(self, cache_dir, cache_name, device='cpu', dl_kwargs=None, **kwargs):
    print('device', device)
    print('kwargs', kwargs) #{'bs': 128, 'num_workers': 3, 'pin_memory': False, 'shuffle_train': True, 'srtkey_fc': False}
    _dl_type = self._dl_type
    hf_dsets = self.hf_dsets
    print('dl_kwargs', dl_kwargs)

    if dl_kwargs is None: dl_kwargs = [{} for _ in range(len(hf_dsets))]
    elif isinstance(dl_kwargs, dict):
        dl_kwargs = [ dl_kwargs[split] if split in dl_kwargs else {} for split in hf_dsets]
    # infer cache file names for each dataloader if needed
    dl_type = kwargs.pop('dl_type', _dl_type)
    print('dl_type', dl_type)
    if dl_type==MySortedDL and cache_name:
        assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        if not cache_name.endswith('.json'): cache_name += '.json'
        for i, split in enumerate(hf_dsets):
            filled_cache_name = dl_kwargs[i].pop('cache_name', cache_name.format(split=split))
            if 'cache_file' not in dl_kwargs[i]:
                dl_kwargs[i]['cache_file'] = cache_dir/filled_cache_name
    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)

    # when corpus like glue/ax has only testset, set it to non-train setting
    if list(hf_dsets.keys())[0].startswith('test'):
        kwargs['shuffle_train'] = False
        kwargs['drop_last'] = False
    
    print('kwargs', kwargs) #drop_last: false だけふえた
    return FilteredBase.dataloaders(self, dl_kwargs=dl_kwargs, device=device, **kwargs)
