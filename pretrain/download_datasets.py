from functools import partial
import datasets
from _utils.electra_dataprocessor import ELECTRADataProcessor


def download_dset(c, hf_tokenizer, cache_dir, num_proc):
    dsets = []
    ELECTRAProcessor = partial(
        ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)
    # Wikipedia
    if 'wikipedia' in c.datas:
        print('load/download wiki dataset')
        wiki = datasets.load_dataset(
            'wikipedia', '20200501.en', cache_dir=cache_dir)['train']
        print('load/create data from wiki dataset for ELECTRA')
        e_wiki = ELECTRAProcessor(wiki).map(
            cache_file_name=f"1000_electra_wiki_{c.max_length}.arrow", num_proc=num_proc)
        dsets.append(e_wiki)

    # OpenWebText
    if 'openwebtext' in c.datas:
        print('load/download OpenWebText Corpus')
        owt = datasets.load_dataset(
            'openwebtext', cache_dir=cache_dir)['train']
        print('load/create data from OpenWebText Corpus for ELECTRA')
        e_owt = ELECTRAProcessor(owt, apply_cleaning=False).map(
            cache_file_name=f"electra_owt_{c.max_length}.arrow", num_proc=num_proc)
        dsets.append(e_owt)

    assert len(dsets) == len(c.datas)

    train_dset = datasets.concatenate_datasets(dsets)
    return train_dset
