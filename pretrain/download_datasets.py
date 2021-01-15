from functools import partial
import datasets
from _utils.electra_dataprocessor import ELECTRADataProcessor
from hugdatafast import HF_Datasets
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
    dls = hf_dsets.dataloaders(
            bs=c.bs,
            num_workers=c.num_workers, pin_memory=False,
            shuffle_train=True,
            srtkey_fc=False,
            cache_dir='./datasets/electra_dataloader',
            cache_name='dl_{split}.json')
    return dls
