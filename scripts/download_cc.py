import gc
from itertools import chain
from pathlib import Path
from multiprocessing import cpu_count, Pool

from datasets import load_dataset, DownloadConfig
import more_itertools
import spacy
from timeoutcontext import timeout
import tqdm



SPACY_PIPELINES = {
    'de': 'de_core_news_sm',
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'fr': 'fr_core_news_sm',
    'it': 'it_core_news_sm',
    'ja': 'ja_core_news_sm',
    'nl': 'nl_core_news_sm',
    'zh': 'zh_core_web_sm',
}


datadir = Path('cc_data')
datadir.mkdir(exist_ok=True)
size = 300_000_000
batch_size = cpu_count()
reload_thresh = 10_000

nlp = None
opencc_converter = None

double_quotes = "«»“”„‟❝❞❠⹂〝〞〟＂🙶🙷🙸"
double_quotes_nlz = '"' * len(double_quotes)
single_quotes = "‘’‛❛❜❟"
single_quotes_nlz = "'" * len(single_quotes)
hyphens = "֊᠆‐‑‧⁃﹣－"
hyphens_nlz = '-' * len(hyphens)

from_str = double_quotes + single_quotes + hyphens
to_str = double_quotes_nlz + single_quotes_nlz + hyphens_nlz

translation_table = {
    ord(x): ord(y) 
    for x,y in zip(from_str, to_str)
}

for lang in list(SPACY_PIPELINES.keys()):
    if (datadir / f'{lang}.txt').is_file():
        print(f'File for language {lang} already exists, skipping.')
        del SPACY_PIPELINES[lang]

def init_worker(pipeline):
    if pipeline.startswith('zh'):
        import opencc
        global opencc_converter
        opencc_converter = opencc.OpenCC('t2s.json') 
    global nlp
    nlp = spacy.load(pipeline, exclude=['ner', 'lemmatizer', 'morphologizer', 'attribute_ruler'])
    nlp.max_length = 1000000 * 2
    

def process(block):
    global opencc_converter
    with timeout(60):
        try:
            sents = nlp(block).sents
            sents = ((tok.text.strip().lower().translate(translation_table) for tok in sent) for sent in sents)
            sents = (list(filter(None, sent)) for sent in sents)
            sents = list(filter(None, sents))
        except:
            sents = []
    if (len(sents) > 0) and (opencc_converter is not None):
        sents = [[opencc_converter.convert(tok) for tok in sent] for sent in sents]
    return sents
        
    
def get_examples(dataset, pipeline):
    texts = (item['text'] for item in dataset.shuffle())
    blocks = filter(None, texts)
    with Pool(cpu_count(), init_worker, (pipeline,), reload_thresh) as pool:
        calls = pool.imap_unordered(process, blocks, batch_size)
        sents = chain.from_iterable(calls)
        yield from sents
        
        
for lang in tqdm.tqdm(SPACY_PIPELINES, desc='langs'):
    with open(datadir / f'{lang}.txt', 'w') as ostr:
        dataset = load_dataset(
            'oscar', 
            f"unshuffled_deduplicated_{lang}", 
            split='train', 
            streaming=True, 
            download_config=DownloadConfig(num_proc=1), # cpu_count()),
        )
        for idx, sentence in zip(
            tqdm.trange(size, total=size, unit_scale=True, unit_divisor=1_000, desc=lang, smoothing=0.0), 
            get_examples(dataset, SPACY_PIPELINES[lang])
        ):
            if idx >= size : break
            print(*sentence, file=ostr)
        
