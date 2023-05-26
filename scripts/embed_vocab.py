import pathlib

import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)
_max_chars = 10_000

def yield_by_char(filename):
	with open(filename, 'r') as istr:
		data = map(str.strip, istr)
		data = sorted(data, key=len)
		accum, ch_count = [], 0
		for word in tqdm.tqdm(data, desc=filename, leave=False):
			accum.append(word)
			ch_count += len(word)
			if ch_count >=_max_chars:
				yield accum
				accum, ch_count = [], 0
		if len(accum) > 0:
			yield accum

layer_groups = [(12, 13), (0, 1), (0, 13), (9, 13), (1, 5), (5, 9)]

for model_type in tqdm.tqdm(['uncased'], desc='model'):
	tokenizer = AutoTokenizer.from_pretrained(f'bert-base-multilingual-{model_type}')
	model = AutoModel.from_pretrained(f'bert-base-multilingual-{model_type}')
	model = model.to('cuda')
	model.eval()
	for lang in tqdm.tqdm(['en'], desc='lang'):
		for vocab_size in tqdm.tqdm(['mc50'], desc='vocab', leave=False):
			vocab_file = f'{lang}-{vocab_size}.txt'
			for s, e in layer_groups:
				 (pathlib.Path(f'mbert-{model_type}-vecto') / lang / f'{s}-{e}').mkdir(parents=True, exist_ok=True)
			vector_files = [open(f'mbert-{model_type}-vecto/{lang}/{s}-{e}/model.txt', 'w') for s, e in layer_groups]
			for batch in yield_by_char(vocab_file):
				inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to('cuda')
				vectors = torch.stack(model(**inputs, output_hidden_states=True).hidden_states, dim=1)
				B, L, S, H = vectors.shape
				assert L == 13
				vectors = vectors.masked_fill((inputs.input_ids == tokenizer.pad_token_id).view(B, 1, S, 1), 0.0).sum(2)
				for word, vector in zip(batch, vectors):
					for idx in range(len(layer_groups)):
						s, e = layer_groups[idx]
						fh = vector_files[idx]
						pooled_vector = vector[s:e].mean(0).cpu().tolist()
						print(word, *pooled_vector, file=fh)
			for fh in vector_files:
				fh.close()
