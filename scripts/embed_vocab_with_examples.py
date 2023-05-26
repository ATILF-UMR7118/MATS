import pathlib

import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)

layer_groups = [(12, 13), (0, 1), (0, 13), (9, 13), (1, 5), (5, 9)]

for model_type in tqdm.tqdm(['uncased'], desc='model'):
	tokenizer = AutoTokenizer.from_pretrained(f'bert-base-multilingual-{model_type}')
	model = AutoModel.from_pretrained(f'bert-base-multilingual-{model_type}')
	model = model.to('cuda')
	model.eval()
	for lang in tqdm.tqdm(['en'], desc='lang'):
		for vocab_size in ['mc50']:
			vocab_file = f'{lang}-{vocab_size}.txt'
			with open(vocab_file, 'r') as istr:
				word2ctxt = {line.strip(): [] for line in tqdm.tqdm(istr, leave=False, desc=f'read {lang} vocab')}
				word2count = {word: 0 for word in word2ctxt}
			with open('../../mats-data/txt/' + lang + '.txt', 'r') as istr:
				pbar = tqdm.trange(len(word2ctxt) * 10, leave=False, desc=f'find {lang} contexts')
				for sentence in map(str.split, map(str.strip, istr)):
					for word in set(sentence):
						if word in word2count:
							word2count[word] += 1
							word2ctxt[word].append(sentence)
							if word2count[word] >= 10:
								del word2count[word]
							pbar.update()
					if len(word2count) == 0:
						break
				pbar.close()
			for s, e in layer_groups:
				 (pathlib.Path(f'mbert-{model_type}-vecto-ctxt') / lang / f'{s}-{e}').mkdir(parents=True, exist_ok=True)
			vector_files = [open(f'mbert-{model_type}-vecto-ctxt/{lang}/{s}-{e}/model.txt', 'w') for s, e in layer_groups]
			for word, ctxts in tqdm.tqdm(word2ctxt.items(), desc=f'embed {lang}', total=len(word2ctxt)):
				inputs = tokenizer(ctxts, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True).to('cuda')
				vectors = torch.stack(model(**inputs, output_hidden_states=True).hidden_states, dim=1)
				B, L, S, H = vectors.shape
				assert L == 13
				word_id = torch.tensor([next(idx for idx in range(len(c)) if c[idx] == word) for c in ctxts], device='cuda')
				token_maps = torch.nn.utils.rnn.pad_sequence(
					[
						torch.tensor(
							[-1, *tokenizer(c, is_split_into_words=True, truncation=True).word_ids()[1:-1], -1],
							device='cuda',
						)
						for c in ctxts
					],
					padding_value=-1,
					batch_first=True,
				)
				to_retrieve = token_maps.view(B, S) == word_id.view(B, 1)
				vectors = vectors.masked_fill(~to_retrieve.view(B, 1, S, 1), 0.0).sum(2)
				B_, L_, H_ = vectors.size()
				assert (B_, L_, H_) == (B, L, H)
				for idx in range(len(layer_groups)):
					s, e = layer_groups[idx]
					fh = vector_files[idx]
					pooled_vector = vectors[:,s:e,:].reshape(-1, H).mean(0).cpu().tolist()
					print(word, *pooled_vector, file=fh)
			for fh in vector_files:
				fh.close()
