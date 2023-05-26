import pathlib
import pandas as pd
import tqdm
from transformers import pipeline

bats_dir = pathlib.Path('/scratch/project_2006235/dataset/BATS/uncased')
lang = 'de'

for model_type in tqdm.tqdm(['cased', 'uncased']):
	pipe = pipeline(task='fill-mask', model=f'bert-base-multilingual-{model_type}', device=0, top_k=1)

	def get_lens(bs):
		return {len(pipe.tokenizer.encode(b, add_special_tokens=False)) for b in bs}

	def get_template_with_quotes(varnum, a, b, c, len_d):
		mask = " ".join([pipe.tokenizer.mask_token] * len_d)
		if varnum == 0:
			return f'"{a}" verhält sich zu "{b}" wie "{c}" zu "{mask}".'
		if varnum == 1:
			return f'"{a}" ist für "{b}" was "{c}" für "{mask}" ist.'
		if varnum == 2:
			return f'"{a}" ist so zu "{b}" wie "{c}" zu "{mask}" ist.'
		if varnum == 3:
			return f'"{a}" steht in Relation zu "{b}" so wie "{c}" zu "{mask}".'

	def get_template_no_quotes(lang, a, b, c, len_d):
		mask = " ".join([pipe.tokenizer.mask_token] * len_d)
		if varnum == 0:
			return f'{a} verhält sich zu {b} wie {c} zu {mask}.'
		if varnum == 1:
			return f'{a} ist für {b} was {c} für {mask} ist.'
		if varnum == 2:
			return f'{a} ist so zu {b} wie {c} zu {mask} ist.'
		if varnum == 3:
			return f'{a} steht in Relation zu {b} so wie {c} zu {mask}.'

	for use_quotes in tqdm.tqdm([True, False], desc='quotes'):
		get_template = get_template_with_quotes if use_quotes else get_template_no_quotes
		def unmask_with_length(varnum, a1, b1, a2, length_b2):
			preds = pipe(get_template(varnum, a1, b1, a2, length_b2))
			if length_b2 == 1: preds = [preds]
			token_strs = [p[0]['token_str'] for p in preds]
			token_strs = ' '.join(token_strs)
			token_strs = token_strs.replace(' ##', '')
			return token_strs

		all_results = []
		for varnum in tqdm.trange(4, desc='vars'):
			mats_dir = bats_dir / lang
			for file in tqdm.tqdm(sorted(mats_dir.glob('**/*.txt')), desc=lang):
				with open(file) as fh:
					pairs = map(str.split, filter(None, map(str.strip, fh)))
					pairs = [(a, sorted(set(filter(None, b.split('/'))))) for a, b in pairs]
				res_subcat = []
				all_preds = []
				for i1, (a1, b1s) in tqdm.tqdm(list(enumerate(pairs)), desc=file.name, leave=False):
					for i2, (a2, b2s) in tqdm.tqdm(list(enumerate(pairs)), desc=file.name, leave=False):
						if i1 != i2:
							ok = False
							for length in get_lens(b2s):
								preds = [unmask_with_length(varnum, a1, b1, a2, length) for b1 in b1s]
								ok = ok or any(p in b2s for p in preds)
								all_preds.extend(preds)
							res_subcat.append(int(ok))
				all_results.append({
					'varnum': varnum,
					'subcategory': file.name,
					'category': file.parent.name,
					'accuracy': (sum(res_subcat) / len(res_subcat)),
					'n_correct': sum(res_subcat),
					'n_items': len(res_subcat),
					'details': all_preds,
				})
		pd.DataFrame.from_records(all_results).to_csv(f'mbert-{model_type}-unmask-de-variants-quotes{use_quotes}.csv')
