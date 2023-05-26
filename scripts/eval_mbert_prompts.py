import pathlib
import pandas as pd
import tqdm
from transformers import pipeline

bats_dir = pathlib.Path('/scratch/project_2006235/dataset/BATS/uncased')

for model_type in ['uncased']:
	pipe = pipeline(task='fill-mask', model=f'bert-base-multilingual-{model_type}', device=0, top_k=1)

	def get_lens(bs):
		return {len(pipe.tokenizer.encode(b, add_special_tokens=False)) for b in bs}

	def get_template_with_quotes(lang, a, b, c, len_d):
		mask = " ".join([pipe.tokenizer.mask_token] * len_d)
		if lang == 'en':
			return f'"{a}" is to "{b}" as "{c}" is to "{mask}".'
		if lang == 'fr':
			return f'"{a}" est à "{b}" ce que "{c}" est à "{mask}".'
		if lang == 'it':
			return f'"{a}" sta a "{b}" come "{c}" sta a "{mask}".'
		if lang == 'es':
			return f'"{a}" es a "{b}" como "{c}" es a "{mask}".'
		if lang == 'zh':
			return f'「{a}」与「{b}」的关系就像「{c}」与「{mask}」的关系。'
		if lang == 'de':
			return f'"{a}" verhält sich zu "{b}" wie "{c}" zu "{mask}".'
		if lang == 'nl':
			return f'"{a}" staat tot "{b}" zoals "{c}" staat tot "{mask}".'

	def get_template_no_quotes(lang, a, b, c, len_d):
		mask = " ".join([pipe.tokenizer.mask_token] * len_d)
		if lang == 'en':
			return f'{a} is to {b} as {c} is to {mask}.'
		if lang == 'fr':
			return f'{a} est à {b} ce que {c} est à {mask}.'
		if lang == 'it':
			return f'{a} sta a {b} come {c} sta a {mask}.'
		if lang == 'es':
			return f'{a} es a {b} como {c} es a {mask}.'
		if lang == 'zh':
			return f'{a}与{b}的关系就像{c}与{mask}的关系。'
		if lang == 'de':
			return f'{a} verhält sich zu {b} wie {c} zu {mask}.'
		if lang == 'nl':
			return f'{a} staat tot {b} zoals {c} staat tot {mask}.'

	for use_quotes in tqdm.tqdm([True, False], desc='quotes'):
		get_template = get_template_with_quotes if use_quotes else get_template_no_quotes
		def unmask_with_length(lang, a1, b1, a2, length_b2):
			preds = pipe(get_template(lang, a1, b1, a2, length_b2))
			if length_b2 == 1: preds = [preds]
			token_strs = [p[0]['token_str'] for p in preds]
			token_strs = ' '.join(token_strs)
			token_strs = token_strs.replace(' ##', '')
			return token_strs

		all_results = []
		for lang in tqdm.tqdm(sorted(['en'])): #'de', 'fr', 'es', 'it', 'zh', 'nl'])):
			mats_dir = bats_dir / lang
			for file in tqdm.tqdm(sorted(mats_dir.glob('**/*.txt')), desc=lang):
				with open(file) as fh:
					pairs = map(str.split, filter(None, map(str.strip, fh)))
					pairs = [(a, set(filter(None, b.split('/')))) for a, b in pairs]
				res_subcat = []
				for i1, (a1, b1s) in tqdm.tqdm(list(enumerate(pairs)), desc=file.name, leave=False):
					for i2, (a2, b2s) in tqdm.tqdm(list(enumerate(pairs)), desc=file.name, leave=False):
						if i1 != i2:
							ok = False
							for length in get_lens(b2s):
								ok = ok or any(map(lambda b1: unmask_with_length(lang, a1, b1, a2, length) in b2s, b1s))
							res_subcat.append(int(ok))
				all_results.append({
					'lang': lang,
					'subcategory': file.name,
					'category': file.parent.name,
					'accuracy': (sum(res_subcat) / len(res_subcat)),
					'n_correct': sum(res_subcat),
					'n_items': len(res_subcat)
				})
		pd.DataFrame.from_records(all_results).to_csv(f'mbert-{model_type}-unmask-alllangs-en-quotes{use_quotes}.csv')
