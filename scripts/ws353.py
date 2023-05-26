import gensim
import numpy as np
import scipy.stats
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lang', type=str)
parser.add_argument('model', type=str)
parser.add_argument('--vecto-format', action='store_true')
args = parser.parse_args()

if args.vecto_format:
  model = gensim.models.KeyedVectors.load_word2vec_format(args.model, no_header=True)
else:
  model = gensim.models.word2vec.Word2Vec.load(args.model).wv
df = pd.read_csv(args.lang + '-ws353.dataset', sep=';', index_col='#id')[['word1', 'word2', 'score']]
def cosine(a, b):
  try:
    a = np.array(model[a.lower()])
    b = np.array(model[b.lower()])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  except KeyError:
    return None

df['cos'] =  df.apply(lambda row: cosine(row['word1'], row['word2']), axis=1)
len_before = len(df)
df = df.dropna()
len_after = len(df)
print(args.model, *scipy.stats.spearmanr(df['cos'], df['score']), len_before, len_after)
