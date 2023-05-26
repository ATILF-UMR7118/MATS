import datetime
import multiprocessing as mp
import pathlib

import gensim
import tqdm




def _decode(line):
    return line.decode("utf-8").strip()

class TQDMEpochLogger(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self, save_dir, *_, disable=False):
        self._pbar = None
        self.save_dir = pathlib.Path(save_dir)
        self.epoch = 0
        self.disable = disable
    def on_train_begin(self, *_):
        self._pbar = tqdm.trange(5, leave=True, position=0, desc="train", disable=self.disable)
    def on_train_end(self, *_):
        self._pbar.close()
    def on_epoch_end(self, model):
        #callbacks = model.callbacks
        #model.callbacks = None
        self.epoch += 1
        if self.disable: print(f"{datetime.datetime.now()}: Epoch {self.epoch} done.")
        (self.save_dir / f'epoch_{self.epoch}').mkdir(parents=True, exist_ok=True)
        model.save(
            str(self.save_dir / f'epoch_{self.epoch}' / f'model_e{self.epoch}.txt'),
            # write_header=False
        )
        #model.callbacks = callbacks
        self._pbar.update()

class Dataset():
    def __init__(self, path, disable=False):
        self._path = path
        self.disable = disable
        self._size = None
    def __iter__(self):
        with open(self._path, 'rb') as lines:
            lines = map(_decode, lines)
            lines = filter(None, lines)
            lines = map(str.split, lines)
            pbar = tqdm.tqdm(
                lines,
                total=self._size,
                leave=False,
                desc="lines",
                position=1,
                unit_scale=True,
                disable=self.disable,
            )
            if self._size is None:
                for i, obj in enumerate(pbar, start=1):
                    yield obj
                self._size = i
            else:
                yield from pbar
            pbar.close()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_text_path', type=pathlib.Path, required=True)
	parser.add_argument('--model_path', type=pathlib.Path, required=True)
	parser.add_argument('--negative_examples', type=int, default=5)
	parser.add_argument('--nsexponent', type=float, default=0.75)
	parser.add_argument('--shrinkwindows', action='store_true')
	parser.add_argument('--mincount', type=int, default=5)
	parser.add_argument('--window', type=int, default=10)
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()

	print(f"{datetime.datetime.now()}: Handling {args.input_text_path}")
	corpus = Dataset(args.input_text_path, disable=not args.verbose)
	pbar = tqdm.trange(1, position=0, leave=False, desc="vocab", disable=not args.verbose)
	logger = TQDMEpochLogger(args.model_path, disable=not args.verbose)
	model = gensim.models.word2vec.Word2Vec(
		sentences=corpus,
		min_count=args.mincount,
		callbacks=(logger,),
		workers=mp.cpu_count(),
		negative=args.negative_examples,
		window=args.window,
		vector_size=50,
		epochs=5,
		ns_exponent=args.nsexponent,
		shrink_windows=args.shrinkwindows,
	)
	pbar.close()
	print(f"{datetime.datetime.now()}: Training complete.")
