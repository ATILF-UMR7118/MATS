import pprint, pathlib, sys, tempfile

import gensim

import vecto.embeddings
from vecto.benchmarks.analogy import Benchmark as Analogy
from vecto.data import Dataset

class LossPrint():
	pass

def run_analogy(model_path, analogy_path, result_path):
    print("running analogy")
    embs = vecto.embeddings.load_from_dir(model_path)
    analogy = Analogy(method="3CosAdd")
    dataset = Dataset(analogy_path)
    result = analogy.run(embs, dataset)
    print("saving data")
    with open(result_path, "w") as fp:
        pprint.pprint(result, fp)
    for block in result:
        pprint.pprint(block['result'])
        pprint.pprint(block['experiment_setup'])
        print()
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run MATS on some model")
    parser.add_argument("--to-vecto", action='store_true')
    parser.add_argument("model_path", type=str)
    parser.add_argument("analogy_path", type=str)
    parser.add_argument("result_path", type=str)
    args = parser.parse_args()
    if pathlib.Path(args.result_path).is_file():
        print("skipping existing file", args.result_path)
    else:
        if args.to_vecto:
            print(f"converting model {args.model_path} to vecto format")
            vecto_dir_path = pathlib.Path(args.model_path) / "vecto"
            vecto_dir_path.mkdir(exist_ok=True)
            if not (vecto_dir_path / "model.txt").is_file():
                print('converting model')
                model = str(next(pathlib.Path(args.model_path).glob('*.txt')))
                gensim.models.Word2Vec.load(model).wv.save_word2vec_format(str(vecto_dir_path / "model.txt"))
            model_path = str(vecto_dir_path)
        else:
            model_path = args.model_path
        print("start")
        results = run_analogy(model_path, args.analogy_path, args.result_path)
        single_score = sum(block['result']['accuracy'] for block in results) / len(results)
        print(f"Model {args.model_path}, single score: {single_score}")
        print("end.")
