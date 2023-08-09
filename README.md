# MATS

This is the repository for the Multilingual Analogy Test Set, made available along our *SEM 2023 paper _„Mann“ is to “Donna” as「国王」is to « Reine »: Adapting the Analogy Task for Multilingual and Contextual Embeddings_. It's based on the original BATS dataset from Gladkova et al. (2016), which you can find [here](https://vecto.space/projects/BATS/).

The dataset spans six languages (Dutch, French, German, Italian, Mandarin, Spanish). All the data is available in the `dataset/` directory. Each language has its own subdirectory; language-specific directories follow the same structure as the original BATS dataset.

This repository also contains the scripts used in our experiments. They are mainly provided for documentation and replicability purposes.

## Cite this paper

You can find the original paper at this [link](https://aclanthology.org/2023.starsem-1.25/).
A copy of the camera-ready version of our paper is also available under `docs/mickus-etal-2023-MATS.pdf`.

If you used this resource in your work, please cite our paper:
```
@inproceedings{mickus-etal-2023-mann,
    title = "{M}ann is to Donna as 「国王」 is to Reine: Adapting the Analogy Task for Multilingual and Contextual Embeddings",
    author = "Mickus, Timothee  and
      Calò, Eduardo  and
      Jacqmin, Léo  and
      Paperno, Denis  and
      Constant, Mathieu",
    booktitle = "Proceedings of the The 12th Joint Conference on Lexical and Computational Semantics (*SEM 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.starsem-1.25",
    pages = "270--283",
    abstract = "How does the word analogy task fit in the modern NLP landscape? Given the rarity of comparable multilingual benchmarks and the lack of a consensual evaluation protocol for contextual models, this remains an open question. In this paper, we introduce MATS: a multilingual analogy dataset, covering forty analogical relations in six languages, and evaluate human as well as static and contextual embedding performances on the task. We find that not all analogical relations are equally straightforward for humans, static models remain competitive with contextual embeddings, and optimal settings vary across languages and analogical relations. Several key challenges remain, including creating benchmarks that align with human reasoning and understanding what drives differences across methodologies.",
}
```
