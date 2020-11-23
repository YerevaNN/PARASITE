# parasite
<img src="parasite.svg" width="300"> 

A parallel sentence preprocessing toolkit

# Interface
The codebase uses `python-fire` to have a flexible, pipelined CLI interface.

The module `parasite.pipeline` implements CLI over all the basic concepts of the codebase.

We recommend using `AlignedBiText from_files` for working with a single bi-text document or `AlignedBiText batch_from_files` to work with multiple files.

# Results
Here is effect of using different components as part of preprocessing, filtering and monotonic alignments pipeline.

All the numbers represent the BLEU score on the WMT20 MEDLINE (local test) set for different data preprocessing configurations (and the exact same architecture and learning parameters).

| **Model**                       | **en → ru** | **ru → en** |
|---------------------------------|:-----------:|:-----------:|
| baseline configuration          | 30.7        | 31.3        |
| \+ greedy alignments            | 30.1        | 31.8        |
| \+ detect subsection names      | 30.7        | 32.3        |
| \+ remove titles                | 31.3        | **32.5**    |
| \+ optimize total similarity    | 30.4        | 32.2        |
| \+ normalize distance matrix    | 30.8        | 32.1        |
| \+ penalize source/target ratio | 31.2        | 31.5        |
| \+ one-to-many (K=3)            | **32.2**    | 32.3        |

Here are training graphs on [Aim](https://github.com/aimhubio/aim) for **en → ru**, averaged across different runs. The best configuration shows *significantly better* results:

<img src="graphs.png" width="900"> 


# Example

To replicate our **best submission (run 2)** (WMT20 Biomedical Translation Task winner models for `en-ru` language pair) preprocessing, please run:

```sh
python -m parasite.pipeline \
    AlignedBiText batch_from_files /datasets/wmt20.biomed.ru-en.medline_train/raw_files/*_en.txt \
        --suffix=".txt" --src-lang="en" --tgt-lang="ru" \
    - apply segmenter reset \
    - apply segmenter scispacy --only-src \
    - apply segmenter razdel --only-tgt \
    - apply segmenter remove-title --only-tgt --blacklist='Резюме' \
    - apply segmenter keyword --only-src --path='examples/medline_keywords/eng_few.txt' \
    - apply segmenter keyword --only-tgt --path='examples/medline_keywords/rus_few.txt' \
    - apply segmenter remove-title --only-src \
    - apply encoder pretrained-transformer "xlm-roberta-large" \
        --normalize=2 --force-lowercase --normalize-length=avg --fp16 \
    - apply aligner greedy-one2one --distance=euclidean \
    --progress \
    - split --mapping-path="examples/wmt20.biomed.ru-en.medline_train.yerevann.splits.txt" \
    - to_files --output-dir="/datasets/wmt20.biomed.ru-en.medline_train/preprocessed_files"
```

In order to replicate our **overall best preprocessing (not submitted, described in the paper)**, you can run:
```sh
python -m parasite.pipeline \
    AlignedBiText batch_from_files /datasets/wmt20.biomed.ru-en.medline_train/raw_files/*_en.txt \
        --suffix=".txt" --src-lang="en" --tgt-lang="ru" \
    - apply segmenter reset \
    - apply segmenter syntok \
    - apply segmenter remove-title --only-tgt --blacklist='Резюме' \
    - apply segmenter keyword --only-src --path='examples/medline_keywords/eng.txt' \
    - apply segmenter keyword --only-tgt --path='examples/medline_keywords/rus.txt' \
    - apply segmenter remove-title --only-src \
    - apply encoder pretrained-transformer "xlm-roberta-large" \
        --encode-windows=3  --normalize-length=avg --fp16 \
    - apply aligner dynamic \
        --max-k=3 --penalty-ratio=2 --distance=euclidean --normalize=True \
    --progress \
    - split --mapping-path="examples/wmt20.biomed.ru-en.medline_train.yerevann.splits.txt" \
    - to_files --output-dir="/datasets/wmt20.biomed.ru-en.medline_train/preprocessed_files"

```
