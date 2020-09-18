# parasite
<img src="parasite.svg" width="300"> 

A parallel sentence preprocessing toolkit

# Interface
The codebase uses `python-fire` to have a flexible, pipelined CLI interface.

The module `parasite.pipeline` implements CLI over all the basic concepts of the codebase.

We recommend using `AlignedBiText from_files` for working with a single bi-text document or `AlignedBiText batch_from_files` to work with multiple files.

# Example

In order to replicate our best preprocessing for 
*WMT20 Biomedical Translation Task winner model*, you can run:
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
    - split --mapping-path="/datasets/wmt20.biomed.ru-en.medline_train/splits.txt" \
    - to_files --output-dir="/datasets/wmt20.biomed.ru-en.medline_train/preprocessed_files"

```