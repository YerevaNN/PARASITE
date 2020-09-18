import os

import numpy as np

from glob import glob
from textwrap import wrap
from tabulate import tabulate

from collections import defaultdict

from typing import List, Union, Iterator, Iterable, Tuple, Dict
from typing import TypeVar, Generic

from .applicator import Applicator


T = TypeVar('T', bound='BiText')


class BatchDocs(Generic[T]):
    def __init__(self,
                 docs: Iterable[Tuple[str, T]],
                 num_docs: int = None):
        self.docs = docs
        if isinstance(docs, list):
            num_docs = len(docs)
        self.num_docs = num_docs

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return self.num_docs

    def __str__(self):
        str_repr = ""
        prefix: str
        doc: T
        for prefix, doc in self.docs:
            str_repr += f"{prefix}\n{doc}\n\n"
        return str_repr

    def to_files(self,
                 output_dir: str,
                 suffix: str = ''):
        prefix: str
        doc: T
        for prefix, doc in self.docs:
            dirname = os.path.dirname(prefix)
            basename = os.path.basename(prefix)
            path_prefix = os.path.join(output_dir, basename)
            doc.to_files(path_prefix, suffix=suffix)

    def apply(self,
              applicator_type: Union['Applicator', str],
              applicator: Union['Applicator', str] = None,
              *args,
              only_src: bool = False,
              only_tgt: bool = False,
              progress: str = None,
              **kwargs) -> T:
        if not isinstance(applicator_type, str):
            assert applicator is None
            fn = applicator_type
        else:
            applicator_cls = Applicator.by_name(applicator_type).by_name(applicator)
            fn = applicator_cls(*args, **kwargs)

        return fn.batch_apply(self,
                              only_src=only_src,
                              only_tgt=only_tgt,
                              progress=progress)

    def split(self,
              mapping_path: str):
        subsets: Dict[str, str] = dict()
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                prefix, _, subset = line.partition('\t')
                subsets[prefix] = subset

        docs = list(self.docs)

        cls, = set(type(doc) for _, doc in docs)
        src_lang, = set(doc.src_lang for _, doc in docs)
        tgt_lang, = set(doc.tgt_lang for _, doc in docs)

        src_lines: Dict[str, List[str]] = defaultdict(list)
        tgt_lines: Dict[str, List[str]] = defaultdict(list)

        for prefix, doc in docs:
            basename = os.path.basename(prefix)
            subset = subsets[basename]
            src_lines[subset] += doc.src_lines
            tgt_lines[subset] += doc.tgt_lines

        merged_docs: List[Tuple[str, T]] = []
        for subset in src_lines.keys() | tgt_lines.keys():
            doc = cls(src=src_lines[subset],
                      tgt=tgt_lines[subset],
                      src_lang=src_lang,
                      tgt_lang=tgt_lang)
            prefix = f'{subset}.{src_lang}-{tgt_lang}.'
            merged_docs.append((prefix, doc))

        return BatchDocs(merged_docs)


class BiText:
    __slots__ = ('src_lang', 'tgt_lang',
                 'src_lines', 'tgt_lines')

    def __init__(self,
                 src: Union[str, List[str]],
                 tgt: Union[str, List[str]],
                 *,
                 src_lang: str,
                 tgt_lang: str):

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.src_lines: List[str]
        self.tgt_lines: List[str]

        if isinstance(src, str):
            self.src_lines = src.split('\n')
        else:
            self.src_lines = src

        if isinstance(tgt, str):
            self.tgt_lines = tgt.split('\n')
        else:
            self.tgt_lines = tgt

    @classmethod
    def read_lines(cls,
                   file_path: str) -> Iterator[str]:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                yield line

    @classmethod
    def write_lines(cls,
                    lines: Iterable[str],
                    file_path: str):
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')

    @classmethod
    def batch_from_files(cls,
                         *prefixes: str,
                         src_lang: str,
                         tgt_lang: str,
                         suffix: str = ''):
        resolved_prefixes: List[str] = []
        for prefix in prefixes:
            if not prefix.endswith(f"{src_lang}{suffix}"):
                prefix = f"{prefix}{src_lang}{suffix}"
            for path in glob(prefix):
                suffix_offset = len(src_lang) + len(suffix)
                resolved_prefix = path[:-suffix_offset]
                resolved_prefixes.append(resolved_prefix)

        generator = (
            (prefix, cls.from_files(prefix,
                                    src_lang=src_lang,
                                    tgt_lang=tgt_lang,
                                    suffix=suffix))
            for prefix in resolved_prefixes
        )
        return BatchDocs(
            generator,
            num_docs=len(prefixes)
        )

    @classmethod
    def from_files(cls,
                   prefix: str,
                   *,
                   src_lang: str,
                   tgt_lang: str,
                   suffix: str = ''):
        src_path = f'{prefix}{src_lang}{suffix}'
        tgt_path = f'{prefix}{tgt_lang}{suffix}'

        src_lines = list(cls.read_lines(src_path))
        tgt_lines = list(cls.read_lines(tgt_path))

        return cls(src=src_lines,
                   tgt=tgt_lines,
                   src_lang=src_lang,
                   tgt_lang=tgt_lang)

    def to_files(self,
                 prefix: str,
                 suffix: str = ''):
        src_path = f'{prefix}{self.src_lang}{suffix}'
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        tgt_path = f'{prefix}{self.tgt_lang}{suffix}'
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        self.write_lines(self.src_lines, src_path)
        self.write_lines(self.tgt_lines, tgt_path)

    def segment(self, segmenter: Union['Applicator', str],
                *args,
                only_src: bool = False,
                only_tgt: bool = False,
                **kwargs) -> 'BiText':
        if isinstance(segmenter, str):
            applicator_cls = Applicator.by_name('segmenter').by_name(segmenter)
            applicator = applicator_cls(**kwargs)
        else:
            assert not kwargs
            applicator = segmenter

        return applicator(self,
                          only_src=only_src,
                          only_tgt=only_tgt)

    def encode(self, encoder: Union['Applicator', str],
               *args,
               **kwargs):
        if isinstance(encoder, str):
            applicator_cls = Applicator.by_name('encoder').by_name(encoder)
            applicator = applicator_cls(**kwargs)
        else:
            assert not kwargs
            applicator = encoder

        return applicator(self)

    @classmethod
    def wrap_row(cls, *cols: str, **kwargs) -> List[str]:
        return [
            '\n'.join(wrap(col, **kwargs))
            for col in cols
        ]

    def __str__(self):
        src_lines = self.src_lines
        tgt_lines = self.tgt_lines

        num_src_lines = len(src_lines)
        num_tgt_lines = len(tgt_lines)
        num_rows = max(num_src_lines, num_tgt_lines)

        src_lines = [''] * (num_rows - num_src_lines) + src_lines
        tgt_lines = [''] * (num_rows - num_tgt_lines) + tgt_lines

        rows = [
            self.wrap_row(src_line, tgt_line)
            for src_line, tgt_line
            in zip(src_lines, tgt_lines)
        ]

        return tabulate(
            rows,
            headers=[self.src_lang, self.tgt_lang],
            tablefmt='grid',
            showindex='always'
        )


class AlignedBiText(BiText):
    def __init__(self,
                 src: Union[str, List[str]],
                 tgt: Union[str, List[str]],
                 *,
                 src_lang: str,
                 tgt_lang: str):
        super().__init__(src, tgt, src_lang=src_lang, tgt_lang=tgt_lang)
        assert len(self.src_lines) == len(self.tgt_lines)


class EncodedBiText(BiText):
    __slots__ = ('src_lang', 'tgt_lang',
                 'src_lines', 'tgt_lines',
                 '_src_embeddings', '_tgt_embeddings',
                 '_src_lines', '_tgt_lines',
                 'num_src_lines', 'num_tgt_lines',
                 'num_src_tokens', 'num_tgt_tokens')

    def __init__(self,
                 src: Union[str, List[str]],
                 tgt: Union[str, List[str]],
                 *,
                 src_lang: str,
                 tgt_lang: str,
                 #  num_src_tokens: int,
                 #  num_tgt_tokens: int,
                 src_embeddings: np.ndarray,
                 tgt_embeddings: np.ndarray,
                 num_src_lines: int = None,
                 num_tgt_lines: int = None):
        if num_src_lines is None:
            num_src_lines = len(src)
        self.num_src_lines = num_src_lines
        if num_tgt_lines is None:
            num_tgt_lines = len(tgt)
        self.num_tgt_lines = num_tgt_lines

        super().__init__(src[:self.num_src_lines],
                         tgt[:self.num_tgt_lines],
                         src_lang=src_lang, tgt_lang=tgt_lang)

        self._src_embeddings = src_embeddings
        self._tgt_embeddings = tgt_embeddings

        self._src_lines = src
        self._tgt_lines = tgt

        # self.num_src_tokens = num_src_tokens
        # self.num_tgt_tokens = num_tgt_tokens

    @property
    def src_embeddings(self):
        return self._src_embeddings[:self.num_src_lines]

    @property
    def tgt_embeddings(self):
        return self._tgt_embeddings[:self.num_tgt_lines]

    def src_windows_embeddings(self):
        return self._src_embeddings

    def tgt_windows_embeddings(self):
        return self._tgt_embeddings

    @property
    def src_windows_lines(self):
        return self._src_lines

    @property
    def tgt_windows_lines(self):
        return self._tgt_lines
