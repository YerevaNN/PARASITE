from abc import abstractmethod
import re
import itertools

from typing import List, Iterable, Iterator, Pattern, Union

from .doc import BiText
from .applicator import Applicator


@Applicator.register('segmenter')
class Segmenter(Applicator):

    def __init__(self, *, reset: bool = False):
        self.reset = reset

    def segment_side(self, lines: List[str],
                     pass_through: bool = False) -> List[str]:
        if pass_through:
            return lines

        if self.reset:
            doc = ' '.join(lines)
            lines = [doc]

        result: List[str] = []
        for line in self.segment(lines):
            line = line.strip()
            if not line:
                continue
            result.append(line)

        return result

    def apply(self,
              doc: BiText,
              *,
              only_src: bool = False,
              only_tgt: bool = False) -> BiText:

        return BiText(self.segment_side(doc.src_lines, pass_through=only_tgt),
                      self.segment_side(doc.tgt_lines, pass_through=only_src),
                      src_lang=doc.src_lang,
                      tgt_lang=doc.tgt_lang)

    @abstractmethod
    def segment(self,
                lines: List[str]) -> Iterable[str]:
        ...


@Segmenter.register('noop')
class NoOpSegmenter(Segmenter):
    def segment(self, lines: List[str]) -> Iterator[str]:
        yield from lines


@Segmenter.register('reset')
class ResetSegmenter(NoOpSegmenter):
    def __init__(self, **kwargs):
        super().__init__(reset=True, **kwargs)


@Segmenter.register('syntok')
class SyntokSegmenter(Segmenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from syntok import segmenter as syntok_segmenter
        self.analyzer = syntok_segmenter.analyze

    def segment_line(self, line: str) -> Iterator[str]:
        for paragraph in self.analyzer(line):
            for snt in paragraph:
                line = ''.join(str(token) for token in snt)
                yield line

    def segment(self, lines: List[str]) -> Iterator[str]:
        for line in lines:
            yield from self.segment_line(line)


@Segmenter.register('spacy')
class SpacySegmenter(Segmenter):

    def __init__(self,
                 spacy_model: str,
                 **kwargs):
        super().__init__(**kwargs)

        import spacy
        self.nlp = spacy.load(spacy_model)

    def segment(self, lines: List[str]) -> Iterator[str]:
        for line in self.nlp.pipe(lines):
            for snt in line.sents:
                yield snt.text


@Segmenter.register('scispacy')
class SciSpacySegmenter(SpacySegmenter):

    MODEL = 'en_core_sci_sm'

    def __init__(self, **kwargs):
        super().__init__(spacy_model=self.MODEL, **kwargs)


@Segmenter.register('razdel')
class RazdelSegmenter(Segmenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from razdel.segmenters.sentenize import SentSegmenter
        self.sent_segmenter = SentSegmenter()

    def segment_line(self, line: str) -> Iterator[str]:
        for snt in self.sent_segmenter(line):
            yield snt.text

    def segment(self, lines: List[str]) -> Iterator[str]:
        for line in lines:
            yield from self.segment_line(line)


@Segmenter.register('keyword')
class KeywordSegmenter(Segmenter):

    def __init__(self, path: str, top: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.top = top
        self.keywords = self.read_keywords(self.path)
        self.regex = self.compile_regex(self.keywords)

    @classmethod
    def build_keywords_pattern(cls, keywords: List[str]) -> str:
        keywords = sorted(keywords, key=len, reverse=True)
        return '|'.join(
            re.escape(keyword)
            for keyword in keywords
        )

    def read_keywords(self, path: str) -> List[str]:
        lines: List[str] = []
        with open(path, 'r') as f:
            for line in itertools.islice(f, self.top):
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
        return lines

    @classmethod
    def compile_regex(cls, keywords: List[str]) -> Pattern[str]:
        keywords_pattern = cls.build_keywords_pattern(keywords)

        start_pattern = r'(?:^|\n)((?:' + keywords_pattern + r')[:\.\s]*?)(?=\n|$|[0-9А-ЯA-Z])'
        end_pattern =   r'(?:^|\s)((?:' + keywords_pattern + r')[:\.\s]*?)(?=\n|$)'

        pattern = r'(?:' + start_pattern + r'|' + end_pattern + r')'
        # print(pattern)
        return re.compile(pattern)

    def segment_line(self, line: str) -> Iterator[str]:
        for snt in self.regex.split(line):
            if snt is None:
                continue
            yield snt

    def segment(self, lines: List[str]) -> Iterator[str]:
        for line in lines:
            yield from self.segment_line(line)


@Segmenter.register('remove_title')
@Segmenter.register('remove-title')
class RemoveTitleSegmenter(Segmenter):

    def __init__(self, blacklist: Union[str, List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        if blacklist is None:
            blacklist = []
        if not isinstance(blacklist, list):
            blacklist = [blacklist]
        self.prefixes: List[str] = blacklist

    def segment_line(self, line: str) -> Iterator[str]:
        if line.startswith('[') and (']' in line[-4:]):
            return
        for prefix in self.prefixes:
            if line.startswith(prefix):
                line = line[len(prefix):]
        yield line

    def segment(self, lines: List[str]):
        for line in lines:
            yield from self.segment_line(line)
