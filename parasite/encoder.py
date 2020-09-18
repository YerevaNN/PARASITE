from abc import abstractmethod
import numpy as np

from tqdm import tqdm

from typing import List, Iterable, Iterator, Any, Tuple

from .doc import BiText, EncodedBiText, BatchDocs
from .applicator import Applicator


@Applicator.register('encoder')
class Encoder(Applicator):
    def __init__(self,
                 batch_size: int = None,
                 encode_windows: int = 1):
        self.batch_size = batch_size
        self.encode_windows = encode_windows

    def generate_windows(self,
                         lines: List[str],
                         window_size: int) -> Iterable[str]:
        num_lines = len(lines)
        for start in range(num_lines - window_size + 1):
            end = start + window_size
            yield ' '.join(lines[start:end])

    def apply(self, doc: BiText, *,
              only_src: bool = False,
              only_tgt: bool = False) -> EncodedBiText:
        if only_src or only_tgt:
            raise NotImplementedError

        num_src_lines = len(doc.src_lines)
        num_tgt_lines = len(doc.tgt_lines)

        src_lines = doc.src_lines
        tgt_lines = doc.tgt_lines

        if self.encode_windows > 1:
            src_lines = []
            tgt_lines = []
            for k in range(1, self.encode_windows + 1):
                src_lines += self.generate_windows(doc.src_lines, k)
                tgt_lines += self.generate_windows(doc.tgt_lines, k)

        src_embeddings, tgt_embeddings = self.encode_both(src_lines,
                                                          tgt_lines)

        return EncodedBiText(src=src_lines,
                             tgt=tgt_lines,
                             src_lang=doc.src_lang,
                             tgt_lang=doc.tgt_lang,
                             src_embeddings=src_embeddings,
                             tgt_embeddings=tgt_embeddings,
                             num_src_lines=num_src_lines,
                             num_tgt_lines=num_tgt_lines)

    def encode_both(self,
                    src_lines: List[str],
                    tgt_lines: List[str]):
        # NOTE We can accelerate further like this:
        # concat_lines = src_lines + tgt_lines
        # concat_embeddings = self.encode(concat_lines)

        # src_embeddings = concat_embeddings[:len(src_lines)]
        # tgt_embeddings = concat_embeddings[len(src_lines):]
        # return src_embeddings, tgt_embeddings

        src_embeddings = self.encode(src_lines)
        tgt_embeddings = self.encode(tgt_lines)
        return src_embeddings, tgt_embeddings

    @classmethod
    def batches(cls,
                iterable: Iterable[Any],
                batch_size: int) -> Iterator[List[Any]]:
        iterator = iter(iterable)
        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    element = next(iterator)
                    batch.append(element)
                yield batch
            except StopIteration:
                if batch:
                    yield batch
                break

    def batch_apply(self,
                    docs: BatchDocs,
                    *,
                    progress: str = None,
                    only_src: bool = False,
                    only_tgt: bool = False):
        # If no `batch_size` is set, then we use no custom
        # batch-apply strategy here.
        if self.batch_size is None:
            return super().batch_apply(docs, progress=progress,
                                       only_src=only_src, only_tgt=only_tgt)

        cls = type(docs)
        num_docs = len(docs)

        generator = self.batch_apply_iter(docs=docs,
                                          only_src=only_src,
                                          only_tgt=only_tgt)
        # The progress is tracked here instread of `batch_apply_iter`
        if progress is not None:
            if not isinstance(progress, str):
                progress = str(self)
            generator = tqdm(generator, desc=progress,
                             total=num_docs)
        return cls(
            generator,
            num_docs=num_docs
        )

    def batch_apply_iter(self,
                         docs: Iterable[Tuple[str, BiText]],
                         *,
                         progress: str = None,
                         only_src: bool = False,
                         only_tgt: bool = False
                         ) -> Iterable[Tuple[str, EncodedBiText]]:
        assert isinstance(self.batch_size, int)

        batch_docs: List[Tuple[str, BiText]]
        for batch_docs in self.batches(docs, self.batch_size):
            # Here we flatten all the lists of lines
            src_lines: List[str] = sum([doc.src_lines for _, doc in batch_docs], [])
            tgt_lines: List[str] = sum([doc.tgt_lines for _, doc in batch_docs], [])
            # We need to make sure that we have exactly one source and target language
            src_lang, = set(doc.src_lang for _, doc in batch_docs)
            tgt_lang, = set(doc.tgt_lang for _, doc in batch_docs)
            # Also, we need to make sure the batch of documents is homogeneous
            doc_cls, = set(type(doc) for _, doc in batch_docs)

            # The strategy is as follows:
            # -> We batch-encode the merged documents
            merged_doc = doc_cls(src=src_lines, tgt=tgt_lines,
                                 src_lang=src_lang, tgt_lang=tgt_lang)
            merged_result = self.apply(merged_doc,
                                       only_src=only_src,
                                       only_tgt=only_tgt)
            # -> Then we distribute encodings to the respecive documents
            src_begin = 0
            tgt_begin = 0
            src_end = 0
            tgt_end = 0
            for prefix, doc in batch_docs:
                src_end = src_begin + len(doc.src_lines)
                tgt_end = tgt_begin + len(doc.tgt_lines)
                src_embeddings = merged_result.src_embeddings[src_begin:src_end]
                tgt_embeddings = merged_result.tgt_embeddings[tgt_begin:tgt_end]

                yield prefix, EncodedBiText(src=doc.src_lines, tgt=doc.tgt_lines,
                                            src_lang=src_lang, tgt_lang=tgt_lang,
                                            src_embeddings=src_embeddings,
                                            tgt_embeddings=tgt_embeddings)
                src_begin = src_end
                tgt_begin = tgt_end

            assert src_end == len(merged_result.src_embeddings)
            assert tgt_end == len(merged_result.tgt_embeddings)

    @abstractmethod
    def encode(self, lines: List[str]) -> np.ndarray:
        ...


@Encoder.register('pretrained-transformer')
class PretrainedTransformerEncoder(Encoder):

    def __init__(self,
                 model_name: str = 'xlm-roberta-large',
                 *,
                 normalize_length: str = None,
                 normalize: int = -1,
                 cuda: bool = None,
                 fp16: bool = False,
                 force_lowercase: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        import torch
        import torch.cuda

        from transformers import AutoTokenizer, AutoModel
        from transformers import PreTrainedTokenizer, PreTrainedModel

        self.tokenizer: PreTrainedTokenizer = \
            AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = \
            AutoModel.from_pretrained(model_name)

        if cuda is None:
            cuda = torch.cuda.is_available()

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        self.fp16 = fp16
        if fp16:
            self.model.half()

        self.force_lowercase = force_lowercase
        self.normalize_length = normalize_length
        self.normalize = normalize

    def encode(self, lines: List[str]) -> Tuple[Iterable[int], np.ndarray]:
        import torch
        import torch.nn.functional as F

        if self.force_lowercase:
            lines = [line.lower() for line in lines]

        encoding = self.tokenizer.batch_encode_plus(lines, return_tensors='pt',
                                                    pad_to_max_length=True,
                                                    return_special_tokens_masks=True)

        if self.cuda:
            # TODO Need to refactor
            for key, value in encoding.items():
                encoding[key] = value.cuda()

        mask: np.ndarray = 1 - encoding['special_tokens_mask']
        del encoding['special_tokens_mask']

        with torch.no_grad():
            output = self.model(**encoding)[0]

        output = output.float()

        mask = mask.unsqueeze(-1)

        if self.normalize_length:
            if self.normalize_length == 'avg':
                output /= mask.sum(dim=1, keepdims=True)
            elif self.normalize_length == 'sqrt':
                output /= mask.sum(dim=1, keepdims=True).float().sqrt() * 100
            else:
                raise NotImplementedError(self.normalize_length)

        embeddings = (output * mask).sum(dim=1)

        if self.normalize >= 0:
            embeddings = F.normalize(embeddings, p=self.normalize, dim=-1)

        # NOTE Oringially we had something like this:
        # num_tokens = mask.sum(dim=1).squeeze(dim=-1)

        return embeddings.cpu().numpy()
