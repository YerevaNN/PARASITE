from abc import abstractmethod
import numpy as np


from overrides import overrides
from typing import List, Union, Tuple, Dict


from .doc import AlignedBiText, EncodedBiText
from .distance import Distance
from .applicator import Applicator
from .resolver import DynamicResolver


@Applicator.register('aligner')
class Aligner(Applicator):
    pass


class DistanceAligner(Aligner):

    def __init__(self,
                 distance: Union[Distance, str],
                 penalty_ratio: int = 1000,
                 **kwargs):
        super().__init__()

        self.penalty_ratio = penalty_ratio

        self.distance: Distance
        if isinstance(distance, str):
            self.distance = Distance.by_name(distance)(**kwargs)
        else:
            assert not kwargs
            self.distance = distance

        self.Z2 = 1.8 * 10000
        self.Z1 = 2.2 * 10000

    @overrides
    def apply(self, doc: EncodedBiText, *,
              only_src: bool = False,
              only_tgt: bool = False) -> AlignedBiText:
        if only_src or only_tgt:
            raise NotImplementedError

        return self.align(doc)

    def resolve(self,
                m: np.ndarray,
                doc: EncodedBiText) -> AlignedBiText:
        src_lines: List[str] = []
        tgt_lines: List[str] = []

        nonzero_indices = np.nonzero(m > self.Z2)
        for i, j in zip(*nonzero_indices):
            src_lines.append(doc.src_windows_lines[i])
            tgt_lines.append(doc.tgt_windows_lines[j])

        return AlignedBiText(src=src_lines, tgt=tgt_lines,
                             src_lang=doc.src_lang, tgt_lang=doc.tgt_lang)

    def ratio_penalty_matrix(self,
                             src_lines: List[str],
                             tgt_lines: List[str],
                             penalty: float):
        # TODO right now this is char-level, not wordpieces
        src_lens = np.array([len(line.split()) for line in src_lines]) + 1
        tgt_lens = np.array([len(line.split()) for line in tgt_lines]) + 1

        src_lens = np.expand_dims(src_lens, axis=1)
        tgt_lens = np.expand_dims(tgt_lens, axis=0)

        mask = np.logical_or(src_lens > self.penalty_ratio * tgt_lens,
                             tgt_lens > src_lens * self.penalty_ratio)

        return mask * penalty

    # @overrides
    def align(self, doc: EncodedBiText):
        # Shape: (num_src_lines, num_tgt_lines)
        distances = self.distance(doc.src_embeddings,
                                  doc.tgt_embeddings)
        current = distances.copy()

        current += self.ratio_penalty_matrix(src_lines=doc.src_lines,
                                             tgt_lines=doc.tgt_lines,
                                             penalty=self.Z2 / 2)

        current = self.solve(current)

        aligned = self.resolve(current, doc)

        return aligned

    @abstractmethod
    def solve(self, matrix: np.ndarray) -> np.ndarray:
        ...


@Aligner.register('greedy-one2one')
class GreedyOne2OneDistanceAligner(DistanceAligner):

    def decoding_step(self, m):
        m = m.copy()
        i, j = np.unravel_index(m.argmin(), m.shape)
        m[:i + 1, j:] = self.Z2
        m[i:, :j + 1] = self.Z2
        m[i, j] = self.Z1
        return m

    def solve(self, matrix: np.ndarray) -> np.ndarray:
        state = matrix.copy()

        while state.min() < self.Z2:
            state = self.decoding_step(state)

        return state


@Aligner.register('dynamic-one2one')
class DynamicOne2OneDistanceAligner(DistanceAligner):

    def solve(self,
              matrix: np.ndarray) -> np.ndarray:

        matrix = 100 - matrix

        cache: Dict[Tuple[int, int], Tuple[float, Tuple]] = dict()

        n, m = matrix.shape

        def helper(i: int, j: int) -> Tuple[float, Tuple]:
            if i < 0 or j < 0:
                return 0, ()

            if (i, j) in cache:
                return cache[i, j]

            up_score, up_path = helper(i - 1, j)
            left_score, left_path = helper(i, j - 1)
            diag_score, diag_path = helper(i - 1, j - 1)
            diag_score += matrix[i, j]
            diag_path = ((i, j), diag_path)

            if up_score > left_score and up_score > diag_score:
                cache[i, j] = up_score, up_path
            elif left_score > up_score and left_score > diag_score:
                cache[i, j] = left_score, left_path
            else:
                cache[i, j] = diag_score, diag_path

            return cache[i, j]

        best, path = helper(n - 1, m - 1)

        matrix[...] = self.Z2
        while path:
            (i, j), path = path
            matrix[i, j] = self.Z1

        return matrix


@Aligner.register('dynamic')
class DynamicDistanceAligner(DistanceAligner):

    def __init__(self,
                 max_k: int = 3,
                 windows_importance: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_k = max_k
        self.windows_importance = windows_importance

    def solve(self,
              matrix: np.ndarray,
              num_src_lines: int = None,
              num_tgt_lines: int = None) -> np.ndarray:

        resolver = DynamicResolver(matrix,
                                   num_src_lines=num_src_lines,
                                   num_tgt_lines=num_tgt_lines,
                                   max_k=self.max_k,
                                   windows_importance=self.windows_importance)

        best, path = resolver.resolve()

        matrix[...] = self.Z2
        while path:
            (i, j), path = path
            matrix[i, j] = self.Z1

        return matrix

    @overrides
    def align(self, doc: EncodedBiText):
        # Shape: (num_src_lines, num_tgt_lines)
        distances = self.distance(doc.src_windows_embeddings(),
                                  doc.tgt_windows_embeddings())

        current = distances.copy()

        current += self.ratio_penalty_matrix(src_lines=doc.src_windows_lines,
                                             tgt_lines=doc.tgt_windows_lines,
                                             penalty=self.Z2 / 2)

        current = self.solve(current,
                             num_src_lines=doc.num_src_lines,
                             num_tgt_lines=doc.num_tgt_lines)

        aligned = self.resolve(current, doc)

        return aligned
