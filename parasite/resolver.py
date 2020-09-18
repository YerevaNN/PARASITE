import numpy as np

from functools import lru_cache

from typing import Tuple


class DynamicResolver:
    def __init__(self,
                 matrix: np.ndarray,
                 *,
                 num_src_lines: int = None,
                 num_tgt_lines: int = None,
                 max_k: int = 3,
                 windows_importance: bool = False
                 ):
        self.matrix = 100 - matrix
        self.max_k = max_k
        self.windows_importance = windows_importance
        self.n, self.m = matrix.shape
        self.num_src_lines = num_src_lines or self.n
        self.num_tgt_lines = num_tgt_lines or self.m

    def __call__(self) -> Tuple[float, Tuple]:
        best, path = self.resolve()
        return best, path

    @lru_cache(maxsize=None)
    def offset(self,
               begin: int,
               end: int,
               num_lines: int) -> int:
        if end - begin == 1:
            return begin
        num_window_elements = num_lines - (end - begin) + 2
        prev_offset = self.offset(begin, end - 1, num_lines)
        return prev_offset + num_window_elements

    def extract_candidate(self,
                          i: int, src_window_size: int,
                          j: int, tgt_window_size: int,) -> Tuple[float, Tuple]:

        from_i = i - src_window_size
        from_j = j - tgt_window_size

        if from_i < 0 or from_j < 0:
            return 0, ()

        candidate_score, candidate_path = self.resolve(from_i, from_j)

        if src_window_size == 0 or tgt_window_size == 0:
            return candidate_score, candidate_path

        offset_i = self.offset(from_i, i, self.num_src_lines)
        offset_j = self.offset(from_j, j, self.num_tgt_lines)

        if offset_i >= self.n or offset_j >= self.m:
            return 0, ()

        added_score = self.matrix[offset_i, offset_j]
        if self.windows_importance:
            added_score *= (src_window_size + tgt_window_size)
        candidate_score += added_score
        candidate_path = ((offset_i, offset_j), candidate_path)

        return candidate_score, candidate_path

    @lru_cache(maxsize=None)
    def resolve(self,
                i: int = None, j: int = None) -> Tuple[float, Tuple]:
        if i is None:
            i = self.num_src_lines
        if j is None:
            j = self.num_tgt_lines

        if i <= 0 or j <= 0:
            return 0, ()

        best_score: float = 0.0
        best_path: Tuple = ()

        for src_window_size in range(self.max_k + 1):
            for tgt_window_size in range(self.max_k + 1):
                if src_window_size == 0 and tgt_window_size == 0:
                    continue
                if src_window_size > 1 and tgt_window_size > 1:
                    continue

                candidate = self.extract_candidate(i, src_window_size,
                                                   j, tgt_window_size)
                candidate_score, candidate_path = candidate
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_path = candidate_path

        return best_score, best_path
