from abc import abstractmethod
import numpy as np

from registrable import Registrable


class Distance(Registrable):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self,
                 a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        ...


@Distance.register('euclidean')
class EuclideanDistance(Distance):
    def __init__(self,
                 p: float = 2,
                 normalize: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.normalize = normalize

    def __call__(self,
                 a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        # Shape: (num_a_vectors, 1, embedding_dim)
        a = np.expand_dims(a, axis=1)
        # Shape: (1, num_b_vectors, embedding_dim)
        b = np.expand_dims(b, axis=0)
        # Shape: (num_a_vectors, num_b_vectors)
        distances: np.ndarray = np.linalg.norm(a - b, ord=2, axis=-1)

        if self.normalize:
            distances /= np.sqrt(distances.mean(axis=0, keepdims=True) * distances.mean(axis=1, keepdims=True))

        return distances


@Distance.register('cosine')
class CosineDistance(Distance):
    def __cal__(self,
                a: np.ndarray,
                b: np.ndarray):
        from sklearn.metrics.pairwise import cosine_distances
        return cosine_distances(a, b)


@Distance.register('margin-cosine')
class MarginCosineSimilarity(Distance):
    def __call__(self,
                 a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        # Shape: (num_a_vectors, num_b_vectors)
        similarities: np.ndarray = cosine_similarity(a, b)
        # Shape: (num_a_vectors, 1)
        mean_from_a = similarities.mean(axis=1, keepdims=True)
        # Shape: (1, num_b_vectors)
        mean_from_b = similarities.mean(axis=0, keepdims=True)

        similarities /= (mean_from_a + mean_from_b) / 2

        return similarities.max() - similarities
