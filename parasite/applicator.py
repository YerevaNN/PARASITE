from tqdm import tqdm

from abc import abstractmethod
from registrable import Registrable


class Applicator(Registrable):

    def __call__(self, doc,
                 *,
                 only_src: bool = False,
                 only_tgt: bool = False):
        return self.apply(doc,
                          only_src=only_src,
                          only_tgt=only_tgt)

    @abstractmethod
    def apply(self, doc,
              *,
              only_src: bool = False,
              only_tgt: bool = False):
        ...

    def batch_apply(self,
                    docs,
                    *,
                    progress: str = None,
                    only_src: bool = False,
                    only_tgt: bool = False):
        cls = type(docs)
        num_docs = len(docs)
        generator = (
            (prefix, self.apply(doc,
                                only_src=only_src,
                                only_tgt=only_tgt))  # Here we had a bug "achqis"
            for prefix, doc in docs
        )
        if progress is not None:
            if not isinstance(progress, str):
                progress = str(self)
            generator = tqdm(generator, desc=progress,
                             total=num_docs)
        return cls(
            generator,
            num_docs=num_docs
        )
