import numpy as np
from torch.utils.data import Dataset

from baseg.samplers.single import TiledSampler
from baseg.samplers.utils import IndexedBounds


class RandomTiledBatchSampler(TiledSampler):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        batch_size: int,
        length: int = None,
        seed: int = None,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        super().__init__(dataset, tile_size, length=length)
        self.batch_size = batch_size
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random
        self.indices = self._rng.choice(len(self.dataset), len(self), replace=True)

    def __len__(self):
        return super().__len__() // self.batch_size

    def __iter__(self):
        # sample a random image, then sample a random tile from that image
        for i in self.indices:
            width, height = self.shapes[i]
            x = self._rng.randint(0, width - self.tile_size + 1, self.batch_size)
            y = self._rng.randint(0, width - self.tile_size + 1, self.batch_size)
            yield [IndexedBounds(i, (x[j], y[j], self.tile_size, self.tile_size)) for j in range(self.batch_size)]
