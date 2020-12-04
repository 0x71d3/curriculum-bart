import math
from collections import Counter

import torch
from torch.utils.data import Sampler


class CompetenceBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        batch_size,
        drop_last,
        epoch,
        num_epochs,
        difficulties,
        init_competence=0.01,
        form='sqrt'
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_samples = len(self.data_source)
        self.num_batches = (
            self.num_samples // self.batch_size if self.drop_last
            else (self.num_samples - 1) // self.batch_size + 1
        )

        self.epoch = epoch
        self.num_epochs = num_epochs
        
        self.difficulties = torch.as_tensor(difficulties, dtype=torch.double)
        assert len(self.difficulties) == self.num_samples
        self.init_competence = init_competence
        self.form = form
        assert self.form in ['linear', 'sqrt']

    def __iter__(self):
        t = self.epoch * self.num_batches
        T = self.num_epochs * self.num_batches
        c_0 = self.init_competence

        i2d = self.difficulties

        counter = Counter(i2d.tolist())
        sorted_d = sorted(counter)
        j2d = torch.as_tensor(sorted_d, dtype=torch.double)  # difficulty
        j2f = torch.as_tensor([counter[d] for d in sorted_d], dtype=torch.int64)  # frequency

        j2cdf = j2f.cumsum(dim=0, dtype=torch.double) / j2f.sum()

        for idx in range(self.num_batches):
            c = (
                min(1, t * (1 - c_0) / T + c_0) if self.form == 'linear'
                else min(1, math.sqrt(t * (1 - c_0 ** 2) / T + c_0 ** 2))
            )

            j_max = ((j2cdf <= c).sum() - 1).item()
            assert j_max >= 0
            d_max = j2d[j_max]

            i_max = ((i2d <= d_max).sum() - 1).item()

            yield torch.randint(
                high=i_max + 1,
                size=(
                    min(
                        self.batch_size,
                        self.num_samples - idx * self.batch_size
                    ),
                ),
                dtype=torch.int64
            ).tolist()

            t += 1  # next step

    def __len__(self):
        return self.num_batches
