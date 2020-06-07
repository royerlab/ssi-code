import math

import dask
import numpy

from utils.log.log import lprint, lsection
from utils.normaliser.base import NormaliserBase


class PercentileNormaliser(NormaliserBase):
    """
        Percentile Normaliser

    """

    percent: float

    def __init__(self, percentile: float = None, **kwargs):
        """
        Constructs a normaliser
        """

        super().__init__(**kwargs)

        self.percentile = percentile

    def calibrate(self, array):

        with lsection(f"Calibrating array using percentile method"):
            self.original_dtype = array.dtype

            if self.percentile is None:
                # We compute an ideal percentile for this array given the size:
                size = array.size
                p = min(0.00001, math.sqrt(size) / size)
            else:
                p = self.percentile

            lprint(f"Using percentile value: {p}")

            if hasattr(array, '__dask_keys__'):
                self.rmin = dask.array.percentile(array.flatten(), 100 * p).compute()
                self.rmax = dask.array.percentile(
                    array.flatten(), 100 - 100 * p
                ).compute()
            else:
                self.rmin = numpy.percentile(array, 100 * p)
                self.rmax = numpy.percentile(array, 100 - 100 * p)

            lprint(f"Range for normalisation: [{self.rmin}, {self.rmax}]")
