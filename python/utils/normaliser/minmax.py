import dask
import numpy

from utils.log.log import lsection, lprint
from utils.normaliser.base import NormaliserBase


class MinMaxNormaliser(NormaliserBase):
    """
        Min-Max Normaliser
    """

    def __init__(self, **kwargs):
        """
        Constructs a normaliser
        """
        super().__init__(**kwargs)

    def calibrate(self, array):
        with lsection(f"Calibrating array using minmax method"):
            self.original_dtype = array.dtype

            if hasattr(array, '__dask_keys__'):
                self.rmin = dask.array.min(array.flatten()).compute()
                self.rmax = dask.array.max(array.flatten()).compute()
            else:
                self.rmin = numpy.min(array)
                self.rmax = numpy.max(array)

            lprint(f"Range for normalisation: [{self.rmin}, {self.rmax}]")
