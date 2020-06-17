from ssi.utils.normaliser.base import NormaliserBase


class IdentityNormaliser(NormaliserBase):
    """
    Identity Normaliser
    """

    def __init__(self, **kwargs):
        """
        Constructs a normaliser
        """
        super().__init__(**kwargs)

    def calibrate(self, array):
        self.original_dtype = array.dtype
