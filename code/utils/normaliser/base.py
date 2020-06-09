from abc import ABC, abstractmethod

import numexpr
import numpy


class NormaliserBase(ABC):
    """
        Normaliser base class

    """

    epsilon: float
    leave_as_float: bool
    clip: bool
    original_dtype: numpy.dtype

    def __init__(
            self, clip=True, epsilon=1e-21, shape_normalisation=True, transform=None
    ):
        """
        Constructs a normaliser
        """

        self.epsilon = epsilon
        self.clip = clip
        self.shape_normalisation = shape_normalisation
        self.transform = '' if transform is None else transform

        self.rmin = None
        self.rmax = None

        self.axis_permutation = None
        self.permutated_image_shape = None

    @abstractmethod
    def calibrate(self, array):
        """
        Calibrates this normaliser given an array.

        :param array: array to use for calibration
        :type array: ndarray
        """
        raise NotImplementedError()

    def normalise(self, array, batch_dims=None, channel_dims=None):
        """
        Normalises the given array in-place (if possible).

        :param array: array to normaliser
        :type array: ndarray
        """

        if self.shape_normalisation:
            (
                array,
                self.axis_permutation,
                self.permutated_image_shape,
            ) = self.shape_normalize(
                array, batch_dims=batch_dims, channel_dims=channel_dims
            )

        if array.dtype != numpy.float32:
            array = array.astype(numpy.float32, copy=True)
        else:
            array = numpy.copy(array)

        if self.rmin is not None and self.rmax is not None:
            min_value = numpy.float32(self.rmin)
            max_value = numpy.float32(self.rmax)
            epsilon = numpy.float32(self.epsilon)

            try:
                # We perform operation in-place with numexpr if possible:
                numexpr.evaluate(
                    f"{self.transform}((array - min_value) / ( max_value - min_value + epsilon ))",
                    out=array,
                )
                if self.clip:
                    numexpr.evaluate(
                        "where(array<0,0,where(array>1,1,array))", out=array
                    )

            except ValueError:
                array -= min_value
                array /= max_value - min_value + epsilon
                if self.clip:
                    array = numpy.clip(array, 0, 1)  # , out=array
                if self.transform == 'sqrt':
                    array = numpy.sqrt(array)

        return array

    def denormalise(
            self,
            array: numpy.ndarray,
            denormalise_values=True,
            leave_as_float=False,
            clip=True,
    ):
        """
        Denormalises the given array in-place (if possible).
        :param array: array to denormalise
        :type array: ndarray
        """

        if self.shape_normalisation:
            array = self.shape_denormalize(
                array,
                axes_permutation=self.axis_permutation,
                permutated_image_shape=self.permutated_image_shape,
            )

        # we copy the array to preserve the original array:
        array = numpy.copy(array)

        if denormalise_values:
            if self.rmin is not None and self.rmax is not None:

                min_value = numpy.float32(self.rmin)
                max_value = numpy.float32(self.rmax)
                epsilon = numpy.float32(self.epsilon)

                try:
                    # We perform operation in-place with numexpr if possible:

                    if self.transform == 'sqrt':
                        array = array ** 2

                    if self.clip and clip:
                        numexpr.evaluate(
                            "where(array<0,0,where(array>1,1,array))",
                            out=array,
                            casting='unsafe',
                        )

                    numexpr.evaluate(
                        "array * (max_value - min_value + epsilon) + min_value ",
                        out=array,
                        casting='unsafe',
                    )

                except ValueError:
                    if self.transform == 'sqrt':
                        array = array ** 2
                    if self.clip and clip:
                        array = numpy.clip(array, 0, 1)  # , out=array
                    array *= max_value - min_value + epsilon
                    array += min_value

            if not leave_as_float and self.original_dtype != array.dtype:
                if numpy.issubdtype(self.original_dtype, numpy.integer):
                    # If we cast back to integer, we need to avoid overflows first!
                    type_info = numpy.iinfo(self.original_dtype)

                    if not (self.clip and clip):
                        array = array + (type_info.min - array.min())
                        array = (array * type_info.max) / array.max()

                    array = array.clip(type_info.min, type_info.max, out=array)
                array = array.astype(self.original_dtype)

        return array

    @staticmethod
    def shape_normalize(image, batch_dims=None, channel_dims=None):
        """Permutates batch dimensions to the front and collapse into
        one dimension. Resulting array has to be in the form of (B,...)
        where B is the number of batch dimensions.
        """

        if batch_dims is None:
            batch_dims = (False,) * len(image.shape)
        if channel_dims is None:
            channel_dims = (False,) * len(image.shape)

        # Singleton dimensions are automatically batch dimension, unless it is a channel dimension, trivially.
        batch_dims = tuple(
            True if s == 1 and not c else b
            for b, c, s in zip(batch_dims, channel_dims, image.shape)
        )

        # get indices for different types of dimensions and their length
        batch_indices = [index for index, value in enumerate(batch_dims) if value]
        batch_length = int(numpy.prod([image.shape[index] for index in batch_indices]))

        channel_indices = (
            [index for index, value in enumerate(channel_dims) if value]
            if channel_dims
            else []
        )
        channel_length = int(
            numpy.prod([image.shape[index] for index in channel_indices])
        )

        spacetime_indices = [
            index
            for index in range(len(image.shape))
            if index not in batch_indices + channel_indices
        ]

        # Axes permutation
        axes_permutation = batch_indices + channel_indices + spacetime_indices

        # Bring all batch dimensions to front
        permutated_image = numpy.transpose(image, axes_permutation)

        # Collapse batch dimensions into one, same for channel dimensions
        spacetime_shape = tuple([image.shape[i] for i in spacetime_indices])
        normalized_shape = (batch_length, channel_length) + spacetime_shape
        normalized_shape = tuple(
            s for i, s in enumerate(normalized_shape) if s != 1 or i <= 1
        )

        # Reshape array:
        normalized_image = permutated_image.reshape(normalized_shape)

        return (normalized_image, axes_permutation, permutated_image.shape)

    @staticmethod
    def shape_denormalize(image, axes_permutation, permutated_image_shape):
        """Denormalises the shape of an image from normalized form to the
        original image form.
        """

        spatiotemp_shape = image.shape[2:]

        # Number of spatio-temp dimensions:
        num_spatiotemp_dims = len(spatiotemp_shape)

        # If the input image has a different lengths along the spatio-temporal dimensions,
        # that's fine, we accommodate for it:
        # Note: that's only fine for spatio-temp dim, not batch or channels!
        adapted_permutated_image_shape = list(permutated_image_shape)
        adapted_permutated_image_shape[-num_spatiotemp_dims:] = spatiotemp_shape
        adapted_permutated_image_shape = tuple(adapted_permutated_image_shape)

        # Reshape the input to its permutated shape:
        permutated_image = image.reshape(adapted_permutated_image_shape)

        # Retrieves dimensions back:
        return numpy.transpose(permutated_image, axes=numpy.argsort(axes_permutation))
