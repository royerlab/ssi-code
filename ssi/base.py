import gc
import math
from abc import ABC, abstractmethod
import numpy
import psutil

from ssi.utils.array.nd import nd_split_slices, remove_margin_slice
from ssi.utils.log.log import lsection, lprint
from ssi.utils.normaliser.identity import IdentityNormaliser
from ssi.utils.offcore.offcore import offcore_array


class ImageTranslatorBase(ABC):
    """Image Translator base class
    """

    def __init__(
            self,
            normaliser_type='identity',
            normaliser_transform=None,
            normaliser_clip=True,
            monitor=None,
            blind_spots=None,
            tile_min_margin=8,
            tile_max_margin=None,
            padding=0,
            padding_mode=None,
            max_memory_usage_ratio=0.9,
            max_tilling_overhead=0.1,
    ):
        """
        :param normaliser_type: can have one of three values; 'identity','percentile' and 'minmax'
        :param monitor: monitor object, has to be instance of it.monitor.Monitor class
        """
        # Instantiates normaliser(s):
        if normaliser_type == 'identity':
            self.normalizer_class = IdentityNormaliser
        else:
            raise ValueError('Unknown normalizer type passed!')
        self.normaliser_transform = normaliser_transform
        self.normaliser_clip = normaliser_clip

        self.self_supervised = False
        self.monitor = monitor
        self.blind_spots = blind_spots
        self.tile_max_margin = tile_max_margin
        self.tile_min_margin = tile_min_margin
        self.padding = padding
        self.padding_mode = padding_mode

        self.max_memory_usage_ratio = max_memory_usage_ratio
        self.max_tilling_overhead = max_tilling_overhead
        self.max_voxels_per_tile = 320 ** 3

        self.callback_period = 3
        self.last_callback_time_sec = -math.inf

        self.loss_history = None

    @abstractmethod
    def _train(
            self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        """This function supposed to take normalized input image only
        :param input_image:
        :param target_image:
        :param train_valid_ratio:
        :param callback_period:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """
        Translates an input image into an output image according to the learned function
        :param input_image: input image
        """
        raise NotImplementedError()

    def _estimate_memory_needed_and_available(self, image):
        # By default there is no memory needed and infinite available memory which means no constraints
        return 0, psutil.virtual_memory().total

    def train(
            self,
            input_image,
            target_image=None,
            batch_dims=None,
            channel_dims=None,
            train_valid_ratio=0.1,
            callback_period=3,
            jinv=None,
    ):
        """Train to translate a given input image to a given output image.
        This has a lot of the machinery for batching and more...
        """

        if target_image is None:
            target_image = input_image

        with lsection(
                f"Learning to translate from image of dimensions {str(input_image.shape)} to {str(target_image.shape)} ."
        ):

            lprint('Running garbage collector...')
            gc.collect()

            # If we use the same image for input and output then we are in a self-supervised setting:
            self.self_supervised = input_image is target_image

            if self.self_supervised:
                lprint('Training is self-supervised.')
            else:
                lprint('Training is supervised.')

            if batch_dims is None:  # set default batch_dim value:
                batch_dims = (False,) * len(input_image.shape)

            self.input_normaliser = self.normalizer_class(
                transform=self.normaliser_transform,
                clip=self.normaliser_clip
            )
            self.target_normaliser = (
                self.input_normaliser
                if self.self_supervised
                else self.normalizer_class(transform=self.normaliser_transform,
                                           clip=self.normaliser_clip)
            )

            # Calibrates normaliser(s):
            self.input_normaliser.calibrate(input_image)
            if not self.self_supervised:
                self.target_normaliser.calibrate(target_image)

            # Intensity values normalisation:
            normalised_input_image = self.input_normaliser.normalise(
                input_image, batch_dims=batch_dims, channel_dims=channel_dims
            )
            normalised_target_image = (
                normalised_input_image
                if self.self_supervised
                else self.target_normaliser.normalise(
                    target_image, batch_dims=batch_dims, channel_dims=channel_dims
                )
            )

            # Let's pad the images to avoid border effects:
            # If we do it for translation we also have to do it for training because of
            # location-aware features such as large-scale features or spatial-features.
            normalised_input_image = self._pad_norm_image(normalised_input_image)
            normalised_target_image = self._pad_norm_image(normalised_target_image)

            self._train(
                normalised_input_image,
                normalised_target_image,
                train_valid_ratio=train_valid_ratio,
                callback_period=callback_period,
                jinv=jinv,
            )

    def translate(
            self,
            input_image,
            translated_image=None,
            batch_dims=None,
            channel_dims=None,
            tile_size=None,
            denormalise_values=True,
            leave_as_float=False,
            clip=True,
    ):
        """
        Translates an input image into an output image according to the learned function.
        :param input_image:
        :type input_image:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """

        with lsection(
                f"Predicting output image from input image of dimension {input_image.shape}"
        ):

            # set default batch_dim and channel_dim values:
            if batch_dims is None:
                batch_dims = (False,) * len(input_image.shape)
            if channel_dims is None:
                channel_dims = (False,) * len(input_image.shape)

            # Number of spatio-temporal dimensions:
            num_spatiotemp_dim = sum(
                0 if b or c else 1 for b, c in zip(batch_dims, channel_dims)
            )

            # First we normalise the input values:
            normalised_input_image = self.input_normaliser.normalise(
                input_image, batch_dims=batch_dims, channel_dims=channel_dims
            )

            # When we trained supervised we need to update permutated image shape of target_normaliser
            # This way we can accommodate different sizes of batch dimensions than batch dimensions used for training
            if not self.self_supervised:
                (
                    _,
                    _,
                    self.target_normaliser.permutated_image_shape,
                ) = self.target_normaliser.shape_normalize(
                    input_image, batch_dims=batch_dims, channel_dims=channel_dims
                )

            # Let's pad the input array so we avoid annoying border-effects:
            normalised_input_image = self._pad_norm_image(normalised_input_image)

            # Spatio-temporal shape:
            spatiotemp_shape = normalised_input_image.shape[-num_spatiotemp_dim:]

            normalised_translated_image = None

            if tile_size == 0:
                # we _force_ no tilling, this is _not_ the default.

                # We translate:
                normalised_translated_image = self._translate(
                    normalised_input_image,
                    whole_image_shape=normalised_input_image.shape,
                )

            else:

                # We do need to do tiled inference because of a lack of memory
                # or because a small batch size was requested:

                normalised_input_shape = normalised_input_image.shape

                # We get the tilling strategy:
                # tile_size, shape, min_margin, max_margin
                tilling_strategy, margins = self._get_tilling_strategy_and_margins(
                    normalised_input_image,
                    self.max_voxels_per_tile,
                    self.tile_min_margin,
                    self.tile_max_margin,
                    suggested_tile_size=tile_size,
                )
                lprint(f"Tilling strategy: {tilling_strategy}")
                lprint(f"Margins for tiles: {margins} .")

                # tile slice objects (with and without margins):
                tile_slices_margins = list(
                    nd_split_slices(
                        normalised_input_shape, tilling_strategy, margins=margins
                    )
                )
                tile_slices = list(
                    nd_split_slices(normalised_input_shape, tilling_strategy)
                )

                # Number of tiles:
                number_of_tiles = len(tile_slices)
                lprint(f"Number of tiles (slices): {number_of_tiles}")

                # We create slice list:
                slicezip = zip(tile_slices_margins, tile_slices)

                counter = 1
                for slice_margin_tuple, slice_tuple in slicezip:
                    with lsection(
                            f"Current tile: {counter}/{number_of_tiles}, slice: {slice_tuple} "
                    ):

                        # We first extract the tile image:
                        input_image_tile = normalised_input_image[
                            slice_margin_tuple
                        ].copy()

                        # We do the actual translation:
                        lprint(f"Translating...")
                        translated_image_tile = self._translate(
                            input_image_tile,
                            image_slice=slice_margin_tuple,
                            whole_image_shape=normalised_input_image.shape,
                        )

                        # We compute the slice needed to cut out the margins:
                        lprint(f"Removing margins...")
                        remove_margin_slice_tuple = remove_margin_slice(
                            normalised_input_shape, slice_margin_tuple, slice_tuple
                        )

                        # We allocate -just in time- the translated array if needed:
                        # if the array is already provided, it must of course have the right dimensions...
                        if normalised_translated_image is None:
                            translated_image_shape = (
                                    normalised_input_image.shape[:2] + spatiotemp_shape
                            )
                            normalised_translated_image = offcore_array(
                                shape=translated_image_shape,
                                dtype=translated_image_tile.dtype,
                                max_memory_usage_ratio=self.max_memory_usage_ratio,
                            )

                        # We plug in the batch without margins into the destination image:
                        lprint(f"Inserting translated batch into result image...")
                        normalised_translated_image[
                            slice_tuple
                        ] = translated_image_tile[remove_margin_slice_tuple]

                        counter += 1

            # Let's crop the padding:
            normalised_translated_image = self._crop_norm_image(
                normalised_translated_image
            )

            # Then we denormalise:
            denormalised_translated_image = self.target_normaliser.denormalise(
                normalised_translated_image,
                # denormalise_values=denormalise_values,
                leave_as_float=leave_as_float,
                clip=clip,
            )

            if translated_image is None:
                translated_image = denormalised_translated_image
            else:
                translated_image[...] = denormalised_translated_image

            return translated_image

    def _pad_norm_image(self, normalised_input_image):
        if self.padding > 0:
            # First we compute the amount of padding:
            num_spatiotemp_dim = normalised_input_image.ndim - 2
            padding = ((0, 0), (0, 0)) + (
                (self.padding, self.padding),
            ) * num_spatiotemp_dim
            # value = normalised_input_image.mean()
            padded_normalised_input_image = numpy.pad(
                normalised_input_image,
                pad_width=tuple(padding),
                mode='constant' if self.padding_mode is None else self.padding_mode
                # constant_values=value
            )
            return padded_normalised_input_image

        return normalised_input_image

    def _crop_norm_image(self, normalised_translated_image):
        if self.padding > 0:
            # Let's compute the slice for cropping:
            num_spatiotemp_dim = normalised_translated_image.ndim - 2
            slice_object = (slice(None), slice(None)) + (
                slice(self.padding, -self.padding),
            ) * num_spatiotemp_dim
            cropped_normalised_translated_image = normalised_translated_image[
                slice_object
            ]
            return cropped_normalised_translated_image

        return normalised_translated_image

    def _get_tilling_strategy_and_margins(
            self,
            image,
            max_voxels_per_tile,
            min_margin,
            max_margin,
            suggested_tile_size=None,
    ):

        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:
        with lsection(f"Determine tilling strategy:"):

            suggested_tile_size = (
                math.inf if suggested_tile_size is None else suggested_tile_size
            )

            # image shape:
            shape = image.shape
            num_spatio_temp_dim = num_spatiotemp_dim = len(shape) - 2

            lprint(f"image shape             = {shape}")
            lprint(f"max_voxels_per_tile     = {max_voxels_per_tile}")

            # Estimated amount of memory needed for storing all features:
            (
                estimated_memory_needed,
                total_memory_available,
            ) = self._estimate_memory_needed_and_available(image)
            lprint(f"Estimated amount of memory needed: {estimated_memory_needed}")

            # Available physical memory :
            total_memory_available *= self.max_memory_usage_ratio

            lprint(
                f"Available memory (we reserve 10% for 'comfort'): {total_memory_available}"
            )

            # How much do we need to tile because of memory, if at all?
            split_factor_mem = estimated_memory_needed / total_memory_available
            lprint(
                f"How much do we need to tile because of memory? : {split_factor_mem} times."
            )

            # how much do we have to tile because of the limit on the number of voxels per tile?
            split_factor_max_voxels = image.size / max_voxels_per_tile
            lprint(
                f"How much do we need to tile because of the limit on the number of voxels per tile? : {split_factor_max_voxels} times."
            )

            # how much do we have to tile because of the suggested tile size?
            split_factor_suggested_tile_size = image.size / (
                    suggested_tile_size ** num_spatio_temp_dim
            )
            lprint(
                f"How much do we need to tile because of the suggested tile size? : {split_factor_suggested_tile_size} times."
            )

            # we keep the max:
            desired_split_factor = max(
                split_factor_mem,
                split_factor_max_voxels,
                split_factor_suggested_tile_size,
            )
            # We cannot split less than 1 time:
            desired_split_factor = max(1, int(math.ceil(desired_split_factor)))
            lprint(f"Desired split factor: {desired_split_factor}")

            # Number of batches:
            num_batches = shape[0]

            # Does the number of batches split the data enough?
            if num_batches < desired_split_factor:
                # Not enough splitting happening along the batch dimension, we need to split further:
                # how much?
                rest_split_factor = desired_split_factor / num_batches
                lprint(
                    f"Not enough splitting happening along the batch dimension, we need to split spatio-temp dims by: {rest_split_factor}"
                )

                # let's split the dimensions in a way proportional to their lengths:
                split_per_dim = (rest_split_factor / numpy.prod(shape[2:])) ** (
                        1 / num_spatio_temp_dim
                )

                spatiotemp_tilling_strategy = tuple(
                    max(1, int(math.ceil(split_per_dim * s))) for s in shape[2:]
                )

                # correction_factor = numpy.prod(tuple(s for s in spatiotemp_tilling_strategy if s<1))

                tilling_strategy = (num_batches, 1) + spatiotemp_tilling_strategy
                lprint(f"Preliminary tilling strategy is: {tilling_strategy}")

                # We correct for eventual oversplitting by favouring splitting over the front dimensions:
                current_splitting_factor = 1
                corrected_tilling_strategy = []
                split_factor_reached = False
                for i, s in enumerate(tilling_strategy):

                    if split_factor_reached:
                        corrected_tilling_strategy.append(1)
                    else:
                        corrected_tilling_strategy.append(s)
                        current_splitting_factor *= s

                    if current_splitting_factor >= desired_split_factor:
                        split_factor_reached = True

                tilling_strategy = tuple(corrected_tilling_strategy)

            else:
                tilling_strategy = (desired_split_factor, 1) + tuple(
                    1 for s in shape[2:]
                )

            lprint(f"Tilling strategy is: {tilling_strategy}")

            # Handles defaults:
            if max_margin is None:
                max_margin = math.inf
            if min_margin is None:
                min_margin = 0

            # First we estimate the shape of a tile:

            estimated_tile_shape = tuple(
                int(round(s / ts)) for s, ts in zip(shape[2:], tilling_strategy[2:])
            )
            lprint(f"The estimated tile shape is: {estimated_tile_shape}")

            # Limit margins:
            # We automatically set the margin of the tile size:
            # the max-margin factor guarantees that tilling will incur no more than a given max tiling overhead:
            margin_factor = 0.5 * (
                    ((1 + self.max_tilling_overhead) ** (1 / num_spatiotemp_dim)) - 1
            )
            margins = tuple(int(s * margin_factor) for s in estimated_tile_shape)

            # Limit the margin to something reasonable (provided or automatically computed):
            margins = tuple(min(max_margin, m) for m in margins)
            margins = tuple(max(min_margin, m) for m in margins)

            # We add the batch and channel dimensions:
            margins = (0, 0) + margins

            # We only need margins if we split a dimension:
            margins = tuple(
                (0 if split == 1 else margin)
                for margin, split in zip(margins, tilling_strategy)
            )

            return tilling_strategy, margins
