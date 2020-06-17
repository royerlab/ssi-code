import math
import traceback

from ssi.base import ImageTranslatorBase
from ssi.utils.log.log import lprint
from ssi.utils.offcore.offcore import offcore_array


class ImageTranslatorLRDeconv(ImageTranslatorBase):
    """
        Lucy Richardson Deconvolution

    """

    def __init__(
            self, psf_kernel, max_num_iterations=50, clip=True, backend='scipy', **kwargs
    ):
        """Constructs a Lucy Richardson deconvolution image translator.

        :param psf_kernel: 2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(**kwargs)

        self.psf_kernel_numpy = psf_kernel
        self.clip = clip
        self.backend = backend
        self.max_num_iterations = max_num_iterations
        self.__debug_allocation = False

        if self.padding_mode is None:
            self.padding_mode = 'reflect'

        self.max_voxels_per_tile = 512 ** 3

    def _estimate_memory_needed_and_available(self, image):
        # By default there is no memory needed which means no constraints

        memory_needed, memory_available = super()._estimate_memory_needed_and_available(
            image
        )
        # TODO: this is a rough estimate, it is not cler how much is really needed...
        memory_needed = 6 * image.size * image.dtype.itemsize

        if self.backend == 'cupy' or self.backend == 'scipy-cupy':
            from cupy.cuda.device import Device

            default_device = Device()
            memory_available = default_device.mem_info[0]

        return memory_needed, memory_available

    def _train(
            self, input_image, target_image, train_valid_ratio, callback_period, jinv
    ):
        pass
        # we need to figure out what to do here...

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Internal method that translates an input image on the basis of the trained model.

        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        import numpy

        convolve_method = self._get_convolution_method(
            input_image, self.psf_kernel_numpy
        )
        pad_method = self._get_pad_method(input_image)

        self.psf_kernel = self._convert_array_format_in(
            self.psf_kernel_numpy.astype(numpy.float32)
        )
        self.psf_kernel_mirror = self._convert_array_format_in(
            self.psf_kernel[::-1, ::-1]
        )

        input_image = input_image.astype(numpy.float32, copy=False)

        deconvolved_image = offcore_array(
            shape=input_image.shape, dtype=input_image.dtype
        )

        lprint(f"Number of Lucy-Richardson iterations: {self.max_num_iterations}")

        for batch_index, batch_image in enumerate(input_image):

            for channel_index, channel_image in enumerate(batch_image):

                channel_image = channel_image.clip(0, math.inf)
                channel_image = self._convert_array_format_in(channel_image)

                candidate_deconvolved_image = numpy.full(
                    channel_image.shape, float(numpy.mean(channel_image))
                )

                candidate_deconvolved_image = self._convert_array_format_in(
                    candidate_deconvolved_image
                )

                kernel_shape = self.psf_kernel.shape
                pad_width = tuple(
                    (max(self.padding, (s - 1) // 2), max(self.padding, (s - 1) // 2))
                    for s in kernel_shape
                )

                for i in range(self.max_num_iterations):

                    if self.padding > 0:
                        padded_candidate_deconvolved_image = pad_method(
                            candidate_deconvolved_image,
                            pad_width=pad_width,
                            mode=self.padding_mode,
                        )
                    else:
                        padded_candidate_deconvolved_image = candidate_deconvolved_image

                    convolved = convolve_method(
                        padded_candidate_deconvolved_image,
                        self.psf_kernel,
                        mode='valid' if self.padding else 'same',
                    )

                    convolved[convolved == 0] = 1

                    relative_blur = channel_image / convolved

                    self._debug_allocation(f"after division")

                    if self.padding:
                        relative_blur = numpy.pad(
                            relative_blur, pad_width=pad_width, mode=self.padding_mode
                        )

                    multiplicative_correction = convolve_method(
                        relative_blur,
                        self.psf_kernel_mirror,
                        mode='valid' if self.padding else 'same',
                    )

                    self._debug_allocation(f"after second convolution")

                    candidate_deconvolved_image *= multiplicative_correction

                if self.clip:
                    candidate_deconvolved_image[candidate_deconvolved_image > 1] = 1
                    candidate_deconvolved_image[candidate_deconvolved_image < -1] = -1

                candidate_deconvolved_image = self._convert_array_format_out(
                    candidate_deconvolved_image
                )

                deconvolved_image[
                    batch_index, channel_index
                ] = candidate_deconvolved_image

        return deconvolved_image

    def _convert_array_format_in(self, input_image):
        if (
                self.backend == 'scipy'
                or self.backend == 'gputools'
                or self.backend == "scipy-cupy"
        ):
            return input_image
        elif self.backend == 'cupy':
            import cupy

            return cupy.asarray(input_image)

    def _convert_array_format_out(self, output_image):
        if (
                self.backend == 'scipy'
                or self.backend == 'gputools'
                or self.backend == "scipy-cupy"
        ):
            return output_image
        elif self.backend == 'cupy':
            import cupy

            return cupy.asnumpy(output_image)

    def _get_convolution_method(self, input_image, psf_kernel):

        if self.backend == 'scipy':
            lprint("Using scipy backend.")
            from scipy.signal import convolve

            return convolve

        elif self.backend == "scipy-cupy":
            try:
                lprint("Attempting to use scipy-cupy backend.")
                import scipy
                import cupy

                scipy.fft.set_backend(cupy.fft)
                self.backend = 'scipy'
                lprint("Succeeded to use scipy-cupy backend.")
                return self._get_convolution_method(input_image, psf_kernel)
            except Exception:
                track = traceback.format_exc()
                lprint(track)
                lprint("Failed to use scipy-cupy backend.")
                self.backend = 'cupy'
                return self._get_convolution_method(input_image, psf_kernel)

        elif self.backend == 'gputools':
            try:
                lprint("Attempting to use gputools backend.")
                # testing if gputools works:
                import gputools
                import numpy

                # try something simple and see if it crashes...
                data = numpy.ones((30, 40, 50))
                h = numpy.ones((10, 11, 12))
                out = gputools.convolve(data, h)  # noqa: F841

                def gputools_convolve(in1, in2, mode=None, method=None):
                    return gputools.convolve(in1, in2)

                # gputools backend does not need extra padding:
                self.padding = False

                lprint("Succeeded to use cupy backend.")
                return gputools_convolve

            except Exception:
                track = traceback.format_exc()
                lprint(track)
                lprint("Failed to use gputools backend.")
                pass

        elif self.backend == 'cupy':
            try:
                lprint("Attempting to use cupy backend.")
                # try:
                # testing if gputools works:
                import cupyx.scipy.ndimage

                # try something simple and see if it crashes...
                import cupy

                data = cupy.ones((30, 40, 50))
                h = cupy.ones((10, 11, 12))
                cupyx.scipy.ndimage.convolve(data, h)

                # gputools backend does not need extra padding:
                self.padding = False

                def cupy_convolve(in1, in2, mode=None, method=None):
                    return cupyx.scipy.ndimage.convolve(in1, in2, mode='reflect')

                lprint("Succeeded to use cupy backend.")
                if psf_kernel.size > 500:
                    return self._cupy_convolve_fft
                else:
                    return cupy_convolve

            except Exception:
                track = traceback.format_exc()
                lprint(track)
                lprint("Failed to use cupy backend, trying gputools")
                self.backend = 'gputools'
                return self._get_convolution_method(input_image, psf_kernel)

        lprint("Faling back to scipy backend.")

        # this is scipy's convolve:
        from scipy.signal import convolve

        return convolve

    def _get_pad_method(self, input_image):
        if self.backend == 'scipy' or self.backend == 'scipy-cupy':
            import numpy

            return numpy.pad
        elif self.backend == 'cupy':
            import cupy

            return cupy.pad

    def _cupy_convolve_fft(self, image1, image2, mode=None):

        import cupy
        import numpy

        # TODO: review if this is needed
        cupy.cuda.set_allocator(None)

        self._debug_allocation(f"before FFT")

        is_planning_on = cupy.fft.config.enable_nd_planning
        cupy.fft.config.enable_nd_planning = False

        if image1.ndim == image2.ndim == 0:  # scalar inputs
            return image1 * image2
        elif not image1.ndim == image2.ndim:
            raise ValueError("Dimensions do not match.")
        elif image1.size == 0 or image2.size == 0:  # empty arrays
            return cupy.array([])

        s1 = numpy.asarray(image1.shape)
        s2 = numpy.asarray(image2.shape)

        shape = tuple(s1 + s2 - 1)

        fsize = shape  # tuple(int(2 ** math.ceil(math.log2(x))) for x in tuple(shape))

        image1_fft = cupy.fft.rfftn(image1, fsize)
        image2_fft = cupy.fft.rfftn(image2, fsize)
        ret = cupy.fft.irfftn(image1_fft * image2_fft)
        # ret = ret.astype(cupy.float32) #cupy.real(ret)

        fslice = tuple([slice(0, int(sz)) for sz in shape])
        ret = ret[fslice]

        # if mode=='same':
        newshape = cupy.asarray(image1.shape)
        currshape = cupy.array(ret.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

        ret = ret[tuple(myslice)]

        cupy.fft.config.enable_nd_planning = is_planning_on

        del image1_fft
        del image2_fft

        cupy.get_default_memory_pool().free_all_blocks()

        self._debug_allocation(f"after fft")

        return ret

    def _debug_allocation(self, info):
        if self.__debug_allocation:
            if self.backend == 'cupy':
                import cupy

                lprint(
                    f"CUDA memory usage {info}: {cupy.get_default_memory_pool().used_bytes() / 1e6} MB"
                )
