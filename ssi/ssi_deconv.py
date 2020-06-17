import numpy
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve

from ssi.it_ptcnn import PTCNNImageTranslator
from ssi.models.psf_convolution import PSFConvolutionLayer2D, PSFConvolutionLayer3D
from ssi.utils.log.log import lprint


class SSIDeconvolution(PTCNNImageTranslator):
    """
    Pytorch-based CNN image deconvolution
    """

    def __init__(self, psf_kernel=None, broaden_psf=1, sharpening=0, bounds_loss=0.1, entropy=0, **kwargs):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(**kwargs)

        self.provided_psf_kernel = psf_kernel
        self.broaden_psf = broaden_psf
        self.sharpening = sharpening
        self.bounds_loss = bounds_loss
        self.entropy = entropy

    def _train(self, input_image, target_image, train_valid_ratio=0.1, callback_period=3, jinv=False):

        ndim = input_image.ndim-2
        num_channels = input_image.shape[1]

        self.psf_kernel = self.provided_psf_kernel

        for i in range(self.broaden_psf):

            self.psf_kernel = numpy.pad(self.psf_kernel, (1,), mode='constant', constant_values=0)

            broadening_kernel = None
            if ndim==2:
                broadening_kernel = numpy.array([[0.095, 0.14, 0.095], [0.14, 0.2, 0.14], [0.095, 0.14, 0.095]])
            elif ndim==3:
                broadening_kernel = numpy.array([[[0.095, 0.095, 0.095], [0.095, 0.14, 0.095], [0.095, 0.095, 0.095]],
                                              [[0.095,  0.14, 0.095], [0.14,   0.2, 0.14 ], [0.095,  0.14, 0.095]],
                                              [[0.095, 0.095, 0.095], [0.095, 0.14, 0.095], [0.095, 0.095, 0.095]]])

            broaden_kernel = broadening_kernel / broadening_kernel.sum()
            self.psf_kernel = convolve(
                self.psf_kernel,
                broaden_kernel,
                mode='constant',
            )

        self.psf_kernel /= self.psf_kernel.sum()
        self.psf_kernel = self.psf_kernel.astype(numpy.float32)

        self.psf_kernel = self.psf_kernel
        self.psf_kernel_tensor = torch.from_numpy(
            self.psf_kernel[numpy.newaxis, numpy.newaxis, ...]
        ).to(self.device)

        if ndim==2:
            self.psfconv = PSFConvolutionLayer2D(self.psf_kernel, num_channels=num_channels).to(self.device)
        elif ndim==3:
            self.psfconv = PSFConvolutionLayer3D(self.psf_kernel, num_channels=num_channels).to(self.device)

        super()._train(input_image, target_image, train_valid_ratio, callback_period, jinv)


    def _train_loop(self, data_loader, optimizer, loss_function):
        try:
            self.model.kernel_continuity_regularisation = False
        except AttributeError:
            lprint("Cannot deactivate kernel continuity regularisation")

        super()._train_loop(data_loader, optimizer, loss_function)

    def _additional_losses(self, translated_image, forward_model_image):

        loss = 0

        # Bounds loss:
        if self.bounds_loss and self.bounds_loss != 0:
            epsilon = 0 * 1e-8
            bounds_loss = F.relu(-translated_image - epsilon)
            bounds_loss += F.relu(translated_image - 1 - epsilon)
            bounds_loss_value = bounds_loss.mean()
            lprint(f"bounds_loss_value = {bounds_loss_value}")
            loss += self.bounds_loss * bounds_loss_value ** 2

        # Sharpen loss_deconvolution:
        if self.sharpening and self.sharpening != 0:
            image_for_loss = translated_image
            num_elements = image_for_loss[0, 0].nelement()
            sharpening_loss = -torch.norm(image_for_loss, dim=(2, 3), keepdim=True, p=2) / (num_elements ** 2)  # /torch.norm(image_for_loss, dim=(2, 3), keepdim=True, p=1)
            lprint(f"sharpening loss = {sharpening_loss}")
            loss += self.sharpening * sharpening_loss.mean()

        # Max entropy loss:
        if self.entropy and self.entropy != 0:
            entropy_value = entropy(translated_image)
            lprint(f"entropy_value = {entropy_value}")
            loss += -self.entropy * entropy_value

        return loss

    def _forward_model(self, input):
        return self.psfconv(torch.clamp(input, 0, 1))


def entropy(image, normalise=True, epsilon=1e-10, clip=True):
    if clip:
        image = torch.clamp(image, 0, 1)
    image = image / (epsilon + torch.sum(image, dim=(2, 3), keepdim=True)) if normalise else image
    entropy = -torch.where(image > 0, image * (image + epsilon).log(), image.new([0.0]))
    entropy_value = entropy.sum(dim=(2, 3), keepdim=True).mean()
    return entropy_value
