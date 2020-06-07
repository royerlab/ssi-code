import numpy
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d

from it_ptcnn import PTCNNImageTranslator
from models.psf_convolution import PSFConvolutionLayer
from utils.log.log import lprint


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class PTCNNDeconvolution(PTCNNImageTranslator):
    """
        Pytorch-based CNN image deconvolution
    """

    def __init__(self, psf_kernel=None, broaden_psf=2, sharpening=0.1, bounds_loss=0.01, entropy=0.01, num_channels=1, **kwargs):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(**kwargs)

        for i in range(broaden_psf):
            psf_kernel = numpy.pad(psf_kernel, (1,), mode='constant', constant_values=0)
            broaden_kernel = numpy.array([[0.095, 0.14, 0.095], [0.11, 0.179, 0.11], [0.095, 0.14, 0.095]])
            broaden_kernel = broaden_kernel / broaden_kernel.sum()
            psf_kernel = convolve2d(
                psf_kernel,
                broaden_kernel,
                'same',
            )

        psf_kernel /= psf_kernel.sum()
        psf_kernel = psf_kernel.astype(numpy.float32)

        self.psf_kernel = psf_kernel
        self.psf_kernel_tensor = torch.from_numpy(
            self.psf_kernel[numpy.newaxis, numpy.newaxis, ...]
        ).to(self.device)

        self.psfconv = PSFConvolutionLayer(self.psf_kernel, num_channels=num_channels).to(self.device)

        self.enforce_blind_spot = False

        self.sharpening = sharpening
        self.bounds_loss = bounds_loss
        self.entropy = entropy

    def _train_loop(self, data_loader, optimizer, loss_function):
        try:
            self.model.kernel_continuity_regularisation = False
        except AttributeError:
            lprint("Cannot deactivate kernel continuity regularisation")

        super()._train_loop(data_loader, optimizer, loss_function)

    def _additional_losses(self, translated_image, forward_model_image):

        loss = 0

        # non-negativity loss:
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

    # def _translate(self, input_image,  image_slice=None, whole_image_shape=None):
    #     return super()._translate(input_image, image_slice=image_slice, whole_image_shape=whole_image_shape).clip(0, 1)


def entropy(image, normalise=True, epsilon=1e-10, clip=True):
    if clip:
        image = torch.clamp(image, 0, 1)
    image = image / (epsilon + torch.sum(image, dim=(2, 3), keepdim=True)) if normalise else image
    entropy = -torch.where(image > 0, image * (image + epsilon).log(), image.new([0.0]))
    entropy_value = entropy.sum(dim=(2, 3), keepdim=True).mean()
    return entropy_value
