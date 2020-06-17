import time
from os.path import exists
import numpy
import torch
from tifffile import imread, imwrite

from ssi.ssi_deconv import SSIDeconvolution
from ssi.models.unet import UNet
from ssi.utils.io.datasets import add_microscope_blur_2d, add_poisson_gaussian_noise


def demo(image):
    image = image[0:512]

    _, psf_kernel = add_microscope_blur_2d(image[0])

    def degrade(image):
        blurred_image = add_microscope_blur_2d(image)[0]
        noisy_blurred_image = add_poisson_gaussian_noise(blurred_image, alpha=0.001, sigma=0.1, sap=0.0001, quant_bits=10, fix_seed=False)
        return noisy_blurred_image

    degraded_stack_filepath = "/media/royer/data1/aydin_datasets/__benchmark_datasets/_DIV2K_train_HR/div2k_degraded.tiff"
    if not exists(degraded_stack_filepath):
        noisy_blurred_image = numpy.stack([degrade(plane) for plane in image])
        imwrite(degraded_stack_filepath, noisy_blurred_image)
    else:
        noisy_blurred_image = imread(degraded_stack_filepath)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy_blurred_image, name='noisy_blurred_image')
        viewer.add_image(psf_kernel, name='psf_kernel')

    it_deconv = SSIDeconvolution(
        max_epochs=2000,
        patience=64,
        batch_size=1,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
        model_class=UNet,
        masking=True,
        masking_density=0.05,
        loss='l1'
    )

    batch_dim = (True, False, False)

    start = time.time()
    it_deconv.train(noisy_blurred_image, batch_dims=batch_dim)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    deconvolved_image = it_deconv.translate(noisy_blurred_image, batch_dims=batch_dim)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image = numpy.clip(image, 0, 1)
    deconvolved_image = numpy.clip(deconvolved_image, 0, 1)

    torch.save(it_deconv.model.state_dict(), "/media/royer/data1/aydin_datasets/__benchmark_datasets/_DIV2K_train_HR/div2k.unet.ptm")

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy_blurred_image, name='noisy_blurred_image')
        viewer.add_image(deconvolved_image, name='ssi_deconvolved_image')


image = imread("/media/royer/data1/aydin_datasets/__benchmark_datasets/_DIV2K_train_HR/div2k.tiff")

demo(image)
