# flake8: noqa
import time
from os import listdir
from os.path import join, isfile

import numpy
from imageio import imread

from it_ptcnn_deconv import PTCNNDeconvolution
from lr_deconv import ImageTranslatorLRDeconv
from models.autoencoder import AutoEncoder
from utils.io.datasets import normalise, add_poisson_gaussian_noise, add_microscope_blur_2d
from utils.metrics.image_metrics import spectral_mutual_information, mutual_information, psnr, ssim

generic_2d_mono_raw_folder = '/home/royer/workspace/python/ssi-code/python/benchmark/images/generic_2d_all/'


def get_benchmark_image(type, name):
    folder = join(generic_2d_mono_raw_folder, type)
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    filename = [f for f in files if name in f][0]
    filepath = join(folder, filename)
    array = imread(filepath)
    return array, filename


def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def demo(image_clipped):
    image_clipped = normalise(image_clipped.astype(numpy.float32))
    blurred_image, psf_kernel = add_microscope_blur_2d(image_clipped)
    # noisy_blurred_image = add_noise(blurred_image, intensity=None, variance=0.01, sap=0.01, clip=True)
    noisy_blurred_image = add_poisson_gaussian_noise(blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10)

    lr = ImageTranslatorLRDeconv(
        psf_kernel=psf_kernel, max_num_iterations=30, backend="cupy"
    )
    lr.train(noisy_blurred_image)
    lr_deconvolved_image = lr.translate(noisy_blurred_image)

    # import napari
    #
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(blurred_image, name='blurred')
    #     viewer.add_image(noisy_blurred_image, name='noisy_blurred_image')
    #     viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
    #     viewer.add_image(psf_kernel, name='psf_kernel')

    it_deconv = PTCNNDeconvolution(
        max_epochs=3000,
        patience=100,
        batch_size=8,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
        model_class=AutoEncoder,
        masking=True,
        masking_density=0.05,
        loss='l2',
        bounds_loss=0.1,
        sharpening=0,
        entropy=0,
        broaden_psf=1
    )

    start = time.time()
    it_deconv.train(noisy_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    deconvolved_image = it_deconv.translate(noisy_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image_clipped = numpy.clip(image_clipped, 0, 1)
    lr_deconvolved_image_clipped = numpy.clip(lr_deconvolved_image, 0, 1)
    deconvolved_image_clipped = numpy.clip(deconvolved_image, 0, 1)

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
    printscore(
        "blurry image          :   ",
        psnr(image_clipped, blurred_image),
        spectral_mutual_information(image_clipped, blurred_image),
        mutual_information(image_clipped, blurred_image),
        ssim(image_clipped, blurred_image),
    )

    printscore(
        "noisy and blurry image:   ",
        psnr(image_clipped, noisy_blurred_image),
        spectral_mutual_information(image_clipped, noisy_blurred_image),
        mutual_information(image_clipped, noisy_blurred_image),
        ssim(image_clipped, noisy_blurred_image),
    )

    printscore(
        "lr deconv             :    ",
        psnr(image_clipped, lr_deconvolved_image_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_clipped),
        ssim(image_clipped, lr_deconvolved_image_clipped),
    )

    printscore(
        "ssi deconv            : ",
        psnr(image_clipped, deconvolved_image_clipped),
        spectral_mutual_information(image_clipped, deconvolved_image_clipped),
        mutual_information(image_clipped, deconvolved_image_clipped),
        ssim(image_clipped, deconvolved_image_clipped),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(noisy_blurred_image, name='noisy_blurred_image')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
        viewer.add_image(deconvolved_image, name='ssi_deconvolved_image')


# image = characters()
image, _ = get_benchmark_image('gt', 'mitochondria')
# image = newyork()
# image, name = get_generic_2d_mono_raw(1)
# print(name)
# image  = dots()
# [-512:-1, 256:256+512]

demo(image)
