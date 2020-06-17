import os
import zipfile
from enum import Enum
from os.path import join, exists
import gdown
import numpy
import skimage
from imageio import imread
from numpy.random.mtrand import normal, uniform
from scipy.ndimage import binary_dilation
from scipy.signal import convolve
from scipy.signal import convolve2d
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from ssi.utils.io.folders import get_cache_folder
from ssi.utils.log.log import lprint
from ssi.utils.psf.simple_microscope_psf import SimpleMicroscopePSF


datasets_folder = join(get_cache_folder(), 'data')

try:
    os.makedirs(datasets_folder)
except Exception:
    pass


# Convenience methods to add noise and blur:

def normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def add_poisson_gaussian_noise(image, alpha=5, sigma=0.01, sap=0.0, quant_bits=8, dtype=numpy.float32, clip=True, fix_seed=True
                               ):
    if fix_seed:
        numpy.random.seed(0)
    rnd = normal(size=image.shape)
    rnd_bool = uniform(size=image.shape) < sap

    noisy = image + numpy.sqrt(alpha * image + sigma ** 2) * rnd
    noisy = noisy * (1 - rnd_bool) + rnd_bool * uniform(size=image.shape)
    noisy = numpy.around((2 ** quant_bits) * noisy) / 2 ** quant_bits
    noisy = numpy.clip(noisy, 0, 1) if clip else noisy
    noisy = noisy.astype(dtype)
    return noisy


def add_noise(
        image, intensity=5, variance=0.01, sap=0.0, dtype=numpy.float32, clip=True
):
    numpy.random.seed(0)
    noisy = image
    if intensity is not None:
        noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance, seed=0, clip=clip)
    noisy = random_noise(noisy, mode="s&p", amount=sap, seed=0, clip=clip)
    noisy = noisy.astype(dtype)
    return noisy


def add_blur_2d(image, k=17, sigma=5, multi_channel=False):
    from numpy import pi, exp, sqrt
    #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    probs = [exp(-z * z / (2 * sigma * sigma)) / sqrt(2 * pi * sigma * sigma) for z in range(-k, k + 1)]
    psf_kernel = numpy.outer(probs, probs)

    def conv(_image):
        return convolve2d(_image, psf_kernel, mode='same').astype(numpy.float32)

    if multi_channel:
        image = numpy.moveaxis(image.copy(), -1, 0)
        return numpy.moveaxis(numpy.stack([conv(channel) for channel in image]), 0, -1), psf_kernel
    else:
        return conv(image), psf_kernel


def add_microscope_blur_2d(image, dz=0, multi_channel=False):
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    psf_kernel = psf_xyz_array[dz]
    psf_kernel /= psf_kernel.sum()

    def conv(_image):
        return convolve2d(_image, psf_kernel, mode='same').astype(numpy.float32)

    if multi_channel:
        image = numpy.moveaxis(image.copy(), -1, 0)
        return numpy.moveaxis(numpy.stack([conv(channel) for channel in image]), 0, -1), psf_kernel
    else:
        return conv(image), psf_kernel


def add_microscope_blur_3d(image):
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    psf_kernel = psf_xyz_array
    psf_kernel /= psf_kernel.sum()
    return convolve(image, psf_kernel, mode='same'), psf_kernel


# Example datasets


def lizard():
    return examples_single.generic_lizard.get_array()


def camera():
    return skimage.data.camera().astype(numpy.float32)


def newyork():
    return examples_single.generic_newyork.get_array()


def pollen():
    return examples_single.generic_pollen.get_array()


def scafoldings():
    return examples_single.generic_scafoldings.get_array()


def characters():
    return 1 - examples_single.generic_characters.get_array()


def andromeda():
    return examples_single.generic_andromeda.get_array()


def fibsem(full=False):
    array = examples_single.scheffer_fibsem.get_array()
    if not full:
        array = array[0:1024, 0:1024]
    return array


def dots():
    image = numpy.random.rand(512, 512) < 0.005  # andromeda()#[256:-256, 256:-256]
    image = 0.8 * binary_dilation(image).astype(numpy.float32)
    image[0:256, 0:256] += 0.1
    image.clip(0, 1)
    return image


class examples_single(Enum):
    def get_path(self):
        download_from_gdrive(*self.value, datasets_folder)
        return join(datasets_folder, self.value[1])

    def get_array(self):
        array = imread(self.get_path())
        return array

    # XY natural images (2D monochrome):
    generic_crowd = ('13UHK8MjhBviv31mAW2isdG4G-aGaNJIj', 'crowd.tif')
    generic_mandrill = ('1B33ELiFuCV0OJ6IHh7Ix9lvImwI_QkR-', 'mandrill.tif')
    generic_newyork = ('15Nuu_NU3iNuoPRmpFbrGIY0VT0iCmuKu', 'newyork.png')
    generic_lizard = ('1GUc6jy5QH5DaiUskCrPrf64YBOLzT6j1', 'lizard.png')
    generic_pollen = ('1S0o2NWtD1shB5DfGRIqOFxTLOi8cHQD-', 'pollen.png')
    generic_scafoldings = ('1ZiWhHnkuaQH-BS8B71y00wkN1Ylo38nY', 'scafoldings.png')
    generic_andromeda = ('1Zl3DtkwUlZSbvpxGILexiIoLW1JOdJh8', 'andromeda.png')

    # Characters (2D monochrome, inverted):
    generic_characters = ('1ZWkHFI2iddKa9qv6tft4QZlCoDS5fLMK', 'characters.jpg')


def download_from_gdrive(
        id, name, dest_folder=datasets_folder, overwrite=False, unzip=False
):
    try:
        os.makedirs(dest_folder)
    except Exception:
        pass

    url = f'https://drive.google.com/uc?id={id}'
    output_path = join(dest_folder, name)
    if overwrite or not exists(output_path):
        lprint(f"Downloading file {output_path} as it does not exist yet.")
        gdown.download(url, output_path, quiet=False)

        if unzip:
            lprint(f"Unzipping file {output_path}...")
            zip_ref = zipfile.ZipFile(output_path, 'r')
            zip_ref.extractall(dest_folder)
            zip_ref.close()
            # os.remove(output_path)

        return output_path
    else:
        lprint(f"Not downloading file {output_path} as it already exists.")
        return None


def downloaded_example(substring):
    for example in examples_single.get_list():
        if substring in example.value[1]:
            print(download_from_gdrive(*example.value))
