import math

import numpy
from numpy.linalg import norm
from scipy.fft import dct
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def ssim(image_a, image_b):
    return structural_similarity(image_a, image_b, multichannel=image_a.ndim == 3)


def psnr(image_true, image_test):
    return peak_signal_noise_ratio(image_true, image_test)


def spectral_psnr(image_true, image_test):
    norm_true_image = image_true / norm(image_true.flatten(), 2)
    norm_test_image = image_test / norm(image_test.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_true_image, axis=0), axis=1)
    dct_norm_test_image = dct(dct(norm_test_image, axis=0), axis=1)

    norm_dct_norm_true_image = dct_norm_true_image / norm(
        dct_norm_true_image.flatten(), 2
    )
    norm_dct_norm_test_image = dct_norm_test_image / norm(
        dct_norm_test_image.flatten(), 2
    )

    norm_true_image = math.log1p(abs(norm_dct_norm_true_image))
    norm_test_image = math.log1p(abs(norm_dct_norm_test_image))

    psnr = peak_signal_noise_ratio(norm_true_image, norm_test_image)
    return psnr


def spectral_mutual_information(image_a, image_b, normalised=True):
    norm_image_a = image_a / norm(image_a.flatten(), 2)
    norm_image_b = image_b / norm(image_b.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_image_a, axis=0), axis=1)
    dct_norm_test_image = dct(dct(norm_image_b, axis=0), axis=1)

    return mutual_information(
        dct_norm_true_image, dct_norm_test_image, normalised=normalised
    )


def mutual_information(image_a, image_b, bins=256, normalised=True):
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    mi = mutual_info_from_contingency(c_xy)
    mi = mi / joint_entropy_from_contingency(c_xy) if normalised else mi
    return mi


def joint_entropy_from_contingency(contingency):
    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # normalised contingency, i.e. probability:
    p = nz_val / contingency_sum

    # log contingency:
    log_p = numpy.log2(p)

    # Joint entropy:
    joint_entropy = -p * log_p

    return joint_entropy.sum()


def mutual_info_from_contingency(contingency):
    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # marginals:
    pi = numpy.ravel(contingency.sum(axis=1))
    pj = numpy.ravel(contingency.sum(axis=0))

    #
    log_contingency_nm = numpy.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(numpy.int64, copy=False) * pj.take(nzy).astype(
        numpy.int64, copy=False
    )
    log_outer = -numpy.log2(outer) + numpy.log2(pi.sum()) + numpy.log2(pj.sum())
    mi = (
            contingency_nm * (log_contingency_nm - numpy.log2(contingency_sum))
            + contingency_nm * log_outer
    )
    return mi.sum()
