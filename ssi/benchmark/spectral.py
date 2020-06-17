import numpy
import scipy
import scipy.signal


def fft_mirror(image):
    flipx = numpy.flip(image, axis=1)
    flipy = numpy.flip(image, axis=0)
    flipxy = numpy.flip(flipx, axis=0)

    image_mirrored = numpy.block([[flipxy, flipy, flipxy], [flipx, image, flipx], [flipxy, flipy, flipxy]])

    h0 = scipy.signal.tukey(image_mirrored.shape[0], alpha=0.3)
    h1 = scipy.signal.tukey(image_mirrored.shape[1], alpha=0.3)
    window = numpy.sqrt(numpy.outer(h0, h1))

    image_mirrored_apodised = window * image_mirrored

    return image_mirrored_apodised


def spectrum(image, log=True):
    image_mirror = fft_mirror(image)
    dft = numpy.fft.fftshift(numpy.fft.fft2(image_mirror))
    dft_mod = numpy.absolute(dft)
    dft_mod = dft_mod * dft_mod
    if log:
        dft_mod_log = numpy.log1p(dft_mod)
        return dft_mod_log, image_mirror
    else:
        return dft_mod, image_mirror


def radial_profile(data, center):
    y, x = numpy.indices((data.shape))
    r = numpy.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = numpy.rint(r)
    r = r.astype(numpy.int)

    tbin = numpy.bincount(r.ravel(), data.ravel())
    nr = numpy.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def smooth(x, window_len=11, window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode=mode)
    y = y[(window_len // 2):-(window_len // 2)]
    return y


def resolution_estimate(image, dc_cutoff=0.05, otf_cutoff=0.5, smooth_window_length=5):
    spectrum_image, _ = spectrum(image)
    length = spectrum_image.shape[0]
    spectrum_image_profile = radial_profile(spectrum_image, (length / 2, length / 2))

    dc_cut_off_index = int(dc_cutoff * length / 2)
    otf_cut_off_index = int(otf_cutoff * length / 2)
    noise_floor = 0.5 * numpy.median(spectrum_image[0:(length - otf_cut_off_index), :])
    noise_floor += 0.5 * numpy.median(spectrum_image[:, 0:(length - otf_cut_off_index)])

    corrected_profile = smooth(spectrum_image_profile, smooth_window_length)
    corrected_profile = numpy.maximum(corrected_profile - noise_floor, 0)

    noise_robust_std = numpy.max(numpy.abs(corrected_profile[otf_cut_off_index:]), axis=0)

    noise_cut_off_index = otf_cut_off_index
    for i in range(otf_cut_off_index, 0, -1):
        value = corrected_profile[i]
        if value > 2 * noise_robust_std:
            noise_cut_off_index = i
            break

    X = numpy.linspace(dc_cut_off_index, noise_cut_off_index, 1 + noise_cut_off_index - dc_cut_off_index)[..., numpy.newaxis]
    y = corrected_profile[dc_cut_off_index:noise_cut_off_index + 1]
    from sklearn.linear_model import TheilSenRegressor
    reg = TheilSenRegressor(random_state=0).fit(X, y)
    # print(reg.coef_)
    # print(reg.intercept_)
    # print(reg.score(X, y))

    resolution_limit = float(-reg.intercept_ / reg.coef_)

    return spectrum_image, spectrum_image_profile, corrected_profile, resolution_limit,

# def resolution_estimate_old(image, cutoff=0.5, smooth_window_length=17, threshold=0.5):
#     spectrum_image, _ = spectrum(image, log=False)
#     shape = spectrum_image.shape
#
#     cut_off_index = int(cutoff*shape[0] / 2)
#     noise_floor = 0.5*numpy.mean(spectrum_image[0:cut_off_index,:])
#     noise_floor += 0.5 * numpy.mean(spectrum_image[:, 0:cut_off_index])
#     noise_floor = numpy.log1p(noise_floor)
#
#     spectrum_image_profile = numpy.log1p(radial_profile(spectrum_image, (shape[0] / 2, shape[1] / 2)))
#
#     smoothed_profile = smooth(spectrum_image_profile, smooth_window_length)
#
#     signal_support_function = numpy.maximum(smoothed_profile - noise_floor, 0)
#
#     noise_robust_std = numpy.max(numpy.abs(signal_support_function[cut_off_index:]), axis=0)
#
#     resolution_limit = 0
#     for i in range(0, signal_support_function.shape[0], 1):
#         value = signal_support_function[i]
#         if value <= threshold*noise_robust_std:
#             resolution_limit = i
#             break
#
#     return numpy.log1p(spectrum_image), spectrum_image_profile, signal_support_function, threshold, resolution_limit,
