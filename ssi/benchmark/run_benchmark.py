import csv
import os
import time
from os.path import join, isfile, exists
import matplotlib
import numpy
from PIL.Image import fromarray
from imageio import imread
from skimage.transform import resize

from ssi.benchmark.spectral import spectrum
from ssi.ssi_deconv import SSIDeconvolution
from ssi.lr_deconv import ImageTranslatorLRDeconv
from ssi.models.unet import UNet
from ssi.tv_restoration.chambole_pock import cp_restoration
from ssi.tv_restoration.conjugate_gradient import cg_restoration
from ssi.utils.io.datasets import normalise, add_microscope_blur_2d, add_poisson_gaussian_noise
from ssi.utils.metrics.image_metrics import psnr, ssim, mutual_information, spectral_mutual_information


def save_png(filepath, image):
    image = image.copy()
    image *= 255
    image = image.astype(numpy.uint8)
    img = fromarray(image)
    img.save(filepath)


def restore_ssi(image, psf_kernel, masking=True):
    it_deconv = SSIDeconvolution(
        max_epochs=3000,
        patience=300,
        batch_size=8,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
        model_class=UNet,
        masking=True,
        masking_density=0.05,
        loss='l2',
        bounds_loss=0.1,
        sharpening=0,
        entropy=0,
        broaden_psf=1,
        num_channels=3 if image.ndim == 3 else 1
    )

    channel_dims = (False, False, True) if image.ndim == 3 else None

    start = time.time()
    it_deconv.train(image, channel_dims=channel_dims, jinv=masking)
    stop = time.time()
    train_time = stop - start
    print(f"Training: elapsed time:  {train_time} ")

    start = time.time()
    restored_image = it_deconv.translate(image, channel_dims=channel_dims)
    stop = time.time()
    inf_time = stop - start
    print(f"inference: elapsed time:  {inf_time} ")

    restored_image = numpy.clip(restored_image, 0, 1)

    return restored_image, train_time, inf_time


def restore_dl(image, psf_kernel):
    return restore_ssi(image, psf_kernel, masking=False)


def restore_lr_low(image, psf_kernel):
    return restore_lr_(image, psf_kernel, 5)


def restore_lr_mid(image, psf_kernel):
    return restore_lr_(image, psf_kernel, 10)


def restore_lr_high(image, psf_kernel):
    return restore_lr_(image, psf_kernel, 20)


def restore_lr_(image, psf_kernel, num_iterations):
    lr = ImageTranslatorLRDeconv(
        psf_kernel=psf_kernel, max_num_iterations=num_iterations, backend="cupy"
    )

    channel_dims = (False, False, True) if image.ndim == 3 else None

    lr.train(image, channel_dims=channel_dims)

    start = time.time()
    restored_image = lr.translate(image, channel_dims=channel_dims)
    stop = time.time()

    inference_time = stop - start
    print(f"Inference: elapsed time:  {inference_time} ")

    return restored_image, 0, inference_time


def restore_tv_cg(image, psf_kernel):
    start = time.time()
    restored_image = cg_restoration(image,
                                    kernel=psf_kernel,
                                    num_iterations=100,
                                    lmbda=4.5e-3,
                                    mu=1e-10)
    stop = time.time()

    inference_time = stop - start
    print(f"Inference: elapsed time:  {inference_time} ")

    restored_image = restored_image.clip(0, 1)
    restored_image = restored_image.astype(numpy.float32)

    return restored_image, 0, inference_time


def restore_tv_cp(image, psf_kernel):
    start = time.time()
    restored_image = cp_restoration(image,
                                    kernel=psf_kernel,
                                    num_iterations=200,
                                    beta=2.5e-3)
    stop = time.time()

    inference_time = stop - start
    print(f"Inference: elapsed time:  {inference_time} ")

    restored_image = restored_image.clip(0, 1)
    restored_image = restored_image.astype(numpy.float32)

    return restored_image, 0, inference_time


def benchmark_on_image(run_name, folder, image_name, image, methods):
    def printscore(header, val1, val2, val3, val4):
        print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")

    image = normalise(image.astype(numpy.float32))

    gt_numpy_filepath = join(join(folder, 'gt_numpy'), f'{image_name}' + '.npy')
    numpy.save(gt_numpy_filepath, image)

    blurred_image, psf_kernel = add_microscope_blur_2d(image, multi_channel=image.ndim == 3)

    noisy_blurred_image = add_poisson_gaussian_noise(blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10)

    blurrynoisy_numpy_filepath = join(join(folder, 'blurrynoisy_numpy'), f'{image_name}' + '.npy')
    numpy.save(blurrynoisy_numpy_filepath, noisy_blurred_image)

    blurry_filepath = join(join(folder, 'blurry'), image_name)
    save_png(blurry_filepath, blurred_image)

    blurrynoisy_filepath = join(join(folder, 'blurrynoisy'), image_name)
    save_png(blurrynoisy_filepath, noisy_blurred_image)

    method_names = [method.__name__ for method in methods]

    # We restore the images with all methods:

    restored_image_list = []

    with open(join(folder, f"timming_{run_name}.tsv"), "a") as timming_file:

        for restore in methods:

            restored_cached_filepath = join(join(folder, 'restored_cache_numpy'), f'{run_name}_{restore.__name__}_' + image_name + '.npy')

            if exists(restored_cached_filepath):
                print(f"File: {restored_cached_filepath} does exists: skipping restoration.")
                restored_image = numpy.load(restored_cached_filepath)
            else:
                print(f"File: {restored_cached_filepath} does not exists, restoration started.")
                restored_image, train_time, inf_time = restore(noisy_blurred_image, psf_kernel)
                numpy.save(restored_cached_filepath, restored_image)
                timming_file.write(f"{image_name}\t{restore.__name__}\t{train_time}\t{inf_time}\n")

            restored_image_list.append(restored_image)

            restored_filepath = join(join(folder, 'restored'), f'{run_name}_{restore.__name__}_' + image_name)
            save_png(restored_filepath, restored_image)

    # We compute scores:
    with open(join(folder, f"scores_{run_name}.tsv"), "a") as scores_file:

        blurred_psnr_value = psnr(image, blurred_image)
        blurred_ssim_value = ssim(image, blurred_image)
        blurred_mi_value = mutual_information(image, blurred_image)
        blurred_smi_value = spectral_mutual_information(image, blurred_image)

        noisy_blurred_psnr_value = psnr(image, noisy_blurred_image)
        noisy_blurred_ssim_value = ssim(image, noisy_blurred_image)
        noisy_blurred_mi_value = mutual_information(image, noisy_blurred_image)
        noisy_blurred_smi_value = spectral_mutual_information(image, noisy_blurred_image)

        scores_file.write(f"{image_name}\tblurry\t{blurred_psnr_value}\t{blurred_ssim_value}\t{blurred_mi_value}\t{blurred_smi_value}\n")
        scores_file.write(f"{image_name}\tnoisy&blurred\t{noisy_blurred_psnr_value}\t{noisy_blurred_ssim_value}\t{noisy_blurred_mi_value}\t{noisy_blurred_smi_value}\n")

        print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
        printscore(
            "blurry image                       \t\t: ",
            blurred_psnr_value,
            blurred_ssim_value,
            blurred_mi_value,
            blurred_smi_value,
        )

        printscore(
            "noisy and blurry image             \t\t: ",
            noisy_blurred_psnr_value,
            noisy_blurred_ssim_value,
            noisy_blurred_mi_value,
            noisy_blurred_smi_value,
        )

        for restore in methods:
            restored_filepath = join(join(folder, 'restored_cache_numpy'), f'{run_name}_{restore.__name__}_' + image_name + '.npy')
            restored_image = numpy.load(restored_filepath)

            psnr_value = psnr(image, restored_image)
            ssim_value = ssim(image, restored_image)
            mi_value = mutual_information(image, restored_image)
            smi_value = spectral_mutual_information(image, restored_image)

            printscore(
                f"restored with {restore.__name__}  \t\t: ",
                psnr_value, ssim_value, mi_value, smi_value
            )

            scores_file.write(f"{image_name}\t{restore.__name__}\t{psnr_value}\t{ssim_value}\t{mi_value}\t{smi_value}\n")


def compute_averages(run_name, folder, methods):
    method_names = list([method.__name__ for method in methods])

    desc = \
        {'blurry': 'blurry',
         'noisy&blurred': 'blurry\&noisy (input)',
         'restore_tv_cg': 'Conjugate Gradient TV',
         'restore_tv_cp': 'Chambole Pock TV ',
         'restore_lr_low': 'Lucy Richardson $n=5$',
         'restore_lr_mid': 'Lucy Richardson $n=10$',
         'restore_lr_high': 'Lucy Richardson $n=20$',
         'restore_dl': 'SSI UNet \emph{no masking}',
         'restore_ssi': 'SSI UNet',
         }

    # We now compute average timings:
    method2train = {}
    method2inf = {}
    timming_tsv_filepath = join(folder, f"timming_{run_name}.tsv")
    if exists(timming_tsv_filepath):
        timming_tsv_file = open(timming_tsv_filepath)
        timming_tsv = csv.reader(timming_tsv_file, delimiter="\t")

        for row in timming_tsv:
            print(row)
            method = row[1]
            train_time = row[2]
            inf_time = row[3]

            if method not in method2train:
                method2train[method] = []

            if method not in method2inf:
                method2inf[method] = []

            method2train[method].append(float(train_time))
            method2inf[method].append(float(inf_time))

    with open(join(folder, f"timming_summary_{run_name}.csv"), "w") as timming_file:
        timming_file.write(f"method, training time, inference time\n")
        for method_name in method_names:
            average_train_time = numpy.mean(method2train[method_name])
            average_inf_time = numpy.mean(method2inf[method_name])
            timming_file.write(f"{desc[method_name]}, {average_train_time:.2f}, {average_inf_time:.2f}\n")

    # We now compute average scores:

    method_names.insert(0, 'noisy&blurred')
    method_names.insert(0, 'blurry')

    method2psnr = {}
    method2ssim = {}
    method2mi = {}
    method2smi = {}
    scores_tsv_filepath = join(folder, f"scores_{run_name}.tsv")
    if exists(scores_tsv_filepath):
        scores_tsv_file = open(scores_tsv_filepath)
        scores_tsv = csv.reader(scores_tsv_file, delimiter="\t")

        # psnr_value}\t{ssim_value}\t{mi_value}\t{smi_value}

        for row in scores_tsv:
            print(row)
            method = row[1]
            psnr_value = row[2]
            ssim_value = row[3]
            mi_value = row[4]
            smi_value = row[5]

            if method not in method2psnr:
                method2psnr[method] = []
                method2ssim[method] = []
                method2mi[method] = []
                method2smi[method] = []

            method2psnr[method].append(float(psnr_value))
            method2ssim[method].append(float(ssim_value))
            method2mi[method].append(float(mi_value))
            method2smi[method].append(float(smi_value))

    with open(join(folder, f"scores_summary_{run_name}.csv"), "w") as scores_file:
        scores_file.write(f"method, PSNR, SSIM, MI, SMI\n")

        average_psnrs = {}
        average_ssims = {}
        average_mis = {}
        average_smis = {}

        max_psnr = -1
        max_ssim = -1
        max_mi = -1
        max_smi = -1

        for method_name in method_names:

            average_psnr = numpy.mean(method2psnr[method_name])
            average_ssim = numpy.mean(method2ssim[method_name])
            average_mi = numpy.mean(method2mi[method_name])
            average_smi = numpy.mean(method2smi[method_name])

            if method_name != 'blurry' and method_name != 'noisy&blurred':
                max_psnr = max(max_psnr, average_psnr)
                max_ssim = max(max_ssim, average_ssim)
                max_mi = max(max_mi, average_mi)
                max_smi = max(max_smi, average_smi)

            average_psnrs[method_name] = average_psnr
            average_ssims[method_name] = average_ssim
            average_mis[method_name] = average_mi
            average_smis[method_name] = average_smi

        for method_name in method_names:
            average_psnr = average_psnrs[method_name]
            average_ssim = average_ssims[method_name]
            average_mi = average_mis[method_name]
            average_smi = average_smis[method_name]

            psnr = f'{average_psnr:.1f}'
            ssim = f'{average_ssim:.2f}'
            mi = f'{average_mi:.2f}'
            smi = f'{average_smi:.2f}'

            psnr = '\\textbf{' + psnr + '}' if average_psnr == max_psnr else psnr
            ssim = '\\textbf{' + ssim + '}' if average_ssim == max_ssim else ssim
            mi = '\\textbf{' + mi + '}' if average_mi == max_mi else mi
            smi = '\\textbf{' + smi + '}' if average_smi == max_smi else smi

            scores_file.write(f"{desc[method_name]}, {psnr}, {ssim}, {mi}, {smi}\n")


def compute_spectra(folder, source_sub_folder, target_sub_folder, add_prefix=''):
    images_folder = join(folder, source_sub_folder)
    files = [f for f in os.listdir(images_folder) if isfile(join(images_folder, f))]

    def compute_spectra_one_image(folder, source_sub_folder, image_name, target_sub_folder):
        print(f"Begin processsing file {image_name}")
        images_folder = join(folder, source_sub_folder)
        filepath = join(images_folder, image_name)
        image = numpy.load(filepath)

        spectrum_image_name_numpy = 'spectrum_' + image_name
        spectrum_image_path_numpy = join(join(folder, target_sub_folder + '_numpy'), spectrum_image_name_numpy)

        if not exists(spectrum_image_path_numpy):
            spectrum_image = spectrum(image)[0]
            numpy.save(spectrum_image_path_numpy, spectrum_image)
        else:
            spectrum_image = numpy.load(spectrum_image_path_numpy)

        spectrum_image_name = 'spectrum_' + add_prefix + image_name.replace('.npy', '')
        spectrum_image_path = join(join(folder, target_sub_folder), spectrum_image_name)

        spectrum_image_resize = resize(spectrum_image, (512, 512), anti_aliasing=True)

        matplotlib.image.imsave(spectrum_image_path, spectrum_image_resize, cmap='magma', vmin=0, vmax=20)
        print(f"End processsing file {image_name}")

    # Parallel(n_jobs=12) (delayed(compute_spectra_one_image)(folder, source_sub_folder, image_name, target_sub_folder) for image_name in files)
    # , prefer="threads"
    for image_name in files:
        compute_spectra_one_image(folder, source_sub_folder, image_name, target_sub_folder)


def run_benchmark_on_folder(run_name, folder, methods=None):
    gt_folder = join(folder, 'gt')
    files = [f for f in os.listdir(gt_folder) if isfile(join(gt_folder, f)) and '.png' in f]

    for image_name in files:
        print(f"Reading gt image: {image_name}")
        gt_filepath = join(gt_folder, image_name)
        image = imread(gt_filepath)
        benchmark_on_image(run_name, folder, image_name, image, methods=methods)


dirname = os.path.dirname(__file__)
image_folder = os.path.join(dirname, 'images')

generic_2d_mono_folder = join(image_folder, 'generic_2d_mono')
generic_2d_rgb_folder = join(image_folder, 'generic_2d_rgb')
generic_2d_all_folder = join(image_folder, 'generic_2d_all')

# run_benchmark_on_folder('best', generic_2d_all_folder, methods=[restore_tv_cg])

run_benchmark_on_folder('best', generic_2d_all_folder, methods=[restore_lr_low, restore_lr_mid, restore_lr_high])
run_benchmark_on_folder('best', generic_2d_all_folder, methods=[restore_ssi])
run_benchmark_on_folder('best', generic_2d_all_folder, methods=[restore_dl])
run_benchmark_on_folder('best', generic_2d_all_folder, methods=[restore_tv_cp, restore_tv_cg])  # restore_tv_cp,
#
compute_averages('best', generic_2d_all_folder, methods=[restore_tv_cg,
                                                         restore_tv_cp,
                                                         restore_lr_low,
                                                         restore_lr_mid,
                                                         restore_lr_high,
                                                         restore_dl,
                                                         restore_ssi])  # restore_tv_cg

compute_spectra(generic_2d_all_folder, 'gt_numpy', 'gt_spectra')
compute_spectra(generic_2d_all_folder, 'blurrynoisy_numpy', 'blurrynoisy_spectra', add_prefix='blurrynoisy_')
compute_spectra(generic_2d_all_folder, 'restored_cache_numpy', 'restored_spectra')


