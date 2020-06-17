from os import listdir
from os.path import isfile, join, basename
from shutil import copy

import numpy
from imageio import imread, imwrite


def find_files_in_folder(folder, name, exclude=None):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    selected_files = [join(folder, f) for f in files if name in f]
    if exclude is not None:
        selected_files = [f for f in selected_files if exclude not in f]
    return list(selected_files)


def collect(folder, filepaths, prepend=''):
    for filepath in filepaths:
        filename = basename(filepath)
        copy(filepath, join(folder, prepend + filename))


def get_images(folder, target_folder, name):
    collect(target_folder, find_files_in_folder(join(folder, 'gt'), name), prepend='gt_')
    collect(target_folder, find_files_in_folder(join(folder, 'blurry'), name), prepend='b_')
    collect(target_folder, find_files_in_folder(join(folder, 'blurrynoisy'), name), prepend='bn_')
    collect(target_folder, find_files_in_folder(join(folder, 'restored'), name))

    collect(target_folder, find_files_in_folder(join(folder, 'restored_spectra'), name))
    collect(target_folder, find_files_in_folder(join(folder, 'gt_spectra'), name))
    collect(target_folder, find_files_in_folder(join(folder, 'blurrynoisy_spectra'), name))


def crop_images(folder, name, crop_prefix, center, extent, exclude=None):
    y, x = center
    h, w = extent
    slice = numpy.s_[y - h:y + h, x - w:x + w]

    files = find_files_in_folder(folder, name, exclude=crop_prefix)
    files_base_names = [basename(f) for f in files]

    images = list([imread(f) for f in files])

    for image, filename in zip(images, files_base_names):
        print(f"Cropping image: {filename} of size:{image.shape} with slice:{slice}")
        if exclude is not None and exclude in filename:
            continue
        image = image[slice]
        filepath = join(folder, crop_prefix + '_' + filename)
        imwrite(filepath, image)


source_folder = "/home/royer/workspace/ssi/ssi-ssi/ssi/benchmark/images/generic_2d_all"
target_folder = "/home/royer/workspace/ssi/ssi-ssi/ssi/benchmark/images/_figure_panels"

get_images(source_folder, target_folder, name='characters')
crop_images(target_folder, 'characters', crop_prefix='crop1', center=(466, 784), extent=(80, 80), exclude='spectrum')

get_images(source_folder, target_folder, name='usaf')
crop_images(target_folder, 'usaf', crop_prefix='crop1', center=(371, 463), extent=(80, 80), exclude='spectrum')

get_images(source_folder, target_folder, name='scafoldings')
crop_images(target_folder, 'scafoldings', crop_prefix='crop1', center=(764, 507), extent=(80, 80), exclude='spectrum')

get_images(source_folder, target_folder, name='drosophilaslice')
crop_images(target_folder, 'drosophilaslice', crop_prefix='crop1', center=(963, 82), extent=(80, 80), exclude='spectrum')
