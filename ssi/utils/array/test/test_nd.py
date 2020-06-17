import numpy
from skimage.data import binary_blobs

from ssi.utils.array.nd import (
    nd_split_slices,
    remove_margin_slice,
    extract_tiles,
)


def test_nd_split_slices():
    array_shape = (96, 17, 117, 45)
    nb_slices = (5, 3, 7, 5)

    slice_tuple_list = list(nd_split_slices(array_shape, nb_slices, do_shuffle=True))

    print(str(slice_tuple_list))

    array_source = numpy.random.choice(a=[False, True], size=array_shape)
    array_target = numpy.random.choice(a=[False, True], size=array_shape)

    for slice_tuple in slice_tuple_list:
        array_target[slice_tuple] = array_source[slice_tuple]
        pass

    assert numpy.all(array_source == array_target)


def test_nd_split_slices_with_margins():
    array_shape = (96, 17, 117, 45)
    nb_slices = (5, 3, 7, 5)
    margins = (4, 5, 6, 7)

    slice_tuple_list = list(nd_split_slices(array_shape, nb_slices, do_shuffle=False))
    slice_margin_tuple_list = list(
        nd_split_slices(array_shape, nb_slices, do_shuffle=False, margins=margins)
    )

    # print(str(slice_tuple_list))

    array_source = numpy.random.choice(a=[0, 1], size=array_shape)
    array_target = numpy.zeros(array_shape, dtype=numpy.int64)

    for slice_tuple, slice_margin_tuple in zip(
            slice_tuple_list, slice_margin_tuple_list
    ):
        sliced_array_with_margin = array_source[slice_margin_tuple]
        remove_margin_slice_tuple = remove_margin_slice(
            array_shape, slice_margin_tuple, slice_tuple
        )
        sliced_array_removed_margin = sliced_array_with_margin[
            remove_margin_slice_tuple
        ]
        # print(array_target[slice_tuple].shape)
        # print(sliced_array_removed_margin.shape)
        array_target[slice_tuple] += sliced_array_removed_margin

    assert numpy.all(array_source == array_target)


def test_extract_tiles():
    blob_image = binary_blobs(length=512, seed=1, n_dim=2)

    tiles = extract_tiles(blob_image, tile_size=64, extraction_step=64, flatten=True)

    print(tiles.shape)
    print(len(tiles))
