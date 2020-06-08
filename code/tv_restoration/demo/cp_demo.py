from tv_restoration.chambole_pock import cp_restoration
from utils.io.datasets import normalise, camera, add_microscope_blur_2d, add_poisson_gaussian_noise


def demo():
    image = normalise(camera().astype('f'))
    blurred_image, psf_kernel = add_microscope_blur_2d(image)
    noisy_blurred_image = add_poisson_gaussian_noise(blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10)

    deconvolved_image = cp_restoration(noisy_blurred_image,
                                       kernel=psf_kernel,
                                       num_iterations=200,
                                       beta=2e-3)

    deconvolved_image = deconvolved_image.clip(0, 1)

    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(noisy_blurred_image, name='noisy_blurred_image')
        viewer.add_image(deconvolved_image, name='deconvolved_image')


demo()
