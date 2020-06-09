from tv_restoration.conjugate_gradient import conjugate_gradient_TV
from tv_restoration.convo_operators import gaussian1D, ConvolutionOperator
from utils.io.datasets import normalise, camera, add_noise


def original_demo():
    Lambda = 1e-7  # weight of TV regularization
    mu = 1e-10  # parameter of TV smoothing
    n_it = 500  # number of iterations

    image = normalise(camera().astype('f'))

    kern = gaussian1D(2.6)
    K = ConvolutionOperator(kern)
    P = lambda x: K * x
    PT = lambda x: K.T() * x

    blurred_image = add_noise(P(image), intensity=None, variance=0.00)

    en, deconvolved_image = conjugate_gradient_TV(P, PT, blurred_image, Lambda, mu, n_it)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(en)
    plt.show()

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(deconvolved_image, name='deconvolved_image')


original_demo()
