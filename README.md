# Image Deconvolution via Noise-Tolerant Self-Supervised Inversion

You can find the latest version of our full paper [here](https://royerlab.github.io/ssi-code/paper/Noise_Tolerant_Self_Supervised_Inversion.pdf).

## Authors

Hirofumi Kobayashi [@liilii_tweet](https://twitter.com/liilii_tweet)

Ahmet Can Solak [@_ahmetcansolak](https://twitter.com/_ahmetcansolak)

Joshua Batson [@thebasepoint](https://twitter.com/thebasepoint)

Loic A. Royer [@loicaroyer](https://twitter.com/loicaroyer)

## Abstract

We propose a general framework for solving inverse problems in the presence of noise that requires no signal prior, no noise estimate, and no clean training data. The only assumptions are that the forward model is available, differentiable and that the noise exhibits statistical independence across different measurement dimensions. We build upon the theory of 'J-invariant' functions  [Batson & Royer 2019](https://arxiv.org/abs/1901.11365) and show how self-supervised denoising \emph{à la} Noise2Self is a special case of learning a noise-tolerant pseudo-inverse of the identity. We demonstrate our approach by showing how a convolutional neural network can be taught in a self-supervised manner to deconvolve images and surpass in image quality classical inversion schemes such as Lucy-Richardson deconvolution.

## Get started

##### Get the project:
```bash
$ git clone https://github.com/royerlab/ssi-code
$ cd ssi-code
```

#### Setting up an environment:
We recommend that you create a dedicated conda environment for SSI:

```bash
$ conda create --name ssi python=3.7
$ conda activate ssi
```

#### Install the generic dependencies:
```bash
$ pip install -r requirements.txt

# (For CUDA 10.0)
$ pip install cupy-cuda100

# (For CUDA 10.1)
$ pip install cupy-cuda101

# (For CUDA 10.2)
$ pip install cupy-cuda102
```

#### Install CUDA dependencies:

If you are using a conda environment:
```bash
$ conda install cudatoolkit=CUDA_VERSION
```

If you are NOT using a conda environment make 
sure you have all CUDA drivers installed properly
on your system for the later options.

#### Run the demo:
You can find the demo in `code/demo/demo.py`.
You can run the demo by:
```bash
python -m code.demo.demo
```

You can also change the image to run the demo with:
```bash
python -m code.demo.demo characters
```
We recommend to trythe following images: 'drosophila' (default), 'usaf', 'characters'.


This should go fast if your GPU is reasonably recent.

Once done, a [napari](https://napari.org/) window will open to let you compare
the images. Please note that due to the stochastic nature of CNN training, and
because we use so little training data, and also because perhaps we have not fully
understood how to train our nets, we occasionally observe failed runs.

Things to observe: Varying the number of iterations for Lucy-Richardson (LR) lets you explore the trade-off between sharpness and noise reduction. Yet, LR has trouble to acheive both. In particular, you can see that the Spectral Mutual Information (SMI) goes down dramatically as you go towards low iterations, but the PSNR varies in the opposite way. That's because while you have good noise reduction at low iterations, you loose fidelity in the high-frequencies of the image. Why? LR reconstructs images by first starting with the low frequencies and then slowly refines the higher ones -- that's when trouble arises and noise gets amplified. Different comparison metrics quantify different aspects of image similarity, SMI (introduced in this paper) is good at telling us if the images are dissimilar (or similar) in the frequency domain. Our approach, Self-Supervised Inversion (SSI) will acheive a good trade-off in comparison. 

## Feedback welcome!

Feedback, pull-requests, and ideas to improve this work are very welcome and will be duly acknowledged.

## How to cite this work?

##### Image Deconvolution via Noise-Tolerant Self-Supervised Inversion.
Hirofumi Kobayashi, Ahmet Can Solak, Joshua Batson, Loic A. Royer.

arXiv submission pending.

## License

BSD 3-Clause
