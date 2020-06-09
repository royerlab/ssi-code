# Image Deconvolution via Noise-Tolerant Self-Supervised Inversion
Image Deconvolution via Noise-Tolerant Self-Supervised Inversion Paper.
Here you can find our [full paper](https://royerlab.github.io/ssi-code/paper/Noise_Tolerant_Self_Supervised_Inversion.pdf).

## Authors

Hirofumi Kobayashi [@liilii_tweet](https://twitter.com/liilii_tweet)

Ahmet Can Solak [@_ahmetcansolak](https://twitter.com/_ahmetcansolak)

Joshua Batson [@thebasepoint](https://twitter.com/thebasepoint)

Loic A. Royer [@loicaroyer](https://twitter.com/loicaroyer)

## Get started

##### Get the project:
```bash
$ git clone https://github.com/royerlab/ssi-code
$ cd ssi-code
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

## How to cite this work?

Image Deconvolution via Noise-Tolerant Self-Supervised Inversion,
Hirofumi Kobayashi, Ahmet Can Solak, Joshua Batson, Loic A. Royer,

Arxiv submission pending.

## License

BSD 3-Clause
