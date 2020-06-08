# Noise-Tolerant Self-Supervised Inversion
Noise-Tolerant Self-Supervised Inversion Paper.
Here you can find our [full paper](paper/Noise_Tolerant_Self_Supervised_Inversion.pdf).

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
$ conda install cudatoolkit==CUDA_VERSION
```

If you are NOT using a conda environment make 
sure you have all CUDA drivers installed properly
on your system for the later options.

#### Run the demo:
You can find the demo in `code/demo/demo.py`.

## How to cite this work?

TBD

## License

TBD

