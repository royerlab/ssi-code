# Noise-Tolerant Self-Supervised Inversion
Noise-Tolerant Self-Supervised Inversion Paper

## Get started

##### Get the project:
```bash
$ git clone https://github.com/royerlab/ssi-code
$ cd ssi-code
```

#### Install the generic dependencies:
```bash
$ pip install -r requirements.txt
```

#### Install CUDA specific dependencies:

If you are using a conda environment:
```bash
$ conda install -c conda-forge cupy cudatoolkit
```

If you are NOT using a conda environment:
```bash
# (For CUDA 10.0)
$ pip install cupy-cuda100

# (For CUDA 10.1)
$ pip install cupy-cuda101

# (For CUDA 10.2)
$ pip install cupy-cuda102
```

Make sure you have all CUDA drivers installed properly
on your system for the later options.



#### Run the demo:
```bash
$ python code/demo/demo.py
```

## How to cite this work?

TBD

## License

TBD

