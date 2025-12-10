# Artifact Appendix

Paper title: **Breaking BAD? Better Call SAUL! - Breaking and Fixing Bloom Filters with Added Diffusion**

Requested Badge(s):
  - [X] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**

## Description 
Replace this with the following:

This repository contains the accompanying artifacts for the paper

*Breaking BAD? Better Call SAUL! – Breaking and Fixing Bloom Filters with Added Diffusion*

to be presented at PETS 2026, Calgary, Canada.
The artifact contains the entire source code of the experiments described in our paper, as well as a reference implementation of your newly proposed scheme.
Datasets are provided if their licensing permits it. For non-free datasets a download link is included in the ReadMe-File.
The following code files will run the experiments:

- ``hgma_bad_benchmark.py`` runs the HGMA-Attack against BAD encoded data.
- ``hgma_saul_benchmark.py`` runs the HGMA-Attack against SAUL encoded data.
- ``indistinguishability_experiments.py`` conducts the indistinguishability experiments as outlined in the security definition for BAD and SAUL. 
- ``matching_benchmark.py`` evaluates the linkage quality of SAUL, BAD and Bloom Filters.
- ``bad_utils.py`` contains various helper functions.

If you are interested in (re-)using the reference implementation of SAUL, you can find it in ``./encoders/saul_encoder.py``.

### Security/Privacy Issues and Ethical Concerns

To the best of our knowledge, there are no adverse affects of running our artifact. It does not interfere with your system configuration.

## Basic Requirements 

### Hardware Requirements

You will need a GPU to run the experiments described in our paper. The larger the datasets you want to run our code on,
the more powerful the system has to be. To fully replicate our results you will need:
- \>= 128 GB of RAM.
- \>= 20 GB of disk space (HDD sufficient).
- A CUDA-enabled GPU with \>= 24 GB of VRAM is strongly recommended. Make sure that your GPU has a [compute capability](https://developer.nvidia.com/cuda-gpus) >= 3.7.

Using the reference implementation of SAUL only is not particularly demanding. It should run well on a modern laptop.

### Software Requirements

You will need the following Software requirements:

- CUDA
- Python >= 3.11
- Libraries as listed in requirements.txt 

The following instructions are designed for running the code on a bare-metal Ubuntu 24.04 LTS. Note that we recommend running our code in Docker (See Section Environment).

**1. Install System Dependencies**
1) If you want to use the GPU, run ``nvidia-smi`` to check if you have a GPU driver installed. If not, install the [latest version](https://www.nvidia.com/download/index.aspx).
2) Install Python and pip if necessary. On Ubuntu/Debian run ``sudo apt install python3 python3-pip``.
3) Make sure that you have ``pkg-config`` and FreeType installed. Otherwise, the installation of matplotlib might fail. On Ubuntu/Debian run ``sudo apt install pkg-config libfreetype6-dev``.
4) Install the g++ compiler for improved performance. On Ubuntu run ``sudo apt install g++``.
5) Install the [oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). On Ubuntu run ``sudo apt install intel-mkl``.

**2. Install Python Dependencies**
We recommend using a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for package management to avoid version conflicts.
1) Install [PyTorch](https://pytorch.org/get-started/locally/). If you want to use the GPU, choose the latest (highest) 
CUDA version. Otherwise, select CPU. 
2) Run ``pip install -r requirements.txt`` to install the remaining dependencies.
3) [Verify](https://pytorch.org/get-started/locally/#linux-verification) your install.


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

Replace the following with estimated values for:

- The overall human and compute times required to run the artifact.
- The overall disk space consumed by the artifact.

This helps reviewers schedule the evaluation in their time plan and others in
general to see if everything is running as intended. This should also be
specified at a finer granularity for each experiment (see below).

## Environment

We recommend running our code in a Docker container, as this means you won't have to worry about
dependencies or version conflicts.
If you prefer to run the code directly on your machine, follow the instructions provided above.

You may install Docker by following the [official instructions](https://docs.docker.com/get-started/get-docker/).

**A note for Windows users:** Make sure to select WSL2 as the subsystem for Docker, otherwise
you won't be able to use the GPU.

### Accessibility

The artifact is publicly availabe on GitHub:
[https://github.com/SchaeferJ/BreakingBAD](https://github.com/SchaeferJ/BreakingBAD)

### Set up the environment

1) Clone this repository: ``git clone https://github.com/SchaeferJ/BreakingBAD``
2) Open a terminal, navigate to this repository and run ``docker build -t saul:1.0 ./``.
3) Wait a few minutes while Docker builds the image.
4) Run ``docker run -it --gpus all --name saul-artifact saul:1.0`` in your terminal.
5) That's it!

If everything worked, you should now see a shell connected to your docker container. It looks something like

``root@bfeff35dda4a:/usr/app# ``

Type ``ls`` to view the contents of the directory you're currently in. The output should look like this:
````
 aligners       embedders     encoders                matchers
 bad_utils      Dockerfile    hgma_bad_benchmark.py   hgma_bad_benchmark.py
 ````
You can interact with the docker container just like with any other Linux OS.
To exit the docker container, simply type ``exit`` into the terminal and hit enter.

**Note:** ``docker run`` will always create a new docker container, so you do not have access
to any files you created in previous runs. Use ``docker start -i saul-artifact`` instead to
resume working with the container you already created.

Describe the expected results where it makes sense to do so.

## Limitations

Note that the following experiments involve randomness:

- The H-GMA relies on random walks and random initialization of embeddings.
- The indistinguishability experiments rely on randomly generated datasets.

Not all randomness can be controlled by setting a seed. For instance, the Random Walks will always slightly differ, even if all seeds are kept constant.
This is a known issue of the underlying PecanPy-Library. For fully deterministic results, the node2vec library has to be used and multiprocessing must be 
disabled, both of which would cause a severe degradation of performance.
Hence, results obtained from running this artifact might (slightly) differ from the ones reported in the paper. 

## Notes on Reusability

The artifact is fully reusable. Attacks can be run against other encoding schemes, as long as their implementations follow the abstract class provided.
The reference implementation of SAUL is provided as a separate module and can be used for encoding arbitraty textual data.
