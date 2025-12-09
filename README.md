# Breaking BAD? Better call SAUL!
___
This repository contains the accompanying artifacts for the paper

*Breaking BAD? Better Call SAUL! – Breaking and Fixing Bloom Filters with Added Diffusion*

to be presented at PETS 2026, Calgary, Canada.
Please follow the instructions below to set up your system.
___
## System Requirements
You will need a GPU to run the experiments described in our paper. The larger the datasets you want to run our code on,
the more powerful the system has to be. To fully replicate our results you will need:
- \>= 128 GB of RAM.
- \>= 20 GB of disk space (HDD sufficient).
- A CUDA-enabled GPU with \>= 24 GB of VRAM is strongly recommended. Make sure that your GPU has a [compute capability](https://developer.nvidia.com/cuda-gpus) >= 3.7.
___
## Set up the Environment
We recommend running our code in a Docker container, as this means you won't have to worry about
dependencies or version conflicts.
If you prefer to run the code directly on your machine, requirements.txt lists the Python dependencies. They can be directly installed via pip.

1) Install Docker by following the [official instructions](https://docs.docker.com/get-started/get-docker/).
2) Open a terminal, navigate to this repository and run ``docker build -t saul:1.0 ./``.
3) Wait a few minutes while Docker builds the image.
4) Run ``docker run -it --gpus all --name gma-artifact saul:1.0`` in your terminal.
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

**A note for Windows users:** Make sure to select WSL2 as the subsystem for Docker, otherwise
you won't be able to use the GPU.
