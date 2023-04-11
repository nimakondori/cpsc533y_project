ARG cuda_version=11.6
ARG torch_version=1.13.1
FROM pytorch/pytorch:$torch_version-cuda${cuda_version}-cudnn8-runtime

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install vim git 

# Install fuzzy finder 
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install

RUN pip install pandas numpy tqdm matplotlib colorlog scipy torch-geometric attributedict

# # Install miniconda
# ENV CONDA_DIR /opt/new_conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh  -b -p $CONDA_DIR

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH




