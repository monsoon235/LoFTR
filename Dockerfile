FROM gpuci/miniconda-cuda:10.1-devel-ubuntu18.04

WORKDIR /root

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN echo -e "\
channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud\n\
" > .condarc

RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

# 把你的 enviroment.yml 按以下示例复制到下面，每行后加 \
# 这里以 loftr 环境为例
# 注意 bitahub 上除了 3090 卡之外的卡最高支持到 cuda10.1
# 3090 最高支持 cuda11.4
RUN echo -e "\
name: loftr\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - defaults\n\
dependencies:\n\
  - python=3.8\n\
  - cudatoolkit=10.1\n\
  - pytorch=1.8.1\n\
" > environment.yml

RUN echo -e "\
opencv_python==4.4.0.46\n\
albumentations==0.5.1 --no-binary=imgaug,albumentations\n\
ray>=1.0.1\n\
einops==0.3.0\n\
kornia==0.4.1\n\
loguru==0.5.3\n\
yacs>=0.1.8\n\
tqdm\n\
autopep8\n\
pylint\n\
ipython\n\
jupyterlab\n\
matplotlib\n\
h5py==3.1.0\n\
pytorch-lightning==1.3.5\n\
joblib>=1.0.1\n\
" > requirements.txt

# 创建环境
RUN apt update && \
    apt install --no-install-recommends -y openssh-server libglib2.0-0 libgl1-mesa-glx && \
    apt autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    conda env create -f environment.yml && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate loftr && \
    conda clean -y -a && \
    pip install --no-cache-dir -r requirements.txt

# 登录时自动激活
RUN sed -i 's/conda activate base/conda activate loftr/g' .bashrc

# 然后执行 docker build -t 环境名 .    (不要漏了最后的 . )
# 然后执行 docker run --rm -it --gpus all 环境名 看看环境是否正确
# 如果正确即可在 https://www.bitahub.com/newmirror 上传此 Dockerfile 创建环境
