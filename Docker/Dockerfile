FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV http_proxy="http://firewall.ina.fr:81" https_proxy="http://firewall.ina.fr:81" no_proxy="localhost,127.0.0.1,localaddress,.ina.fr,priv.ina"

# Set working directory
WORKDIR /usr/src/app
COPY environment.yml .

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH


# Set working directory
WORKDIR /usr/src/app
COPY environment.yml .

RUN conda create -c rdkit -n myenv rdkit
RUN conda install pytorch==1.7.0 cudatoolkit=10.2 -n myenv -c pytorch

#RUN conda env create -f environment.yml
RUN echo "source activate myenv" >> ~/.bashrc
#RUN ls /opt/conda/envs/myenv/

ENV PATH  /opt/conda/envs/myenv/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
RUN pip install torch-geometric

RUN pip  install  ogb
RUN echo "Make sure ogb is installed:"
RUN python -c "from ogb.lsc import PCQM4MDataset"

RUN pip install tensorboard
RUN pip install pathos

#RUN cat /etc/resolv.conf

# Copy application source
COPY ./main.py /usr/src/app

EXPOSE 8000

ENTRYPOINT ["python"]
CMD [ "main.py", "/usr/src/kdd/config.json" ]
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "main.py",  "/usr/src/kdd/config.json"]

