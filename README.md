## Self Supervised 2D Human Pose Prediction

This code implementation is the self-supervised 2D
pose prediction method to predict the pose in the next frame
from previous sequential images.

### Getting Started

#### Install Requriements

Create a python 3.7 environement, eg:

    conda create -n pytorch-openpose python=3.7
    conda activate pytorch-openpose

Install pytorch by following the quick start guide here (use pip) https://download.pytorch.org/whl/torch_stable.html

Install other requirements with pip

    pip install -r requirements.txt

#### Download the pretrained Openpose Model

* [dropbox](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0)
* [baiduyun](https://pan.baidu.com/s/1IlkvuSi0ocNckwbnUe7j-g)
* [google drive](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG?usp=sharing)

You should only download the pretrained model named "body_pose_model.pth"

Download the pytorch models and put them in a directory named `model` in the project root directory

#### Download the sequential dataset MPII

* [official site](http://human-pose.mpi-inf.mpg.de/#download)

You can download any batches as you like.

Download the MPII dataset, unzip them and put them in a directory named `data` in the project root directory

#### Train and Run the demo

train our model from scratch:

    python train.py

NOTICE: You should get the saved parameters in a directory named `ckpt` in the project root directory after you run
at least one epoch.

to run a demo

    python our_demo.py

If you already have the saved model in `ckpt`, you can directly run `our_demo.py` to explore the model.




