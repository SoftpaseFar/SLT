# 安装教程

```bash
# 1. Create a conda virtual environment.
conda create -n alphapose python=3.7 -y
conda activate alphapose

# 2. Install specific pytorch version
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 3. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

# 4. install dependencies
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
sudo apt-get install libyaml-dev
pip install cython==0.27.3 ninja easydict halpecocotools munkres natsort opencv-python pyyaml scipy tensorboardx  terminaltables timm==0.1.20 tqdm visdom jinja2 typeguard pycocotools
################Only For Ubuntu 18.04#################
locale-gen C.UTF-8
# if locale-gen not found
sudo apt-get install locales
export LANG=C.UTF-8
######################################################

# 5. install AlphaPose 
python setup.py build develop

# 6. Install PyTorch3D (Optional, only for visualization)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install pytorch3d

从github的项目页[下载](https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view)yolov3-spp.weights到AlphaPose/detector/yolo/data ，如果没有这个文件夹就创建一个。

[下载](https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/docs/MODEL_ZOO.md)FastPose预训练模型到AlphaPose/pretrained_models

! 将self_demo_inference.py放到
! self.yaml放到
! self_model.pth
! 图片
python scripts/self_demo_inference.py --cfg configs/halpe_136/resnet/self.yaml --checkpoint pretrained_models/self_model.pth --indir examples/demo/dev --save_img --format cmu
python scripts/self_demo_inference.py --cfg configs/halpe_136/resnet/self.yaml --checkpoint pretrained_models/self_model.pth --indir examples/demo/dev  --format cmu
python scripts/self_demo_inference.py --cfg configs/halpe_136/resnet/self.yaml --checkpoint pretrained_models/self_model.pth --indir ~/autodl-fs/SLT/data/Phonexi2014T/features/train/ --format cmu
python scripts/self_demo_inference.py --cfg configs/halpe_136/resnet/self.yaml --checkpoint pretrained_models/self_model.pth --indir ~/autodl-fs/SLT/data/Phonexi2014T/features/train/ --chdir ./examples/train/  --format cmu

! 视频
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video examples/demo/test_video.mp4 --save_video
python scripts/self_demo_inference.py --cfg configs/halpe_136/resnet/self.yaml --checkpoint pretrained_models/self_model.pth --video examples/demo/how2sign  --format cmu

```
