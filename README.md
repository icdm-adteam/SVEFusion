# SVEFusion: Salient Voxel Enhancement for 3D Object Detection of LiDAR and 4D Radar Fusion

:wave: This is the official repository for **SVEFusion**. 

<p align="center">
  <img src="images/network.png" width="800"/>
</p>

# News
* 2025/3/29: Code is coming soon...

# Getting Started
This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

## Installation

### 1. Clone the source code 
```
git clone https://github.com/icdm-adteam/SVEFusion.git
cd SVEFusion
```

### 2. Create conda environment and set up the base dependencies
```
conda create --name svefusion python=3.8
conda activate svefusion
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install spconv-cu118
```

### 3. Install pcdet
```
python setup.py develop
```

### 4. Install required environment
```
pip install -r requirements.txt
```

# Acknowledgements
Thank for the excellent 3D object detection codebases [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

Thank for the excellent Spatially Sparse Convolution Library [spconv](https://github.com/traveller59/spconv)

Thank for the excellent 4D radar dataset [VoD Dataset](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/GETTING_STARTED.md) to download dataset.