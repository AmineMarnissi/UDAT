# Feature distribution alignments for object detection in the thermal domain
![architecture](https://github.com/AmineMarnissi/UDAT/blob/main/journal_flowchart_juin_v1.png)

## Introduction
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) respository to setup the environment. In this project, we use Pytorch 1.0.1 and CUDA version is 10.0.130. 

## Datasets
### Datasets Preparation
* **KAIST:** Download the [Thermal KAIST](https://www.cityscapes-dataset.com/) and [Visible KAIST](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data) dataset.
* **FLIR:** Download the [Thermal FLIR](https://drive.google.com/drive/u/3/folders/1aeCO2XCXgf2f2U3B99fk4htI8-9DHdMw) and [Visible FLIR](https://drive.google.com/drive/u/3/folders/1tgI86nBdbkKMHLTpKjBSAnOTW2qdOV4B) dataset.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e. ResNet101. Please download the model from:
* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0),  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download and write the path in **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python test_net.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net resnet101  \
       --load_name path_to_model
```
