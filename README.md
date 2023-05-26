# Feature distribution alignments for object detection in the thermal domain
### :book: Feature distribution alignments for object detection in the thermal domain

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://aminemarnissi.github.io/projects/vcj.html)] &emsp; [Demo] <br>
> [Mohamed Amine Marnissi](https://aminemarnissi.github.io/)
> ...

<p align="center">
  <img src="https://github.com/AmineMarnissi/UDAT/blob/main/journal_flowchart_juin_v1.png">
</p>

---

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>

1. Clone repo

    ```bash
    git clone https://github.com/AmineMarnissi/UDAT.git
    cd UDAT
    ```

1. Install dependent packages

    ```bash
    # Create the environment from the environment.yml file:
    conda env create -f environment.yml
    
    # Activate the new environment:
    conda activate UDAT
    
    # Verify that the new environment was installed correctly:
    conda env list
    ```

## Datasets
* **KAIST:** Download the [Thermal KAIST](https://drive.google.com/drive/u/3/folders/1PYr6RyLvRO5s0UPoo4bG94AhKSp7a4zL) and [Visible KAIST](https://drive.google.com/drive/u/3/folders/1XO8WwTNTzjbAvX771Pov6wGXhnJzYE1y) dataset.
* **FLIR:** Download the [Thermal FLIR](https://drive.google.com/drive/u/3/folders/1aeCO2XCXgf2f2U3B99fk4htI8-9DHdMw) and [Visible FLIR](https://drive.google.com/drive/u/3/folders/1tgI86nBdbkKMHLTpKjBSAnOTW2qdOV4B) dataset.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e. ResNet50. Please download the model from:
* **ResNet50:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0),  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download and make the model in ```data/pretrained_model/```.
## Training
```
bash kaist_train.sh

bash flir_train.sh
```
## Test
```
```
## Demo
<p align="center">
  <img src="https://github.com/AmineMarnissi/UDAT/blob/main/demo.gif">
</p>

## BibTeX

    @article{article,
    author = {Marnissi, Mohamed and Fradi, Hajer and Sahbani, Anis and ESSOUKRI BEN AMARA, Najoua},
    year = {2022},
    month = {02},
    title = {Feature distribution alignments for object detection in the thermal domain},
    journal = {The Visual Computer}
    }

## :e-mail: Contact

If you have any question, please email `mohamed.amine.marnissi@gmail.com`.
