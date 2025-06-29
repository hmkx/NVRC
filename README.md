# NVRC
NVRC: Neural Video Representation Compression (NeurIPS 2024)

by Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, and David Bull.


Welcome to the repository for NVRC. NVRC is the first INR-based neural video codec outperforming VTM (RA) for long sequence videos. This repository provides implementation of our proposed method and the training scripts/configurations.

[Project page](https://hmkx.github.io/nvrc/)

[arXiv](https://arxiv.org/abs/2409.07414)

This work is also based on prior work [HiNeRV](https://hmkx.github.io/hinerv/).

## TODO

- Add YUV configurations
- Provide experiment results
- Provide evaluation code with input bitstream

## Install Environment
```
. install.sh
```

## Usage
Prepare the RGB Dataset
First, convert each video into a sequence of PNG images. For example, using FFmpeg:

```
VIDEO_ID=<your_video_id>
mkdir -p $VIDEO_ID
ffmpeg -video_size 1920x1080 -pixel_format yuv420p -i ${VIDEO_ID}.yuv ${VIDEO_ID}/%04d.png
```

Prepare the YUV Dataset
For raw YUV inputs, simply rename your file to the format:
```
${VIDEO_ID}_1920x1080_yuv420p.yuv
```

## Training
A sample script for overfitting on the UVG dataset (RGB) is provided:

```
VIDEO_ID=Beauty
LAMB=1.0
SCALE=s
LR_S1=2e-3
LR_S2=1e-4
GRAD_ACCUM=1
BATCH_SIZE=144

bash scripts/train/overfitting_uvg_nvrc.sh 0 ${VIDEO_ID} ${LAMB} ${SCALE} ${LR_S1} ${LR_S2} ${GRAD_ACCUM} ${BATCH_SIZE}
```
Feel free to adjust these parameters and scripts for your own datasets and experiments.


## Acknowledgements

Part of this implementation is based on the code from [CompressAI](https://github.com/InterDigitalInc/CompressAI/tree/master), [C3](https://github.com/google-deepmind/c3_neural_compression), and [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models).


## Citation
If you find this work useful, please consider citing:
```
@inproceedings{
  author       = {Ho Man Kwan and Ge Gao and Fan Zhang and Andrew Gower and David Bull},
  title        = {HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation},
  booktitle    = {NeurIPS},
  year         = {2023}
}

@inproceedings{
  author       = {Ho Man Kwan and Ge Gao and Fan Zhang and Andrew Gower and David Bull},
  title        = {NVRC: Neural video representation compression},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```