conda create -n NVRC
conda activate NVRC
conda install python==3.13.2
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install compressai==1.2.6 accelerate==1.3.0 pytorch-msssim==1.0.0 timm==0.9.16 deepspeed==0.16.2