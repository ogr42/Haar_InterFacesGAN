# Haar_InterFacesGAN
It's version 2.0, version 1.0 is  there: 
    https://colab.research.google.com/drive/1stjkMUhcTj3xy8KYjqxvFM3NRVV7nkwf

StyleGAN3 Inference Notebook
Source:
https://colab.research.google.com/github/yuval-alaluf/stylegan3-editing/blob/master/notebooks/inference_playground.ipynb
https://github.com/AnonSubm2021/TransStyleGAN


CLI code by installing modules BEFORE running script:

mkdir haar_interfacesgan
chdir haar_interfacesgan
git clone https://github.com/yuval-alaluf/stylegan3-editing
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/'haarcascade_frontalface_default.xml
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 –force
pip install pyrallis
