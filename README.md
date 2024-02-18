<hr style="height:1px" />

<!-- <p align="center"> <img src="./pics/horse2zebra.gif" width="100%" /> </p> -->

<hr style="height:1px" />

<!-- # `<p align="center">` CycleGAN - Tensorflow 2 `</p>` -->
# CycleGAN - Tensorflow 2
Tensorflow 2 implementation of CycleGAN.
<!-- Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) -->
<!-- Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.* -->
## Example results

<!-- ### summer2winter -->

<!-- row 1: summer -> winter -> reconstructed summer, row 2: winter -> summer -> reconstructed winter -->

<p align="center"> <img src="./pics/patient64542_study1_frontal.jpg" width="100%" /> </p>
<p align="center"> <img src="./pics/patient64542_study1_lateral.jpg" width="100%" /> </p>

## Usage

- Software environment:
  ```console
  # Python with the following version and libraries.
  conda create -n r2c-gan python=3.7.11
  conda activate r2c-gan
  conda install tensorflow-gpu=2.4.1
  conda install scikit-image tqdm scikit-learn pydot
  conda install -c conda-forge oyaml
  pip install tensorflow-addons==0.13.0
  pip install numpy==1.19.2
  ```
- Dataset
  ```console
  sh ./download_dataset.sh CheXpert
  python unzip.py
  ```
- Example of training

  ```console
  python train.py
  ```
  - tensorboard for loss visualization

    ```console
    tensorboard --logdir ./output/summaries --port 6006
    ```
- Example of testing

  ```console
  python test.py
  ```
# Result

- The result will be saved in the `./output` directory.
- The result will be saved in this [link ](https://tuni-my.sharepoint.com/:f:/g/personal/long_nguyen_tuni_fi1/EscN9XgA1MdDnc6AumEsKGIB3qatPi-jxZXLs5X60HiAdQ?e=rMXbCN)
