# Air Learning Reinforcement Learning

In this page we give step by step instruction to install airlearning reinforcement learning package and get it up and running. If you have read the paper, then this page will give instruction on setting up the portion highlighted in red in the Air Learning Infrastructure:

![](https://github.com/harvard-edge/airlearning-rl/blob/master/docs/images/airlearning-rl.png)

## System requirements:
The following instructions were tested on Windows 10, Ubuntu 16.04 WSL (Windows 10)

* Windows 10
* Tensorflow-gpu 1.8 (Tested)
* CUDA/CuDNN 

([Here](https://medium.com/@lmoroney_40129/installing-tensorflow-with-gpu-on-windows-10-3309fec55a00) is a really good step by step instruction to install Tensorflow/CUDA/CuDNN on Windows 10)

P.S: For Ubuntu users, we have tested this on Ubuntu 16.04, 18.04, Ubuntu Mate platforms. The instructions are the same as below with one caveat:-You need two machines. The first machine will render the [Air Learning Environment Generator](https://github.com/harvard-edge/airlearning-ue4/tree/b4f27ea457936609745ddad1191ab8c54f8799ac). This portion is currently tested on Windows 10 machine (We will port it to Ubuntu soon). The second machine will be used to train the reinforcment learning algorithm. This second machine can be Windows 10 or Ubuntu. 

On the other hand, if you have a Windows 10 machine, you can run both rendering and RL training on a single machine. So are providing instruction to install the RL training on Windows 10 here.

## Installation Instruction

### Step 1: Install Dependencies
Assuming you have installed Python, Pip, Tensorflow/CUDA/CuDNN from correctly from [here](https://medium.com/@lmoroney_40129/installing-tensorflow-with-gpu-on-windows-10-3309fec55a00), get the following packages:

```pip install msgpack-rpc-python airsim keras-rl h5py Pillow gym opencv-python eventlet matplotlib PyDrive pandas```

### Step 2: Install Air Learning RL
Clone the Air Learning Project

```$ git clone --recursive https://github.com/harvard-edge/airlearning.git```
```$ cd airlearning/airlearning-rl```

Lets call the directory where airlearning is cloned as <AIRLEARNING_ROOT>
### Step 3: Setup the machine_dependent_settings.py file

You need to point to the directory where Unreal Project files are installed. Please install them before you follow the instructions below.The instructions for installing Air Learning Environment Generator is [here](https://github.com/harvard-edge/airlearning-ue4/tree/b4f27ea457936609745ddad1191ab8c54f8799ac). 

Here is a sample machine_dependent_settings.py file. Please use this as a template and point to the location where you have installed the airlearning-ue4 project

``` $ cd <AIRLEARNING_ROOT>/airlearning-rl/settings_folder/```
``` $ vim machine_dependent_settings.py```

