# Image Captioning using Attention

This project is inspired from the paper [Show Attend and Tell](https://arxiv.org/abs/1502.03044)

To evaluate the code from the features, jump to the evaluation section. 

### Prerequisites to run the code
 * CUDA 8 is required. (Note CUDA 9 is not yet supported by Tensorflow. If tensorflow needs to be installed using CUDA 9, we need to install it from the source.
 * Caffe ( For computing image features using pretrained VGG19 model)
 * Tensorflow (Model is trained using tensorflow)
 * Keras
 * Python modules - numpy, pandas, matplotlib, scikit-image 
 * OpenCV used for resizing images. Also it is one of the prerequisite for Caffe.
 
 (Note: It is necessary that NVIDIA drivers and CUDA is set up on your system)
 
### Installing Caffe
You might have to install cuDNN version compatible with the CUDA version installed on the machine. This can be done by following the steps mentioned in this [link](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

Follow this [link](https://gist.github.com/arundasan91/b432cb011d1c45b65222d0fac5f9232c) for installing Caffe. Install the one without Anaconda. Make sure you install gpu version of caffe. This will require editing of Makefile.config.

(Note while installing OpenCV, there might be an error while building examples. In case you get an error, try making the BUILD_EXAMPLES = OFF and INSTALL_C_EXAMPLES=OFF in the cmake command. Also add -D WITH_CUDA=OFF in the cmake command). 

You can skip installation of Cython, which is one of the step in the link.

We need to comment the line which says CPU_ONLY = 1. 

Also for using gpu, we might have to uncomment the line USE_CUDNN=1. This is to build using cudnn(This step is necessary). 

Set BLAS to atlas in the Makefile.config

In case you get an error of no hdf5.hpp found during the make command of caffe, add the following lines in the Makefile.config
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu/hdf5/serial
```

If you get an error of comput-20 not compatible, just remove the following lines from the Makefile.config
```
-gencode arch=compute_20,code=sm_20 \
-gencode arch=compute_20,code=sm_21 \
```

If you get an error saying 'cannot find -lcblas' or 'cannot find -latlas', run the following command
```
sudo apt-get install libatlas-base-dev
```

You may need to upgrade pip to version >9. This can be done using the following command.
```
pip install --upgrade pip
```

This step will be encountered in the process of installation. 

Following the link you should be able to install caffe.

### Installing Tensorflow
For Python2.7, run the following commands to install tensorflow gpu version
```
sudo apt-get install python-pip python-dev
sudo pip  install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
```
To check if tensorflow is installed, go to the python terminal and run the following command
```
import tensorflow
```
This should run without error.
### Installing Keras
Run the following command in the terminal:
```
pip install keras
```
### Installing python modules
We can install the python modules using the pip command, eg.
```
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-image
```
### Installing OpenCV
Note OpenCV is one of the prerequisite for Caffe, and thus the installation is done as part of the Caffe installation. But in case, we need to install OpenCV explicitly, we can do so by running the bash script given in this [link](https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh). (Note while installing OpenCV, there might be an error while building examples. In case you get an error, try making the BUILD_EXAMPLES = OFF and INSTALL_C_EXAMPLES=OFF in the cmake command of the script). To run the script, use the following command
```
sudo bash install-opencv.sh
```
install-opencv.sh is the name of the bash script.

### Downloading the dataset for training
Flickr30k Dataset is used for trainig the model. You can download the dataset from this [link](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/). Extract the dataset in the data folder of the code. There should be a folder named flickr30k_images and a file named results_20130124.token. Now we are ready for training the model.

### Downloading the VGG19 caffe model
Download the vgg19 caffe model and the vgg19.prototxt. 

You can use this [link](http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel) for downloading the vgg caffe model. Name the downloaded model as VGG_ILSVRC_19_layers.caffemodel

For prototxt file go to this [link](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt), copy the contents of the raw file and paste it into a file. Name that file as VGG_ILSVRC_19_layers_deploy.prototxt

You have to place both the files in the data folder.

(Note the prototxt file is already present in the data folder of this repository).

### Preprocessing the data
For preprocessing of data, two files are used, namely utility.py and preprocess.py. Before, preprocessing make sure the following 4 things are present in the data folder.
  * results_20130124.token
  * flickr30k_images folder
  * VGG_ILSVRC_19_layers_deploy.prototxt
  * VGG_ILSVRC_19_layers.caffemodel

You have to edit the utilities.py to add appropriate caffe model path.
Once the above prerequisites are met, run the following command in the terminal to preprocess images. 
```
python preprocess.py
```
After running above command, it will create features.npy and annotations.pickle in the data folder. These two files will be used for training of model.

### Training model
To train the model, run the following command
```
python trainModel.py model/ data/features.npy data/annotations.pickle
```
* The first argument specifies the path at which the model will be saved after every epoch.
* The second argument specifies the path of features.npy, which gets created in the preprocessing step.
* The third argument specifies the path of annotations.pickle, which gets created in the preprocessing step.

This will start the training of the model and you can observe the loss for each batch (batchsize=80). Model is saved after every epoch. The code runs for 1000 epochs.(Note: It takes a long time for training for 1000 epochs, hence the model after 80th epoch is used for evaluation and it tends to give decent results). 

Training with decay was tested but didnt give good enough results. The line for introducing decay has been commented, but can be used if required.

(Note: Pretrained model-80 is present in the data folder for evaluation purposes.)

### Evaluation of model
Since training for 1000 epochs takes longer time, model after 80th epoch is used for evaluation and it tends to give decent results. 

For evaluation of an image, we need to preprocess the image to get its VGG features. This can be done using the CreateDataForSingleImage.py. Run the following command to compute the features.
```
python CreateDataForSingleImage.py ImageFeatures/manInSnow.npy Images/manInSnow.jpg
```
* The first argument specifies the path at which the features would be stored.
* The second argument specifies the path of the image whose features needs to be computed.

(Note: Features for few images are present in the data folder for evaluation of the model-80.)

Now we need to run the following command to get the caption and visualize the attention over image. This can be done using the following command.
```
python testModel.py model/model-80 Images/manInSnow.jpg ImageFeatures/manInSnow.npy data/annotations.pickle
```
* The first argument specifies the saved model path.
* The second argument specifies the image path
* The third argument specifies the image features path
* The fourth argument specifies the annotations path

Running the command would yield the caption and also display the attention maps over images.
