import numpy as np
import sys
from utilities import *

feat_path = sys.argv[1]
image_path = sys.argv[2]

vgg_model = 'data/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = 'data/VGG_ILSVRC_19_layers_deploy.prototxt'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

feats = cnn.get_features([image_path], layers='conv5_3', layer_sizes=[512,14,14])
np.save(feat_path, feats)

