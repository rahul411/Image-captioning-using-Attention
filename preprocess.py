import pandas as pd
import numpy as np
import os
from utilities import *

vgg_model = 'data/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = 'data/VGG_ILSVRC_19_layers_deploy.prototxt'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

annotation_path = 'data/results_20130124.token'
flickr_image_path = 'data/flickr30k_images/'
feat_path = 'data/features.npy'
annotation_result_path = 'data/annotations.pickle'


annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

unique_images = annotations['image'].unique()
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)

feats = cnn.get_features(unique_images)
np.save(feat_path, feats)
