import pandas as pd
import tensorflow as tf
import numpy as np
import  os
import math
from keras.preprocessing import sequence
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cPickle as pkl

import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io
from cnn_util import *

from PIL import Image

# feat_path = 'data/feats.npy'
annotation_path = 'data/annotations.pickle'
model_path = 'model-18'
test_feat = 'data/Train-1000092795.npy'
test_image_path = 'data/flickr30k_images/1000092795.jpg'
learning_rate = 0.01
# decay_rate = 
n_epochs = 1000    
dim_embed = 256
dim_ctx = 512
dim_hidden = 256
ctx_shape = [196,512]
n_lstm_steps = 30
batch_size = 1#80
stddev = 1.0
# bias_init_vector = None


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

print('.................Loading data.............')
# feats = np.load(feat_path)
annotation_data = pd.read_pickle(annotation_path)
captions = annotation_data['caption'].values
print(captions)
print('...............Done Loading...............')

wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

# np.save('data/ixtoword', ixtoword)
# print("no of words", len(wordtoix))
n_words = len(wordtoix)  #update this
maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
n_lstm_steps = maxlen + 1



graph = tf.Graph()
with graph.as_default():
    context = tf.placeholder(tf.float32,shape=(batch_size,ctx_shape[0],ctx_shape[1]))

    with tf.device("/cpu:0"):
        Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

    init_hidden_W = tf.Variable(tf.truncated_normal([dim_ctx, dim_hidden], stddev=stddev/math.sqrt(float(dim_ctx))), name='init_hidden_W')
    init_hidden_b = tf.Variable(tf.zeros([dim_hidden]), name='init_hidden_b')

    init_memory_W = tf.Variable(tf.truncated_normal([dim_ctx, dim_hidden], stddev=stddev/math.sqrt(float(dim_ctx))), name='init_memory_W')
    init_memory_b = tf.Variable(tf.zeros([dim_hidden]), name='init_memory_b')

    lstm_W = tf.Variable(tf.truncated_normal([dim_embed, dim_hidden*4], stddev=stddev/math.sqrt(float(dim_embed))), name='lstm_W')
    lstm_U = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden*4], stddev=stddev/math.sqrt(float(dim_hidden))), name='lstm_U')
    lstm_b = tf.Variable(tf.zeros([dim_hidden*4]), name='lstm_b')

    image_encode_W = tf.Variable(tf.truncated_normal([dim_ctx, dim_hidden*4], stddev=stddev/math.sqrt(float(dim_ctx))), name='image_encode_W')

    image_att_W = tf.Variable(tf.truncated_normal([dim_ctx, dim_ctx], stddev=stddev/math.sqrt(float(dim_ctx))), name='image_att_W')
    hidden_att_W = tf.Variable(tf.truncated_normal([dim_hidden, dim_ctx], stddev=stddev/math.sqrt(float(dim_hidden))), name='hidden_att_W')
    pre_att_b = tf.Variable(tf.zeros([dim_ctx]), name='pre_att_b')

    att_W = tf.Variable(tf.truncated_normal([dim_ctx, 1], stddev=stddev/math.sqrt(float(dim_ctx))), name='att_W')
    att_b = tf.Variable(tf.zeros([1]), name='att_b')

    decode_lstm_W = tf.Variable(tf.truncated_normal([dim_hidden, dim_embed], stddev=stddev/math.sqrt(float(dim_hidden))), name='decode_lstm_W')
    decode_lstm_b = tf.Variable(tf.zeros([dim_embed]), name='decode_lstm_b')

    decode_word_W = tf.Variable(tf.truncated_normal([dim_embed, n_words], stddev=stddev/math.sqrt(float(dim_embed))), name='decode_word_W')

    if bias_init_vector is not None:
        decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
    else:
        decode_word_b = tf.Variable(tf.zeros([n_words]), name='decode_word_b')


    def model(context):

        c = tf.nn.tanh(tf.matmul(tf.reduce_mean(context,1), init_memory_W) + init_memory_b)  #batchSize x 256
        h  = tf.nn.tanh(tf.matmul(tf.reduce_mean(context,1), init_hidden_W) + init_hidden_b)  #batchSize x 256
        loss = 0
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([batch_size, dim_embed])  #batchSize x 256
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for ind in range(n_lstm_steps):

                context_encode = tf.reshape(context,[-1,512])  #batchSize*196 x 512
                context_encode = tf.matmul(context_encode, image_att_W) #batchSize*196 x 512
                context_encode = tf.reshape(context_encode,[-1,196,512]) #batchSize x 196 x 512

                ################################Computing the attention weights############################################
                
                h1 = tf.expand_dims(tf.matmul(h,hidden_att_W),1) #batchSize x 1 x 512
                eti = context_encode + h1 + pre_att_b   #batchSize x 196 x 512
                eti = tf.nn.tanh(eti) #batchSize x 196 x 512
                eti = tf.reshape(eti,[-1,512])  #batchSize*196 x 512
                eti = tf.matmul(eti,att_W) + att_b #batchSize*196 x 1
                eti = tf.reshape(eti,[-1,196])  #batchSize x 196
                alphas = tf.nn.softmax(eti)  #batchSize x 196 
                alpha_list.append(alphas)
                ##########################################################################################################

                ##############################Computing the LSTM gate outputs###########################################
                z = tf.reduce_sum(tf.multiply(context,tf.expand_dims(alphas,2)),1) #batchSize x 512

                xt = tf.matmul(word_emb, lstm_W) + lstm_b     #batchSize x dim_hidden*4
                ensemble = xt + tf.matmul(h,lstm_U) + tf.matmul(z,image_encode_W)      #batchSize x dim_hidden*4
                it, ft, ot, gt = tf.split(ensemble,4,1)
                it = tf.nn.sigmoid(it)
                ft = tf.nn.sigmoid(ft)
                ot = tf.nn.sigmoid(ot)
                gt = tf.nn.tanh(gt)
                c = tf.multiply(ft,c) + tf.multiply(it,gt) #batchSize x 256 -> ft
                h = tf.multiply(ot,tf.nn.tanh(c))         #batchSize x 256
                ########################################################################################################

                logits = tf.matmul(h, decode_lstm_W) + decode_lstm_b
                logits = tf.nn.relu(logits)

                logit_words = tf.matmul(logits, decode_word_W) + decode_word_b
                max_prob_word = tf.argmax(logit_words, 1)
                
                with tf.device("/cpu:0"):
                    word_emb = tf.nn.embedding_lookup(Wemb, max_prob_word)

                generated_words.append(max_prob_word)
                logit_list.append(logit_words)

            return context, generated_words, logit_list, alpha_list
    context, generated_words, logit_list, alpha_list = model(context)    

        

##########################################Testing Process Begins############################################################

sess = tf.InteractiveSession()

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess,model_path)

    feat = np.load(test_feat).reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)


    generated_word_index, alpha_list = sess.run([generated_words,alpha_list], feed_dict={context:feat})
    # alpha_list_val = sess.run([alpha_list], feed_dict={context:feat})
    # print(alpha_list)
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1
    print(generated_words)
    generated_words = generated_words[:punctuation]
    alpha_list = alpha_list[:punctuation]
    print(generated_words)
    img = mpimg.imread(test_image_path)
    # plt.imshow(img)
    # plt.title(' '.join(generated_words))
    # plt.show()
    # alpha_list_val = alpha_list_val[:punctuation]
    # generated_words, alpha_list_val
    print(len(alpha_list))
    img = crop_image(test_image_path)

    alphas = np.array(alpha_list).swapaxes(1,2)
    print(alphas.shape)
    n_words = alphas.shape[0] + 1
    w = np.round(np.sqrt(n_words))
    h = np.ceil(np.float32(n_words) / w)

    plt.subplot(w, h, 1)
    plt.imshow(img)
    plt.axis('off')
    # plt.show()
    smooth = True

    for ii in xrange(alphas.shape[0]):
        plt.subplot(w, h, ii+2)
        lab = generated_words[ii]
    
        plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, lab, color='black', fontsize=13)
        plt.imshow(img)
    
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alphas[ii,:,0].reshape(14,14), upscale=16, sigma=20)
        else:
            alpha_img = skimage.transform.resize(alphas[ii,:,0].reshape(14,14), [img.shape[0], img.shape[1]])
        
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

    