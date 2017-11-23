import pandas as pd
import tensorflow as tf
import numpy as np
import  os
import math
from keras.preprocessing import sequence

feat_path = 'data/feats.npy'
annotation_path = 'data/annotations.pickle'
model_path = 'modelNewFinal/'

learning_rate = 0.001
# decay_rate = 
n_epochs = 1000    
dim_embed = 256
dim_ctx = 512
dim_hidden = 256
ctx_shape = [196,512]
n_lstm_steps = 30
batch_size = 80
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
feats = np.load(feat_path)
annotation_data = pd.read_pickle(annotation_path)
captions = annotation_data['caption'].values
print(captions)
print('...............Done Loading...............')

wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

np.save('data/ixtoword', ixtoword)
# print("no of words", len(wordtoix))
n_words = len(wordtoix)  #update this
maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
n_lstm_steps = maxlen + 1



graph = tf.Graph()
with graph.as_default():
    context = tf.placeholder(tf.float32,shape=(batch_size,ctx_shape[0],ctx_shape[1]))
    sentence = tf.placeholder(tf.int32, shape=(batch_size,n_lstm_steps))
    mask = tf.placeholder(tf.float32,shape=(batch_size,n_lstm_steps))

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


    def model(context,sentence,mask):

        c0 = tf.nn.tanh(tf.matmul(tf.reduce_mean(context,1), init_memory_W) + init_memory_b)  #batchSize x 256
        h0  = tf.nn.tanh(tf.matmul(tf.reduce_mean(context,1), init_hidden_W) + init_hidden_b)  #batchSize x 256
        loss = 0
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for ind in range(n_lstm_steps):
                if ind == 0:    
                    #if first step, then the word_emb input to LSTM would be zeros
                    word_emb = tf.zeros([batch_size, dim_embed])  #batchSize x 256
                    #First step, c0 and h0, computed above is used as c and h
                    c = c0
                    h = h0
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        word_emb = tf.nn.embedding_lookup(Wemb, sentence[:,ind-1])

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
                logits = tf.nn.dropout(logits, 0.5)

                logit_words = tf.matmul(logits, decode_word_W) + decode_word_b

                labels = tf.expand_dims(sentence[:,ind], 1)
                indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
                concated = tf.concat([indices, labels],1)
                onehot_labels = tf.sparse_to_dense( concated, tf.stack([batch_size, n_words]), 1.0, 0.0)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * mask[:,ind]

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss

    loss = model(context,sentence,mask)
    global_step = tf.Variable(50)  # count the number of steps taken.
    # learning_rate = tf.train.exponential_decay(learning_rate, global_step, 50,0.9)  #learning_rate*0.97^(global_step/50)
    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        

##########################################Training Process Begins############################################################

sess = tf.InteractiveSession()

index = list(annotation_data.index)
# print(index)
np.random.shuffle(index)
annotation_data = annotation_data.ix[index]

captions = annotation_data['caption'].values
# print(captions)
image_id = annotation_data['image_id'].values
with tf.device('/device:GPU:2'):
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver(max_to_keep=50)

        tf.initialize_all_variables().run()

        for epoch in range(n_epochs):
            for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

                current_feats = feats[image_id[start:end]]
                current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)

                current_captions = captions[start:end]
                current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

                current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
                # print(current_caption_matrix)
                current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
                nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

                for ind, row in enumerate(current_mask_matrix):
                    row[:nonzeros[ind]] = 1
                
                _, loss_value = sess.run([optimizer, loss], feed_dict={
                    context:current_feats,
                    sentence:current_caption_matrix,
                    mask:current_mask_matrix})

                print "Current Cost: ", loss_value
            # learning_rate *= 0.95    

            print ("Epoch ", epoch, " is done. Saving the model ... ")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
