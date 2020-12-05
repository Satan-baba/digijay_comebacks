#training on dig_comebacks.txt
#Add more data to the training file "dig comebacks.txt" to improve accuracy

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import collections
import time


def model(n_input, training_iters, LSTM_layers, n_hidden):
    tf.reset_default_graph()

    def time_elapsed(t):

        if t<60:
            return str(t) + " second(s)"
        elif t<(60*60):
            return str(t/60) + " minute(s)"
        else:
            return str(t/(60*60)) + " hour(s)"



    def read_data(fname):

        with open(fname) as f:
            content = f.readlines()
        
        content = [x.strip() for x in content]
        content = [word for i in range(len(content)) for word in content[i].split()]
        content = np.array(content)
        return content


   
    def build_dataset(words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
   
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    

    def module_RNN(x, weights, biases):

        x = tf.reshape(x, [-1, n_input])

        x = tf.split(x,n_input,1)

    
        if LSTM_layers == 1:
            rnn_cell = rnn.BasicLSTMCell(n_hidden)

        else:  
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    learning_rate = 0.001


    training_iters = training_iters

    display_step = 1000
    n_input = n_input
    n_output = 5
    n_hidden = n_hidden
    training_file = 'dig_comebacks.txt'
    

    training_data = read_data(training_file)
    print("Loaded training data...")

    dictionary, reverse_dictionary = build_dataset(training_data)
    vocab_size = len(dictionary)

    
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])

    logs_path = 'C:\dig LSTM'
    writer  = tf.summary.FileWriter(logs_path)

    
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))}
    biases  = {'out': tf.Variable(tf.random_normal([vocab_size]))}

    
    pred = module_RNN(x, weights, biases)

    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    
    init = tf.global_variables_initializer()

    start_time = time.time()

    
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0, n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)

            
            symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            
            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

            
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

            loss_total += loss
            acc_total += acc

            
            if (step+1) % display_step == 0:
                print("Iter = " + str(step+1) + ", average loss= " + \
                      "{:0.6f}".format(loss_total/display_step) + ", average accuracy= " + \
                      "{:0.2f}%".format(100*acc_total/display_step))
            
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]

                print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
            step += 1
            offset += (n_input+1)
        print("Optimization Finished!")

        t_elapsed = time.time() - start_time
        print("Elapsed time: ", time_elapsed(t_elapsed))

        prompt = "insert %s words: " % n_input
        if n_input == 1:
            sentence = "hi"
        elif n_input == 3:
            sentence = "how are you?"
        sentence = sentence.strip()
        words = sentence.split(' ')

    
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]

            # how many words

            for i in range(n_output):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)

            print(sentence)
        except:
            print("cannot be found in dictionary")




#Checking parameters

c = 0


for LSTM_layer in (1,2):
    for n_hidden in (2,3,4):
        c = c+1
        model(1, 10000, LSTM_layer, n_hidden)
        print((1, 10000, LSTM_layer, n_hidden))
        print("\n" + "--------" + str(c) + "-----------------" + "\n")

# learning_rate = 0.001
# training_iters = 8000
# display_step = 10
# n_input = 1
# n_output = 8
# n_hidden = 4            
# model(n_input, training_iters, 1, n_hidden)
