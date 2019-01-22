"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow.

Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
import numpy as np
import tensorflow as tf
import random
import collections
import time


start_time = time.time()

# Target log path
# logs_path = 'rnn_logs'
# writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = 'belling_the_cat.txt'

# Save model
export_dir = 'LSTM_test/model_{}'.format(int(time.time()))
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


def read_data(fname):
    """Read text into array of symbols."""
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content))
               for word in content[i].split()]
    content = np.array(content)
    return content


# Load training data
training_data = read_data(training_file)
print("Loaded training data...")


def build_dataset(words):
    """Build dataset dictionary and reverse dictionary for inputs/outputs."""
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 3
# number of units in RNN cell
# (dimensionality of the hidden and the output states)
n_hidden = 512

# tf Graph input
"""
(sample, time_steps, features) represents the tensor you will feed into LSTM
sample: size of your minibatch: How many examples you give at once
time_steps: length of a sequence
features: dimension of each element of the time-series.
"""
x = tf.placeholder("float", [None, n_input, 1], name="myInput")
print(type(x))
"""(sample, number_of_categories)"""
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases (transform output to probability vector)
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def LSTM(x, weights, biases):
    """Define the network."""
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    # rnn_cell = rnn.MultiRNNCell(
    #     [rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

    # generate prediction
    # outputs is a list of outputs for each input
    # state is the final state
    outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = LSTM(x, weights, biases)
print(type(pred))


def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)


addNameToTensor(pred, "myOutput")

# Loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    # Reporting variables
    acc_total = 0
    loss_total = 0

    # writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        # Generate input vector of dimension [1, n_input]
        symbols_in_keys = [
            [dictionary[str(training_data[i])]]
            for i in range(offset, offset+n_input)]
        # Reshape input to [1, n_input, 1]
        symbols_in_keys = np.reshape(
            np.array(symbols_in_keys), [-1, n_input, 1])

        # Build labels (vector of word probabilities where (p(4th word)=1)
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0  # noqa
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        # Evaluate the optimizer, accuracy, cost and pred
        _, acc, loss, onehot_pred = session.run(
            [optimizer, accuracy, cost, pred],
            feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

        # Code for reporting
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) +
                  ", Average Loss in {} steps= ".format(display_step) +
                  "{:.6f}".format(loss_total/display_step) +
                  ", Average Accuracy in {} steps= ".format(display_step) +
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i]
                          for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[
                int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" %
                  (symbols_in, symbols_out, symbols_out_pred))

        step += 1
        offset += (n_input+1)

    signature = tf.saved_model.predict_signature_def(
        inputs={'myInput': x},
        outputs={'myOutput': pred})
    # using custom tag instead of: tags=[tag_constants.SERVING]
    builder.add_meta_graph_and_variables(
        sess=session,
        tags=["myTag"],
        signature_def_map={'predict': signature})
    builder.save()

    print("Optimization Finished!")
    print("Elapsed time: ", time.time() - start_time)

    # For visualisation follow these commands
    # print("Run on command line.")
    # print("\ttensorboard --logdir=%s" % (logs_path))
    # print("Point your web browser to: http://localhost:6006/")

    # Story generation
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])]
                               for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = ("%s %s" %
                            (sentence, reverse_dictionary[onehot_pred_index]))
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except KeyError:
            print("Word not in dictionary")
