"""Script fo training the gesture recognition LSTM network."""
import numpy as np
import tensorflow as tf
import time
import random
import matplotlib.pyplot as plt

from prepare_inputs import (
    calculate_temporal_features,
    load_LMDHG_from_file,
    load_DHG_dataset
)


def addNameToTensor(someTensor, theName):
    """Add tags to tensors for model export."""
    return tf.identity(someTensor, name=theName)


# Save model
export_dir = 'model_{}'.format(int(time.time()))
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


# Load training data
start_time = time.time()

# LMDHG
# all_gestures = []
# all_labels = []
# for i in range(1, 51):
#     new_gestures, new_labels = load_LMDHG_from_file(i)
#     all_gestures.append(new_gestures)
#     all_labels.append(new_labels)
# all_gestures = np.concatenate(all_gestures)
# all_labels = np.concatenate(all_labels)
# training_data_len = int(0.9 * len(all_gestures))

# DHG
all_gestures, all_labels = load_DHG_dataset()
training_data_len = len(all_gestures)-1

print("Loaded training data in {} seconds".format(time.time() - start_time))

# LMDHG label dictionary
label_dict = {
    "REPOS": 0,  # NAG
    "POINTER": 1,  # Point to
    "POINTER_PROLONGE": 1,  # "Point extended"???
    "ATTRAPER": 2,  # Catch
    "SECOUER_POING_LEVE": 3,  # Shake with 2 hands
    "ATTRAPER_MAIN_LEVEE": 4,  # Catch with 2 hands
    "SECOUER_BAS": 5,  # Shake down
    "SECOUER": 6,  # Shake
    "C": 7,  # Draw C
    "POINTER_MAIN_LEVEE": 8,  # Point to with 2 hands
    "ZOOM": 9,  # Zoom
    "DEFILER_DOIGT": 10,  # Scroll
    "LIGNE": 11,  # Draw line
    "TRANCHER": 12,  # Slice
    "PIVOTER": 13,  # Rotate
    "CISEAUX": 12,  # scissors?????
}

# Define parameters
window_size = 400  # the actual size will be window_size/frame_step
frame_step = 1
batch_size = 3
learning_rate = 0.001
training_iters = 100
display_step = 10
n_input = 228
# n_output = len(label_dict)  # LMDHG n_output
n_output = max(all_labels)+1  # DHG n_output
print(n_output)
# number of units in RNN cell
# (dimensionality of the hidden and the output states)
n_hidden = 512
random.seed(10)

# tf Graph input
"""(sample, time_steps, features) represents the tensor you will feed into LSTM
sample: size of your minibatch: How many examples you give at once
time_steps: length of a sequence
features: dimension of each element of the time-series."""
x = tf.placeholder("float", [None, n_input, 1], name="myInput")
"""(sample, number_of_categories)"""
y = tf.placeholder("float", [None, n_output])

# RNN output node weights and biases (transform output to probability vector)
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_output]))}
biases = {'out': tf.Variable(tf.random_normal([n_output]))}


def LSTM(x, weights, biases):
    """Define the network."""
    x = tf.reshape(x, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # outputs is a list of outputs for each input
    # state is the final state
    outputs, states = tf.nn.static_rnn(LSTM_cell, x, dtype=tf.float32)
    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def get_training_data():
    """Get the input and label arrays for one training sample."""
    gesture_no = random.randint(0, training_data_len)

    # Generate input vector of dimension [1, n_input]
    frames = all_gestures[gesture_no]
    while len(frames) < 4:
        gesture_no = random.randint(0, training_data_len)
        frames = all_gestures[gesture_no][::frame_step]
    with np.errstate(divide='ignore', invalid='ignore'):
        # print(len(frames), gesture_no, all_labels[gesture_no])
        features = calculate_temporal_features(frames)
    if len(features[0]) != n_input:
        raise Exception("Input dimension doesn't match {}"
                        .format(len(features[0])))

    # Build labels
    actual_label = all_labels[gesture_no]
    onehot_label = np.zeros(n_output, dtype=float)
    # onehot_label[label_dict[actual_label]] = 1.0
    onehot_label[actual_label] = 1.0
    onehot_label = np.reshape(onehot_label, [-1, n_output])

    return features, onehot_label


def get_test_data():
    """Get the input and actual label number for one test sample."""
    # gesture_no = random.randint(training_data_len+1, len(all_gestures)-1)
    # DHG gestures are not mixed. Can't as easily separate training and testing
    gesture_no = random.randint(0, training_data_len)

    # Generate input vector of dimension [1, n_input]
    frames = all_gestures[gesture_no][::frame_step]
    while len(frames) < 4:
        # gesture_no = random.randint(training_data_len+1, len(all_gestures)-1)
        gesture_no = random.randint(0, training_data_len)
        frames = all_gestures[gesture_no][::frame_step]
    with np.errstate(divide='ignore', invalid='ignore'):
        features = calculate_temporal_features(frames)
    if len(features[0]) != n_input:
        raise Exception("Input dimension doesn't match {}"
                        .format(len(features[0])))

    # Build labels
    # actual_label = label_dict[all_labels[gesture_no]]  # LMDHG
    actual_label = all_labels[gesture_no]  # DHG

    return features, actual_label


pred = LSTM(x, weights, biases)
addNameToTensor(pred, "myOutput")

# Loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
start_time = time.time()

# Launch the graph
loss_plot = np.zeros(training_iters)
with tf.Session() as session:
    session.run(init)
    step = 0
    # Reporting variables
    acc_total = 0
    loss_total = 0

    while step < training_iters:

        input_batch = np.zeros((batch_size, n_input, 1))
        output_batch = np.zeros((batch_size, n_output))
        for i in range(batch_size):
            input_batch[i, :, :], output_batch[i, :] = get_training_data()

        # Evaluate the optimizer, accuracy, cost and pred
        _, acc, loss, onehot_pred = session.run(
            [optimizer, accuracy, cost, pred],
            feed_dict={x: input_batch, y: output_batch})

        # Code for reporting
        loss_plot[step] = loss
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) +
                  ", Average Loss in {} steps= ".format(display_step) +
                  "{:.6f}".format(loss_total/display_step) +
                  ", Average Accuracy in {} steps= ".format(display_step) +
                  "{:.2f}%".format(100*acc_total/display_step))
            actual_label = np.argmax(output_batch[:, :], axis=1)
            onehot_label_pred = tf.argmax(onehot_pred, 1).eval()
            print("[%s] predicted vs [%s]" %
                  (onehot_label_pred, actual_label))
            loss_total = 0
            acc_total = 0

        step += 1

    print("Optimization Finished!")
    print("Elapsed time: ", time.time() - start_time)

    signature = tf.saved_model.predict_signature_def(
        inputs={'myInput': x},
        outputs={'myOutput': pred})
    # using custom tag instead of: tags=[tag_constants.SERVING]
    builder.add_meta_graph_and_variables(
        sess=session,
        tags=["myTag"],
        signature_def_map={'predict': signature})
    builder.save()

    # Test
    for t in range(10):
        features, actual_label = get_test_data()
        onehot_pred = session.run(pred, feed_dict={x: features})
        print(onehot_pred)
        onehot_label_pred = int(tf.argmax(onehot_pred, 1).eval())
        print("[%s] predicted vs [%s]" %
              (onehot_label_pred, actual_label))

    plt.plot(np.arange(0, len(loss_plot)), loss_plot)
    plt.ylim(bottom=0)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()
