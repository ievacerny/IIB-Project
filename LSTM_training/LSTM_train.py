"""Script fo training the gesture recognition LSTM network."""
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from collections import Counter

from prepare_inputs import (
    calculate_features,
    load_LMDHG_from_file,
    load_DHG_dataset,
    load_my_dataset
)

# ------------------------ PARAMETERS -----------------------------------------
# Network parameters
frame_step = 1  # Used to change the frame rate of the input
batch_size = 10
learning_rate = 0.0005
training_iters = 1000
display_step = 50
testing_iters = 50
# Dimensionality parameters
n_frames = 40
n_dimension = 24
n_output = 15
n_hidden = 512  # Dimension of the hidden state

np.random.seed(10)

# ------------------------ SAVE MODEL -----------------------------------------
# export_dir = 'model_{}'.format(int(time.time()))
# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


# ------------------------ LOAD DATA ------------------------------------------
def load_LMDHG(training_prop):
    """Load and split LMDHG data into training and testing data."""
    all_gestures = []
    all_labels = []
    for i in range(1, 51):
        new_gestures, new_labels = load_LMDHG_from_file(i)
        all_gestures.append(new_gestures)
        all_labels.append(new_labels)
    training_data = (
        np.concatenate(all_gestures[:int(training_prop*len(all_gestures))]),
        np.concatenate(all_labels[:int(training_prop*len(all_gestures))])
    )
    testing_data = (
        np.concatenate(all_gestures[int(training_prop*len(all_gestures)):]),
        np.concatenate(all_labels[int(training_prop*len(all_gestures)):])
    )
    print(training_data[0].shape, testing_data[0].shape)
    return training_data, testing_data


def load_custom(training_prop):
    """Load and split my own data into training and testing data."""
    all_gestures = []
    all_labels = []
    for i in range(1, 6):
        new_gestures, new_labels = load_my_dataset(i)
        all_gestures.append(new_gestures)
        all_labels.append(new_labels)
    training_data = (
        np.concatenate(all_gestures[:int(training_prop*len(all_gestures))]),
        np.concatenate(all_labels[:int(training_prop*len(all_gestures))])
    )
    testing_data = (
        np.concatenate(all_gestures[int(training_prop*len(all_gestures)):]),
        np.concatenate(all_labels[int(training_prop*len(all_gestures)):])
    )
    print(training_data[0].shape, testing_data[0].shape)
    return training_data, testing_data


def load_DHG(training_prop):
    """Load and split DHG data into training and testing data.

    ************Not adapted for the new network***********
    """
    all_gestures, all_labels = load_DHG_dataset()
    indices = np.arange(len(all_gestures))
    np.random.shuffle(indices)
    training_data = (
        all_gestures[indices[:int(training_prop*len(all_gestures))]],
        all_labels[indices[:int(training_prop*len(all_gestures))]]
    )
    testing_data = (
        all_gestures[indices[int(training_prop*len(all_gestures)):]],
        all_labels[indices[int(training_prop*len(all_gestures)):]]
    )
    return training_data, testing_data


# Load training data
start_time = time.time()
training_data, testing_data = load_custom(0.8)
print("Loaded training data in {} seconds".format(time.time() - start_time))

# import sys
# sys.exit("Stop point")


# ------------------------ NETWORK DEFINITION --------------------------------
def LSTM(x, weights, biases):
    """Define network structure - predicition function."""
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell, output_keep_prob=0.5)
    # Requires a sequence (type list or tuple) of inputs, each of n_dimension
    x = tf.reshape(x, [-1, n_frames * n_dimension])
    x = tf.split(x, n_frames, 1)
    # outputs - sequence of outputs (of h_ts), only need the last one
    # state - C_t of last time step
    outputs, states = tf.nn.static_rnn(LSTM_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Input data x and output label pdf y to be fed into LSTM during training
# (batch size, sequence length * single datapoint dimension)
x = tf.placeholder("float", [None, n_frames, n_dimension], name="myInput")
# (batch size, number fo categories)
y = tf.placeholder("float", [None, n_output])

# Linear regression layer parameters to be optimized
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_output]))}
biases = {'out': tf.Variable(tf.random_normal([n_output]))}

# Prediction
pred = tf.identity(LSTM(x, weights, biases), name="myOutput")

# Loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialisation method
init = tf.global_variables_initializer()


# ------------------ PREPARE INPUT AND OUTPUT --------------------------------
def get_window_start(testing=False):
    """Generate exhaustively random window_start positions."""
    if testing:
        data = testing_data
    else:
        data = training_data
    if frame_step > 1:
        start_positions = np.arange(
            0,
            len(data[0])-n_frames*frame_step,
            int(frame_step/2))
    else:
        start_positions = np.arange(
            0,
            len(data[0])-n_frames*frame_step)
    np.random.shuffle(start_positions)
    for position in start_positions:
            yield position
    print("Done with all videos")


def get_data(generator, testing=False):
    """Get the input and label arrays for a window starting at window_start."""
    if testing:
        data = testing_data
    else:
        data = training_data
    # Input
    window_start = next(generator)
    frames = data[0][
        window_start:window_start+n_frames*frame_step:frame_step]
    features = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for frame in frames:
            features.append(calculate_features(frame))
    if len(frames) != n_frames:
        raise Exception("Number of frames is not as expected: {}"
                        .format(len(frames)))
    # Labels
    onehot_label = np.zeros((1, n_output), dtype=float)
    labels = data[1][
        window_start:window_start+n_frames:frame_step]
    actual_label = Counter(labels).most_common(1)[0][0]
    if actual_label > 14:
        print(actual_label)
        print(Counter(labels).most_common(1))
        raise ValueError("Multiple most common")
    onehot_label[0, actual_label] = 1.0

    return np.array(features), onehot_label


# ------------------------ RUN SESSION ----------------------------------------
start_time = time.time()
loss_plot = np.zeros(training_iters)

with tf.Session() as session:
    session.run(init)
    step = 0
    # Reporting variables
    acc_total = 0
    loss_total = 0
    accuracy_graph_train = []
    accuracy_graph_test = []
    train_gen = get_window_start()
    test_gen = get_window_start(True)

    while step < training_iters:

        # ------------- Training
        input_batch = np.zeros((batch_size, n_frames, n_dimension))
        output_batch = np.zeros((batch_size, n_output))
        for i in range(batch_size):
            input_batch[i, :, :], output_batch[i, :] = get_data(train_gen)

        # Evaluate the optimizer, accuracy, cost and pred
        _, acc, loss, onehot_pred = session.run(
            [optimizer, accuracy, cost, pred],
            feed_dict={x: input_batch, y: output_batch})

        loss_plot[step] = loss
        step += 1

        # Code for reporting (actually it's like epoch)
        # -------------- Epoch end + testing
        loss_total += loss
        acc_total += acc
        if (step) % display_step == 0:
            # Report numbers
            print("Iter= " + str(step) +
                  ", Average Loss in {} steps= ".format(display_step) +
                  "{:.6f}".format(loss_total/display_step) +
                  ", Average Accuracy in {} steps= ".format(display_step) +
                  "{:.2f}%".format(100*acc_total/display_step))
            actual_label = np.argmax(output_batch[:, :], axis=1)
            onehot_label_pred = tf.argmax(onehot_pred, 1).eval()
            print("[%s] predicted vs [%s]" %
                  (onehot_label_pred, actual_label))
            accuracy_graph_train.append(acc_total/display_step)
            # Do some validation after this epoch
            accuracy_testing = 0
            for t in range(testing_iters):
                features, onehot_label = get_data(test_gen, testing=True)
                features = np.reshape(features, [-1, n_frames, n_dimension])
                onehot_pred, acc = session.run(
                    [pred, accuracy],
                    feed_dict={x: features, y: onehot_label})
                accuracy_testing += acc
            print("Validation accuracy: {}".format(
                accuracy_testing/testing_iters))
            accuracy_graph_test.append(accuracy_testing/testing_iters)
            loss_total = 0
            acc_total = 0

    print("Optimization Finished!")
    print("Elapsed time: ", time.time() - start_time)

    # -------------------- SAVE MODEL ----------------------------------------
    # signature = tf.saved_model.predict_signature_def(
    #     inputs={'myInput': x},
    #     outputs={'myOutput': pred})
    # using custom tag instead of: tags=[tag_constants.SERVING]
    # builder.add_meta_graph_and_variables(
    #     sess=session,
    #     tags=["myTag"],
    #     signature_def_map={'predict': signature})
    # builder.save()

    # ---------------------PLOT GRAPHS ----------------------------------------
    plt.figure()
    plt.plot(np.arange(0, len(loss_plot)), loss_plot)
    plt.ylim([0, 5])
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()
    plt.plot(np.arange(0, len(accuracy_graph_test)), accuracy_graph_test,
             label="Test")
    plt.plot(np.arange(0, len(accuracy_graph_train)), accuracy_graph_train,
             label="Train")
    plt.xlim([0, max(len(accuracy_graph_test), len(accuracy_graph_train))])
    plt.ylim([0, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.gca().legend()
    plt.show()
