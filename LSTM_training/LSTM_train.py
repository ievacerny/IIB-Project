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
    load_my_dataset,
    set_mapping
)

# ------------------------ PARAMETERS -----------------------------------------
# Network parameters
frame_step = 5  # Used to change the frame rate of the input
batch_size = 10
learning_rate = 0.0005
training_iters = 1000
display_step = 50
testing_iters = 50
final_testing_iters = 500
# Dimensionality parameters
n_frames = 8
n_dimension = 40
n_output = 6
n_hidden = 512  # Dimension of the hidden state


# ------------------------ SAVE MODEL -----------------------------------------
# export_dir = 'model_{}'.format(int(time.time()))
export_dir = "model_I-1000_LD-2_S-5_F-8"
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


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
    file_numbers = list(range(1, 21))
    np.random.shuffle(file_numbers)
    print(file_numbers)
    for i in file_numbers:
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
U, S, V = np.linalg.svd(training_data[0])
print("Loaded training data in {} seconds".format(time.time() - start_time))


# ------------------------ HYPERPARAMETERS --------------------------------
# for learning_rate in learning_rate_opts:
#     for n_frames in n_frames_opts:
#         for n_dimension in n_dimension_opts:
# for learning_rate, n_frames, n_dimension in option_sets:
# for learning_rate in learning_rates:

# tf.reset_default_graph()
# hyper_string = "L-{0}_F-{1}_D-{2}_Gest-".format(
#     learning_rate, n_frames, n_dimension)

set_mapping(V[:n_dimension, :])
print(training_data[0].shape, testing_data[0].shape)
np.random.seed(10)
np.savetxt(export_dir+'/mapping.csv', V[:n_dimension, :],
           delimiter=',', fmt='%f')


# ------------------------ NETWORK DEFINITION --------------------------------
def LSTM(x, weights, biases):
    """Define network structure - predicition function."""
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell,output_keep_prob=0.5)
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

# Summary collections
tf.summary.histogram("true_output", y)
tf.summary.histogram("weights", weights["out"])
tf.summary.histogram("biases", biases["out"])
tf.summary.histogram("prediction", pred)
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)

# Initialisation method
init = tf.global_variables_initializer()


# ------------------ PREPARE INPUT AND OUTPUT --------------------------------
def get_window_start(testing=False):
    """Generate exhaustively random window_start positions."""
    if testing:
        data = testing_data
    else:
        data = training_data
    epoch = 0
    while True:
        if frame_step > 1:
            start_positions = np.arange(
                0,
                len(data[0])-n_frames*frame_step)
        else:
            start_positions = np.arange(
                0,
                len(data[0])-n_frames*frame_step)
        np.random.shuffle(start_positions)
        for position in start_positions:
                yield position
        epoch += 1
        print("EPOCH {}. Done with all videos".format(epoch))


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
    for frame in frames:
        features.append(calculate_features(frame))
    # Labels
    onehot_label = np.zeros((1, n_output), dtype=float)
    labels = data[1][
        window_start:window_start+n_frames:frame_step]
    # actual_label = Counter(labels).most_common(1)[0][0]
    actual_label = labels[-2]
    # actual_label = np.bincount(labels, minlength=n_output) / n_frames
    # actual_label = np.reshape(actual_label, [1, n_output])
    onehot_label[0, actual_label] = 1.0

    return np.array(features), onehot_label


# ---------------------- TRAIN AND TEST FUNCTIONS -----------------------------
def train(session, train_generator, acc_total, loss_total, writer, it=0):
    """Do one optimization step."""
    input_batch = np.zeros((batch_size, n_frames, n_dimension))
    output_batch = np.zeros((batch_size, n_output))
    for i in range(batch_size):
        input_batch[i, :, :], output_batch[i, :] = get_data(train_generator)

    if (it+1) % display_step == 0:
        merge = tf.summary.merge_all()
        # Evaluate the optimizer, accuracy, cost and pred
        summary, _, acc, loss, onehot_pred = session.run(
            [merge, optimizer, accuracy, cost, pred],
            feed_dict={x: input_batch, y: output_batch})
        writer.add_summary(summary, it)
    else:
        # Evaluate the optimizer, accuracy, cost and pred
        _, acc, loss, onehot_pred = session.run(
            [optimizer, accuracy, cost, pred],
            feed_dict={x: input_batch, y: output_batch})

    loss_total += loss
    acc_total += acc

    return output_batch, onehot_pred, acc_total, loss_total


def epoch_test(session, test_generator):
    """Do testing after one epoch."""
    accuracy_testing = 0
    for t in range(testing_iters):
        features, onehot_label = get_data(test_generator, testing=True)
        features = np.reshape(features, [-1, n_frames, n_dimension])
        onehot_pred, acc = session.run(
            [pred, accuracy],
            feed_dict={x: features, y: onehot_label})
        accuracy_testing += acc

    return accuracy_testing/testing_iters


def final_test(session, test_generator, save=False, writers=None, step=0):
    """Do final testing after training has finished."""
    accuracy_testing = 0
    # Rows: TP, TN, FP, FN
    rec_numbers = np.zeros((4, n_output), dtype='int32')
    gesture_counts = np.zeros(n_output, dtype='int32')
    for t in range(final_testing_iters):
        # Predict
        features, onehot_label = get_data(test_generator, testing=True)
        features = np.reshape(features, [-1, n_frames, n_dimension])
        onehot_pred, acc = session.run(
            [pred, accuracy],
            feed_dict={x: features, y: onehot_label})
        # Get labels
        actual_label = np.argmax(onehot_label, axis=1)[0]
        pred_label = tf.argmax(onehot_pred, 1).eval()[0]
        # Update recognition numbers and gesture counts
        gesture_counts[actual_label] += 1
        if actual_label == pred_label:
            rec_numbers[0, actual_label] += 1
            # True negatives calculated later
        else:
            rec_numbers[3, actual_label] += 1
            rec_numbers[2, pred_label] += 1
        accuracy_testing += acc
    # Calculate true negatives
    rec_numbers[1, :] = final_testing_iters - np.sum(
        rec_numbers[[0, 2, 3], :], axis=0)

    # Calculate recognition rate, TPR and FPR
    rr = (rec_numbers[0, :] + rec_numbers[1, :]) / final_testing_iters
    tpr = rec_numbers[0, :] / (rec_numbers[0, :] + rec_numbers[3, :])
    fpr = rec_numbers[2, :] / (rec_numbers[1, :] + rec_numbers[2, :])

    if writers is not None:
        test_summary = tf.Summary()
        test_summary.value.add(
            tag='Overall gesture accuracy', simple_value=np.sum(rr[1:]))
        test_summary.value.add(
            tag='Overall TPR', simple_value=np.sum(tpr[1:]))
        test_summary.value.add(
            tag='Overall FPR', simple_value=np.sum(fpr[1:]))
        writers[0].add_summary(test_summary, step)
        for i in range(1, 6):
            test_summary = tf.Summary()
            test_summary.value.add(
                tag='Gesture accuracy', simple_value=rr[i])
            test_summary.value.add(
                tag='Gesture TPR', simple_value=tpr[i])
            test_summary.value.add(
                tag='Gesture FPR', simple_value=fpr[i])
            writers[i].add_summary(test_summary, step)

    # Report
    report_string = (
        "---------- Testing accuracy {}\n".format(accuracy_testing/final_testing_iters) +  # noqa
        "   " + " ".join("{:>8}".format(n) for n in np.arange(0, n_output)) + "\n"  # noqa
        "GC " + " ".join("{:>8}".format(n) for n in gesture_counts) + "\n"
        "--\n"
    )
    for i, term in enumerate(["TP", "TN", "FP", "FN"]):
        report_string = report_string + "{0} {1}\n".format(
            term,
            " ".join("{:>8}".format(n) for n in rec_numbers[i, :]))
    report_string = report_string + (
        "--\n"
        "RR " + " ".join("{:>8.5f}".format(n) for n in rr) + "\n"
        "TPR" + " ".join("{:>8.5f}".format(n) for n in tpr) + "\n"
        "FPR" + " ".join("{:>8.5f}".format(n) for n in fpr) + "\n"
    )
    if save:
        with open(export_dir+"/test.txt", 'w+') as f:
            f.write(report_string)
    else:
        print(report_string)


def plot_graphs(accuracy_train, accuracy_test, loss, save=False):
    """Plot loss and accuracy graphs."""
    plt.figure()
    plt.plot(np.arange(0, len(loss)), loss)
    plt.ylim([0, 3])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if save:
        plt.savefig(export_dir+"/loss.png")
    else:
        plt.show()

    plt.figure()
    plt.plot(np.arange(0, len(accuracy_test)), accuracy_test, label="Test")
    plt.plot(np.arange(0, len(accuracy_train)), accuracy_train, label="Train")
    plt.xlim([0, max(len(accuracy_test), len(accuracy_train))])
    plt.ylim([0, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.gca().legend()
    if save:
        plt.savefig(export_dir+"/accuracy.png")
    else:
        plt.show()


# ------------------------ RUN SESSION ----------------------------------------
start_time = time.time()

# Data collection for graphs
accuracy_graph_train = []
accuracy_graph_test = []
loss_graph = []

with tf.Session() as session:
    # Initialise
    session.run(init)
    step, acc_total, loss_total = 0, 0, 0
    train_gen = get_window_start()
    test_gen = get_window_start(True)
    # ------------------------ Logging
    writer = tf.summary.FileWriter(
        export_dir+"/logs",
        session.graph)

    while step < training_iters:
        # ------------- Training
        true_output, prediction, acc_total, loss_total = train(
            session, train_gen, acc_total, loss_total, writer, step)
        step += 1
        # -------------- Epoch end
        if step % display_step == 0:
            # Report last iteration prediction
            actual_label = np.argmax(true_output[:, :], axis=1)
            onehot_label_pred = tf.argmax(prediction, 1).eval()
            print("Iter={}:\n".format(step) +
                  "   {} predicted\n".format(
                    " ".join("{:>3}".format(l) for l in onehot_label_pred)) +
                  "   {} true".format(
                    " ".join("{:>3}".format(l) for l in actual_label)))
            # Reset numbers
            average_loss = loss_total/display_step
            average_acc = acc_total/display_step
            acc_total, loss_total = 0, 0
            # -------------- Testing
            average_test_acc = epoch_test(session, test_gen)
            test_summary = tf.Summary()
            test_summary.value.add(
                tag='Test accuracy', simple_value=average_test_acc)
            writer.add_summary(test_summary, step)
            # -------------- Report
            print("Iter={}, ".format(step) +
                  "Average loss={:.6f}, ".format(average_loss) +
                  "Average accuracy={:.4f}, ".format(average_acc) +
                  "Validation accuracy={:.4f}".format(average_test_acc))
            # Save numbers for plotting
            accuracy_graph_train.append(average_acc)
            accuracy_graph_test.append(average_test_acc)
            loss_graph.append(average_loss)

    print("Elapsed time: ", time.time() - start_time)

    # -------------- After training validation
    final_test(session, test_gen, save=True)
    # -------------- Plot loss and accuracy
    plot_graphs(accuracy_graph_train, accuracy_graph_test, loss_graph,
                save=True)

    # -------------------- SAVE MODEL ----------------------------------------
    signature = tf.saved_model.predict_signature_def(
        inputs={'myInput': x},
        outputs={'myOutput': pred})
    builder.add_meta_graph_and_variables(
        sess=session,
        tags=["myTag"],
        signature_def_map={'predict': signature})
    builder.save()
