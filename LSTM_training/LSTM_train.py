"""Script fo training the gesture recognition LSTM network."""
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from os.path import join as pjoin
import scipy
from collections import Counter  # noqa

from prepare_inputs import (
    calculate_features,
    # load_LMDHG_from_file,
    # load_DHG_dataset,
    # load_my_dataset,
    set_mapping,
)

LOGGING = True


# ------------------------ PARAMETERS -----------------------------------------
class Config():
    """Container of all needed parameters."""

    def __init__(
        self,
        # Network parameters
        frame_step=1,
        batch_size=10,
        learning_rate=0.001,
        training_iters=5000,
        display_step=50,
        testing_iters=50,
        final_testing_iters=2000,
        # Dimensionality parameters
        n_frames=6,
        n_dimension=40,
        n_output=6,
        n_hidden=512,  # Dimension of the hidden state
        delay=3,
    ):
        """Initialise the parameters."""
        self.frame_step = frame_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.display_step = display_step
        self.testing_iters = testing_iters
        self.final_testing_iters = final_testing_iters
        self.n_frames = n_frames
        self.n_dimension = n_dimension
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.delay = delay
        self.validate_values()

    def validate_values(self):
        """Check if parameter values are valid."""
        if self.frame_step == 0:
            raise Exception("frame_step can't be zero.")
        if self.n_frames == 0:
            raise Exception("n_frames can't be zero.")


# Network parameters
frame_step = 8  # Used to change the frame rate of the input
batch_size = 10
learning_rate = 0.001
training_iters = 5000
display_step = 50
testing_iters = 50
final_testing_iters = 2000
# Dimensionality parameters
n_frames = 6
n_dimension = 40
n_output = 6
n_hidden = 512  # Dimension of the hidden state
delay = 3

np.random.seed(7)
# frame_step_opts = [3, 5, 8, 10]
# learning_rate_opts = [0.01, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]
# n_frames_opts = [6]
# delay_opts = [3]

# ------------------------ SAVE MODEL -----------------------------------------
# export_dir = 'model_{}'.format(int(time.time()))
# export_dir = "hyperparameters3/"
# export_dir = "model_I-5000_L-0.001_Random"
# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


def _log(*args):
    if LOGGING:
        print(args)


# ---------------------- SVD MAP CALCULATION ----------------------------------
def calculateMapping(data, data_folder=None, code_test=False):
    """Do SVD on specified data and save it to file.

    Only need to do this once for all of data. Then simply load it from file.
    """
    if data_folder is None:
        data_folder = pjoin("..", "Database", "MyDatabase")

    # Need to use a sparse algorithm because of memory limitations
    U, S, V = scipy.sparse.linalg.svds(data, k=min(data.shape)-1, which='LM')
    # Sparse algorithm doesn't sort the singular values, need to sort manually
    indices = np.argsort(S)[::-1]

    np.savetxt(pjoin(data_folder, "svd_V.csv"), V[indices, :], delimiter=",")
    np.savetxt(pjoin(data_folder, "svd_S.csv"), S[indices], delimiter=",")

    # If code testing return for inspection
    if code_test:
        return U[:, indices], S[indices], V[indices, :]


# ------------------- DATA LOADING FUNCTIONS ----------------------------------
def loadFiles(file_numbers, data_folder=None):
    """Load coordinates and labels from data files of specified numbers."""
    if data_folder is None:
        data_folder = pjoin("..", "Database", "MyDatabase")

    labels = np.empty(60000, dtype='int32')
    last_vid = 0
    for file_no in file_numbers:
        if file_no < 100:
            fname = "vid_{}_labels.txt".format(file_no)
        else:
            fname = "random_{}_labels.txt".format(file_no % 100)
        label_info = np.loadtxt(pjoin(data_folder, fname), dtype='int32')

        prev_start, prev_lbl = 0, 0
        for start, lbl in label_info:
            if start != 0:
                labels[prev_start+last_vid:start+last_vid] = prev_lbl
            prev_start, prev_lbl = start, lbl
        last_vid += start
    labels = labels[:last_vid]

    gestures = np.zeros((len(labels), 249))
    prev_video = 0
    for file_no in file_numbers:
        if file_no < 100:
            fname = "vid_{}.csv".format(file_no)
        else:
            fname = "random_{}.csv".format(file_no % 100)

        gestures_data = np.loadtxt(pjoin(data_folder, fname), delimiter=',')
        data_len = gestures_data.shape[0]
        gestures[prev_video:prev_video+data_len, :] = gestures_data
        prev_video += data_len

    if prev_video != len(gestures):
        raise Exception("Label and gesture files don't match in length.")

    return gestures, labels


def loadData(testing_prop, number_of_vids=None, data_folder=None):
    """Load and split my own data into training and testing data."""
    if number_of_vids is None:
        number_of_vids = (21, 107)

    file_numbers = list(range(1, number_of_vids[0]))
    np.random.shuffle(file_numbers)
    file_numbers.extend(list(range(101, number_of_vids[1])))
    no_testing = int(round(testing_prop*len(file_numbers)))

    if no_testing == 0:
        raise Exception("No videos for testing.")
    if no_testing == len(file_numbers):
        raise Exception("No videos for training.")

    training_data = loadFiles(file_numbers[no_testing:],
                              data_folder=data_folder)
    testing_data = loadFiles(file_numbers[:no_testing],
                             data_folder=data_folder)

    _log("Training: ", file_numbers[no_testing:])
    _log("Testing: ", file_numbers[:no_testing])

    return training_data, testing_data


def loadMapping(config, data_folder=None):
    """Read mapping from file and save it for feature calculations."""
    if data_folder is None:
        data_folder = pjoin("..", "Database", "MyDatabase")

    mapping = np.genfromtxt(pjoin(data_folder, "svd_V.csv"), delimiter=',')
    return mapping[:config.n_dimension, :].T


# ------------ INPUT AND OUTPUT PREPARATION FUNCTIONS -------------------------
def getWindowStart(config, testing=False):
    """Generate exhaustively random window_start positions. And repeats."""
    if testing:
        data = testing_data
    else:
        data = training_data
    start_positions = np.arange(
        (len(data[0]) + 1) -  # Range not inclusive (hence +1)
        config.n_frames -  # Number of needed frames
        (config.n_frames-1)*(config.frame_step-1)  # Gaps between frames
    )
    if len(start_positions) == 0:
        raise Exception("No starting positions in the generator.")
    epoch = 0
    while True:
        np.random.shuffle(start_positions)
        for position in start_positions:
                yield position
        epoch += 1
        _log("EPOCH {}. Done with all videos".format(epoch))


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
        window_start:window_start+n_frames*frame_step:frame_step]
    # actual_label = Counter(labels).most_common(1)[0][0]
    actual_label = labels[-1*delay]
    # actual_label = np.bincount(labels, minlength=n_output) / n_frames
    # actual_label = np.reshape(actual_label, [1, n_output])
    onehot_label[0, actual_label] = 1.0

    return np.array(features), onehot_label








if __name__ == '__main__':
    print("I'm here")
    # Load training data
    start_time = time.time()
    training_data, testing_data = load_custom(0.2)
    print("Loaded training data in {} seconds".format(time.time() - start_time))


    # ------------------------ HYPERPARAMETERS --------------------------------
    # for learning_rate in learning_rate_opts:
    #     for n_frames in n_frames_opts:
    #         for n_dimension in n_dimension_opts:
    # for learning_rate, n_frames, n_dimension in option_sets:
    # for learning_rate in learning_rates:

    # for n_frames in n_frames_opts:
    #     for learning_rate in learning_rate_opts:
    #     # for frame_step in frame_step_opts:
    #         for delay in delay_opts:

    # tf.reset_default_graph()
    # hyper_string = "F-{0}_S-{1}_DL-{2}_L-{3}_5000".format(
    #     n_frames, frame_step, delay, learning_rate)
    hyper_string = ""

    set_mapping(n_dimension)
    print(training_data[0].shape, testing_data[0].shape)
    np.random.seed(10)


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
            with open(export_dir + hyper_string + "/test.txt", 'w+') as f:
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
            plt.savefig(export_dir + hyper_string + "/loss.png")
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
            plt.savefig(export_dir + hyper_string + "/accuracy.png")
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
        train_gen = getWindowStart()
        test_gen = getWindowStart(True)
        # ------------------------ Logging
        writer = tf.summary.FileWriter(
            export_dir + hyper_string + "/logs",
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
