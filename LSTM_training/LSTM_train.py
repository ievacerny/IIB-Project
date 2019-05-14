"""Script fo training the gesture recognition LSTM network."""
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import scipy
import tensorflow as tf
import time

LOGGING = True


def _log(*args):
    if LOGGING:
        print(args)


# ------------------------ PARAMETERS -----------------------------------------
class Config():
    """Container of all needed parameters and certain ."""

    def __init__(
        self,
        # Network parameters
        frame_step=1,
        batch_size=10,
        learning_rate=0.001,
        training_iters=5000,
        display_step=50,
        testing_iters=50,
        final_testing_iters=50,  # Number of iterations per gesture
        # Dimensionality parameters
        n_frames=40,
        n_dimension=45,
        n_output=15,
        n_hidden=512,  # Dimension of the hidden state
        delay=13,
        mapping=None,
        label_type='delay',
        export_dir='model_{}'.format(int(time.time())),
        save_model=False,
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
        self.mapping = mapping
        self.label_type = label_type
        self.export_dir = export_dir
        self.validate_values()
        self.save_model = save_model
        if save_model:
            self.builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    def validate_values(self):
        """Check if parameter values are valid."""
        if self.frame_step == 0:
            raise Exception("frame_step can't be zero.")
        if self.n_frames == 0:
            raise Exception("n_frames can't be zero.")
        if self.label_type not in ['delay', 'majority', 'bincount']:
            raise Exception("Not a valid label_type. Choose from: " +
                            "'delay', 'majority', 'bincount'")

    def get_hyperstring(self):
        """Build and return a hyperparameter string."""
        return "LR-{0}_H-{1}_F-{2}_D-{3}_S-{4}_I-{5}".format(
            self.learning_rate,
            self.n_hidden,
            self.n_frames,
            self.n_dimension,
            self.frame_step,
            self.training_iters)


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


def getData(config, generator, testing=False):
    """Get the input and label arrays for a window starting at window_start."""
    if testing:
        data = testing_data
    else:
        data = training_data
    window_start = next(generator)
    # Input
    frames = data[0][
        window_start:
        window_start+config.n_frames*config.frame_step:
        config.frame_step]
    features = np.matmul(frames, config.mapping)
    # Labels
    frame_labels = data[1][
        window_start:
        window_start+config.n_frames*config.frame_step:
        config.frame_step]
    if config.label_type == "bincount":
        label = (np.bincount(frame_labels, minlength=config.n_output) /
                 config.n_frames)
        label = np.reshape(label, [1, config.n_output])
    else:
        label = np.zeros((1, config.n_output), dtype=float)
        if config.label_type == 'delay':
            actual_label = frame_labels[-1*config.delay]
        else:
            actual_label = Counter(frame_labels).most_common(1)[0][0]
        label[0, actual_label] = 1.0

    return features, label


# ---------------------- TRAIN AND TEST FUNCTIONS -----------------------------
def trainOneStep(session, train_generator, acc_total, loss_total, writer,
                 config, network, it=0):
    """Do one optimization step."""
    # Prepare input and output batches
    input_batch = np.zeros(
        (config.batch_size, config.n_frames, config.n_dimension))
    output_batch = np.zeros((config.batch_size, config.n_output))
    for i in range(config.batch_size):
        input_batch[i, :, :], output_batch[i, :] = getData(
            config, train_generator)
    # Every display_steps get the summary on the training progress
    if (it+1) % config.display_step == 0:
        merge = tf.summary.merge_all()
        summary, _, acc, loss, network_output = session.run(
            [merge, network.optimizer, network.accuracy,
             network.cost, network.pred],
            feed_dict={network.x: input_batch, network.y: output_batch})
        writer.add_summary(summary, it)
    # Otherwise, just do the training step
    else:
        _, acc, loss, network_output = session.run(
            [network.optimizer, network.accuracy, network.cost, network.pred],
            feed_dict={network.x: input_batch, network.y: output_batch})
    # Track total loss and accuracy
    loss_total += loss
    acc_total += acc

    return output_batch, network_output, acc_total, loss_total


def epoch_test(session, test_generator, config, network):
    """Do testing after one epoch."""
    accuracy_testing = 0
    for t in range(config.testing_iters):
        features, onehot_label = getData(config, test_generator, testing=True)
        features = np.reshape(features,
                              [-1, config.n_frames, config.n_dimension])
        onehot_pred, acc = session.run(
            [network.pred, network.accuracy],
            feed_dict={network.x: features, network.y: onehot_label})
        accuracy_testing += acc

    return accuracy_testing/config.testing_iters


def final_test(session, test_generator, config, network, save=False,
               writers=None, step=0):
    """Do final testing after training has finished."""
    accuracy_testing = 0
    # Rows: TP, TN, FP, FN
    rec_numbers = np.zeros((4, config.n_output), dtype='int32')
    gesture_counts = np.zeros(config.n_output, dtype='int32')
    confusion_matrix = np.zeros((config.n_output, config.n_output))
    total_iterations = 0
    while True:
        # Get data
        features, onehot_label = getData(config, test_generator, testing=True)
        # Check if need more predictions for this gesture
        actual_label = np.argmax(onehot_label, axis=1)[0]
        if (actual_label != 0 and
                gesture_counts[actual_label] >= config.final_testing_iters):
            print(gesture_counts)
            if sum(gesture_counts[1:]) == (
                    config.final_testing_iters*(config.n_output-1)):
                break
            else:
                continue
        total_iterations += 1
        # Predict
        features = np.reshape(features,
                              [-1, config.n_frames, config.n_dimension])
        onehot_pred, acc = session.run(
            [network.pred, network.accuracy],
            feed_dict={network.x: features, network.y: onehot_label})
        # Get labels
        pred_label = tf.argmax(onehot_pred, 1).eval()[0]
        # Update recognition numbers and gesture counts
        confusion_matrix[actual_label, pred_label] += 1
        gesture_counts[actual_label] += 1
        if actual_label == pred_label:
            rec_numbers[0, actual_label] += 1
            # True negatives calculated later
        else:
            rec_numbers[3, actual_label] += 1
            rec_numbers[2, pred_label] += 1
        accuracy_testing += acc
    # Calculate true negatives
    rec_numbers[1, :] = total_iterations - np.sum(
        rec_numbers[[0, 2, 3], :], axis=0)

    # Calculate recognition rate, TPR and FPR
    rr = (rec_numbers[0, :] + rec_numbers[1, :]) / (total_iterations)
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
        "---------- Testing accuracy {}\n"
            .format(accuracy_testing/total_iterations) +  # noqa
        "   " + " ".join("{:>8}".format(n) for n in np.arange(0, config.n_output)) + "\n"  # noqa
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
    report_string += "---------- Confusion matrix ----------\n"
    for i in range(config.n_output):
        report_string += np.array2string(
            confusion_matrix[i], precision=4, suppress_small=True)
        report_string += '\n'
    report_string += "---------- Normalised confusion matrix ----------\n"
    for i in range(config.n_output):
        report_string += np.array2string(
            (confusion_matrix[i]/sum(confusion_matrix[i])),
            precision=4, suppress_small=True, floatmode='fixed')
        report_string += '\n'

    if save:
        with open(config.export_dir + "/test.txt", 'w+') as f:
            f.write(report_string)
    else:
        print(report_string)


def plot_graphs(accuracy_train, accuracy_test, loss, config, save=False):
    """Plot loss and accuracy graphs."""
    plt.figure()
    plt.plot(np.arange(0, len(loss)), loss)
    plt.ylim([0, 3])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if save:
        plt.savefig(config.export_dir + "/loss.png")
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
        plt.savefig(config.export_dir + "/accuracy.png")
    else:
        plt.show()


class NetworkModel():
    """Definition of the graph of the network."""

    def __init__(self, config):
        """Build the graph."""
        def LSTM(x, weights, biases):
            """Define cell structure - predicition function."""
            LSTM_cell = tf.nn.rnn_cell.LSTMCell(config.n_hidden)
            # LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(
            #     LSTM_cell, output_keep_prob=0.5)
            # Requires a sequence (type list or tuple) of inputs,
            # each of n_dimension
            x = tf.reshape(x, [-1, config.n_frames * config.n_dimension])
            x = tf.split(x, config.n_frames, 1)
            # outputs - sequence of outputs (of h_ts), only need the last one
            # state - C_t of last time step
            outputs, states = tf.nn.static_rnn(LSTM_cell, x, dtype=tf.float32)
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        # Input data x and output label pdf y to be fed into LSTM
        # (batch size, sequence length * single datapoint dimension)
        self.x = tf.placeholder(
            "float",
            [None, config.n_frames, config.n_dimension],
            name="myInput")
        # (batch size, number fo categories)
        self.y = tf.placeholder("float", [None, config.n_output])

        # Linear regression layer parameters to be optimized
        self.weights = {'out': tf.Variable(tf.random_normal(
            [config.n_hidden, config.n_output]))}
        self.biases = {'out': tf.Variable(tf.random_normal([config.n_output]))}

        # Prediction
        self.pred = tf.identity(LSTM(self.x, self.weights, self.biases),
                                name="myOutput")

        # Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate).minimize(self.cost)

        # Model evaluation
        if config.label_type == 'bincount':
            _log("Bincount norm prediction accuracy measurement")
            probabilities = tf.nn.softmax(self.pred)
            correct_pred = 1 - tf.norm(probabilities - self.y, axis=1)
        else:
            correct_pred = tf.equal(
                tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Summary collections
        tf.summary.histogram("true_output", self.y)
        tf.summary.histogram("weights", self.weights["out"])
        tf.summary.histogram("biases", self.biases["out"])
        tf.summary.histogram("prediction", self.pred)
        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)

        # Initialisation method
        self.init = tf.global_variables_initializer()


def doTrainingSession(training_data, testing_data, config):
    """Train network. Main function that encapsulates all the training."""
    tf.reset_default_graph()
    network = NetworkModel(config)

    start_time = time.time()

    # Data collection for graphs
    accuracy_graph_train = []
    accuracy_graph_test = []
    loss_graph = []

    with tf.Session() as session:
        # ------------------------ Initialise
        session.run(network.init)
        step, acc_total, loss_total = 0, 0, 0
        train_gen = getWindowStart(config)
        test_gen = getWindowStart(config, True)
        # ------------------------ Logging
        writer = tf.summary.FileWriter(config.export_dir + "/logs",
                                       session.graph)

        while step < config.training_iters:
            # ------------- Training
            true_output, prediction, acc_total, loss_total = trainOneStep(
                session, train_gen, acc_total, loss_total, writer, config,
                network, step)
            step += 1
            # -------------- Epoch end
            if step % config.display_step == 0:
                # Report last iteration prediction
                actual_label = np.argmax(true_output[:, :], axis=1)
                onehot_label_pred = tf.argmax(prediction, 1).eval()
                print("Iter={}:\n".format(step) +
                      "   {} predicted\n".format(
                        " ".join("{:>3}".format(l) for l in onehot_label_pred)) +  # noqa
                      "   {} true".format(
                        " ".join("{:>3}".format(l) for l in actual_label)))
                # Reset numbers
                average_loss = loss_total/config.display_step
                average_acc = acc_total/config.display_step
                acc_total, loss_total = 0, 0
                # -------------- Testing
                average_test_acc = epoch_test(session, test_gen,
                                              config, network)
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
        final_test(session, test_gen, config, network, save=True)
        # -------------- Plot loss and accuracy
        plot_graphs(accuracy_graph_train, accuracy_graph_test, loss_graph,
                    config, save=True)

        # -------------------- SAVE MODEL -----------------------------------
        if config.save_model:
            signature = tf.saved_model.predict_signature_def(
                inputs={'myInput': network.x},
                outputs={'myOutput': network.pred})
            config.builder.add_meta_graph_and_variables(
                sess=session,
                tags=["myTag"],
                signature_def_map={'predict': signature})
            config.builder.save()


# frame_step_opts = [3, 5, 8, 10]
# learning_rate_opts = [0.01, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]
# n_frames_opts = [6]
# delay_opts = [3]


if __name__ == '__main__':

    data_folder = pjoin("..", "Database", "MyDatabase14")

    # Load training data
    start_time = time.time()
    # np.random.seed(7)
    training_data, testing_data = loadData(
        0.2,
        (6, 102), data_folder)
    _log("Loaded training data in {} seconds".format(time.time() - start_time))

    # calculateMapping(training_data[0], data_folder)

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

    # If doing hyperparameters, DON'T save the model and change the
    # export_dir = "hyperparameters/" + config.get_hyperstring()
    config = Config(
        learning_rate=0.0005,
        training_iters=2000,
        final_testing_iters=20,
        batch_size=10,
        delay=20,
        save_model=True
    )
    config.mapping = loadMapping(config, data_folder)
    doTrainingSession(training_data, testing_data, config)
    print("Finished")
