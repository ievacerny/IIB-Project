"""LSTM prediction server."""
import socket
import numpy as np
import tensorflow as tf

# ---------------------------- PARAMETERS -------------------------------------
HOST = '127.0.0.1'
PORT = 65432  # (non-privileged ports are > 1023)
model_path = r"model_I-5000_L-0.001_Random"
data_path = r"..\\Database\\MyDatabase\\"
n_frames = 6
n_dimension = 40
threshold = 0.8


mapping = np.loadtxt(data_path+'/svd_V.csv', dtype='float', delimiter=',')
mapping_t = mapping[:n_dimension, :].T

# Initialise input data
input_data = []
for i in range(n_frames):
    input_data.append(np.zeros(n_dimension))


def barplot(probabilities, predicted_label):
    """Plot probabilities on the command line."""
    probabilities = (probabilities / 0.01).astype(int)
    print("-" * 103)
    for i in range(6):
        if predicted_label == i:
            print("{}: ".format(i) + "█" * probabilities[i] + " <--")
        else:
            print("{}: ".format(i) + "█" * probabilities[i])


# ---------------------------- DATA SERVING -----------------------------------
with tf.Session() as session:
    tf.saved_model.load(session, ["myTag"], model_path)
    x = tf.get_default_graph().get_tensor_by_name("myInput:0")
    pred = tf.get_default_graph().get_tensor_by_name("myOutput:0")
    zero_frame = np.zeros(n_dimension)
    prev_prediction = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket:
        socket.bind((HOST, PORT))
        socket.listen()
        connection, address = socket.accept()
        with connection:
            print('Socket connected by', address)
            while True:
                data = connection.recv(4096)
                if not data:
                    print("Client disconnected")
                    break
                # print("Received data", data)
                if (len(data) == 1 and data == b'0'):
                    input_data.append(zero_frame)
                else:
                    new_frame = np.fromstring(data, dtype=float, sep=',')
                    input_data.append(np.matmul(new_frame, mapping_t))

                input_data.pop(0)
                features = np.reshape(
                    input_data, [-1, n_frames, n_dimension])
                raw_output = session.run(pred, feed_dict={x: features})

                prediction_prob = tf.nn.softmax(raw_output).eval()
                prediction = np.argmax(prediction_prob)
                prev_prediction = prediction
                if prediction_prob[0][prediction] < threshold:
                    prev_prediction = 0
                    prediction = 0
                elif prev_prediction != prediction:
                    prediction = 0
                barplot(prediction_prob[0], prediction)
                prediction = str.encode(str(prediction))
                connection.sendall(prediction)
                # print("Sent prediction", prediction)
