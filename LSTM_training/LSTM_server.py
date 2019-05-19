"""LSTM prediction server."""
import numpy as np
import socket
import tensorflow as tf

# ---------------------------- PARAMETERS -------------------------------------
HOST = '127.0.0.1'
PORT = 65432  # (non-privileged ports are > 1023)
model_path = r"final_model"
data_path = r"..\\Database\\MyDatabase\\"
n_frames = 30
n_dimension = 45
threshold = 0.9


# ------------------------------ LOGGING --------------------------------------
def barplot(probabilities, predicted_label):
    """Plot probabilities on the command line."""
    probabilities = (probabilities / 0.01).astype(int)
    print("-" * 103)
    for i in range(6):
        if predicted_label == i:
            print("{}: ".format(i) + "█" * probabilities[i] + " <--")
        else:
            print("{}: ".format(i) + "█" * probabilities[i])


# ---------------------------- INITIALISE -------------------------------------
# Load mapping
mapping = np.loadtxt(data_path+'/svd_V.csv', dtype='float', delimiter=',')
mapping_t = mapping[:n_dimension, :].T

# Initialise input data
zero_frame = np.zeros(n_dimension)
input_data = []
for i in range(n_frames):
    input_data.append(zero_frame)
prev_prediction = 0

# ---------------------------- DATA SERVING -----------------------------------
with tf.Session() as session:
    # Load graph
    tf.saved_model.load(session, ["myTag"], model_path)
    x = tf.get_default_graph().get_tensor_by_name("myInput:0")
    pred = tf.get_default_graph().get_tensor_by_name("myOutput:0")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket:
        # Connect socket
        print('Waiting for connection...')
        socket.bind((HOST, PORT))
        socket.listen()
        connection, address = socket.accept()
        with connection:
            print('Socket connected by', address)
            while True:
                # Receive frame data
                data = connection.recv(4096)
                if not data:
                    print("Client disconnected")
                    break

                # Add new frame data
                if (len(data) == 1 and data == b'0'):
                    input_data.append(zero_frame)
                else:
                    new_frame = np.fromstring(data, dtype=float, sep=',')
                    input_data.append(np.matmul(new_frame, mapping_t))
                input_data.pop(0)

                # Predict
                features = np.reshape(
                    input_data, [-1, n_frames, n_dimension])
                raw_output = session.run(pred, feed_dict={x: features})
                prediction = np.argmax(raw_output)
                # Calculate probabilities
                sum = np.sum(np.exp(raw_output))
                prediction_prob = np.exp(raw_output[0]) / sum

                # Post-processing
                if prediction_prob[prediction] < threshold:
                    prev_prediction = 0
                    prediction = 0
                elif prev_prediction != prediction:
                    prev_prediction = prediction
                    prediction = 0
                else:
                    prev_prediction = prediction
                barplot(prediction_prob, prediction)

                # Send gesture code
                connection.sendall(str.encode(str(prediction)))
