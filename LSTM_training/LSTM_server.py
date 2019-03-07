import socket
import numpy as np
from os.path import join as pjoin
import sys
import tensorflow as tf

# ---------------------------- PARAMETERS -------------------------------------
HOST = '127.0.0.1'
PORT = 65432  # (non-privileged ports are > 1023)
model_path = r"model_I-1000_LD-2_S-5_F-8"
data_path = r"..\\Database\\MyDatabase\\"
n_frames = 8
n_dimension = 40


mapping = np.loadtxt(model_path+'/mapping.csv', dtype='float', delimiter=',')
mapping_t = mapping.T

# Initialise input data
input_data = []
for i in range(n_frames):
    input_data.append(np.zeros(n_dimension))


# ---------------------------- DATA SERVING -----------------------------------
def slide_window(new_frame):
    """Slide the window over input data."""
    windowed_data = []
    # Initialise with zeros
    for i in range(n_frames):
        windowed_data.append(np.zeros(n_dimension))
    yield windowed_data
    i = 0
    while i < len(windowed_data):
        windowed_data.append(np.matmul(new_frame, mapping_t))
        windowed_data.pop(0)
        yield windowed_data
        i += 1


with tf.Session() as session:
    tf.saved_model.load(session, ["myTag"], model_path)
    x = tf.get_default_graph().get_tensor_by_name("myInput:0")
    pred = tf.get_default_graph().get_tensor_by_name("myOutput:0")

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
                if (len(data) == 1 and data == '0'):
                    print("Fake zero array")
                new_frame = np.fromstring(data, dtype=float, sep=',')
                print("Data has shape of ", new_frame.shape)

                input_data.append(np.matmul(new_frame, mapping_t))
                input_data.pop(0)
                features = np.reshape(
                    input_data, [-1, n_frames, n_dimension])
                raw_output = session.run(pred, feed_dict={x: features})
                prediction = tf.nn.softmax(raw_output).eval()
                prediction = str.encode(np.array2string(
                    prediction, precision=3, suppress_small=True))

                connection.sendall(prediction)
                print("Sent prediction", prediction)
