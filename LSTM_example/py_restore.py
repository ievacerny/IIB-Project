import tensorflow as tf
import numpy as np

model_path = r"LSTM_test\\model_1548158314"

with tf.Session() as session:

    tf.saved_model.load(
        session,
        ["myTag"],
        model_path)

    x = tf.get_default_graph().get_tensor_by_name("myInput:0")
    pred = tf.get_default_graph().get_tensor_by_name("myOutput:0")
    symbols_in_keys = [1, 17, 18]
    keys = np.reshape(np.array(symbols_in_keys), [-1, 3, 1])
    onehot_pred = session.run(pred, feed_dict={x: keys})
    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
    print(onehot_pred_index)
