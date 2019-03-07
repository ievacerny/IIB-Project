import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import join as pjoin
import sys
import tensorflow as tf

# ---------------------------- PARAMETERS -------------------------------------
model_path = r"model_I-1000_LD-2_S-5_F-8"
data_path = r"..\\Database\\MyDatabase\\"
n_frames = 8
n_dimension = 40
frame_step = 5

# ---------------------------- READ ARGUMENTS ---------------------------------
if len(sys.argv) == 2:
    vid_no = sys.argv[1]
else:
    vid_no = 1


# ---------------------------- LOAD DATA FILE ---------------------------------
def load_data(vid_no=1):
    """Load gesture data from specified video."""
    gestures = np.genfromtxt(pjoin(data_path, "vid_{}.csv".format(vid_no)),
                             delimiter=',')
    # Wrist and finger tip positions
    position_points_x = [i for i in range(0, 21, 3)]
    position_points_y = [i+1 for i in range(0, 21, 3)]
    position_points_z = [i+2 for i in range(0, 21, 3)]
    # Bone beginning positions
    position_points_x.extend([49+10*i for i in range(20)])
    position_points_y.extend([50+10*i for i in range(20)])
    position_points_z.extend([51+10*i for i in range(20)])
    # Normalise data
    non_zero_gestures_norm = np.copy(gestures)
    non_zero_gestures_norm[:, position_points_x] = (
        gestures[:, position_points_x] - gestures[:, :1])
    non_zero_gestures_norm[:, position_points_y] = (
        gestures[:, position_points_y] - gestures[:, 1:2])
    non_zero_gestures_norm[:, position_points_z] = (
        gestures[:, position_points_z] - gestures[:, 2:3])
    gestures = non_zero_gestures_norm

    return gestures


# ---------------------------- DATA SERVING -----------------------------------
def slide_window(full_data):
    """Slide the window over input data."""
    windowed_data = []
    # Initialise with zeros
    for i in range(n_frames):
        windowed_data.append(np.zeros(n_dimension))
    yield windowed_data
    i = 0
    while i < len(full_data):
        windowed_data.append(np.matmul(full_data[i], mapping_t))
        windowed_data.pop(0)
        yield windowed_data
        i += 1


# ---------------------------- PLOT FUNCTIONS ---------------------------------
def plot_hand(ax1, time_idx):
    """Plot the hand skeleton."""
    ax1.set_title("Hand position at frame {}".format(time_idx))
    frame = full_data[time_idx]
    ax1.set_xlim(-0.2, 0.4)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_zlim(-0.6, 0)
    ax1.plot(frame[[0, 3]], frame[[1, 4]], frame[[2, 5]], ':', label="Wrist")
    for i in range(2, 7):
        ax1.plot(
            frame[[0, 49+40*(i-2), 59+40*(i-2), 69+40*(i-2), 79+40*(i-2),
                   6+3*(i-2)]],
            frame[[0, 50+40*(i-2), 60+40*(i-2), 70+40*(i-2), 80+40*(i-2),
                   7+3*(i-2)]],
            frame[[0, 51+40*(i-2), 61+40*(i-2), 71+40*(i-2), 81+40*(i-2),
                   8+3*(i-2)]],
            label="Finger {}".format(i))
    plt.legend()


def plot_softmax(ax2, time_idx):
    """Plot the softmax values."""
    ax2.set_title("Softmax values at frame {}".format(time_idx))
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Softmax values")
    ax2.set_xlabel("Gesture code")
    prediction = predictions[time_idx]
    barlist = ax2.bar(np.arange(6), prediction)
    predicted_label = np.argmax(prediction)
    barlist[predicted_label].set_color('r')


# ---------------------------- PLOTTING ---------------------------------------
def on_key(event):
    """Deal with the matplolib event."""
    global time_idx, ax1, ax2
    if event.key == "right":
        if time_idx == len(full_data) - 1:
            return
        time_idx += 1
    elif event.key == "left":
        if time_idx == 0:
            return
        time_idx -= 1
    else:
        return
    # Plot hand
    elev, azim = ax1.elev, ax1.azim
    event.canvas.figure.clear()
    ax1 = event.canvas.figure.add_subplot(2, 1, 1, projection='3d')
    ax1.view_init(elev=elev, azim=azim)
    plot_hand(ax1, time_idx)
    # Plot predictions
    ax2 = event.canvas.figure.add_subplot(2, 1, 2)
    plot_softmax(ax2, time_idx)
    event.canvas.draw()


# ---------------------------- MAIN CODE --------------------------------------

# Load and prep data
full_data = load_data(vid_no)
full_data = full_data[::frame_step]
mapping = np.loadtxt(model_path+'/mapping.csv', dtype='float', delimiter=',')
mapping_t = mapping.T
slider = slide_window(full_data)
next(slider)
predictions = np.zeros((len(full_data), 6))

# Get all predictions
with tf.Session() as session:
    tf.saved_model.load(session, ["myTag"], model_path)
    x = tf.get_default_graph().get_tensor_by_name("myInput:0")
    pred = tf.get_default_graph().get_tensor_by_name("myOutput:0")
    idx = 0
    while True:
        try:
            features = next(slider)
            features = np.reshape(features, [-1, n_frames, n_dimension])
            raw_output = session.run(pred, feed_dict={x: features})
            predictions[idx, :] = tf.nn.softmax(raw_output).eval()
            # next(slider)
            # predictions[idx, :] = np.random.rand(6)
            idx += 1
        except StopIteration:
            print("Video finished")
            break

# Plot the gestures and the predictions
time_idx = 0
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)
ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax1.view_init(elev=105, azim=-89.5)
plot_hand(ax1, time_idx)
ax2 = fig.add_subplot(2, 1, 2)
plot_softmax(ax2, time_idx)
plt.show()
