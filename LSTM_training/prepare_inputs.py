"""Prepare the dataset to be used as inputs for trainig."""
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
# import matplotlib.pyplot as plt
from collections import Counter


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


# mapping = np.loadtxt('mapping.csv', dtype='float', delimiter=',')
# mapping_t = mapping.T
mapping_t = np.empty((1, 1))


def set_mapping(mapping):
    global mapping_t
    mapping_t = mapping.T


def load_my_dataset(file_no, data_folder=None, position_only=False):
    """Load coordinates and labels from data file of specified number."""
    if data_folder is None:
        data_folder = pjoin("..", "Database", "MyDatabase")

    gestures = np.genfromtxt(pjoin(data_folder, "vid_{}.csv".format(file_no)),
                             delimiter=',')
    if position_only:
        gestures = gestures[:, :21]
        gestures = np.reshape(gestures, [gestures.shape[0], 7, 3])
        gestures = gestures - gestures[:, :1, :]
    else:
        # Wrist and finger tip positions
        position_points_x = [i for i in range(0, 21, 3)]
        position_points_y = [i+1 for i in range(0, 21, 3)]
        position_points_z = [i+2 for i in range(0, 21, 3)]
        # Bone beginning positions
        position_points_x.extend([49+10*i for i in range(20)])
        position_points_y.extend([50+10*i for i in range(20)])
        position_points_z.extend([51+10*i for i in range(20)])

        non_zero_gestures_norm = np.copy(gestures)
        non_zero_gestures_norm[:, position_points_x] = gestures[:, position_points_x] - gestures[:, :1]
        non_zero_gestures_norm[:, position_points_y] = gestures[:, position_points_y] - gestures[:, 1:2]
        non_zero_gestures_norm[:, position_points_z] = gestures[:, position_points_z] - gestures[:, 2:3]

        gestures = non_zero_gestures_norm

    labels = np.empty(gestures.shape[0], dtype='int32')
    label_info = np.loadtxt(
        pjoin(data_folder, "vid_{}_labels.txt".format(file_no)), dtype='int32')
    prev_start, prev_lbl = 0, 0
    for start, lbl in label_info:
        if start != 0:
            labels[prev_start:start] = prev_lbl
        prev_start, prev_lbl = start, lbl
    print(Counter(labels))

    return gestures, labels


def load_LMDHG_from_file(file_no, data_folder=None):
    """Load coordinates and labels from data file of specified number."""
    if data_folder is None:
        data_folder = pjoin("..", "Database", "LMDHG")

    data = sio.loadmat(pjoin(data_folder, "DataFile{}.mat".format(file_no)),
                       squeeze_me=True)

    gestures = np.array(data['skeleton'], dtype="object")
    labels = np.empty(len(data['skeleton']), dtype="object")

    idx = 0
    for first, last in data['Anotations']:
        labels[first-1:last] = label_dict[data['labels'][idx]]
        idx += 1

    return gestures, labels


def load_DHG_dataset(no_instances=None, data_folder=None, frame_length=150):
    """Load coordinates and labels of all DHG dataset."""
    if data_folder is None:
        data_folder = pjoin("..", "Database", "DHG2016")

    gestures = []
    labels = []
    idx = 0
    lengths = []
    with open(pjoin(data_folder, "informations_troncage_sequences.txt")) as f:
        for line in f:
            g, f, s, e, beg, end = line.split()
            beg, end = int(beg), int(end)
            lengths.append(end-beg)
            gesture_fname = pjoin(
                data_folder,
                "gesture_{}".format(g),
                "finger_{}".format(f),
                "subject_{}".format(s),
                "essai_{}".format(e),
                "skeleton_world.txt"
            )
            with open(gesture_fname) as fg:
                frames = np.empty(end+100, dtype='object')
                frame_id = 0
                for frame in fg:
                    points = frame.split()
                    points = np.array(points, dtype='float32')
                    frames[frame_id] = np.reshape(points, [-1, 3])
                    frame_id += 1
                # Cut non action frames
                frames = frames[beg-1:end]
            gestures.append(frames)
            labels.append(int(g)-1)
            idx += 1
            if no_instances is not None and idx == no_instances:
                break

    gestures = np.array(gestures)
    labels = np.array(labels)
    # plt.hist(lengths)
    # plt.show()
    print(min(lengths), max(lengths))
    return gestures, labels


def calculate_features(points):
    features = np.matmul(points, mapping_t)
    # features = points[:-1]
    return features


def calculate_old_features(points):
    """Obtain features from one frame."""
    # C = points[23, :]  # central point - the palm
    # wrist_vec = points[24, :] - C
    C = points[0, :]
    wrist_vec = points[1, :]
    wrist_mag = np.sqrt(abs(wrist_vec.dot(wrist_vec)))
    # fingers_vec = points[[5, 9, 13, 17, 21], :] - C
    # fingers_vec = points[[29, 33, 37, 41, 45], :] - C
    fingers_vec = points[[2, 3, 4, 5, 6], :]
    fingers_mag = np.linalg.norm(fingers_vec, axis=1)

    features = np.zeros(24)

    features[:15] = fingers_vec.flatten()

    idx1 = 15
    idx2 = 19
    for i in range(len(fingers_vec)):
        finger = fingers_vec[i]
        if wrist_mag != 0 and fingers_mag[i] != 0:  # Else leave as zero
            features[idx1] = np.arccos(np.dot(finger, wrist_vec) /
                                       (fingers_mag[i] * wrist_mag))
        if i != 0 and fingers_mag[i]*fingers_mag[i-1] != 0:
            features[idx2] = np.arccos(np.dot(finger, fingers_vec[i-1]) /
                                       (fingers_mag[i] * fingers_mag[i-1]))
        idx1 += 1
        idx2 += 1
    return features


# --------------------------- OLD FEATURES ------------------------------------
# LMDHG
# points_of_interest = [0, 1, 6, 10, 14, 18, 22, 23, 24, 29, 33, 37, 41, 45]
# DHG
# points_of_interest = [0, 1, 5, 9, 13, 17, 21]
points_of_interest = np.arange(0, 22)


def calculate_old_features(pattern):
    """Obtain features from a set of frames."""
    features = np.zeros(57)

    last_idx = len(pattern) - 1
    box = np.array([
        np.max(pattern[:, 0]) - np.min(pattern[:, 0]),
        np.max(pattern[:, 1]) - np.min(pattern[:, 1]),
        np.max(pattern[:, 2]) - np.min(pattern[:, 2])])
    if np.sqrt(box.dot(box)) == 0:  # Neither of the hands is in view
        return features
    centre = 0.5 * box
    max_box = max(box)
    differences = np.diff(pattern, axis=0)
    differences_sq = differences**2
    path_length = np.sum(np.sqrt(np.sum(differences_sq, axis=1)))

    features[0:3] = (pattern[0] - centre) / max_box + 0.5
    features[3:6] = (pattern[last_idx] - centre) / max_box + 0.5
    point_to_last_vec = pattern[last_idx] - pattern[0]
    # Internet says that sqrt and dot is significantly faster that norm
    features[6] = np.sqrt(abs(point_to_last_vec.dot(point_to_last_vec)))
    features[7] = np.dot(point_to_last_vec,
                         np.array([1, 0, 0])) / features[6]
    features[8] = np.dot(point_to_last_vec,
                         np.array([0, 1, 0])) / features[6]
    features[9] = np.dot(point_to_last_vec,
                         np.array([0, 0, 1])) / features[6]
    features[10] = features[6] / path_length
    initial_vec = pattern[2] = pattern[0]
    initial_vec_length = np.sqrt(abs(initial_vec.dot(initial_vec)))
    features[11] = np.dot(initial_vec,
                          np.array([1, 0, 0])) / initial_vec_length
    features[12] = np.dot(initial_vec,
                          np.array([0, 1, 0])) / initial_vec_length
    features[13] = np.dot(initial_vec,
                          np.array([0, 0, 1])) / initial_vec_length
    middle = pattern[int(last_idx/2)]
    first_last_sum = pattern[0] + pattern[last_idx]
    features[14:17] = (middle - 0.5*first_last_sum) / box
    # Features[17:20] are computationaly expensive and re about downstrokes
    # which probably are not very useful in gesture recognition
    features[17:20] = 1
    features[20] = np.arctan(box[1] / box[0])
    features[21] = np.arctan(box[2] / box[1])
    features[22] = np.arctan(box[1] / box[2])
    features[23] = path_length
    features[24] = np.sum(box) / path_length
    central_gravity = np.sum(pattern, axis=0) / (last_idx+1)
    features[25] = np.linalg.norm(central_gravity - pattern) / (last_idx+1)
    features[26] = np.sum(
        np.arctan(differences[:, 0]/differences[:, 2])) / last_idx
    features[27] = np.sum(
        np.arctan(differences[:, 1]/differences[:, 0])) / last_idx
    features[28] = np.sum(
        np.arctan(differences[:, 2]/differences[:, 1])) / last_idx
    bins = np.arange(9) * np.pi/8
    alphas_xy = np.arccos(differences[:, 0]/np.sqrt(
        differences_sq[:, 0] + differences_sq[:, 1]))
    hist, _ = np.histogram(alphas_xy, bins, (0, np.pi))
    features[29] = (hist[0] + hist[4]) / (last_idx+1)
    features[30] = (hist[1] + hist[5]) / (last_idx+1)
    features[31] = (hist[2] + hist[6]) / (last_idx+1)
    features[32] = (hist[3] + hist[7]) / (last_idx+1)
    alphas_yz = np.arccos(differences[:, 1]/np.sqrt(
        differences_sq[:, 1] + differences_sq[:, 2]))
    hist, _ = np.histogram(alphas_yz, bins, (0, np.pi))
    features[33] = (hist[0] + hist[4]) / (last_idx+1)
    features[34] = (hist[1] + hist[5]) / (last_idx+1)
    features[35] = (hist[2] + hist[6]) / (last_idx+1)
    features[36] = (hist[3] + hist[7]) / (last_idx+1)
    alphas_zx = np.arccos(differences[:, 2]/np.sqrt(
        differences_sq[:, 2] + differences_sq[:, 0]))
    hist, _ = np.histogram(alphas_zx, bins, (0, np.pi))
    features[37] = (hist[0] + hist[4]) / (last_idx+1)
    features[38] = (hist[1] + hist[5]) / (last_idx+1)
    features[39] = (hist[2] + hist[6]) / (last_idx+1)
    features[40] = (hist[3] + hist[7]) / (last_idx+1)

    theta_1 = theta_angles(pattern, 1)
    features[41] = np.sum(theta_1)
    features[42] = np.sum(np.sin(theta_1)**2)
    phi_1 = phi_angles(pattern, 1)
    features[43] = np.sum(phi_1)
    features[44] = np.sum(np.sin(phi_1)**2)
    theta_2 = theta_angles(pattern, 2)
    features[45] = np.sum(np.sin(theta_2)**2)
    features[46] = np.max(theta_2)
    phi_2 = phi_angles(pattern, 2)
    features[47] = np.sum(np.sin(phi_2)**2)
    features[48] = np.max(phi_2)
    gamma = 0.25  # parameter
    bins = bins = np.arange(5) * np.pi/4
    psi = gamma * theta_1 + (1-gamma) * theta_2
    hist, _ = np.histogram(psi, bins, (0, np.pi))
    features[49:53] = hist / (last_idx+1)
    chi = gamma * phi_1 + (1-gamma) * phi_2
    hist, _ = np.histogram(chi, bins, (0, np.pi))
    features[53:57] = hist / (last_idx+1)
    # last few features are very expensive, see how it goes without them
    features = np.nan_to_num(features)

    return features


def theta_angles(pattern, k):
    """Calculate all theta angles with k parameter."""
    n = len(pattern)
    angles = np.zeros(n)
    for i in range(k, n-k):
        v1 = pattern[i] - pattern[i-k]
        v1_mag = np.sqrt(abs(v1.dot(v1)))
        v2 = pattern[i+k] - pattern[i]
        v2_mag = np.sqrt(abs(v2.dot(v2)))
        if v1_mag != 0 and v2_mag != 0:  # Else leave as zero
            angles[i-k] = np.arccos(np.dot(v1, v2) / (v1_mag * v2_mag))
    return angles


def phi_angles(pattern, k):
    """Calculate all phi angles with k parameter."""
    n = len(pattern)
    angles = np.zeros(n)
    for i in range(k, n-2*k):
        n1 = np.cross(pattern[i+k]-pattern[i], pattern[i-k]-pattern[i])
        n1_mag = np.sqrt(n1.dot(n1))
        if n1_mag != 0:  # Else leave the angle as zero
            n1 = n1 / n1_mag
        n2 = np.cross(pattern[i+2*k]-pattern[i+k], pattern[i]-pattern[i+k])
        n2_mag = np.sqrt(n2.dot(n2))
        if n2_mag != 0:
            n2 = n2 / n2_mag
        normal_dot = n1.dot(n2)
        # Currently cannot tell if angle is positive or negative (+-180)
        angles[i-k] = np.arccos(normal_dot)
    return angles
