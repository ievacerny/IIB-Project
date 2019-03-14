"""Visualise data recording."""
# Used implicitly by projection='3d'
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys

# ------------- READ ARGUMENTS
if len(sys.argv) == 2:
    i = sys.argv[1]
else:
    i = 1

# ------------- LOAD DATA
frames = []
with open("../Database/MyDatabase/random_{}.csv".format(i), 'r') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        points = []
        for i in range(7):
            points.append([float(row[i*3]),
                           float(row[i*3+1]),
                           -1*float(row[i*3+2])])
        for i in range(20):
            points.append([float(row[49+i*10]),
                           float(row[50+i*10]),
                           -1*float(row[51+i*10])])
        frames.append(points)

frames = np.array(frames)
frames[:, 1:, :] = frames[:, 1:, :] + frames[:, :1, :]
print(frames.shape)


# -------------- EVENT HANDLERS
def onclick(event):
    """Change frames via arrow clicks."""
    global fno
    if event.key == "left":
        if fno > 0:
            fno -= 1
            ax = event.canvas.figure.gca()
            elev, azim = ax.elev, ax.azim
            event.canvas.figure.clear()
            ax = event.canvas.figure.gca(projection='3d')
            ax.view_init(elev=elev, azim=azim)
            plot_hand(ax, fno)
            event.canvas.draw()
    elif event.key == "right":
        if fno < frames.shape[0] - 1:
            fno += 1
            ax = event.canvas.figure.gca()
            elev, azim = ax.elev, ax.azim
            event.canvas.figure.clear()
            ax = event.canvas.figure.gca(projection='3d')
            ax.view_init(elev=elev, azim=azim)
            plot_hand(ax, fno)
            event.canvas.draw()


# ---------------- MAIN CODE
def plot_hand(ax, fno):
    """Plot hand from data."""
    ax.set_title(str(fno))
    frame = frames[fno, :, :]
    ax.set_xlim(-0.2, 0.4)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.6, 0)
    ax.plot(frame[[0, 1], 0], frame[[0, 1], 1], frame[[0, 1], 2], ':',
            label="Wrist")
    for i in range(2, 7):
        ax.plot(frame[[0, 7+4*(i-2), 8+4*(i-2), 9+4*(i-2), 10+4*(i-2), i], 0],
                frame[[0, 7+4*(i-2), 8+4*(i-2), 9+4*(i-2), 10+4*(i-2), i], 1],
                frame[[0, 7+4*(i-2), 8+4*(i-2), 9+4*(i-2), 10+4*(i-2), i], 2],
                label="Finger {}".format(i))
    plt.legend()


fno = 0
fig = plt.figure()
ax = Axes3D(fig)
fig.canvas.mpl_connect('key_press_event', onclick)
plot_hand(ax, fno)
plt.show()
