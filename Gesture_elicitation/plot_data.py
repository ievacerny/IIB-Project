import matplotlib.pyplot as plt
import numpy as np

frames = np.loadtxt("test.csv", delimiter=',')


def onclick(event):
    global fno
    if event.key == "left":
        if fno > 0:
            fno -= 1
            ax2.clear()
            ax2.set_title(fno)
            ax2.axis('off')
            ax2.plot(np.arange(0, frames.shape[0]), frames[:, fno])
            event.canvas.draw()
    if event.key == "right":
        if fno < frames.shape[1] - 1:
            fno += 1
            ax2.clear()
            ax2.set_title(fno)
            ax2.axis('off')
            ax2.plot(np.arange(0, frames.shape[0]), frames[:, fno])
            event.canvas.draw()


fno = 0
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 7))
fig.canvas.mpl_connect('key_press_event', onclick)

ax1.plot(np.arange(0, frames.shape[0]), frames[:, -1], label="Gesture")
ax1.set_title("Gesture presence. Each gesture done 3 times, gestures done in a row")  # noqa
plt.xlabel("Frame number")

for i in range(frames.shape[1]-1):
    ax2.clear()
    ax2.set_title("Data column {}".format(i))
    ax2.axis('off')
    mean = np.mean(frames[:, i])
    std = np.std(frames[:, i])
    ax2.set_ylim(mean-3*std, mean+3*std)
    ax2.plot(np.arange(0, frames.shape[0]), frames[:, i])
    plt.savefig("point_patterns/point{}".format(i), dpi=300)
