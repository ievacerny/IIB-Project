from random import shuffle
import sys

gestures = [
    "UNDO - CCW circle",
    "REDO - CW circle",
    "COPY - letter C",
    "PASTE - letter V",
    "DELETE - flick hand to the left",
    "UNDO - CCW circle",
    "REDO - CW circle",
    "COPY - letter C",
    "PASTE - letter V",
    "DELETE - flick hand to the left",
]

zero_class = [
    "Touch plane in few places",
    "Select some text with 2 fingers",
    "Rotate hand from palm down to palm up",
    "Wrist rotations",
    "Anything random",
    "One finger bending",
    "Show yes",
    "Palm forward slide to the left",
    "Palm forward slide to the right",
    "Bend 4 finger with palm to the side",
    "Make a fist at various wrist rotations",
    "Extend and relax hand multiple times"
]

if len(sys.argv) == 2 and int(sys.argv[1]) == 0:
    shuffle(zero_class)
    for motion in zero_class:
        print(motion)
else:
    shuffle(gestures)
    for gesture in gestures:
        print(gesture)
