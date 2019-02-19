from random import shuffle
import sys

gestures = [
    "1 Flick hand to the left",
    "2 Flick hand to the right",
    "3 CCW circle with 1 finger",
    "4 CW circle with 1 finger",
    "5 Palm forwards, bend 4 fingers",
    "6 Palm upwards, bend 4 fingers",
    "7 Grab text out of the plane",
    "8 Push hand into the plane",
    "9 Contract hand and turn left",
    "10 Contract hand and turn right",
    "11 Form C",
    "12 Form V",
    "13 Wipe blackboard motion twice",
    "14 Cross text out with 1 finger",
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
