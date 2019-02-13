from random import shuffle

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
    "Touch plane in few places",
    "Select some text with 2 fingers",
    "Rotate hand from palm down to palm up",
    "Anything random",
    "One finger bending",
    "Show yes",
    "Palm forward slide to the left",
    "Palm forward slide to the right",
    "Flick finger from left to right",
    "Bend 4 finger with palm to the side",
    "Finger bending in row",
]

shuffle(gestures)
for gesture in gestures:
    print(gesture)
