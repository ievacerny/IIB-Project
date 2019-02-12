from random import shuffle

gestures = [
  "1 Form C",
  "1 Form V",
  "8 Palm upwards, bend 4 fingers",
  "9 Contract hand and turn left",
  "9 Contract hand and turn right",
  "10 Palm forwards, bend 4 fingers",
  "13 Cross text out with 1 finger",
  "16 Wipe blackboard motion twice",
  "17 CCW circle with 1 finger",
  "17 CW circle with 1 finger",
  "20 Flick hand to the left",
  "20 Flick hand to the right",
  "29 Push hand into the plane",
  "29 Grab text out of the plane",
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
