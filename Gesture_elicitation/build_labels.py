import numpy as np

""" VIDEO 1
Bend 4 finger with palm to the side
9 Contract hand and turn left
13 Wipe blackboard motion twice
Flick finger from left to right
11 Form C
Anything random
Rotate hand from palm down to palm up
7 Grab text out of the plane
Finger bending in row
One finger bending
14 Cross text out with 1 finger
3 CCW circle with 1 finger
12 Form V
4 CW circle with 1 finger
Palm forward slide to the left
2 Flick hand to the right
5 Palm forwards, bend 4 fingers
Select some text with 2 fingers
8 Push hand into the plane
Palm forward slide to the right
1 Flick hand to the left
Touch plane in few places
6 Palm upwards, bend 4 fingers
Show yes
10 Contract hand and turn right
"""

vid_1 = [
  (0, 0),
  (260, 9),
  (365, 0),
  (450, 13),
  (515, 0),
  (945, 11),
  (1058, 0),
  (1784, 7),
  (1820, 0),
  (1883, 7),
  (1935, 0),
  (2620, 14),
  (2650, 0),
  (2793, 3),
  (2830, 0),
  (2913, 12),
  (3023, 0),
  (3078, 4),
  (3111, 0),
  (3630, 2),
  (3670, 0),
  (3974, 5),
  (4002, 0),
  (5007, 8),
  (5030, 0),
  (5383, 1),
  (5408, 0),
  (5890, 6),
  (5910, 0),
  (5976, 6),
  (5995, 0),
  (6226, 10),
  (6294, 0),
  (6336, -1)
]

with open("vid_1_labels.txt", 'w') as f:
    np.savetxt(f, vid_1, fmt="%u")

""" VIDEO 2
Anything random
Touch plane in few places
2 Flick hand to the right
Rotate hand from palm down to palm up
Bend 4 finger with palm to the side
Select some text with 2 fingers
10 Contract hand and turn right
5 Palm forwards, bend 4 fingers
4 CW circle with 1 finger
9 Contract hand and turn left
7 Grab text out of the plane
8 Push hand into the plane
11 Form C
Flick finger from left to right
14 Cross text out with 1 finger
13 Wipe blackboard motion twice
Palm forward slide to the left
One finger bending
12 Form V
1 Flick hand to the left
6 Palm upwards, bend 4 fingers
Finger bending in row
Show yes
3 CCW circle with 1 finger
Palm forward slide to the right
"""

vid_2 = [
    (0, 0),
    (1398, 2),
    (1413, 0),
    (2687, 10),
    (2753, 0),
    (3073, 5),
    (3094, 0),
    (3350, 4),
    (3407, 0),
    (3618, 9),
    (3660, 0),
    (3913, 7),
    (3940, 0),
    (4062, 8),
    (4080, 0),
    (4162, 11),
    (4278, 0),
    (4784, 14),
    (4811, 0),
    (4921, 13),
    (4957, 0),
    (5521, 12),
    (5593, 0),
    (5648, 1),
    (5657, 0),
    (5769, 1),
    (5777, 0),
    (6141, 6),
    (6169, 0),
    (6587, 3),
    (6630, 0),
    (7053, -1)
]

with open("vid_2_labels.txt", 'w') as f:
    np.savetxt(f, vid_2, fmt="%u")

""" VIDEO 3
10 Contract hand and turn right
12 Form V
Flick finger from left to right
7 Grab text out of the plane
Bend 4 finger with palm to the side
Select some text with 2 fingers
1 Flick hand to the left
11 Form C
Finger bending in row
4 CW circle with 1 finger
6 Palm upwards, bend 4 fingers
3 CCW circle with 1 finger
Anything random
13 Wipe blackboard motion twice
8 Push hand into the plane
9 Contract hand and turn left
Palm forward slide to the right
Show yes
14 Cross text out with 1 finger
2 Flick hand to the right
Rotate hand from palm down to palm up
Touch plane in few places
5 Palm forwards, bend 4 fingers
One finger bending
Palm forward slide to the left
"""

vid_3 = [
    (0, 0),
    (218, 10),
    (280, 0),
    (373, 12),
    (418, 0),
    (654, 7),
    (680, 0),
    (1173, 1),
    (1184, 0),
    (1273, 11),
    (1323, 0),
    (1646, 4),
    (1680, 0),
    (1803, 6),
    (1828, 0),
    (1970, 3),
    (1997, 0),
    (2661, 13),
    (2695, 0),
    (2888, 8),
    (2903, 0),
    (3051, 9),
    (3125, 0),
    (3475, 14),
    (3494, 0),
    (3729, 2),
    (3740, 0),
    (4239, 5),
    (4254, 0),
    (4793, -1)
]

with open("vid_3_labels.txt", 'w') as f:
    np.savetxt(f, vid_3, fmt="%u")

""" VIDEO 4
Flick finger from left to right
5 Palm forwards, bend 4 fingers
1 Flick hand to the left
Bend 4 finger with palm to the side
12 Form V
9 Contract hand and turn left
Touch plane in few places
10 Contract hand and turn right
14 Cross text out with 1 finger
7 Grab text out of the plane
Palm forward slide to the right
Palm forward slide to the left
Anything random
13 Wipe blackboard motion twice
8 Push hand into the plane
One finger bending
3 CCW circle with 1 finger
Show yes
4 CW circle with 1 finger
Select some text with 2 fingers
Rotate hand from palm down to palm up
Finger bending in row
--- 6 Palm upwards, bend 4 fingers - not preset, lagged
11 Form C
2 Flick hand to the right
"""

vid_4 = [
    (0, 0),
    (457, 5),
    (475, 0),
    (645, 1),
    (660, 0),
    (915, 12),
    (971, 0),
    (1023, 9),
    (1103, 0),
    (1442, 10),
    (1500, 0),
    (1673, 14),
    (1710, 0),
    (1795, 7),
    (1817, 0),
    (2784, 13),
    (2817, 0),
    (3018, 8),
    (3030, 0),
    (3344, 3),
    (3380, 0),
    (3532, 4),
    (3565, 0),
    (4585, 11),
    (4678, 0),
    (4784, 2),
    (4798, 0),
    (4858, -1)
]

with open("vid_4_labels.txt", 'w') as f:
    np.savetxt(f, vid_4, fmt="%u")

""" VIDEO 5
Flick finger from left to right
5 Palm forwards, bend 4 fingers
Select some text with 2 fingers
11 Form C
8 Push hand into the plane
Show yes
14 Cross text out with 1 finger
2 Flick hand to the right
Touch plane in few places
One finger bending
1 Flick hand to the left
10 Contract hand and turn right
Palm forward slide to the right
13 Wipe blackboard motion twice
Finger bending in row
7 Grab text out of the plane
3 CCW circle with 1 finger
Palm forward slide to the left
Anything random
Rotate hand from palm down to palm up
4 CW circle with 1 finger
12 Form V
6 Palm upwards, bend 4 fingers
Bend 4 finger with palm to the side
9 Contract hand and turn left
"""

vid_5 = [
    (0, 0),
    (347, 5),
    (365, 0),
    (947, 11),
    (1020, 0),
    (1140, 8),
    (1150, 0),
    (1320, 14),
    (1357, 0),
    (1467, 2),
    (1477, 0),
    (2292, 1),
    (2303, 0),
    (2425, 10),
    (2505, 0),
    (2662, 13),
    (2700, 0),
    (3352, 7),
    (3375, 0),
    (3562, 3),
    (3588, 0),
    (4709, 4),
    (4736, 0),
    (4808, 12),
    (4870, 0),
    (5045, 6),
    (5067, 0),
    (5517, 9),
    (5580, 0),
    (5629, -1)
]

with open("vid_5_labels.txt", 'w') as f:
    np.savetxt(f, vid_5, fmt="%u")
