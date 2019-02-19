import numpy as np

""" VIDEO 1
3 CCW circle with 1 finger
2 Flick hand to the right
13 Wipe blackboard motion twice
1 Flick hand to the left
4 CW circle with 1 finger
6 Palm upwards, bend 4 fingers
7 Grab text out of the plane
10 Contract hand and turn right
14 Cross text out with 1 finger
12 Form V
8 Push hand into the plane
11 Form C
9 Contract hand and turn left
5 Palm forwards, bend 4 fingers
"""

vid_1 = [
  (0, 0),
  (68, 3),
  (100, 0),
  (171, 2),
  (178, 0),
  (256, 13),
  (288, 0),
  (339, 1),
  (351, 0),
  (400, 4),
  (439, 0),
  (580, 6),
  (600, 0),
  (653, 7),
  (679, 0),
  (796, 10),
  (844, 0),
  (912, 14),
  (943, 0),
  (988, 12),
  (1014, 0),
  (1066, 8),
  (1073, 0),
  (1130, 11),
  (1163, 0),
  (1252, 9),
  (1307, 0),
  (1430, 5),
  (1450, 0),
  (1485, -1)
]


with open("vid_1_labels.txt", 'w') as f:
    np.savetxt(f, vid_1, fmt="%u")

""" VIDEO 2
2 Flick hand to the right
6 Palm upwards, bend 4 fingers
5 Palm forwards, bend 4 fingers
13 Wipe blackboard motion twice
9 Contract hand and turn left
8 Push hand into the plane
4 CW circle with 1 finger
1 Flick hand to the left
14 Cross text out with 1 finger
10 Contract hand and turn right
11 Form C
7 Grab text out of the plane
12 Form V
3 CCW circle with 1 finger
"""

vid_2 = [
    (0, 0),
    (161, 2),
    (170, 0),
    (269, 6),
    (288, 0),
    (354, 5),
    (374, 0),
    (414, 13),
    (450, 0),
    (533, 9),
    (590, 0),
    (669, 8),
    (679, 0),
    (821, 4),
    (859, 0),
    (929, 1),
    (942, 0),
    (1070, 14),
    (1097, 0),
    (1158, 10),
    (1211, 0),
    (1278, 11),
    (1293, 0),
    (1385, 7),
    (1411, 0),
    (1472, 12),
    (1486, 0),
    (1673, 3),
    (1698, 0),
    (1735, -1)
]

with open("vid_2_labels.txt", 'w') as f:
    np.savetxt(f, vid_2, fmt="%u")

""" VIDEO 3
9 Contract hand and turn left
13 Wipe blackboard motion twice
6 Palm upwards, bend 4 fingers
3 CCW circle with 1 finger
2 Flick hand to the right
8 Push hand into the plane
12 Form V
14 Cross text out with 1 finger
1 Flick hand to the left
7 Grab text out of the plane
5 Palm forwards, bend 4 fingers
10 Contract hand and turn right
4 CW circle with 1 finger
11 Form C
"""

vid_3 = [
    (0, 0),
    (34, 9),
    (79, 0),
    (138, 13),
    (174, 0),
    (306, 6),
    (328, 0),
    (403, 3),
    (437, 0),
    (509, 2),
    (519, 0),
    (583, 8),
    (596, 0),
    (670, 12),
    (692, 0),
    (746, 14),
    (779, 0),
    (844, 1),
    (856, 0),
    (904, 7),
    (933, 0),
    (1065, 5),
    (1085, 0),
    (1147, 10),
    (1191, 0),
    (1244, 4),
    (1274, 0),
    (1322, 11),
    (1373, 0),
    (1403, -1)
]

with open("vid_3_labels.txt", 'w') as f:
    np.savetxt(f, vid_3, fmt="%u")

""" VIDEO 4
8 Push hand into the plane
6 Palm upwards, bend 4 fingers
11 Form C
5 Palm forwards, bend 4 fingers
7 Grab text out of the plane
13 Wipe blackboard motion twice
3 CCW circle with 1 finger
2 Flick hand to the right
9 Contract hand and turn left
4 CW circle with 1 finger
10 Contract hand and turn right
1 Flick hand to the left
12 Form V
14 Cross text out with 1 finger
"""

vid_4 = [
    (0, 0),
    (24, 8),
    (35, 0),
    (121, 6),
    (1455, 0),
    (201, 11),
    (231, 0),
    (281, 5),
    (306, 0),
    (339, 7),
    (370, 0),
    (434, 13),
    (475, 0),
    (530, 3),
    (558, 0),
    (680, 2),
    (690, 0),
    (789, 9),
    (836, 0),
    (896, 4),
    (925, 0),
    (1040, 10),
    (1101, 0),
    (1173, 1),
    (1184, 0),
    (1256, 12),
    (1297, 0),
    (1332, 14),
    (1367, 0),
    (1408, -1)
]

with open("vid_4_labels.txt", 'w') as f:
    np.savetxt(f, vid_4, fmt="%u")

""" VIDEO 5
6 Palm upwards, bend 4 fingers
10 Contract hand and turn right
4 CW circle with 1 finger
9 Contract hand and turn left
7 Grab text out of the plane
12 Form V
2 Flick hand to the right
3 CCW circle with 1 finger
11 Form C
14 Cross text out with 1 finger
8 Push hand into the plane
1 Flick hand to the left
13 Wipe blackboard motion twice
5 Palm forwards, bend 4 fingers
"""

vid_5 = [
    (0, 0),
    (32, 6),
    (52, 0),
    (128, 10),
    (189, 0),
    (293, 4),
    (321, 0),
    (388, 9),
    (432, 0),
    (500, 7),
    (527, 0),
    (612, 12),
    (636, 0),
    (691, 2),
    (702, 0),
    (817, 3),
    (843, 0),
    (995, 11),
    (1041, 0),
    (1096, 14),
    (1126, 0),
    (1179, 8),
    (1190, 0),
    (1262, 1),
    (1272, 0),
    (1341, 13),
    (1380, 0),
    (1475, 5),
    (1496, 0),
    (1547, -1)
]

with open("vid_5_labels.txt", 'w') as f:
    np.savetxt(f, vid_5, fmt="%u")

""" RANDOM 1
Palm forward slide to the left
Bend 4 finger with palm to the side
One finger bending
Touch plane in few places
Palm forward slide to the right
Show yes
Wrist rotations
Anything random
Select some text with 2 fingers
Rotate hand from palm down to palm up
Make a fist at various wrist rotations
Extend and relax hand multiple times
"""

random_1 = [
    (0, 0),
    (4887, -1)
]

with open("random_1_labels.txt", 'w') as f:
    np.savetxt(f, random_1, fmt="%u")
