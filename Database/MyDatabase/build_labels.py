import numpy as np

random = np.empty(6, dtype='object')

random[0] = [(0, 0), (4887, -1)]  # Ieva
random[1] = [(0, 0), (4188, -1)]  # Steve
random[2] = [(0, 0), (3079, -1)]  # Fin
random[3] = [(0, 0), (2857, -1)]  # Monica
random[4] = [(0, 0), (4664, -1)]  # Static hand (Ieva)
random[5] = [(0, 0), (1000, -1)]  # Zeros

for i in range(len(random)):
    with open("random_{}_labels.txt".format(i+1), 'w') as f:
        np.savetxt(f, random[i], fmt="%u")


vid = np.empty(20, dtype='object')

# Ieva
vid[0] = [
    (0, 0),
    (24, 2), (72, 0),    # REDO - CW circle
    (109, 3), (155, 0),    # COPY - letter C
    (180, 5), (205, 0),    # DELETE - flick hand to the left
    (260, 5), (290, 0),    # DELETE - flick hand to the left
    (329, 1), (380, 0),    # UNDO - CCW circle
    (395, 2), (468, 0),    # REDO - CW circle
    (505, 4), (567, 0),    # PASTE - letter V
    (600, 3), (645, 0),    # COPY - letter C
    (669, 1), (730, 0),    # UNDO - CCW circle
    (758, 4), (819, 0),    # PASTE - letter V
    (830, -1)
]

vid[1] = [
    (0, 0),
    (11, 1), (80, 0),    # UNDO - CCW circle
    (130, 3), (193, 0),    # COPY - letter C
    (232, 4), (279, 0),    # PASTE - letter V
    (319, 5), (344, 0),    # DELETE - flick hand to the left
    (404, 2), (471, 0),    # REDO - CW circle
    (493, 2), (562, 0),    # REDO - CW circle
    (602, 5), (640, 0),    # DELETE - flick hand to the left
    (688, 1), (760, 0),    # UNDO - CCW circle
    (820, 4), (866, 0),    # PASTE - letter V
    (891, 3), (952, 0),    # COPY - letter C
    (986, -1)
]

vid[2] = [
    (0, 0),
    (19, 1), (67, 0),    # UNDO - CCW circle
    (119, 4), (168, 0),    # PASTE - letter V
    (189, 2), (250, 0),    # REDO - CW circle
    (305, 4), (350, 0),    # PASTE - letter V
    (395, 3), (435, 0),    # COPY - letter C
    (461, 5), (483, 0),    # DELETE - flick hand to the left
    (534, 2), (587, 0),    # REDO - CW circle
    (623, 3), (677, 0),    # COPY - letter C
    (693, 1), (740, 0),    # UNDO - CCW circle
    (782, 5), (807, 0),    # DELETE - flick hand to the left
    (815, -1)
]

vid[3] = [
    (0, 0),
    (22, 1), (74, 0),    # UNDO - CCW circle
    (124, 4), (191, 0),    # PASTE - letter V
    (221, 5), (245, 0),    # DELETE - flick hand to the left
    (301, 2), (366, 0),    # REDO - CW circleh
    (388, 2), (460, 0),    # REDO - CW circle
    (488, 1), (548, 0),    # UNDO - CCW circle
    (597, 4), (668, 0),    # PASTE - letter V
    (695, 5), (717, 0),    # DELETE - flick hand to the left
    (784, 3), (836, 0),    # COPY - letter C
    (912, 3), (951, 0),    # COPY - letter C
    (973, -1)
]

vid[4] = [
    (0, 0),
    (12, 4), (81, 0),    # PASTE - letter V
    (90, 2), (136, 0),    # REDO - CW circle
    (193, 3), (220, 0),    # COPY - letter C
    (242, 1), (295, 0),    # UNDO - CCW circle
    (316, 1), (365, 0),    # UNDO - CCW circle
    (411, 5), (431, 0),    # DELETE - flick hand to the left
    (480, 4), (548, 0),    # PASTE - letter V
    (584, 3), (625, 0),    # COPY - letter C
    (651, 5), (677, 0),    # DELETE - flick hand to the left
    (720, 2), (777, 0),    # REDO - CW circle
    (809, -1)
]

# Steve
vid[5] = [
    (0, 0),
    (15, 2), (78, 0),    # REDO - CW circle
    (148, 3), (212, 0),    # COPY - letter C
    (238, 4), (316, 0),    # PASTE - letter V
    (335, 5), (360, 0),    # DELETE - flick hand to the left
    (420, 4), (508, 0),    # PASTE - letter V
    (536, 3), (621, 0),    # COPY - letter C
    (632, 5), (651, 0),    # DELETE - flick hand to the left
    (768, 1), (835, 0),    # UNDO - CCW circle
    (876, 1), (938, 0),    # UNDO - CCW circle
    (983, 2), (1052, 0),    # REDO - CW circle
    (1070, -1)
]

vid[6] = [
    (0, 0),
    (35, 2), (100, 0),    # REDO - CW circle
    (136, 2), (202, 0),    # REDO - CW circle
    (278, 5), (307, 0),    # DELETE - flick hand to the left
    (379, 3), (468, 0),    # COPY - letter C
    (486, 5), (509, 0),    # DELETE - flick hand to the left
    (645, 1), (711, 0),    # UNDO - CCW circle
    (819, 4), (892, 0),    # PASTE - letter V
    (920, 3), (999, 0),    # COPY - letter C
    (1026, 4), (1106, 0),    # PASTE - letter V
    (1188, 1), (1270, 0),    # UNDO - CCW circle
    (1300, -1)
]

vid[7] = [
    (0, 0),
    (29, 3), (128, 0),    # COPY - letter C
    (191, 2), (260, 0),    # REDO - CW circle
    (339, 4), (449, 0),    # PASTE - letter V
    (541, 1), (616, 0),    # UNDO - CCW circle
    (705, 3), (798, 0),    # COPY - letter C
    (870, 1), (935, 0),    # UNDO - CCW circle
    (1018, 5), (1044, 0),    # DELETE - flick hand to the left
    (1139, 4), (1199, 0),    # PASTE - letter V
    (1240, 5), (1261, 0),    # DELETE - flick hand to the left
    (1411, 2), (1497, 0),    # REDO - CW circle
    (1520, -1)
]

vid[8] = [
    (0, 0),
    (6, 2), (77, 0),    # REDO - CW circle
    (183, 4), (266, 0),    # PASTE - letter V
    (344, 1), (416, 0),    # UNDO - CCW circle
    (500, 3), (588, 0),    # COPY - letter C
    (630, 5), (650, 0),    # DELETE - flick hand to the left
    (762, 3), (852, 0),    # COPY - letter C
    (909, 1), (986, 0),    # UNDO - CCW circle
    (1082, 4), (1151, 0),    # PASTE - letter V
    (1184, 5), (1205, 0),    # DELETE - flick hand to the left
    (1343, 2), (1431, 0),    # REDO - CW circle
    (1450, -1)
]

vid[9] = [
    (0, 0),
    (5, 1), (76, 0),    # UNDO - CCW circle
    (163, 4), (276, 0),    # PASTE - letter V
    (307, 3), (397, 0),    # COPY - letter C
    (460, 2), (538, 0),    # REDO - CW circle
    (584, 2), (669, 0),    # REDO - CW circle
    (763, 5), (800, 0),    # DELETE - flick hand to the left
    (905, 4), (996, 0),    # PASTE - letter V
    (1090, 1), (1156, 0),    # UNDO - CCW circle
    (1275, 5), (1303, 0),    # DELETE - flick hand to the left
    (1406, 3),    # COPY - letter C
    (1512, -1)
]

# Tomas
vid[10] = [
    (0, 0),
    (13, 5), (48, 0),    # DELETE - flick hand to the left
    (152, 1), (214, 0),    # UNDO - CCW circle
    (227, 2), (300, 0),    # REDO - CW circle
    (248, 3), (394, 0),    # COPY - letter C
    (418, 4), (466, 0),    # PASTE - letter V
    (494, 3), (549, 0),    # COPY - letter C
    (561, 5), (593, 0),    # DELETE - flick hand to the left
    (642, 4), (701, 0),    # PASTE - letter V
    (722, 2), (787, 0),    # REDO - CW circle
    (800, 1), (878, 0),    # UNDO - CCW circle
    (900, -1)
]

vid[11] = [
    (0, 0),
    (26, 3), (70, 0),    # COPY - letter C
    (140, 1), (202, 0),    # UNDO - CCW circle
    (237, 2), (289, 0),    # REDO - CW circle
    (326, 2), (378, 0),    # REDO - CW circle
    (466, 4), (549, 0),    # PASTE - letter V
    (575, 3), (653, 0),    # COPY - letter C
    (684, 1), (745, 0),    # UNDO - CCW circle
    (873, 5), (896, 0),    # DELETE - flick hand to the left
    (970, 4), (1035, 0),    # PASTE - letter V
    (1056, 5), (1087, 0),    # DELETE - flick hand to the left
    (1110, -1)
]

vid[12] = [
    (0, 0),
    (32, 1), (94, 0),    # UNDO - CCW circle
    (125, 2), (200, 0),    # REDO - CW circle
    (240, 4), (282, 0),    # PASTE - letter V
    (303, 5), (335, 0),    # DELETE - flick hand to the left
    (390, 4), (444, 0),    # PASTE - letter V
    (465, 2), (518, 0),    # REDO - CW circle
    (540, 1), (604, 0),    # UNDO - CCW circle
    (629, 5), (659, 0),    # DELETE - flick hand to the left
    (715, 3), (755, 0),    # COPY - letter C
    (826, 3), (869, 0),    # COPY - letter C
    (880, -1)
]

vid[13] = [
    (0, 0),
    (10, 5), (41, 0),    # DELETE - flick hand to the left
    (108, 4), (168, 0),    # PASTE - letter V
    (182, 2), (243, 0),    # REDO - CW circle
    (265, 1), (335, 0),    # UNDO - CCW circle
    (370, 5), (395, 0),    # DELETE - flick hand to the left
    (453, 2), (507, 0),    # REDO - CW circle
    (550, 3), (586, 0),    # COPY - letter C
    (650, 3), (710, 0),    # COPY - letter C
    (735, 4), (810, 0),    # PASTE - letter V
    (889, 1), (943, 0),    # UNDO - CCW circle
    (960, -1)
]

vid[14] = [
    (0, 0),
    (16, 1), (91, 0),    # UNDO - CCW circle
    (137, 5), (164, 0),    # DELETE - flick hand to the left
    (227, 3), (302, 0),    # COPY - letter C
    (325, 5), (363, 0),    # DELETE - flick hand to the left
    (418, 3), (459, 0),    # COPY - letter C
    (493, 4), (554, 0),    # PASTE - letter V
    (573, 2), (648, 0),    # REDO - CW circle
    (682, 2), (758, 0),    # REDO - CW circle
    (815, 4), (881, 0),    # PASTE - letter V
    (895, 1), (979, 0),    # UNDO - CCW circle
    (1000, -1)
]

# Jamie
vid[15] = [
    (0, 0),
    (12, 2), (74, 0),    # REDO - CW circle
    (157, 1), (224, 0),    # UNDO - CCW circle
    (339, 3), (413, 0),    # COPY - letter C
    (450, 5), (486, 0),    # DELETE - flick hand to the left
    (552, 3), (678, 0),    # COPY - letter C
    (875, 2), (945, 0),    # REDO - CW circle
    (1023, 1), (1090, 0),    # UNDO - CCW circle
    (1163, 4), (1213, 0),    # PASTE - letter V
    (1236, 5), (1262, 0),    # DELETE - flick hand to the left
    (1326, 4), (1388, 0),    # PASTE - letter V
    (1410, -1)
]

vid[16] = [
    (0, 0),
    (35, 3), (80, 0),    # COPY - letter C
    (104, 5), (135, 0),    # DELETE - flick hand to the left
    (277, 1), (335, 0),    # UNDO - CCW circle
    (419, 2), (480, 0),    # REDO - CW circle
    (539, 1), (604, 0),    # UNDO - CCW circle
    (653, 4), (695, 0),    # PASTE - letter V
    (731, 3), (783, 0),    # COPY - letter C
    (822, 4), (862, 0),    # PASTE - letter V
    (893, 5), (921, 0),    # DELETE - flick hand to the left
    (1016, 2), (1080, 0),    # REDO - CW circle
    (1100, -1)
]

vid[17] = [
    (0, 0),
    (15, 1), (65, 0),    # UNDO - CCW circle
    (140, 3), (209, 0),    # COPY - letter C
    (243, 4), (306, 0),    # PASTE - letter V
    (335, 5), (365, 0),    # DELETE - flick hand to the left
    (468, 2), (530, 0),    # REDO - CW circle
    (614, 3), (659, 0),    # COPY - letter C
    (740, 2), (818, 0),    # REDO - CW circle
    (870, 4), (945, 0),    # PASTE - letter V
    (1034, 1), (1094, 0),    # UNDO - CCW circle
    (1202, 5), (1222, 0),    # DELETE - flick hand to the left
    (1235, -1)
]

vid[18] = [
    (0, 0),
    (34, 3), (111, 0),    # COPY - letter C
    (260, 2), (326, 0),    # REDO - CW circle
    (411, 5), (442, 0),    # DELETE - flick hand to the left
    (495, 4), (557, 0),    # PASTE - letter V
    (598, 5), (626, 0),    # DELETE - flick hand to the left
    (918, 1), (978, 0),    # UNDO - CCW circle
    (1037, 1), (1101, 0),    # UNDO - CCW circle
    (1151, 2), (1234, 0),    # REDO - CW circle
    (1334, 3), (1387, 0),    # COPY - letter C
    (1417, 4), (1491, 0),    # PASTE - letter V
    (1510, -1)
]

vid[19] = [
    (0, 0),
    (25, 2), (98, 0),    # REDO - CW circle
    (176, 1), (248, 0),    # UNDO - CCW circle
    (330, 4), (392, 0),    # PASTE - letter V
    (563, 1), (627, 0),    # UNDO - CCW circle
    (712, 2), (807, 0),    # REDO - CW circle
    (922, 5), (954, 0),    # DELETE - flick hand to the left
    (1086, 3), (1166, 0),    # COPY - letter C
    (1262, 3), (1354, 0),    # COPY - letter C
    (1450, 4), (1544, 0),    # PASTE - letter V
    (1606, 5), (1634, 0),    # DELETE - flick hand to the left
    (1650, -1)
]

for i in range(len(vid)):
    with open("vid_{}_labels.txt".format(i+1), 'w') as f:
        np.savetxt(f, vid[i], fmt="%u")
