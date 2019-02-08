from collections import Counter
import csv

gestures = {}
with open("agreement_gestures.csv", 'r') as csvf:
    reader = csv.reader(csvf)
    next(reader)
    for row in reader:
        print(row[0])
        print(Counter(row[1:]))
