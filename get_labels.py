import pandas as pd
import csv

with open('/home/yankun/spamGAN/dividedData/labeled70/val.csv', mode='r') as origin:
    with open('/home/yankun/spamGAN/dividedData/labeled70/val_label.txt', mode='w') as f:
        reader = csv.DictReader(origin)
        for row in reader:
            f.write(row['Label'])
            f.write('\n')






