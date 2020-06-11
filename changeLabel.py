import pandas as pd

df = pd.read_csv('/home/yankun/dividedData/labeledTrain.csv')
df.loc[(df.Label == -1), 'Label'] = 1
df.to_csv('/home/yankun/dividedData/labeledTrain.csv')
