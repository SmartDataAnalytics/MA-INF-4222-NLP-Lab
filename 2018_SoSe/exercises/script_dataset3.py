import random
import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

ds1 = sys.argv[1]
ds2 = sys.argv[2]

def get_dataset3_split(dataset1_in, dataset2_in):
    try:
        print('processing datasets')
        print('ds1=', dataset1_in)
        print('ds2=', dataset2_in)

        print('-- fake news')
        df1 = pd.read_csv(dataset1_in, sep=',', usecols=['title','text','label'])
        df1['claim'] = df1[['title', 'text']].apply(lambda x: '. '.join(x), axis=1)
        del df1['title']
        del df1['text']
        df1.rename(index=str, columns={'label': 'y'}, inplace=True)
        print(df1.keys())
        print(len(df1[df1['y']=='REAL']))
        print(len(df1[df1['y']=='FAKE']))
        df1['y'] = np.where(df1['y'] == 'FAKE', 'false', 'true')
        print(len(df1))

        print('-- liar liar')
        df2 = pd.read_csv(dataset2_in, sep='\t', header=None, usecols=[1,2], names=['y', 'claim'])
        print(df2.keys())
        print(set(df2.y), len(df2))
        print(len(df2[df2['y'] == 'true']))
        print(len(df2[df2['y'] == 'false']))
        df2=df2[(df2['y'] == 'true') | (df2['y'] == 'false')]
        print(set(df2.y), len(df2))

        df3=pd.concat([df1, df2], ignore_index=True)

        print(df3['y'].value_counts())
        print('done')
        return train_test_split(df3['claim'], df3['y'], test_size=0.30, random_state=4222)
    except Exception as e:
        print(e)

ds3_train, ds3_test, ds3_y_train, ds3_y_test = get_dataset3_split(ds1,ds2)
print(len(ds3_train))
print(len(ds3_test))
print(ds3_y_train.get_values()[1], ds3_train.get_values()[1])
