import pandas as pd

def get_df():
  f = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/features.csv'
  df = pd.read_csv(f)
  return df

def convert_data(df):
  testset = df['filen'].str.contains('test')
  interictal = df['filen'].str.contains('interictal')
  subject=['_'.join(f.split('_')[:2]) for f in df.loc[:,'filen']]
  df.insert(0, 'interictal', interictal)
  df.insert(0, 'test', testset)
  df.insert(0, 'subject', subject)
  '''
  data == dictionary of dictionaries
  {'SUBJECT_0':{'TEST':df, 'TRAIN':df}, ...
   'SUBJECT_N:{'TEST':df, 'TRAIN':df}}
  '''
  data = {}
  for sub in set(subject):
    data[sub] = {'test':df.query('test and subject==@sub').reset_index(), 'train':df.query('~test and subject==@sub').reset_index()}
  return data

  '''
  #new dataframe of filename and interictal status
  df1 = df.loc[0:,['interictal', 'test', 'filen']]
  #new dataframe of filename and non-interictal
  preictal = df1.query('~interictal and ~test')
  not_test = df1.query('~test')
  test_data = df.query('test')
  '''

def main():
  df = get_df()
  data = convert_data(df)
  return data

if __name__ == '__main__':
  main()
