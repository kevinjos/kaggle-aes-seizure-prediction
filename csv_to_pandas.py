import pandas as pd


def convert_data():
	df = pd.read_csv('features.csv')
	test = df['filen'].str.contains('test')
	idf = df['filen'].str.contains('interictal')
	df.insert(0, 'interictal', idf)
	df.insert(1, 'test', test)
	#new dataframe of filename and interictal status
	df1 = df.loc[0:,['interictal', 'test', 'filen']]
	#new dataframe of filename and non-interictal
	preictal = df1.query('~interictal and ~test')
	not_test = df1.query('~test')
	test_data = df.query('test')


