import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import json
import statistics
from collections import Counter

'''
Task description:
choose 3 ML predictive models  to solve ONE problem with the Covid-19 dataset and compare their results

Problem: Outcome Projection

-> Classification: Quick discharge / Severe and Unstable / Death
-> Regression: 


Models
- Light Gradient Boosted Model / Random Forest
- Logistic Regression
- SVM
- Multivariate Linear Regression
'''
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for np types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DataPreprocess:
	'''
	Reads epidemiology dataset from csv and prepares it for ML processing
	'''
	def __init__(self, fileName):
		self.df = pd.read_csv(f'./{fileName}', low_memory=False, dtype={'age': 'string'})

	def standardize_age(self):
		'''
		Change all age values to float
		'''
		new_age_col = []
		for age in self.df['age']:
			if re.match('[0-9]+\s*\-\s*[0-9]*', str(age)):
				split_range = age.split('-')
				lb = float(age.split('-')[0])
				try:
					ub = float(age.split('-')[1])
					age_possibilities = [i for i in range(int(lb), int(ub)+1)]
					mean_age = statistics.mean(age_possibilities)
				except:
					new_age_col.append(lb)
					continue

				new_age_col.append(mean_age)

			else:
				new_age_col.append(age)

		self.df.drop(['age'], axis=1, inplace=True)
		self.df['age'] = new_age_col
		self.df = self.df.astype({'age': np.float64})

	def standardize_outcome(self):
		'''
		Combining redundant values in outcome column => 0: Dead, 1: Alive
		'''
		self.df['outcome'] = np.where(self.df['outcome'] == 'death', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'died', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Death', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'dead', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Dead', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Died', 0, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Deceased', 0, self.df['outcome'])

		self.df['outcome'] = np.where(self.df['outcome'] == 'discharged', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'discharge', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Discharged', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Discharged from hospital', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'not hospitalized', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'recovered', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'recovering at home 03.03.2020', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'released from quarantine', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'stable', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Symptoms only improved with cough. Currently hospitalized for follow-up.', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Recovered', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'stable condition', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Stable', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Under treatment', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Receiving Treatment', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Migrated', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Migrated_Other', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Hospitalized', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'critical condition, intubated as of 14.02.2020', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'severe', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'treated in an intensive care unit (14.02.2020)', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'Critical condition', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'severe illness', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'unstable', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'critical condition', 1, self.df['outcome'])

		# self.df['outcome'] = np.where(self.df['outcome'] == 'critical condition, intubated as of 14.02.2020', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'severe', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'treated in an intensive care unit (14.02.2020)', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'Critical condition', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'severe illness', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'unstable', 'Severe', self.df['outcome'])
		# self.df['outcome'] = np.where(self.df['outcome'] == 'critical condition', 'Severe', self.df['outcome'])

	def standardize_symptoms(self, num_rows):
		'''
		Categorize symptoms and create binary columns for each category
		'''
		new_data = {'Asymptomatic':np.zeros(num_rows, dtype=int), 
					'Respiratory_Issues':np.zeros(num_rows, dtype=int), 
					'Cardiac_Issues':np.zeros(num_rows, dtype=int), 
					'Pneumonia':np.zeros(num_rows, dtype=int), 
					'Organ_Failure':np.zeros(num_rows, dtype=int), 
					'General_Unwellness':np.zeros(num_rows, dtype=int), 
					'anorexia, fatigue':np.zeros(num_rows, dtype=int), 
					'obnubilation, somnolence':np.zeros(num_rows, dtype=int), 
					'lesions_on_chest_radiographs':np.zeros(num_rows, dtype=int),
					'primary_myelofibrosis':np.zeros(num_rows, dtype=int), 
					'significant_clinical_suspicion':np.zeros(num_rows, dtype=int), 
					'none':np.zeros(num_rows, dtype=int), 
					'severe':np.zeros(num_rows, dtype=int)
					}

		for index, sym in enumerate(list(self.df.symptoms)):
			sym = str(sym).lower()
			if re.match('^(acute kidney|acute respiratory|arrhythmia|acute myocardial|respiratory symptoms|myocardial infarction:acute respiratory distress syndrome|fever:cough:acute respiratory distress syndrome|septic shock:cardiogenic shock:acute respiratory distress syndrome|torpid evolution with respiratory distress and severe bronchopneumonia)', sym):
				new_data['Respiratory_Issues'][index] = 1
			elif re.match('^(cardiac|cardio|myocardial infarction:pneumonia:multiple electrolyte imbalance|acute coronary syndrome)', sym):
				new_data['Cardiac_Issues'][index] = 1
			elif 'asymptomatic' in sym:
				new_data['Asymptomatic'][index] = 1
			elif ((re.match('^(pneumonia|chest discomfort|chest distress|dyspnea)+', sym)) or (sym == 'difficulty breathing') or (sym == 'dry cough') or (re.match('(shortness of breath)$|(sensation of chill)$|pneumonia$', sym)) or (sym == 'fever, pneumonia, sore throat') or ('pneumonia' in sym)):
				new_data['Pneumonia'][index] = 1
			elif 'failure' in sym:
				new_data['Organ_Failure'][index] = 1
			elif re.search('(chills|cough|diarrhea|fever|discomfort|fatigue|systemic weakness)', sym):
				new_data['General_Unwellness'][index] = 1
			else:
				if ',' not in sym:
					sym = '_'.join(sym.split(' '))
				if sym in new_data:
					new_data[sym][index] = 1
				else:
					new_data[sym] = np.zeros(num_rows, dtype=int)
					new_data[sym][index] = 1


		self.df = self.df.drop(['symptoms'], axis=1)
		new_symptoms = pd.DataFrame.from_dict(new_data)
		self.df = pd.concat([self.df, new_symptoms], axis=1)

	def replace_chronic_disease_to_bin(self):
		'''
		Replace chronic disease boolean string to binary int
		'''
		self.df['chronic_disease_binary'] = np.where(self.df['chronic_disease_binary'] == 'TRUE', 1, self.df['chronic_disease_binary'])
		self.df['chronic_disease_binary'] = np.where(self.df['chronic_disease_binary'] == 'FALSE', 0, self.df['chronic_disease_binary'])

	def replace_sex_to_bin(self):
		'''
		Replace age string to binary int: 0 => female; 1 => male
		'''
		self.df['sex'] = np.where(self.df['sex'] == 'male', 1, self.df['sex'])
		self.df['sex'] = np.where(self.df['sex'] == 'female', 0, self.df['sex'])

	def clean_df(self):
		try:
			self.df = self.df.drop(['date_onset_symptoms', 'reported_market_exposure','travel_history_dates', 'location', 
									'travel_history_location'], axis=1)
			self.df = self.df.drop(['admin1', 'admin2', 'admin3', 'admin_id', 'travel_history_binary', 'notes_for_discussion', 
									'sequence_available', 'source', 'geo_resolution', 'lives_in_Wuhan', 'latitude', 'longitude', 
									'data_moderator_initials', 'country_new', 'city', 'province', 'additional_information', 
									'date_death_or_discharge', 'date_admission_hospital'
									], axis=1)
		except:
			pass

		self.df = self.df[self.df.outcome != 'https://www.mspbs.gov.py/covid-19.php']
		self.df = self.df[self.df.chronic_disease != 'Iran; Kuala Lumpur, Federal Territory of Kuala Lumpur, Malaysia']
		self.df = self.df.dropna(subset=['outcome', 'age', 'sex', 'country'])
		self.df = self.df.reset_index(drop=True)
		self.df.symptoms.fillna('none', inplace=True)

		self.standardize_outcome()
		self.standardize_age()
		self.standardize_symptoms(self.df.shape[0])
		self.replace_chronic_disease_to_bin()
		self.replace_sex_to_bin()

	def data_explore(self, feature):
		# sns.countplot(x='sex', data=self.df, palette='hls')
		self.df['age'].hist(bins=10)
		plt.savefig('age2_plot.png')
		plt.show()

	def debug(self):
		self.df.to_csv('./modified_latestdata2.csv', encoding='utf-8', index=False)

		# print(self.df.isna().sum())

		# sym = list(self.df.symptoms.unique())
		# sym.sort()
		# with open('symptoms.txt', 'w', encoding='utf-8') as f:
		# 	for i in sym:
		# 		f.write(i + '\n')
		
		# print(Counter(self.df.symptoms), '\n')
		print(self.df.chronic_disease.unique(), '\n')

data = DataPreprocess('modified_latestdata.csv')
data.clean_df()
# data.data_explore('age')
data.debug()