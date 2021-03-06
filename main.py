from operator import mod
import re
import os
from argparse import ArgumentParser
from collections import Counter
from sys import prefix
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import time

import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE

'''
Models
- Gradient Boosted Model / Random Forest
- Logistic Regression
- SVM
'''

class DataCleaningAndPreprocess:
	'''
	Reads epidemiology dataset from csv and prepares it for ML processing
	'''
	def __init__(self, fileName):
		self.df = pd.read_csv(f'./{fileName}', low_memory=False, dtype={'age': 'string'})
		self.ori_df = self.df.copy(deep=True)

	def standardize_age(self):
		'''
		Change all age values to float
		'''
		new_age_col = []
		for age in self.df['age']:
			if re.match('[0-9]+\s*\-\s*[0-9]*', str(age)):
				split_range = age.split('-')
				lb = int(age.split('-')[0])
				try:
					ub = int(age.split('-')[1])
					age_possibilities = [i for i in range(int(lb), int(ub)+1)]
					mean_age = int(statistics.mean(age_possibilities))
				except:
					new_age_col.append(lb)
					continue

				new_age_col.append(mean_age)

			else:
				try:
					age = int(float(age))
					new_age_col.append(age)
				except:
					new_age_col.append(0)

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
		self.df['outcome'] = np.where(self.df['outcome'] == 'Alive', 1, self.df['outcome'])
		self.df['outcome'] = np.where(self.df['outcome'] == 'unstable', 1, self.df['outcome'])
  
		unrelated_rows = self.df[(self.df['outcome'] != 1) & (self.df['outcome'] != 0)].index
		self.df.drop(unrelated_rows, inplace=True)

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
		Col 7 - 20
		'''
		new_data = {'asymptomatic':np.zeros(num_rows, dtype=int), 
					'respiratory_issues':np.zeros(num_rows, dtype=int), 
					'cardiac_issues':np.zeros(num_rows, dtype=int), 
					'pneumonia':np.zeros(num_rows, dtype=int), 
					'organ_failure':np.zeros(num_rows, dtype=int), 
					'general_unwellness':np.zeros(num_rows, dtype=int), 
					'anorexia, fatigue':np.zeros(num_rows, dtype=int), 
					'obnubilation, somnolence':np.zeros(num_rows, dtype=int), 
					'lesions_on_chest_radiographs':np.zeros(num_rows, dtype=int),
					'primary_myelofibrosis':np.zeros(num_rows, dtype=int), 
					'significant_clinical_suspicion':np.zeros(num_rows, dtype=int), 
					'unknown_symptoms':np.zeros(num_rows, dtype=int), 
					'severe':np.zeros(num_rows, dtype=int)
					}

		for index, sym in enumerate(list(self.df.symptoms)):
			sym = str(sym).lower()
			if re.match('^(acute kidney|acute respiratory|arrhythmia|acute myocardial|respiratory symptoms|myocardial infarction:acute respiratory distress syndrome|fever:cough:acute respiratory distress syndrome|septic shock:cardiogenic shock:acute respiratory distress syndrome|torpid evolution with respiratory distress and severe bronchopneumonia)', sym):
				new_data['respiratory_issues'][index] = 1
				if 'pneumonia' in sym:
					new_data['pneumonia'][index] = 1
				elif 'failure' in sym:
					new_data['organ_failure'][index] = 1
			elif re.match('^(cardiac|cardio|myocardial infarction:pneumonia:multiple electrolyte imbalance|acute coronary syndrome)', sym):
				new_data['cardiac_issues'][index] = 1
				if 'respiratory' in sym:
					new_data['respiratory_issues'][index] = 1
			elif 'asymptomatic' in sym:
				new_data['asymptomatic'][index] = 1
			elif ((re.match('^(pneumonia|chest discomfort|chest distress|dyspnea)+', sym)) or (sym == 'difficulty breathing') or (sym == 'dry cough') or (re.match('(shortness of breath)$|(sensation of chill)$|pneumonia$', sym)) or (sym == 'fever, pneumonia, sore throat') or ('pneumonia' in sym)):
				new_data['pneumonia'][index] = 1
				if 'respiratory' in sym:
					new_data['respiratory_issues'][index] = 1
				elif 'failure' in sym:
					new_data['organ_failure'][index] = 1
			elif 'failure' in sym:
				new_data['organ_failure'][index] = 1
				if 'pneumonia' in sym:
					new_data['pneumonia'][index] = 1
			elif re.search('(chills|cough|diarrhea|fever|discomfort|fatigue|systemic weakness)', sym):
				new_data['general_unwellness'][index] = 1
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

	def standardize_chronic_disease(self, num_rows):
		'''
		Categorize chronic diseases and create binary columns for each category
		Col 21 - 45
		'''
		new_data = {
					'COPD':np.zeros(num_rows, dtype=int),
					'hypertension':np.zeros(num_rows, dtype=int),
					'diabetes':np.zeros(num_rows, dtype=int),
					'heart_disease':np.zeros(num_rows, dtype=int),
					'cancer':np.zeros(num_rows, dtype=int),
					'asthma':np.zeros(num_rows, dtype=int),
					'chronic_kidney_disease':np.zeros(num_rows, dtype=int),
					'chronic_bronchitis':np.zeros(num_rows, dtype=int),
					'stenocardia':np.zeros(num_rows, dtype=int),
					'dyslipidemia':np.zeros(num_rows, dtype=int),
					'atherosclerosis':np.zeros(num_rows, dtype=int),
					'prostate_issues':np.zeros(num_rows, dtype=int),
					'hemorrhage':np.zeros(num_rows, dtype=int),
					'cerebral_infarction':np.zeros(num_rows, dtype=int),
					'cerebrovascular_infarct':np.zeros(num_rows, dtype=int),
					'upper_git_bleeding':np.zeros(num_rows, dtype=int),
					'cardiovascular_disease':np.zeros(num_rows, dtype=int),
					'renal_disease':np.zeros(num_rows, dtype=int),
					'hyperthyroidism':np.zeros(num_rows, dtype=int),
					'pre-renal_azotemia':np.zeros(num_rows, dtype=int),
					'obstructive_pulmonary_disease':np.zeros(num_rows, dtype=int),
					'tuberculosis':np.zeros(num_rows, dtype=int),
					"parkinson's_disease":np.zeros(num_rows, dtype=int),
					'unknown_diseases':np.zeros(num_rows, dtype=int)
					}

		keys = [' '.join(i.split('_')).lower() for i in list(new_data.keys())]
		for index, disease in enumerate(list(self.df.chronic_disease)):
			flag = False
			disease = str(disease).lower()
			for key in keys:
				if ((key in disease) or (key == disease)):
					new_data[list(new_data.keys())[keys.index(key)]][index] = 1
					flag = True

			if (('cardiomyopathy' in disease) or ('cardiac' in disease)):
				new_data['heart_disease'][index] = 1
				flag = True
			elif 'copd' in disease:
				new_data['COPD'][index] = 1
				flag = True
			elif 'renal' in disease:
				new_data['renal_disease'][index] = 1
				flag = True
			elif 'hypertensive' in disease:
				new_data['hypertension'][index] = 1
				flag = True
			elif (('prostate' in disease) or ('prostatic' in disease)):
				new_data['prostate_issues'][index] = 1
				flag = True
			else:
				if not flag:
					disease = '_'.join(disease.split(' '))
					new_data[disease] = np.zeros(num_rows, dtype=int)
					new_data[disease][index] = 1

		self.df = self.df.drop(['chronic_disease'], axis=1)
		new_cd = pd.DataFrame.from_dict(new_data)
		self.df = pd.concat([self.df, new_cd], axis=1)

	def replace_chronic_disease_to_bin(self):
		'''
		Replace chronic disease boolean string to binary int
		'''
		self.df['chronic_disease_binary'] = self.df['chronic_disease_binary'].replace(['True'], 1)
		self.df['chronic_disease_binary'] = self.df['chronic_disease_binary'].replace(['False'], 0)

	def replace_sex_to_bin(self):
		'''
		Replace age string to binary int: 0 => female; 1 => male
		'''
		self.df['sex'] = np.where(self.df['sex'] == 'male', 1, self.df['sex'])
		self.df['sex'] = np.where(self.df['sex'] == 'female', 0, self.df['sex'])

	def clean_df(self):
		'''
		Main function to remove empty / unncessary data and standardize the remaining features
		'''
		try:
			self.df = self.df.drop(['admin1', 'admin2', 'admin3', 'admin_id', 'travel_history_binary', 'notes_for_discussion', 'sequence_available', 'source', 'geo_resolution', 'lives_in_Wuhan', 'latitude', 'longitude', 
									'data_moderator_initials', 'country_new', 'city', 'province', 'additional_information', 
									'date_death_or_discharge', 'date_admission_hospital', 'date_onset_symptoms', 'reported_market_exposure',
         							'travel_history_dates', 'location', 'travel_history_location', 'date_confirmation', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35',
       								'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38'
									], axis=1)
		except:
			pass

		self.df = self.df[self.df.outcome != 'https://www.mspbs.gov.py/covid-19.php']
		self.df = self.df[self.df.chronic_disease != 'Iran; Kuala Lumpur, Federal Territory of Kuala Lumpur, Malaysia']
		self.df = self.df.dropna(subset=['outcome', 'age', 'sex', 'country'])
		self.df = self.df.reset_index(drop=True)
		self.df.symptoms.fillna('unknown_symptoms', inplace=True)
		self.df.chronic_disease.fillna('unknown_diseases', inplace=True)

		self.standardize_outcome()
		self.standardize_age()
		self.standardize_symptoms(self.df.shape[0])
		self.standardize_chronic_disease(self.df.shape[0])
		self.replace_chronic_disease_to_bin()
		self.replace_sex_to_bin()
		self.df = self.df.dropna()
		self.df = self.df.reset_index(drop=True)
  
		lab_enc = preprocessing.LabelEncoder()
		self.df.outcome = lab_enc.fit_transform(self.df.outcome)
		self.df = pd.get_dummies(self.df, columns=['country'])

	def split_data(self, use_smote='False'):
		'''
		Splits data into 80:20 training and testing set - option to also use SMOTE to balance classes in training set
		'''
		covid_dataset = self.df
		y = covid_dataset.outcome
		x = covid_dataset.drop(['ID','outcome'], axis=1)
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
		# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
		is_smote = False

		if use_smote == 'True':
			os = SMOTE(random_state=0)
			is_smote = True
			X_cols = X_train.columns
			os_data_X, os_data_y = os.fit_resample(X_train, y_train)
			os_data_X = pd.DataFrame(data=os_data_X, columns=X_cols)
			os_data_y = pd.DataFrame(data=os_data_y, columns=['outcome'])
			X_train = os_data_X
			y_train = os_data_y

		return X_train, X_test, y_train, y_test, is_smote

	def create_corr_matrix(self, y, x, is_plot=False):
		yX = pd.concat([y, x], axis=1)
		yX = yX.rename(columns={0: 'TARGET'})

		yX_corr = yX.corr(method='pearson')
		yX_abs_corr = np.abs(yX_corr)

		if is_plot:
		    plt.figure(figsize=(10, 10))
		    plt.imshow(yX_abs_corr, cmap='RdYlGn', interpolation='none', aspect='auto')
		    plt.colorbar()
		    plt.xticks(range(len(yX_abs_corr)), yX_abs_corr.columns, rotation='vertical')
		    plt.yticks(range(len(yX_abs_corr)), yX_abs_corr.columns)
		    plt.suptitle('Pearson Correlation Heat Map (absolute values)', fontsize=15, fontweight='bold')
		    plt.show()

		return yX, yX_corr, yX_abs_corr

	def data_explore(self):
		'''
		Exploring features in dataset via visualizations
		'''
		# chronic_diseases = self.df.iloc[:, 21:45]
		# for col in chronic_diseases:
		# 	sns.countplot(x=col, data=self.df, palette='hls')
		# # self.df['age'].hist(bins=10)
		# 	# plt.savefig('age2_plot.png')

		## visualizing outcome counts
		sns.countplot(x='outcome', data=self.df, palette='hls')
		plt.savefig('./Data_Exploration/outcome_plot.png')
		
		## visualizing missing values
		plt.figure(figsize=(10,10))
		plt.subplots_adjust(top=0.9, bottom=0.25)
		sns.heatmap(self.ori_df.isnull(), cbar=False)
		plt.savefig('./Data_Exploration/missing_vals')
		plt.show()
		plt.clf()

	def debug(self):
		self.df.to_csv('./modified_latestdata2.csv', encoding='utf-8', index=False)
		# print(self.df.isna().sum())


def check_class_ratio(x_train, x_test, y_train, y_test, is_smote):
	'''
	Shows the distribution of outcomes in the training and testing sets
	'''
	print("x_train.shape {}, y_train.shape {}".format(x_train.shape, y_train.shape))

	if is_smote:
		y_train_df = y_train
	else:
		y_train_df = pd.DataFrame(data=y_train)

	y_test_df = pd.DataFrame(data=y_test)

	train_outcome_counts = y_train_df.value_counts()
	print('Class distribution in training set:')
	print(train_outcome_counts)
	print('Percentage of positive class samples: {}'.format(100 * train_outcome_counts[1] / len(y_train_df)))

	print('\n-------------------------------------------------------------------------------\n')
	print("x_test.shape {}, y_test.shape {}".format(x_test.shape, y_test.shape))

	test_outcome_counts = y_test_df.value_counts()
	print('Class distribution in testing set:')
	print(test_outcome_counts)
	print('Percentage of positive class samples: {}'.format(100 * test_outcome_counts[1] / len(y_test_df)))


def param_tuning_2d_gs_heatmap(grid_search, grid_params, x, y, is_verbose=True):
	'''
	Visualizes hyperparameter tuning scores
	'''
	grid_params_x = grid_params[x]
	grid_params_y = grid_params[y]

	results = pd.DataFrame(grid_search.cv_results_)
	ar_scores = np.array(results.mean_test_score).reshape(len(grid_params_y), len(grid_params_x))
	sns.heatmap(ar_scores, annot=True, fmt='.3f', xticklabels=grid_params_x, yticklabels=grid_params_y)

	plt.suptitle('Param Tuning Grid Search Heatmap')
	plt.xlabel(x)
	plt.ylabel(y)
	plt.show()

	if is_verbose:
		print(f'grid_search.best_score: {grid_search.best_score_}')
		print(f'grid_search.best_estimator_: {grid_search.best_estimator_}')


class implementLogisticRegression:
	def __init__ (self, training_data, testing_data):
		print('Initialising Logistic Regression Model')
		self.X_train = training_data[0]
		self.y_train = np.ravel(training_data[1])
		self.X_test = testing_data[0]
		self.y_test = np.ravel(testing_data[1])
		self.logReg = None
		self.outcome_predict = 0

	def train(self, model_dir):
		print('\nBegin training')
		learning_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1', 'l2']}
		self.logReg = LogisticRegression(class_weight='balanced', dual=False, intercept_scaling=1, 
									max_iter=500, n_jobs=1, random_state=0, tol=0.0001, penalty='l2', 
									solver='liblinear', verbose=0, warm_start=False
									)

		gs_lr = GridSearchCV(self.logReg, learning_params, return_train_score=True)
		gs_lr.fit(self.X_train, self.y_train)
		self.logReg = gs_lr.best_estimator_
		param_tuning_2d_gs_heatmap(gs_lr, learning_params, 'C', 'penalty')

		with open(f'./{model_dir}/lr_trained.pkl', 'wb') as f:
			pickle.dump(self.logReg, f)
		f.close()
		print('Training completed successfully')

	def predict(self):
		print('\nRunning prediction on test data')
		self.outcome_predict = self.logReg.predict(self.X_test)
		# print('Accuracy on test set: {:.2f}'.format(self.logReg.score(self.X_test, self.y_test)))
		print(f'Accuracy: {round(accuracy_score(self.y_test, self.outcome_predict), 3)}')

		c_matrix = confusion_matrix(self.y_test, self.outcome_predict)
		correct_predictions = c_matrix[0][0] + c_matrix[1][1]
		wrong_predictions = c_matrix[0][1] + c_matrix[1][0]
		print(c_matrix)
		print('We have {} correct predictions and {} wrong predictions'.format(correct_predictions, wrong_predictions))

		print(f'Classification report:\n{classification_report(self.y_test, self.outcome_predict)}')


class implementSVM:
	def __init__(self, training_data, testing_data):
		print('Initialising Support Vector Machine')
		self.X_train = training_data[0]
		self.y_train = np.ravel(training_data[1])
		self.X_test = testing_data[0]
		self.y_test = np.ravel(testing_data[1])
		self.svm = None
		self.outcome_predict = 0

	def train(self, model_dir):
		print('\nBegin training')
		self.svm = SVC(probability=True, kernel='linear', C=100)
		self.svm.fit(self.X_train, self.y_train) 
		with open(f'./{model_dir}/svm_trained.pkl', 'wb') as f:
			pickle.dump(self.svm, f)
		f.close()
		print('Training completed successfully')

	def predict(self):
		print('\nRunning prediction on test data')
		self.outcome_predict = self.svm.predict(self.X_test)

		print(f'Accuracy: {round(accuracy_score(self.y_test, self.outcome_predict), 3)}')
		c_matrix = confusion_matrix(self.y_test, self.outcome_predict)
		correct_predictions = c_matrix[0][0] + c_matrix[1][1]
		wrong_predictions = c_matrix[0][1] + c_matrix[1][0]
		print('We have {} positive predictions and {} negative predictions'.format(correct_predictions, wrong_predictions))
		print(f'Classification report:\n{classification_report(self.y_test, self.outcome_predict)}')


class implementGradientBoosting:
	def __init__(self, training_data, testing_data):
		print('Initialising Gradient Boosting Model')
		self.X_train = training_data[0]
		self.y_train = np.ravel(training_data[1])
		self.X_test = testing_data[0]
		self.y_test = np.ravel(testing_data[1])
		self.gbm = None
		self.outcome_predict = 0

	def train(self, lr, model_dir):
		print('\nBegin training')
		learning_params = {'min_samples_leaf':[2, 4, 10, 12], 'learning_rate':[0.001, 0.01, 0.1, 1, 5, 10]}
		self.gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3, random_state=0)
		gs_gbm = GridSearchCV(self.gbm, learning_params, verbose=0, return_train_score=True)
		gs_gbm.fit(self.X_train, self.y_train)
		self.gbm = gs_gbm.best_estimator_
		
		param_tuning_2d_gs_heatmap(gs_gbm, learning_params, 'min_samples_leaf', 'learning_rate')

		with open(f'./{model_dir}/gbm_trained.pkl', 'wb') as f:
			pickle.dump(self.gbm, f)
		f.close()

		# print(f'\nLearning rate: {lr}')
		# print("Accuracy score (training): {0:.3f}".format(self.gbm.score(self.X_train, self.y_train)))
		# print("Accuracy score (validation): {0:.3f}\n".format(self.gbm.score(self.X_val, self.y_val)))

		print('Training completed successfully')

	def predict(self):
		print('\nRunning predictions using test data')
		self.outcome_predict = self.gbm.predict(self.X_test)
		
		print(f'Accuracy: {round(accuracy_score(self.y_test, self.outcome_predict), 3)}')
		c_matrix = confusion_matrix(self.y_test, self.outcome_predict)
		correct_predictions = c_matrix[0][0] + c_matrix[1][1]
		wrong_predictions = c_matrix[0][1] + c_matrix[1][0]
		print(c_matrix)
		print('We have {} correct predictions and {} wrong predictions'.format(correct_predictions, wrong_predictions))
		print(f'Classification report:\n{classification_report(self.y_test, self.outcome_predict)}')


class CompareModels:
	def __init__(self, models, testing_data):
		self.models = models
		self.X_test = testing_data[0]
		self.y_test = testing_data[1]
	
	def plot_ROC_curve(self):	
		print('\nPlotting ROC curve')
		fpr_1, tpr_1, _ = roc_curve(self.y_test, self.models[0][0].predict_proba(self.X_test)[:,1])
		fpr_2, tpr_2, _ = roc_curve(self.y_test, self.models[1][0].predict_proba(self.X_test)[:,1])
		fpr_3, tpr_3, _ = roc_curve(self.y_test, self.models[2][0].predict_proba(self.X_test)[:,1])
		roc_auc_1 = roc_auc_score(self.y_test, self.models[0][1])
		roc_auc_2 = roc_auc_score(self.y_test, self.models[1][1])
		roc_auc_3 = roc_auc_score(self.y_test, self.models[2][1])

		plt.figure()
		plt.subplot(1,2,1)
		# plt.plot(fpr_1, tpr_1, label='{}'.format(self.models[0][2]))
		# plt.plot(fpr_2, tpr_2, label='{}'.format(self.models[1][2]))
		# plt.plot(fpr_3, tpr_3, label='{}'.format(self.models[2][2]))
		plt.plot(fpr_1, tpr_1, label='{} (area = {:0.2f})'.format(self.models[0][2], roc_auc_1))
		plt.plot(fpr_2, tpr_2, label='{} (area = {:0.2f})'.format(self.models[1][2], roc_auc_2))
		plt.plot(fpr_3, tpr_3, label='{} (area = {:0.2f})'.format(self.models[2][2], roc_auc_3))

		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		plt.savefig('ROC Comparison LR SVM GBM')
		plt.grid(True)
		plt.show()

	def plot_auc_boxplot(self):
		print('Plotting AUC boxplot')
		auc_result = []
		model_names = []
		# with open('./precision.txt', 'a+', encoding='utf-8') as f:
		for model, prediction, name in self.models:
			repeat_strat_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=7)
			cross_val_result = cross_val_score(model, self.X_test, self.y_test, cv=repeat_strat_kfold, scoring='roc_auc')
			# f.write(name)
			# f.write(str(cross_val_result))
			auc_result.append(cross_val_result)
			model_names.append(name)
			outcome_msg = '{}: {:0.4f} ({:0.4f})'.format(name, cross_val_result.mean(), cross_val_result.std())
			print(outcome_msg)
		
		bp_fig = plt.figure()
		bp_fig.suptitle('AUC Comparison')
		ax = bp_fig.add_subplot(111)
		plt.boxplot(auc_result)
		ax.set_xticklabels(model_names)
		plt.show()

	def plot_precision_recall_curve(self):
		print('\nPlotting precision recall curve')
		aucs = []
		for model, prediction, name in self.models:
			model_precision, model_recall, _ = precision_recall_curve(self.y_test, model.predict_proba(self.X_test)[:,1])
			for _ in range(0,10):
				aucs.append(auc(model_recall, model_precision))
			model_auc = np.array(aucs).mean()
			model_ap = average_precision_score(self.y_test, model.predict_proba(self.X_test)[:,1])
			plt.plot(model_recall, model_precision, marker='.', label='{} (AP: {:.3f}, AUC: {:.3f})'.format(name, model_ap, model_auc))
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.legend()
		plt.show()
			
	def plot_acc_boxplot(self):
		print('\nPlotting Accuracy boxplot')
		acc_results = []
		model_names = []
		for model, prediction, name in self.models:
			start = time.time()
			repeat_strat_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=7)
			cross_val_result = cross_val_score(model, self.X_test, self.y_test, cv=repeat_strat_kfold, scoring='accuracy')
			print(f'10-fold Stratified CV time for {name}: {time.time() - start}')
			acc_results.append(cross_val_result)
			model_names.append(name)
			outcome_msg = '{}: {:0.4f} ({:0.4f})'.format(name, cross_val_result.mean(), cross_val_result.std())
			print(outcome_msg)

		bp_fig = plt.figure()
		bp_fig.suptitle('Model Score Comparison')
		ax = bp_fig.add_subplot(111)
		plt.boxplot(acc_results)
		ax.set_xticklabels(model_names)
		plt.show()


def main():
	args = parse_arguments()
	data = DataCleaningAndPreprocess('latestdata.csv')
	data.clean_df()
	# data.data_explore()
	data.debug()
	if args.balanced_dataset == 'True':
		model_dir = r'./Models_with_SMOTE'
	else:
		model_dir = r'./Models_without_SMOTE'

	X_train, X_test, y_train, y_test, is_smote = data.split_data(args.balanced_dataset)

	# yX, yX_corr, yX_abs_corr = data.create_corr_matrix(X_train, y_train, True)

	check_class_ratio(X_train, X_test, y_train, y_test, is_smote)

	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	start = time.time()
	lr = implementLogisticRegression([X_train, y_train], [X_test, y_test])

	if (args.load_model == 'True') and (os.path.isfile(f'./{model_dir}/lr_trained.pkl')):
		with open(f'./{model_dir}/lr_trained.pkl', 'rb') as f:
			lr.logReg = pickle.load(f)
		f.close()
	else:
		lr.train(model_dir)
		checkpoint = time.time()
		print(f'Time to train lr: {checkpoint - start}')

	lr.predict()
	print('----------------------------------------------')
	start = time.time()
	gbm = implementGradientBoosting([X_train, y_train], [X_test, y_test])

	if (args.load_model == 'True') and (os.path.isfile(f'./{model_dir}/gbm_trained.pkl')):
		with open(f'./{model_dir}/gbm_trained.pkl', 'rb') as f:
			gbm.gbm = pickle.load(f)
		f.close()
	else:
		gbm.train(0.1, model_dir)
		checkpoint = time.time()
		print(f'Time to train lr: {checkpoint - start}')

	gbm.predict()
	print('----------------------------------------------')
	start = time.time()
	svm = implementSVM([X_train, y_train], [X_test, y_test])

	if (args.load_model == 'True') and (os.path.isfile(f'./{model_dir}/svm_trained.pkl')):
		with open(f'./{model_dir}/svm_trained.pkl', 'rb') as f:
			svm.svm = pickle.load(f)
		f.close()
	else:
		svm.train(model_dir)
		checkpoint = time.time()
		print(f'Time to train lr: {checkpoint - start}')

	svm.predict()
	print('----------------------------------------------')

	model_comparator = CompareModels([(lr.logReg, lr.outcome_predict, 'Logistic Regression Model'), (gbm.gbm, gbm.outcome_predict, 'Gradient Boosting Model'), (svm.svm, svm.outcome_predict, 'Support Vector Machine')], [X_test, y_test])
	cm = args.compare_mode
	if cm != 'None':
		if cm == 'roc' or cm == 'all':
			model_comparator.plot_ROC_curve()
		if cm == 'auc' or cm == 'all':
			model_comparator.plot_auc_boxplot()
		if cm == 'prc' or cm == 'all':
			model_comparator.plot_precision_recall_curve()
		if cm == 'acc' or cm == 'all':
			model_comparator.plot_acc_boxplot()

def parse_arguments():
	parser = ArgumentParser(description='')
	parser.add_argument('-lm', '--load_model',
						type=str,
						default='False',
						help='Choice to load existing ML models or train new ones'
						)
	parser.add_argument('-e', '--eval',
						type=str,
						default='True',
						help='Shows evaluation of models with one test on the testing dataset'
						)
	parser.add_argument('-bd', '--balanced_dataset',
						type=str,
						default='True',
						help='Use SMOTE to balance dataset'
						)
	parser.add_argument('-cm', '--compare_mode',
						type=str,
						default='none',
						choices=['none', 'roc', 'auc', 'prc', 'acc', 'all'],
						help='''
							none: No comparison done;
							roc: Plots ROC;
							auc: Visualizes roc auc calculated from repeated stratifed k-fold cross validation as boxplots;
							prc: Plots PRC;
							acc: Visualizes accuracy calculated from repeated stratifed k-fold cross validation as boxplots;
							all:does all comparison visualization
							'''
						)
	return parser.parse_args()

if __name__ == '__main__':
	main()