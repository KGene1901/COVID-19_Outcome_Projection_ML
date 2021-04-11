import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import json
import statistics
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

'''
Task description:
choose 3 ML predictive models  to solve ONE problem with the Covid-19 dataset and compare their results

Problem: Outcome Projection

Models
- Gradient Boosted Model / Random Forest
- Logistic Regression
- SVM
'''
class NpEncoder(json.JSONEncoder):
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


class DataCleaningAndPreprocess:
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
		self.df.symptoms.fillna('unknown_symptoms', inplace=True)
		self.df.age.fillna('unknown_age', inplace=True)
		self.df.chronic_disease.fillna('unknown_diseases', inplace=True)
		self.df.date_confirmation.fillna('no_date', inplace=True)

		self.standardize_outcome()
		self.standardize_age()
		self.standardize_symptoms(self.df.shape[0])
		self.standardize_chronic_disease(self.df.shape[0])
		self.replace_chronic_disease_to_bin()
		self.replace_sex_to_bin()

	def split_data(self, use_smote=False):
		# covid_dataset = pd.read_csv(f'./{dataFile}', low_memory=False)
		lab_enc = preprocessing.LabelEncoder()
		covid_dataset = self.df
		y = covid_dataset.outcome
		x = covid_dataset.drop(['ID','outcome','date_confirmation'], axis=1)
		x['country'] = lab_enc.fit_transform(x.country)
		y = lab_enc.fit_transform(y)
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
		X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
		is_smote = False

		if use_smote:
			os = SMOTE(random_state=0)
			is_smote = True
			X_cols = X_train.columns
			os_data_X, os_data_y = os.fit_resample(X_train, y_train)
			os_data_X = pd.DataFrame(data=os_data_X, columns=X_cols)
			os_data_y = pd.DataFrame(data=os_data_y, columns=['outcome'])
			X_train = os_data_X
			y_train = os_data_y

		return X_train, X_test, y_train, y_test, X_val, y_val, is_smote

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
		    plt.yticks(range(len(yX_abs_corr)), yX_abs_corr.columns);
		    plt.suptitle('Pearson Correlation Heat Map (absolute values)', fontsize=15, fontweight='bold')
		    plt.show()

		return yX, yX_corr, yX_abs_corr

	def data_explore(self):
		chronic_diseases = self.df.iloc[:, 21:45]
		for col in chronic_diseases:
			sns.countplot(x=col, data=self.df, palette='hls')
		# self.df['age'].hist(bins=10)
			# plt.savefig('age2_plot.png')
			plt.show()

	def debug(self):
		self.df.to_csv('./modified_latestdata2.csv', encoding='utf-8', index=False)
		# print(self.df.isna().sum())


def check_class_ratio(x_train, x_test, y_train, y_test, is_smote):
	print("x_train.shape {}, y_train.shape {}".format(x_train.shape, y_train.shape))

	if is_smote:
		y_train_df = y_train
	else:
		y_train_df = pd.DataFrame(data=y_train.flatten())

	y_test_df = pd.DataFrame(data=y_test.flatten())

	train_outcome_counts = y_train_df.value_counts()
	print('Class distribution in training set:')
	print(train_outcome_counts)
	print('Percentage of positive class samples: {}'.format(100 * train_outcome_counts[1] / len(y_train_df)))

	print('\n-------------------------------------------------------------------------------\n')
	print("x_test.shape {}, y_test.shape {}".format(x_test.shape, y_test.shape))

	test_outcome_counts = y_test_df.value_counts()
	print('Class distribution in training set:')
	print(test_outcome_counts)
	print('Percentage of positive class samples: {}'.format(100 * test_outcome_counts[1] / len(y_test_df)))


class implementLogisticRegression:
	def __init__ (self, training_data, testing_data):
		print('Initialising Logistic Regression Model')
		self.X_train = training_data[0]
		self.y_train = np.ravel(training_data[1])
		self.X_test = testing_data[0]
		self.y_test = np.ravel(testing_data[1])
		self.logReg = None
		self.outcome_predict = 0

	def train(self):
		print('\nBegin training')
		self.logReg = LogisticRegression(class_weight='balanced', dual=False, intercept_scaling=1, 
									max_iter=500, n_jobs=1, random_state=0, tol=0.0001, penalty='l2',
									verbose=0, warm_start=False
									)
		self.logReg.fit(self.X_train, self.y_train)
		print('Training completed successfully')

	def predict(self):
		print('\nRunning prediction on test data')
		self.outcome_predict = self.logReg.predict(self.X_test)
		# print('Accuracy on test set: {:.2f}'.format(self.logReg.score(self.X_test, self.y_test)))
		print(f'Accuracy: {round(accuracy_score(self.y_test, self.outcome_predict), 3)}')

		c_matrix = confusion_matrix(self.y_test, self.outcome_predict)
		correct_predictions = c_matrix[0][0] + c_matrix[1][1]
		wrong_predictions = c_matrix[0][1] + c_matrix[1][0]
		print('We have {} positive predictions and {} negative predictions'.format(correct_predictions, wrong_predictions))

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

	def train(self):
		print('\nBegin training')
		self.svm = SVC(kernel='linear', C=100)
		self.svm.fit(self.X_train, self.y_train)
		print('Training completed successfully')

	def predict(self):
		print('\nRunning prediction on test data')
		self.outcome_predict = self.svm.predict(self.X_test)

		print(f'Accuracy: {round(accuracy_score(self.y_test, self.outcome_predict), 3)}')
		c_matrix = confusion_matrix(self.y_test, self.outcome_predict)
		correct_predictions = c_matrix[0][0] + c_matrix[1][1]
		wrong_predictions = c_matrix[0][1] + c_matrix[1][0]
		print('We have {} positive predictions and {} negative predictions'.format(correct_predictions, wrong_predictions))


class implementGradientBoosting:
	def __init__(self, training_data, testing_data, validation_data):
		print('Initialising Gradient Boosting Model')
		self.X_train = training_data[0]
		self.y_train = np.ravel(training_data[1])
		self.X_test = testing_data[0]
		self.y_test = np.ravel(testing_data[1])
		self.X_val = validation_data[0]
		self.y_val = np.ravel(validation_data[1])
		self.gbm = None
		self.outcome_predict = 0

	def train(self, lr):
		print('\nBegin training')
		self.gbm = GradientBoostingClassifier(n_estimators=30, learning_rate=lr, max_depth=3, random_state=0)
		self.gbm.fit(self.X_train, self.y_train)

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
		print('We have {} positive predictions and {} negative predictions'.format(correct_predictions, wrong_predictions))


def compare_models(lr, svm, gbm):
	# compare ROC for logistic regression and gradient boosting
	print('\nPlotting ROC curve')
	lr_roc_auc = roc_auc_score(lr.y_test, lr.outcome_predict)
	gbm_roc_auc = roc_auc_score(gbm.y_test, gbm.outcome_predict)
	fpr_1, tpr_1, thresholds_1 = roc_curve(lr.y_test, lr.logReg.predict_proba(lr.X_test)[:,1])
	fpr_2, tpr_2, thresholds_2 = roc_curve(gbm.y_test, gbm.gbm.predict_proba(gbm.X_test)[:,1])
	plt.figure()
	plt.plot(fpr_1, tpr_1, label='Logistic Regression Model (area = %0.2f)' % lr_roc_auc)
	plt.plot(fpr_2, tpr_2, label='Gradient Boosting Model (area = %0.2f)' % gbm_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('ROC Comparison LR and GBM')
	plt.show()


def main():
	data = DataCleaningAndPreprocess('modified_latestdata.csv')
	data.clean_df()
	# data.data_explore()
	# data.debug()

	X_train, X_test, y_train, y_test, X_val, y_val, is_smote = data.split_data(False)

	# yX, yX_corr, yX_abs_corr = data.create_corr_matrix(y_train, x_train, True)

	# check_class_ratio(X_train, X_test, y_train, y_test, is_smote)

	implement_lr = implementLogisticRegression([X_train, y_train], [X_test, y_test])
	implement_lr.train()
	implement_lr.predict()
	print('----------------------------------------------')
	implement_svm = implementSVM([X_train, y_train], [X_test, y_test])
	implement_svm.train()
	implement_svm.predict()
	print('----------------------------------------------')
	implement_gbm = implementGradientBoosting([X_train, y_train], [X_test, y_test], [X_val, y_val])
	implement_gbm.train(0.1)
	implement_gbm.predict()

	compare_models(implement_lr, implement_svm, implement_gbm)


if __name__ == '__main__':
	main()