
from nltk import DecisionTreeClassifier
from pandas import read_csv
# Load dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

url  = "tmp_file.csv"
names = ['TMP_Anormal','HMD_Anormal','Temperature_normal', 'Humidity_normal', 'Type']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(36))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('Type').size())
# Split-out validation dataset
array = dataset.values
print(array)
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("model :")
print(X_train)
print(Y_train)
print(X_validation)
print(Y_validation)
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
print("Show")
print(models)
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print("Show")
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	# Make predictions on validation dataset
	model = SVC(gamma='auto')
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	# Evaluate predictions
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
