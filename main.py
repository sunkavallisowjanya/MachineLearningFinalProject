print('Importing all the required libraries...')
# importing all the required libraries and packages
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler,Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
 
warnings.filterwarnings("ignore")
accuracies = {}
# importing the dataset
print('Loading data...')
df = pd.read_csv('emails.csv')
X = df.drop(columns=['Email No','Prediction'])
y = df[['Prediction']]
def temp(scaler,data):
	scaler.fit(data)
	print(f'data is scaled using {scaler}')
	return pd.DataFrame(scaler.transform(data),columns=data.columns)
scalers = {1:StandardScaler(),2:Normalizer(),3:MinMaxScaler(),4:MaxAbsScaler(),5:RobustScaler()}
def scaling():
	# temp()
	print("Choose any one of the below scaling techniques to scale the data")
	print('1. StandardScaler\n2. Normalizer\n3. MinMaxScaler\n4. MaxAbsScaler\n5. RobustScaler\n6. NoScaler')
	option = int(input('Enter your choice : '))
	if option == 6:
		X_scaled = X
	else:
		X_scaled = temp(scalers[option],X)
	# if X.equals(X_scaled):
	# 	print('data is not scaled')
	# else:
	# 	print('data is scaled')
	return train_test_split(X_scaled, y, test_size = 0.2)

def train_and_test(classifier):
		# print(classifier)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		print(classification_report(y_test, y_pred))
		print(confusion_matrix(y_test, y_pred))
		accuracy = accuracy_score(y_pred,y_test)
		print('accuracy is',accuracy)
		accuracies[classifier] = round(accuracy*100,2)
naive_bayes = GaussianNB()
svm = SVC()
decision_tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
boosting = GradientBoostingClassifier()
adaboost = AdaBoostClassifier()
bagging = BaggingClassifier()
randomforest = RandomForestClassifier()
d = {1:naive_bayes,2:svm,3:decision_tree,4:knn,5:boosting,6:adaboost,7:bagging,8:randomforest}
X_train, X_test, y_train, y_test = scaling()

while(True):
	print('Choose any one of the below supervised machine learning models')
	print('1. Navie Bayes\n2. Support Vector Machine\n3. Decision Tree\n4. KNN\n5. Gradient Boosting\n6. AdaBoost\n7. Bagging\n8. Random Forest')
	choice = int(input('Enter your choice : '))
	train_and_test(d[choice])
	repeat = int(input('Do you want to train another model?\n1. Yes\n0.No\nEnter your choice: '))
	if repeat: 
		scale = int(input('Do you want to Scale the data?\n1. Yes\n0. No\nEnter your choice : '))
		if scale:
			# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)
			X_train, X_test, y_train, y_test = scaling()
		continue
	else: break
# print(accuracies)
accuracy_list = []
head = ['Classifier','Accuracy']
print()
print('Accuracies of all the classifiers used are:')
for k,v in accuracies.items():
	accuracy_list.append([k,str(v)+' %'])
print(tabulate(accuracy_list, headers=head, tablefmt="grid"))
max_accuracy = max(accuracies.values())
max_classifier = [i for i in accuracies if accuracies[i] == max_accuracy]
print('')
print(f'{max_classifier[0]} has the highest accuracy of {max_accuracy} %')





