
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer ##loading the dataset
(X_can,y_can)=load_breast_cancer(return_X_y =True)
df=pd.DataFrame(X_can) ## converting it into pandas dataframe
(x_train, x_test, y_train, y_test)=train_test_split(X_can, y_can, random_state=0)
#print(np.shape(x_test))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train) ##scaling the dataset
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(x_test)
from sklearn.svm import LinearSVC ##linear support vector machine as the classifier


clf = LinearSVC(C=4,random_state=60).fit(X_train_scaled, y_train) ##setting the hyper parameter as c=4 for regularisation
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))
from sklearn.metrics import confusion_matrix ## making a confusion matrix 
y_predict=clf.predict(X_test_scaled)
confusion=confusion_matrix(y_test,y_predict)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ##comparing different evaluation methods in this case recall is of high importance
print('precision: {:.2f}'.format(precision_score(y_test,y_predict)))
print('Recall: {:.2f}'.format(recall_score(y_test,y_predict)))
print('accuracy: {:.2f}'.format(accuracy_score(y_test,y_predict)))        	 
print('F1: {:.2f}'.format(f1_score(y_test,y_predict)))

## plotting Precision_recall_curve for the model


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test,y_predict)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]
##plotting the curve
plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()
