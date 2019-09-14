#	this part of the code is responsible for printing the original dataset size along with the feature given out after 
#	dimentional reduction and also the the dataset size after the dimentional redution 
#	next performing the classification using KNN, Random Forest and SVM classifiers and printing out 
#	their accuracy and classification report which consists of precision,recall and f1 score
# 	next generate the graphical comparision of accuracy, precision, recall and f1 score after classification 
# 	with KNN, Random Forest and SVM after dimentional reduction by FSPeSOA, PCA and LDA
#	all these operations are performed for the ION dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('D:\project\ion.csv')
df.shape


data = df.values[:,:df.shape[-1]-1]
target = df.values[:,df.shape[-1]-1]


preprocessor = MinMaxScaler()
data1 = preprocessor.fit_transform(data)
target1 = []
x = {}
for i, j in enumerate(np.unique(target)):
    x[j] = i
target1 = np.array([x[i] for i in target])


x_train, x_test, y_train, y_test = train_test_split(data1, target1, train_size = 0.7, random_state=1999, stratify = target1)



PCA = PCA(n_components = 15) 

x_train_1 = PCA.fit_transform(x_train)
x_test_1 = PCA.fit_transform(x_test)

LDA = LatentDirichletAllocation(n_components = 15)  
x_train_2 = LDA.fit_transform(x_train)
x_test_2 = LDA.fit_transform(x_test)


rf = RandomForestClassifier()
rf.fit(x_train_1, y_train)
y_pred_pca_rf = rf.predict(x_test_1)
accuracy_pca_rf = accuracy_score(y_pred_pca_rf, y_test)
precision_pca_rf = precision_score(y_pred_pca_rf, y_test, average='macro')
recall_pca_rf = recall_score(y_pred_pca_rf, y_test, average='macro')
f1_pca_rf = f1_score(y_pred_pca_rf, y_test, average='macro')



rf.fit(x_train_2, y_train)
y_pred_lda_rf = rf.predict(x_test_2)
accuracy_lda_rf = accuracy_score(y_pred_lda_rf, y_test)
precision_lda_rf = precision_score(y_pred_lda_rf, y_test, average='macro')
recall_lda_rf = recall_score(y_pred_lda_rf, y_test, average='macro')
f1_lda_rf = f1_score(y_pred_lda_rf, y_test, average='macro')

svm = SVC()
svm.fit(x_train_1, y_train)
y_pred_pca_svm = svm.predict(x_test_1)
accuracy_pca_svm = accuracy_score(y_pred_pca_svm, y_test)
precision_pca_svm = precision_score(y_pred_pca_svm, y_test, average='macro')
recall_pca_svm = recall_score(y_pred_pca_svm, y_test, average='macro')
f1_pca_svm = f1_score(y_pred_pca_svm, y_test, average='macro')


svm.fit(x_train_2, y_train)
y_pred_lda_svm = svm.predict(x_test_2)
accuracy_lda_svm = accuracy_score(y_pred_lda_svm, y_test)
precision_lda_svm = precision_score(y_pred_lda_svm, y_test, average='macro')
recall_lda_svm = recall_score(y_pred_lda_svm, y_test, average='macro')
f1_lda_svm = f1_score(y_pred_lda_svm, y_test, average='macro')

log = KNeighborsClassifier(n_neighbors=5)
log.fit(x_train_1, y_train)
y_pred_pca_log = log.predict(x_test_1)
accuracy_pca_log = accuracy_score(y_pred_pca_log, y_test)
precision_pca_log = precision_score(y_pred_pca_log, y_test, average='macro')
recall_pca_log = recall_score(y_pred_pca_log, y_test, average='macro')
f1_pca_log = f1_score(y_pred_pca_log, y_test, average='macro')


log.fit(x_train_2, y_train)
y_pred_lda_log = log.predict(x_test_2)
accuracy_lda_log = accuracy_score(y_pred_lda_log, y_test)
precision_lda_log = precision_score(y_pred_lda_log, y_test, average='macro')
recall_lda_log = recall_score(y_pred_lda_log, y_test, average='macro')
f1_lda_log = f1_score(y_pred_lda_log, y_test, average='macro')

values_acc_lda = [accuracy_lda_log, accuracy_lda_rf, accuracy_lda_svm]
values_acc_pca = [accuracy_pca_log, accuracy_pca_rf, accuracy_pca_svm]
values_pre_lda = [precision_lda_log, precision_lda_rf, precision_lda_svm]
values_pre_pca = [precision_pca_log, precision_pca_rf, precision_pca_svm]
values_re_lda = [recall_lda_log,  recall_lda_rf, recall_lda_svm]
values_re_pca = [recall_pca_log, recall_pca_rf, recall_pca_svm]
values_f1_lda = [f1_lda_log, f1_lda_rf, f1_lda_svm]
values_f1_pca = [f1_pca_log, f1_pca_rf, f1_pca_svm]


from __future__ import division
import numpy as np
import pandas as pd


class Penguin(object):
    def __init__(self, X, percentage):
        self.X = X
        self.percentage = 0.7
        self.variance_ratio = []
        self.low_dataMat = []

    def pos(self, eigVal):
        sortVal = np.sort(eigVal)[-1::-1]
        percentSum, componentNum = 0, 0
        for i in sortVal:
            percentSum += i
            componentNum += 1
            if percentSum >= sum(sortVal) * self.percentage:
                break
        return componentNum

    def _fit(self):
        oxygen=100
        for z in range(oxygen):
            X_mean = np.mean(self.X, axis=0)
            dataMat = self.X - X_mean
            covMat = np.cov(dataMat, rowvar=False)
            eigVal, eigVect = np.linalg.eig(np.mat(covMat))
            new_pos = self.pos(eigVal)
            eigValInd = np.argsort(eigVal)
            eigValInd = eigValInd[-1:-(new_pos + 1):-1]
            n_eigVect = eigVect[:, eigValInd]
            self.low_dataMat = dataMat * n_eigVect
            [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
            print(eigValInd)
            return self.low_dataMat

    def fit(self):
        self._fit()
        return self
    
df = pd.read_csv('D:\project\ion.csv')
df.shape


data = df.values[:,:df.shape[-1]-1]
target = df.values[:,df.shape[-1]-1]

preprocessor = MinMaxScaler()
data1 = preprocessor.fit_transform(data)
target1 = []
x = {}
for i, j in enumerate(np.unique(target)):
    x[j] = i
target1 = np.array([x[i] for i in target])

data, label = data1,target1
data = np.mat(data)

print("ion datasets:")
print("Original dataset = {}*{}".format(data.shape[0], data.shape[1]))
#pca = PCAcomponent(data, 2)
pca = Penguin(data, 1)
pca.fit()
#print(pca.low_dataMat)
#print(pca.variance_ratio)
print(pca.low_dataMat.shape)
x=pca.low_dataMat
x_train, x_test, y_train, y_test = train_test_split(x, target1,  train_size=0.8, random_state=1999)
svm.fit(x_train, y_train)
log.fit(x_train, y_train)
rf.fit(x_train, y_train)


y_pred_png_rf = rf.predict(x_test)
accuracy_png_rf = accuracy_score(y_pred_png_rf, y_test)
precision_png_rf = precision_score(y_pred_png_rf, y_test, average='macro')
recall_png_rf = recall_score(y_pred_png_rf, y_test, average='macro')
f1_png_rf = f1_score(y_pred_png_rf, y_test, average='macro')
print("Random Forest:")
print("Accuracy = {}".format(rf.score(x_test, y_test)))
#print('\n\n')
print('Classification Report\n')
print(classification_report(y_test, y_pred_png_rf))

y_pred_png_svm = svm.predict(x_test)
accuracy_png_svm = accuracy_score(y_pred_png_svm, y_test)
precision_png_svm = precision_score(y_pred_png_svm, y_test, average='macro')
recall_png_svm = recall_score(y_pred_png_svm, y_test, average='macro')
f1_png_svm = f1_score(y_pred_png_svm, y_test, average='macro')
print("SVM:")
print("Accuracy = {}".format(svm.score(x_test, y_test)))
#print('\n')
print('Classification Report\n')
print(classification_report(y_test, y_pred_png_svm))

y_pred_png_log = log.predict(x_test)
accuracy_png_log = accuracy_score(y_pred_png_log, y_test)
precision_png_log = precision_score(y_pred_png_log, y_test, average='macro')
recall_png_log = recall_score(y_pred_png_log, y_test, average='macro')
f1_png_log = f1_score(y_pred_png_log, y_test, average='macro')
print("KNN:")
print("Accuracy = {}".format(log.score(x_test, y_test)))
#print('\n\n')
print('Classification Report\n')
print(classification_report(y_test, y_pred_png_log))

values_acc_png = [accuracy_png_log, accuracy_png_rf, accuracy_png_svm]
values_pre_png = [precision_png_log, precision_png_rf, precision_png_svm]
values_re_png = [recall_png_log, recall_png_rf, recall_png_svm]
values_f1_png = [f1_png_log, f1_png_rf, f1_png_svm]

import pylab as plt
plt.figure(figsize=(15,7))
classifier = [1, 2, 3, 4, 5, 6, 7, 8, 9]
names = ['knn_lda', 'rforest_lda', 'svm_lda', 'knn_pca', 'rforest_pca',  'svm_pca', 'knn_FSPeSO', 'rforest_FSPeSO','svm_FSPeSO']
plt.plot(classifier, values_acc_lda + values_acc_pca + values_acc_png, label = 'accuracy', marker ='*', markersize =8)
plt.plot(classifier, values_pre_lda + values_pre_pca + values_pre_png, label = 'precision', marker = 'o', markersize =8)
plt.plot(classifier, values_re_lda + values_re_pca + values_re_png, label = 'recall', marker = 'v', markersize=8)
plt.plot(classifier, values_f1_lda + values_f1_pca + values_f1_png, label = 'f1', marker ='s', markersize=8)
plt.legend()
plt.ylabel('VALUES')
plt.xlabel('CLASSIFIER')
plt.xticks(classifier, names)
plt.show()

