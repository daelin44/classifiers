from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets as dts
from sklearn import model_selection as ms
from sklearn import metrics as mtr

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def Init_Data(FileName, Target_Last):
    df=pd.read_csv(FileName, header=None)
    if Target_Last:
        y=np.array(df.iloc[:,-1])
        X=np.array(df.iloc[:,:-1][:].replace('?',np.nan))
    else:
        y=np.array(df.iloc[:,0])
        X=np.array(df.iloc[:,1:][:].replace('?',np.nan))        
    imp=SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X=imp.transform(X)
    return X, y
def Init_Data_bcancer():
    df=pd.read_csv('wdbc.data', header=None)
    df=df.iloc[:,1:]
    df=df.replace(to_replace='M',value=1)
    df=df.replace(to_replace='B',value=0)
    y=np.array(df.iloc[:,0])
    X=np.array(df.iloc[:,1:][:].replace('?',np.nan))        
    imp=SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X=imp.transform(X)
    return X, y
def Init_Data_lcancer():
    df=pd.read_csv('lung-cancer.data', header=None)
    y=np.array(df.iloc[:,0])
    y=y-1
    X=np.array(df.iloc[:,1:][:].replace('?',np.nan))        
    imp=SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X=imp.transform(X)
    return X, y
def Use_SKF(X, y):
    skf=ms.RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)
    Z=skf.split(X,y)
    return Z

def Form_Train_Test(Z, X, y):
    X_train, X_test, y_train, y_test = [], [], [], []
    for train_index, test_index in Z:
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    return X_train, X_test, y_train, y_test

def WorkSGD(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(SGDClassifier(),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred

def WorkSGD_bin(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    precision,recall,f1=[],[],[]
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(SGDClassifier(),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        precision.append(mtr.precision_score(y_te, y_pr,zero_division=1))
        recall.append(mtr.recall_score(y_te,y_pr))
        f1.append(mtr.f1_score(y_te,y_pr))
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred, precision, recall, f1
def WorkRF_bin(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    precision,recall,f1=[],[],[]
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        precision.append(mtr.precision_score(y_te, y_pr,zero_division=1))
        recall.append(mtr.recall_score(y_te,y_pr))
        f1.append(mtr.f1_score(y_te,y_pr))
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred, precision, recall, f1

def WorkRF(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred


def WorkP(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(MLPClassifier(solver='sgd', learning_rate='constant',momentum=0, learning_rate_init=0.2),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred

def WorkP_bin(X_train, X_test, y_train, y_test):
    y_real, y_pred = [], []
    precision,recall,f1=[],[],[]
    for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        learned_classifier=Learn_Classifier(MLPClassifier(solver='sgd', learning_rate='constant',momentum=0, learning_rate_init=0.2),X_tr,y_tr)
        y_pr=Predict(learned_classifier, X_te)
        precision.append(mtr.precision_score(y_te, y_pr,zero_division=1))
        recall.append(mtr.recall_score(y_te,y_pr))
        f1.append(mtr.f1_score(y_te,y_pr))
        y_real.append(y_te)
        y_pred.append(y_pr)
    return y_real, y_pred, precision, recall, f1

def Learn_Classifier(classifier, X, y):
    learned_classifier=classifier.fit(X, y)
    return learned_classifier

def Predict(learned_classifier, X):
    y=learned_classifier.predict(X)
    return y

def Accuracy(y_real, y_pred):
    return mtr.accuracy_score(y_real, y_pred)

def Estimate_Accuracy(y_real, y_pred):
    acc=[]
    for y_re, y_pr in zip(y_real, y_pred):
        acc.append(Accuracy(y_re, y_pr))
    return acc



def Test():
    X, y = Init_Data_bcancer()
    Z = Use_SKF(X, y)
    X_train, X_test, y_train, y_test = Form_Train_Test(Z, X, y)
    y_real, y_pred, precision, recall, f1score = WorkSGD_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('\nbreast cancer dataset\n')
    print('AVG of Accuracy with SGD classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")
    y_real, y_pred, precision, recall, f1score = WorkRF_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Random Forest classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")
    y_real, y_pred, precision, recall, f1score = WorkP_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Multi-Layer Perceptron')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")

    X, y = Init_Data_lcancer()
    Z = Use_SKF(X, y)
    X_train, X_test, y_train, y_test = Form_Train_Test(Z, X, y)
    y_real, y_pred, precision, recall, f1score = WorkSGD_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('\nlung cancer dataset\n')
    print('AVG of Accuracy with SGD classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")
    y_real, y_pred, precision, recall, f1score = WorkRF_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Random Forest classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")
    y_real, y_pred, precision, recall, f1score = WorkP_bin(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Multi-Layer Perceptron')
    print(f"{sum(acc)/len(acc):.2f}")
    print('AVG of Precision')
    print(f"{sum(precision)/len(precision):.2f}")
    print('AVG of Recall')
    print(f"{sum(recall)/len(recall):.2f}")
    print('AVG of F1Score')
    print(f"{sum(f1score)/len(f1score):.2f}")

    X, y = Init_Data('wine.data',0)
    Z = Use_SKF(X, y)
    X_train, X_test, y_train, y_test = Form_Train_Test(Z, X, y)
    y_real, y_pred = WorkSGD(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('\nwine dataset\n')
    print('AVG of Accuracy with SGD classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    y_real, y_pred = WorkRF(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Random Forest classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    y_real, y_pred = WorkP(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Multi-Layer Perceptron')
    print(f"{sum(acc)/len(acc):.2f}")

    
    X, y = Init_Data('iris.data',1)
    Z = Use_SKF(X, y)
    X_train, X_test, y_train, y_test = Form_Train_Test(Z, X, y)
    y_real, y_pred = WorkSGD(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('\niris dataset\n')
    print('AVG of Accuracy with SGD classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    y_real, y_pred = WorkRF(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Random Forest classifier')
    print(f"{sum(acc)/len(acc):.2f}")
    y_real, y_pred = WorkP(X_train, X_test, y_train, y_test)
    acc = Estimate_Accuracy(y_real, y_pred)
    print('AVG of Accuracy with Multi-Layer Perceptron')
    print(f"{sum(acc)/len(acc):.2f}")
if __name__=='__main__':
    Test()

    
