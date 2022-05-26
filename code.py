# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:25:43 2022

@author: EBB
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

"""
dataframe = pd.read_csv("train.csv")

backup = dataframe


dataframe =  dataframe.('number', axis=1)
dataframe =  dataframe.('id', axis=1)

describeDataFrame = dataframe.describe()
backup = dataframe

genderMod = dataframe['Gender'].value_counts()

customerTypeMod  = dataframe['Customer Type'].value_counts()
travelMod = dataframe['Type of Travel'].value_counts()
classMod = dataframe['Class'].value_counts()

satisfactionMod = dataframe['satisfaction'].value_counts()



########################## null olanların dynamic olarak kontrolü ve lanması #################
print(dataframe.isnull().sum())
def nullChecker():
    global numerical_cols
    numAttributes = "Gender,Customer Type,Age,Type of Travel,Class,Flight Distance,Inflight wifi service,Departure/Arrival time convenient,Ease of Online booking,Gate location,Food and drink,Online boarding,Seat comfort,Inflight entertainment,On-board service,Leg room service,Baggage handling,Checkin service,Inflight service,Cleanliness,Departure Delay in Minutes,Arrival Delay in Minutes,satisfaction".split(",")

    numAttributes.remove('Gender')
    numAttributes.remove('Customer Type')
    numAttributes.remove('Type of Travel')
    numAttributes.remove('Class')
    numAttributes.remove('satisfaction')
    numerical_cols = numAttributes
    nuls = []
    for i in numAttributes:
        nuls.append(dataframe[dataframe[i].isnull()])
    returnNuls = []
    for i in nuls:
        if len(i.index) !=0:
            returnNuls.append(i)
    return returnNuls

def nullDeleter():
    Nulls = nullChecker()[-1].index
    dataframe.(Nulls, inplace = True)
    Nulls=nullChecker()


nullDeleter()
print(dataframe.info())


################# Verilerin tamamının one hot encoding yapılması ve sayısallaştırılması #################################
dataframe['satisfaction'] = dataframe['satisfaction'].map({'neutral or dissatisfied' : 0, 'satisfied' : 1})
dataframe['Gender'] = dataframe['Gender'].map({'Female' : 0, 'Male' : 1})

dataframe['Type of Travel'] = dataframe['Type of Travel'].map({'Personal Travel' : 0, 'Business travel' : 1})
dataframe['Customer Type'] = dataframe['Customer Type'].map({'Loyal Customer' : 1, 'disloyal Customer' : 0})


hotEncodingNeighbourhood = pd.get_dummies(dataframe['Class'][:])
dataframe = pd.concat([dataframe, hotEncodingNeighbourhood], axis=1, join="inner")
dataframe =  dataframe.('Class', axis=1)

#############################################################################################################################


def normalization(dataframeName):
    listDf = dataframe[dataframeName].tolist()
    argMin = min(listDf)
    argMax = max(listDf)
    differance = argMax-argMin
    return (dataframe[dataframeName]-argMin)/differance
    
    

fig = plt.figure(figsize =(10, 7))
data = dataframe['Departure Delay in Minutes'].to_numpy()
plt.boxplot(data)
plt.title('Departure Delay  Plot\n',fontweight ="bold")
plt.show()


fig = plt.figure(figsize =(10, 7))
data = dataframe['Arrival Delay in Minutes'].to_numpy()
plt.boxplot(data)
plt.title('Arrival Delay in Minutes\n',fontweight ="bold")
plt.show()



outlierSelect = np.array(dataframe['Departure Delay in Minutes'].to_numpy())
outlierSelect.sort()
outlierSelect = outlierSelect[::-1] # max verilerden 4 'ü göze batan 

for i in outlierSelect[0:4]:
    findIndex = dataframe.loc[dataframe['Departure Delay in Minutes'] == i].index
    dataframe.(findIndex, inplace = True)

Temizlemenin kontrolü 
fig = plt.figure(figsize =(10, 7))
data = dataframe['Departure Delay in Minutes'].to_numpy()
plt.boxplot(data)
plt.title('Departure Delay  Plot\n',fontweight ="bold")
plt.show()



outlierSelect = np.array(dataframe['Arrival Delay in Minutes'].to_numpy())
outlierSelect.sort()
outlierSelect = outlierSelect[::-1] # max verilerden 4 'ü göze batan 

for i in outlierSelect[0:6]:
    findIndex = dataframe.loc[dataframe['Arrival Delay in Minutes'] == i].index
    dataframe.drop(findIndex, inplace = True)
    
#Temizlenen Verinin Kontrolü 

fig = plt.figure(figsize =(10, 7))
data = dataframe['Arrival Delay in Minutes'].to_numpy()
plt.boxplot(data)
plt.title('Arrival Delay in Minutes\n',fontweight ="bold")
plt.show()



## Çok az outlierımız var ve bunlar sadece arriver delay in minutes ve deperture delay in minutes içerisinde 
## Bu verilerde outlier dememizin sebebi ortalamadan uzak olanlar bu veride atılmaması gerekmesine rağmen bunlar bariz olarak
## göze batan anomoliler

dataframe['Age']=normalization('Age')
dataframe['Flight Distance']=normalization('Flight Distance')
dataframe['Arrival Delay in Minutes']=normalization('Arrival Delay in Minutes')
dataframe['Departure Delay in Minutes']=normalization('Departure Delay in Minutes')

print(dataframe.isnull().sum())




dataframe.to_csv("trainClear.csv",encoding="utf-8", index=False)
##################### NORMALİZASYONLAR ####################


"""
import seaborn as sns 
dataframe = pd.read_csv("trainClear.csv")
"""

plt.figure(dpi=125)
sns.heatmap(np.round(dataframe.corr(),2),annot=True)
plt.show()

plt.figure(dpi=100,figsize=(10,5))
sns.countplot(dataframe['Gender'],palette='rocket')
plt.xlabel('Gender')
plt.ylabel('Number of Gender')
plt.title('Gender')
"""

"""

def plotS(columnName):
    

    product_cat = dataframe[columnName].value_counts()
    label= [product_cat.index.tolist()]
    plt.pie(product_cat,labels=label[0],explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), autopct='%1.1f%%', startangle=240, colors=['#E8B89C','#CB4854','#DE8165', '#EB7545', '#EB7865', '#DF6566'])
    plt.gcf().set_size_inches(12,6)
    plt.title(columnName)

    plt.show()
plotS('Inflight wifi service')
plotS('Departure/Arrival time convenient')
plotS('Ease of Online booking')

"""





"""
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets, neighbors
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from yellowbrick.datasets import load_occupancy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score



######### Age verisinin standardize edilmesi

x = dataframe.drop(['satisfaction'], axis=1)
y = dataframe['satisfaction']




#verilerin egitim ve test icin bolunmesi
## Verilerin test ve train olarak bölünmesi
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

############### KNN #################### 
print("KNN with Default Parameter ")
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(classification_report(y_test, y_pred))
succes = cross_val_score(estimator= knn, X=x_train, y=y_train, cv=10)
print(succes)
print(succes.mean)

###################################

####################  Decision Tree Class. #################
print("Decision Tree Classifier with Default Parameter ")
dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)



y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)

print("Decision Tree with Change Some Parameters Parameter with GridSearch ")
p = [{'class_weight' : ['balanced',None],'criterion':['gini', 'entropy'],'splitter':['best', 'random'], 'max_features':['auto','sqrt','log2']} ]


gs = GridSearchCV(estimator= dtc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10 ,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)


#################### Logistic Regression ################
print("Logistic Regression with Default Parameter ")
logr = LogisticRegression( max_iter=1000)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)

##################################################

######################### Naive Bayes ########################
print("Naive Bayes Regression with Default Parameter ")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)

############# Random Forest #############
print("Random Forest with Default Parameter ")
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))

print("Random Forest with Change Some Parameters Parameter with GridSearch ")
p = [{'n_estimators':[100,150,200,250],'criterion':['gini','entropy'], 'max_features':['auto']} ]
rfc = RandomForestClassifier(n_jobs=-1)

gs = GridSearchCV(estimator= rfc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=1000,
                           task_type="GPU",
                           devices='0:1')

    
model.fit(x_train,y_train,verbose=True)
y_pred = model.predict(x_test)
succes = cross_val_score(estimator= model, X=x, y=y, cv=2)
print(succes)


from sklearn.ensemble import AdaBoostClassifier

dtc = DecisionTreeClassifier()
ada1 = AdaBoostClassifier(base_estimator = dtc)


#ada1.fit(x_train,y_train)
#y_pred = ada1.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#print(classification_report(y_test, y_pred))
#print(cm)
p = [{'n_estimators':[50,60,70],'learning_rate':[1.0,2.0],'random_state' : [None,5]} ]
gs = GridSearchCV(estimator= ada1, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 3 ,
                  verbose=3,
                  n_jobs=-1
                  )

grid_search = gs.fit(x,y)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_
print(bestResult)
print(bestParams) 



from random import sample
N = 60000
samplingdata = dataframe.sample(frac =.50)  # random sampling 
x = samplingdata.drop(['satisfaction'], axis=1)
y = samplingdata['satisfaction']




#verilerin egitim ve test icin bolunmesi
## Verilerin test ve train olarak bölünmesi


stratifiedDf  = dataframe.groupby('satisfaction', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataframe))))).sample(frac=1).reset_index(drop=True) 
x = stratifiedDf.(['satisfaction'], axis=1)
y = stratifiedDf['satisfaction']
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.ensemble import AdaBoostClassifier
dtc = DecisionTreeClassifier()
ada1 = AdaBoostClassifier(base_estimator = dtc)


#ada1.fit(x_train,y_train)
#y_pred = ada1.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#print(classification_report(y_test, y_pred))
#print(cm)
p = [{'n_estimators':[50,60,70],'learning_rate':[1.0,2.0],'random_state' : [None,5]} ]
gs = GridSearchCV(estimator= ada1, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 3 ,
                  verbose=3,
                  n_jobs=-1
                  )

grid_search = gs.fit(x,y)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_
print(bestResult)
print(bestParams) 
"""
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
"""
sns.distplot(dataframe['Age'], bins=10, color='blue')
plt.title("Age Distribution")
plt.show()
"""
"""
f = (dataframe.loc[:, ['satisfaction', 'Type of Travel', 'Flight Distance', 'Baggage handling', 
                    'Online boarding', 'On-board service', # flight 
                    'Cleanliness', 'Seat comfort', 'Leg room service', 'Inflight entertainment', 
                    'Inflight wifi service' # in-flight
                    ]]).corr();
sns.heatmap(f, cmap='PiYG', annot=True, fmt='.1f');
"""
"""
plt.figure(dpi=100,figsize=(10,5))
sns.countplot(data=dataframe, x='Gender',hue='satisfaction',palette='rocket')
plt.xlabel('Gender Category')
plt.ylabel('Number of Transaction')
plt.title('Gender Group Satisfaction Category ')
"""
temp =  "Flight Distance,Inflight wifi service,Departure/Arrival time convenient,Ease of Online booking,Gate location,Food and drink,Online boarding,Seat comfort,Inflight entertainment,On-board service,Leg room service,Baggage handling,Checkin service,Inflight service,Cleanliness".split(",")
for i in temp:
    dataframe.drop(i)

tt = dataframe.columns()
f = (dataframe.loc[:,[tt]]).corr();
sns.heatmap(f, cmap='PiYG', annot=True, fmt='.1f');