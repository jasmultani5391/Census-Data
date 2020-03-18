#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


import matplotlib.pyplot as plt

import numpy as np
from numpy import array
from numpy import argmax


df1 = pd.read_csv(r'C:\Users\Jasmine\Desktop\Database\US Census 1994\rawdatatest.txt',
                  sep=" ",
                  names=['age',
                         'workclass',
                         'fnlwgt',
                         'education',
                         'educationnum',
                         'maritalstatus',
                         'occupation',
                         'relationship',
                         'race',
                         'sex',
                         'capitalgain',
                         'capitalloss',
                         'hoursperweek',
                         'ogcountry',
                         'salaryrange'],
                  delimiter = ","
                          )

print(df1.head(5))
df1.describe()


# # Cleaning up data even more:
# 
# Which rows have empty values? --> delete them

# In[2]:


workclass_unq = df1['workclass'].unique() #switch '?' to NaN and delete those rows
print(workclass_unq)
df1.workclass = df1['workclass'].replace('[\?,]', np.NaN, regex=True)

age_unq = df1['age'].unique()
print(age_unq)

education_unq = df1['education'].unique()
print(education_unq)

maritalstatus_unq = df1['maritalstatus'].unique()
print(maritalstatus_unq)

occupation_unq = df1['occupation'].unique() #switch '?' to NaN and delete those rows
print(occupation_unq)
df1.occupation = df1['occupation'].replace('[\?,]', np.NaN, regex=True)


race_unq = df1['race'].unique()
print(race_unq)

df1 = df1.dropna()
print(df1.head(5))


# # Turning the multi-class columns into matrix of binaries
# (for workclass, race, maritalstatus)
# 

# In[3]:


workclass_freq = df1.workclass.value_counts()
print(workclass_freq)
df1['wcPrivate'] = df1['workclass'].apply(lambda x: 1 if 'Private' in x else 0)
df1['wcSelfEmployedNotInc'] = df1['workclass'].apply(lambda x: 1 if 'Self-emp-not-inc' in x else 0)
df1['wcLocalGov'] = df1['workclass'].apply(lambda x: 1 if 'Local-gov' in x else 0)
df1['wcStateGov'] = df1['workclass'].apply(lambda x: 1 if 'State-gov' in x else 0)
df1['wcSelfEmployedInc'] = df1['workclass'].apply(lambda x: 1 if 'Self-emp-inc' in x else 0)
df1['wcFederalGov'] = df1['workclass'].apply(lambda x: 1 if 'Federal-gov' in x else 0)
df1['wcWithoutPay'] = df1['workclass'].apply(lambda x: 1 if 'Without-pay' in x else 0)


race_freq = df1.race.value_counts()
print(race_freq)
df1['raceWhite'] = df1['race'].apply(lambda x: 1 if 'White' in x else 0)
df1['raceBlack'] = df1['race'].apply(lambda x: 1 if 'Black' in x else 0)
df1['raceAsianPacific'] = df1['race'].apply(lambda x: 1 if 'Asian-Pac_Islander' in x else 0)
df1['raceAmerIndianEskimo'] = df1['race'].apply(lambda x: 1 if 'Amer-Indian-Eskimo' in x else 0)
df1['raceOther'] = df1['race'].apply(lambda x: 1 if 'Other' in x else 0)

maritalstatus_freq = df1.maritalstatus.value_counts()
print(maritalstatus_freq)
df1['maritalMarriedCivSpouse'] = df1['maritalstatus'].apply(lambda x: 1 if 'Married-civ-spouse' in x else 0)
df1['maritalNeverMarried'] = df1['maritalstatus'].apply(lambda x: 1 if 'Never-married' in x else 0)
df1['maritalDivorced'] = df1['maritalstatus'].apply(lambda x: 1 if 'Divorced' in x else 0)
df1['maritalSeparated'] = df1['maritalstatus'].apply(lambda x: 1 if 'Separated' in x else 0)
df1['maritalWidowed'] = df1['maritalstatus'].apply(lambda x: 1 if 'Widowed' in x else 0)
df1['maritalAbsentSpouse'] = df1['maritalstatus'].apply(lambda x: 1 if 'Married-spouse-absent' in x else 0)
df1['maritalMarriedAfSpouse'] = df1['maritalstatus'].apply(lambda x: 1 if 'Married-AF-spouse' in x else 0)


print(df1.head(5))
print(df1.columns)


# ## For the remaining columns that only have two groups, transforming data into binary matrix is easier

# In[4]:


def genderint(data):
    gend = []
    for i in data:
        if 'Male' in i:
            gend.append(1)
        else:
            gend.append(0)
    return gend

def ogcountryint(data):
    ogc = []
    for i in data:
        if 'United-States' in i:
            ogc.append(1)
        else:
            ogc.append(0)
    return ogc

def label(data):
    lbl = []
    for i in data:
        if '<=50K.'in i:
            lbl.append(0)
        else:
            lbl.append(1)
    return lbl

def normit(column):
    newlist = []
    for i in column:
        a = float(i)
        newlist.append(a)
    minimum = min(newlist)
    maximum = max(newlist)
    normalized = []
    for i in newlist:
        normalized += [(i - minimum)/ (maximum-minimum)]
    return normalized


#can i use preprocessing for normalizing data
df1['agenorm'] = normit(df1['age'])
df1['edunumnorm'] = normit(df1['educationnum'])
df1['genderint'] = genderint(df1['sex'])
df1['capgainnorm'] = normit(df1['capitalgain'])
df1['caplossnorm'] = normit(df1['capitalloss'])
df1['HoursPWknormit']= normit(df1['hoursperweek'])
df1['ogcountryint'] = ogcountryint(df1['ogcountry']) #there are some missing values here... '?'
df1['targetlbl'] = label(df1['salaryrange'])
df1['fnlwgtint'] = normit(df1['fnlwgt'])

#using the following line, I looked up if there are any missing values in my interested features that are listed as '?'
#I found that df1['ogcountry'] is the only column of my featurs of interest that has a '?' for value;
#but since I'm narrowing the values of "is this person originally from the US or not", I feel like categorizing '?' as 0
#should would be nough
(np.unique(df1['ogcountryint']))

#print(df1.head(5))
df1.describe()


# ## Setting up dataframes with varying features
# 
# normDF = includes columns that already had float variables (and normalized) + columns with data between two choices that would easily be converted.
# 
# catDF = columns that have data that include 3+ categories.
# 
# catnormDF = combines all of the above.
# 
# I later try to explore some of the data and visualzie any trends.

# In[7]:


#seting up data frame and exploring the dataframe


allDF = pd.DataFrame(
    {'Age': df1['agenorm'],
     'Education': df1['edunumnorm'],
     'Gender': df1['genderint'],
     'CapGain': df1['capgainnorm'],
     'CapLoss' : df1['caplossnorm'],
     'HoursPWk' : df1['HoursPWknormit'],
     'Ogcountry' : df1['ogcountryint'],
     'Label': df1['targetlbl']
    })

normDF = pd.DataFrame(
    {'Age': df1['agenorm'],
     'Education': df1['edunumnorm'],
     'Gender': df1['genderint'],
     'CapGain': df1['capgainnorm'],
     'CapLoss' : df1['caplossnorm'],
     'HoursPWk' : df1['HoursPWknormit'],
     'Ogcountry' : df1['ogcountryint']
    })

catnormDF = df1[['wcPrivate',
                 'wcSelfEmployedNotInc',
                 'wcLocalGov',
                 'wcStateGov',
                 'wcSelfEmployedInc',
                 'wcFederalGov',
                 'wcWithoutPay',
                 'raceWhite',
                 'raceBlack',
                 'raceAsianPacific',
                 'raceAmerIndianEskimo',
                 'raceOther',
                 'maritalMarriedCivSpouse',
                 'maritalNeverMarried',
                 'maritalDivorced',
                 'maritalSeparated',
                 'maritalWidowed',
                 'maritalAbsentSpouse',
                 'maritalMarriedAfSpouse',
                 'agenorm',
                 'edunumnorm',
                 'genderint',
                 'capgainnorm',
                 'caplossnorm',
                 'HoursPWknormit',
                 'ogcountryint']]

catDF = df1[['wcPrivate',
             'wcSelfEmployedNotInc',
             'wcLocalGov',
             'wcStateGov',
             'wcSelfEmployedInc',
             'wcFederalGov',
             'wcWithoutPay',
             'raceWhite',
             'raceBlack',
             'raceAsianPacific',
             'raceAmerIndianEskimo',
             'raceOther',
             'maritalMarriedCivSpouse',
             'maritalNeverMarried',
             'maritalDivorced',
             'maritalSeparated',
             'maritalWidowed',
             'maritalAbsentSpouse',
             'maritalMarriedAfSpouse']]
#catoriesDF includes workclass, race, marital status


labelDF = df1['targetlbl']

print(normDF.head(10))

#exploring data

plt.scatter(df1['targetlbl'],df1['age'],alpha=0.1)
plt.xlabel('Target')
plt.ylabel('Age')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()

plt.scatter(df1['targetlbl'],df1['educationnum'],alpha=0.1)
plt.xlabel('Target')
plt.ylabel('Education')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()

plt.scatter(df1['targetlbl'],normDF['CapGain'],alpha=0.1)
plt.xlabel('Target')
plt.ylabel('CapGain')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()


#plot your other histogram here



labelcorr = allDF.corr()
labelcorr = labelcorr['Label']
labelcorr = labelcorr.sort_values(ascending=False)
print(labelcorr) # 'education' is heaviest indicator, then 'age', 'gender', 'capgain'
plt.hist(df1['educationnum'], bins=20, alpha = .4, normed=True)



# # KNearest Neighbor

# In[9]:


def bigclassifying(tdata, tlabels, vdata, vlabels):
    classifierlist = {}
    for i in range(1, 101, 1):
        classifier = KNeighborsClassifier(n_neighbors = i)
        classifier.fit(tdata, np.ravel(tlabels))
        classifieraccuracy = classifier.score(vdata, vlabels)
        classifierlist[i] = classifieraccuracy
    return classifierlist


# In[22]:


#nearest neighbors with normDF vs labelDF
tdata, vdata, tlabels, vlabels = train_test_split(normDF, labelDF, test_size = 0.2, random_state = 100)
kAccpairs = bigclassifying(tdata, tlabels, vdata, vlabels)

kNum = list(kAccpairs.keys())
kAcc = list(kAccpairs.values())
kAccPerc = [round(i * 100, 2) for i in kAcc]


maxAcc = max(kAccPerc)
bestK = kNum[kAcc.index(max(kAcc))]

print('Best K is ' + str(bestK) + ' at accuracy of ' + str(maxAcc) + '%')
plt.plot(kNum, kAccPerc)


plt.scatter(bestK, maxAcc, c='red', cmap='jet', s=20, label = ('Best K = ' + str(bestK)))
    
plt.xlabel('K Number')
plt.ylabel('Validation Accuracy (%)') #accuracy is 81.88% with the final weights; is at 82.01% accuracy without fnlweights in NormDF
plt.title('Above or Below 50K line?: best K values to predict household salary')
plt.legend(loc=4)
plt.show()

bestKNN = KNeighborsClassifier(n_neighbors = bestK)
bestKNN.fit(tdata, np.ravel(tlabels))
bestKNN.score(vdata, vlabels)

scores = []

predictlabel= bestKNN.predict(vdata)
precisionKNN = precision_score(vlabels, predictlabel, average='binary')
recallKNN = recall_score(vlabels, predictlabel, average='binary')
F1KNN = f1_score(vlabels, predictlabel, average='binary')

scores.append(precisionKNN)
scores.append(recallKNN)
scores.append(F1KNN)

scorenames = ['precision', 'recall', 'F1']
KNNscores = list(zip(scorenames, scores))

print("Scores for KNN are: " + str(KNNscores))


# In[21]:


#knearest with catDF vs labelDF

tdataII, vdataII, tlabelsII, vlabelsII = train_test_split(catDF, labelDF, test_size = 0.2, random_state = 100)
scaler = StandardScaler()
tdataII = scaler.fit_transform(tdataII)
vdataII = scaler.transform(vdataII)


kAccpairsII = bigclassifying(tdataII, tlabelsII, vdataII, vlabelsII)

kNumII = list(kAccpairsII.keys())
kAccII = list(kAccpairsII.values())
kAccPercII = [round(i * 100, 2) for i in kAccII]


maxAccII = max(kAccPercII)
bestKII = kNumII[kAccII.index(max(kAccII))]

print('Best K is ' + str(bestKII) + ' at accuracy of ' + str(maxAccII) + '%')
plt.plot(kNumII, kAccPercII)


plt.scatter(bestKII, maxAccII, c='red', cmap='jet', s=20, label = ('Best K = ' + str(bestKII)))
    
plt.xlabel('K Number')
plt.ylabel('Validation Accuracy (%)') 
plt.title('Above or Below 50K line?: best K values to predict household salary')
plt.legend(loc=4)
plt.show()

bestKNNII = KNeighborsClassifier(n_neighbors = bestKII)
bestKNNII.fit(tdataII, np.ravel(tlabelsII))

scoresII = []

predictlabelII= bestKNNII.predict(vdataII)
precisionKNNII = precision_score(vlabelsII, predictlabelII, average='binary')
recallKNNII = recall_score(vlabelsII, predictlabelII, average='binary')
F1KNNII = f1_score(vlabelsII, predictlabelII, average='binary')

scoresII.append(precisionKNNII)
scoresII.append(recallKNNII)
scoresII.append(F1KNNII)

scorenamesII = ['precision', 'recall', 'F1']
KNNscoresII = list(zip(scorenamesII, scoresII))

print("Scores for KNN are: " + str(KNNscoresII))


# In[12]:


#knearest with catnormDF vs labelDF

tdataIII, vdataIII, tlabelsIII, vlabelsIII = train_test_split(catnormDF, labelDF, test_size = 0.2, random_state = 100)
scaler = StandardScaler()
tdataIII = scaler.fit_transform(tdataIII)
vdataIII = scaler.transform(vdataIII)


kAccpairsIII = bigclassifying(tdataIII, tlabelsIII, vdataIII, vlabelsIII)

kNumIII = list(kAccpairsIII.keys())
kAccIII = list(kAccpairsIII.values())
kAccPercIII = [round(i * 100, 2) for i in kAccIII]

maxAccIII = max(kAccPercIII)
bestKIII = kNumIII[kAccIII.index(max(kAccIII))]

print('Best K is ' + str(bestKIII) + ' at accuracy of ' + str(maxAccIII) + '%')
plt.plot(kNumIII, kAccPercIII)


plt.scatter(bestKIII, maxAccIII, c='red', cmap='jet', s=20, label = ('Best K = ' + str(bestKIII)))
    
plt.xlabel('K Number')
plt.ylabel('Validation Accuracy (%)') 
plt.title('Above or Below 50K line?: best K values to predict household salary')
plt.legend(loc=4)
plt.show()

bestKNNIII = KNeighborsClassifier(n_neighbors = bestKIII)
bestKNNIII.fit(tdataIII, np.ravel(tlabelsIII))


scoresIII = []

predictlabelIII= bestKNNIII.predict(vdataIII)
precisionKNNIII = precision_score(vlabelsIII, predictlabelIII, average='binary')
recallKNNIII = recall_score(vlabelsIII, predictlabelIII, average='binary')
F1KNNIII = f1_score(vlabelsIII, predictlabelIII, average='binary')

scoresIII.append(precisionKNNIII)
scoresIII.append(recallKNNIII)
scoresIII.append(F1KNNIII)

scorenamesIII = ['precision', 'recall', 'F1']
KNNscoresIII = list(zip(scorenamesIII, scoresIII))

print("Scores for KNN are: " + str(KNNscoresIII))



# # Decision Forest

# In[13]:


#use normDF with decision forest
traindataF, testdataF, trainlblF, testlblF = train_test_split(normDF, labelDF, random_state = 2) #why does random_state need to be 1
forest = RandomForestClassifier(random_state=2)
forest.fit(traindataF, trainlblF)
print('Accuracy of Forest classifier is ' + str(forest.score(testdataF, testlblF)*100)) #81.87% (normDF without finalweights); 79.99% with the finalweights in normDF
print(forest.feature_importances_) #from this printout, we the classifier lets us know that age is the strongest indicator for label (of this set of features)


# In[14]:


#use catDF vs labelDF with decision forest

traindataFII, testdataFII, trainlblFII, testlblFII = train_test_split(catDF, labelDF, random_state = 2) #why does random_state need to be 1
forestII = RandomForestClassifier(random_state=2)
forestII.fit(traindataFII, trainlblFII)
print('Accuracy of ForestII classifier is ' + str((forestII.score(testdataFII, testlblFII)*100))) #81.87% (normDF without finalweights); 79.99% with the finalweights in normDF
forestIIcoeffs = forestII.feature_importances_
print(forestIIcoeffs)


# In[15]:


#use catnormDF vs labelDF for this Decision Forest

traindataFIII, testdataFIII, trainlblFIII, testlblFIII = train_test_split(catnormDF, labelDF, random_state = 2) #why does random_state need to be 1
forestIII = RandomForestClassifier(random_state=2)
forestIII.fit(traindataFIII, trainlblFIII)
print('Accuracy of ForestIII classifier is ' + str((forestIII.score(testdataFIII, testlblFIII)*100))) #81.87% (normDF without finalweights); 79.99% with the finalweights in normDF
forestIIIcoeffs = forestII.feature_importances_
print(forestIIIcoeffs)


# # Logistic Regression

# In[16]:


#use normDF and labelDF for logistic regression
traindataLog, testdataLog, trainlblLog, testlblLog = train_test_split(normDF,labelDF,test_size = 0.2)

#scaler = StandardScaler()
#traindataLog = scaler.fit_transform(traindataLog)
#testdataLog = scaler.transform(testdataLog)
# Create and train the model
model = LogisticRegression()
model.fit(traindataLog, trainlblLog)

# Score the model on the train data
trainmodel = model.score(traindataLog, trainlblLog) #should parameters be the same as .fit()
print('train score is ' + str(round(trainmodel*100,2)))
# Score the model on the test data
testmodel = model.score(testdataLog, testlblLog)
print('test score is ' + str(round(testmodel*100)))

coeffs = list(zip(['age', 'educationnum', 'gender', 'capgain', 'caploss', 'hoursPWK', 'OGcountry'], model.coef_[0]))
print(coeffs) #these coefficients state that capgain and educationnum are the greater indicators of label


# In[17]:


#use catDF and labelDF for logistic regression
traindataLogII, testdataLogII, trainlblLogII, testlblLogII = train_test_split(catDF,labelDF,test_size = 0.2)

scaler = StandardScaler()
traindataLogII = scaler.fit_transform(traindataLogII)
testdataLogII = scaler.transform(testdataLogII)
# Create and train the model
modelII = LogisticRegression()
modelII.fit(traindataLogII, trainlblLogII)

# Score the model on the train data
trainmodelII = modelII.score(traindataLogII, trainlblLogII) #should parameters be the same as .fit()
print('train score is ' + str(round(trainmodelII*100,2)))
# Score the model on the test data
testmodelII = modelII.score(testdataLogII, testlblLogII)
print('test score is ' + str(round(testmodelII*100)))

coeffsII = list(zip(['wcPrivate', 'wcSelfEmployedNotInc', 'wcLocalGov',
       'wcStateGov', 'wcSelfEmployedInc', 'wcFederalGov', 'wcWithoutPay',
       'raceWhite', 'raceBlack', 'raceAsianPacific', 'raceAmerIndianEskimo',
       'raceOther', 'maritalMarriedCivSpouse', 'maritalNeverMarried',
       'maritalDivorced', 'maritalSeparated', 'maritalWidowed',
       'maritalAbsentSpouse', 'maritalMarriedAfSpouse'], modelII.coef_[0]))
print(coeffsII)


# In[18]:


#use catnormDF and labelDF for logistic regression
traindataLogIII, testdataLogIII, trainlblLogIII, testlblLogIII = train_test_split(catnormDF,labelDF,test_size = 0.2)

scaler = StandardScaler()
traindataLogIII = scaler.fit_transform(traindataLogIII)
testdataLogIII = scaler.transform(testdataLogIII)
# Create and train the model
modelIII = LogisticRegression()
modelIII.fit(traindataLogIII, trainlblLogIII)

# Score the model on the train data
trainmodelIII = modelIII.score(traindataLogIII, trainlblLogIII) #should parameters be the same as .fit()
print('train score is ' + str(round(trainmodelIII*100,2)))
# Score the model on the test data
testmodelIII = modelIII.score(testdataLogIII, testlblLogIII)
print('test score is ' + str(round(testmodelIII*100)))

coeffsIII = list(zip(['wcPrivate', 'wcSelfEmployedNotInc', 'wcLocalGov',
       'wcStateGov', 'wcSelfEmployedInc', 'wcFederalGov', 'wcWithoutPay',
       'raceWhite', 'raceBlack', 'raceAsianPacific', 'raceAmerIndianEskimo',
       'raceOther', 'maritalMarriedCivSpouse', 'maritalNeverMarried',
       'maritalDivorced', 'maritalSeparated', 'maritalWidowed',
       'maritalAbsentSpouse', 'maritalMarriedAfSpouse'], modelIII.coef_[0]))
print(coeffsIII)


# In[ ]:


#0 is =<50k, while 1 is >50k
#predict using      [# age, educationnum, gender, capgain, caploss, hoursPWK, OGcountry]
#prediction1 = bestKNN.predict([[0.01, 0.73, 0.00, 0.00, 0.01, 0.00, 1.00]])  #returns 1
#prediction2 = bestKNN.predict([[0.59, 0.89, 0.00, 0.59, 0.00, 0.00, 1.00]]) #returns 1
#prediction3 = bestKNN.predict([[0.59, 0.89, 0.00, 0.00, 0.35, 1.00, 1.00]]) #returns 1, 
#prediction4 = bestKNN.predict([[0.59, 0.89, 0.30, 0.00, 0.60, 1.00, 1.00]]) #returns 1
#prediction5 = bestKNN.predict([[0.59, 0.89, 0.30, 0.00, 0.60, 0.00, 0.00]]) #returns 1
#prediction6 = bestKNN.predict([[0.59, 0.89, 0.00, 0.59, 0.00, 1.00, 1.00]]) #returns 0
#print(prediction1, prediction2, prediction3, prediction4, prediction5, prediction6)

