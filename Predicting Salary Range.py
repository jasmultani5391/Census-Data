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




####################### STEP 1: DATA INTRODUCTION #######################

# In this first step, I want to pull up the data from UCI Machine Learning
# Repository and set it up as a dataframe using pandas. I then want to use
# .describe() to get a first glance of any correlations that may pop up.
df1 = pd.read_csv(
    r'C:\Users\Jasmine\Desktop\Database\US Census 1994\rawdatatest.txt',
    sep=" ",
    names=[
        'age',
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
        'salaryrange'
        ],
    delimiter=","
    )


df1.describe()


# In[5]:


#################### STEP 2: DATA CLEANUP - EXPLORE #####################

# When observing the dataset as is by printing the first 10 lines, we see
# that a few datapoints have '?' written in them. These incomplete lines
# may throw  off the rest of the data, so I replaced the '?' datapoints
# with 'NaN', and later dropped the lines containing 'Nan'. This will
# further clean up the data. I first detect which columns (with 
# qualitative data) by finding the unique elements in each.
workclass_unq = df1['workclass'].unique()
print(workclass_unq)

age_unq = df1['age'].unique()
print(age_unq)

education_unq = df1['education'].unique()
print(education_unq)

maritalstatus_unq = df1['maritalstatus'].unique()
print(maritalstatus_unq)

occupation_unq = df1['occupation'].unique()
print(occupation_unq)

race_unq = df1['race'].unique()
print(race_unq)

# From these last prints, I find out that 'workclass' and 'occupation'
# have datapoints with '?' being used. In this next step, I'll replace
# those datapoints as 'NaN' and drop each line from the main dataframe.
# I want to only use rows with completely filled data points, in case
# one column is dependent on another to predict salary range.
df1.workclass = df1['workclass'].replace('[\?,]', np.NaN, regex=True)
df1.occupation = df1['occupation'].replace('[\?,]', np.NaN, regex=True)
df1 = df1.dropna()
print(df1.head(5))


# In[6]:


##### STEP 2 (CONT): TURN MULTI-CLASS COLUMNS INTO MATRIX OF BINARIES ######

# There are some columns with qualitative data that need to be turned into
# numbers. To avoid having a huge range of numbers, I'll make new columns as
# a range of binaries. I do this for columns 'workclass', 'race', and
# 'maritalstatus'. For example, If the individual is from the Private working
# class, they will have '1' under the wcPrivate column, and '0' in the rest
# of the columns associated with workingclass.

# I will create a new series of columns and place a sign at the beginning of
# name to indicate which of the original columns it's labelling from. For
# example, 'wcPrivate' is labelling that this individual's workclass is
# in the private sector. The 'raceWhite' means that the indivudal is from
# the white racial group.
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


# In[10]:


#### STEP 2 (CONT): CREATE CLASSES TO HELP CLEAN UP DATA ###

class QualToQuant():
    # Using this class to clean up data.
    
    
    # The following method will allow us to normalize
    # the range of datapoints between 0 and 1.
    def normalizer(self, data):
        float_list = []
        for i in data:
            age = float(i)
            float_list.append(age)
        minimum = min(float_list)
        maximum = max(float_list)
        normalized_list = []
        for i in float_list:
            normalized_list += [(i - minimum) / (maximum-minimum)]
        return normalized_list
    
    
    # The following method is to convert the qualitative
    # info of whether individual is male or female into a
    # binary. This will be set up so that if the indivudal
    # is a female, the information will be denoted as '1'.
    def gender_labeler(self, data):
        gender_list = []
        for i in data:
            if 'Female' in i:
                gender_list.append(1)
            else:
                gender_list.append(0)
        return gender_list
    
    
    # The following method is to convert qualitative info
    # on country of origin into a binary. I am most interested
    # in understanding whether being an immigrant vs. a native
    # of the U.S. would influence the prediction. So-- if the
    # person is born in the U.S., the info will be denoted as '1'.
    # The 'og' in 'ogcountry_labeler' is meant to be read as
    # 'original'.
    def ogcountry_labeler(self, data):
        ogcountry_list = []
        for i in data:
            if 'United-States' in i:
                ogcountry_list.append(1)
            else:
                ogcountry_list.append(0)
        return ogcountry_list
    
    
    # The following method will label whether the individual
    # makes less than/equal to $50,000 per year, or more. The
    # former will be denoted as '0' and the latter as '1'. This
    # will later be used as the label to train the algorithm to
    # split up the indivudals into either classes.
    def salary_labeler(self, data):
        salary_list = []
        for i in data:
            if '<=50K' in i:
                salary_list.append(0)
            else:
                salary_list.append(1)
        return salary_list
    
    
    
    
class NearestK():
    
    # The following method is to train the labeled
    # data set 'i' number of times and to make a large
    # dictionary where 'i' is the key, and accuracy of
    # the classifier is the value. This will allow us
    # to find the best K to use in our KNeighborsClassifier.
    def bestK(self, train_data, test_data, train_labels, test_labels):
        k_dictionary = {}
        for i in range(1, 101, 1):
            classifier = KNeighborsClassifier(n_neighbors = i)
            classifier.fit(train_data, np.ravel(train_labels))
            classifier_accuracy = classifier.score(test_data, test_labels)
            k_dictionary[i] = classifier_accuracy
        return k_dictionary
    


# In[24]:


# Now, I can actually normalize  the data for datasets that have a range
# of numbers.
normalized_educationnum = QualToQuant()
df1['normalized_age'] = normalized_educationnum.normalizer(df1['age'])

normalized_educationnum = QualToQuant()
df1['normalized_educationnum'] = normalized_educationnum.normalizer(df1['educationnum'])

normalized_capitalgain = QualToQuant()
df1['normalized_capitalgain'] = normalized_capitalgain.normalizer(df1['capitalgain'])

normalized_capitalloss = QualToQuant()
df1['normalized_capitalloss'] = normalized_capitalloss.normalizer(df1['capitalloss'])

normalized_hoursperweek = QualToQuant()
df1['normalized_hoursperweek'] = normalized_hoursperweek.normalizer(df1['hoursperweek'])

normalized_fnlwgtint = QualToQuant()
df1['normalized_fnlwgtint'] = normalized_fnlwgtint.normalizer(df1['fnlwgt'])

# Now, I can change qualitative numbers into numbers of either 0 or 1
gender_label = QualToQuant()
df1['gender_label'] = gender_label.gender_labeler(df1['sex'])

ogcountry_label = QualToQuant()
df1['ogcountry_label'] = ogcountry_label.ogcountry_labeler(df1['ogcountry'])

salary_label = QualToQuant()
df1['salary_label'] = salary_label.salary_labeler(df1['salaryrange'])




######################### STEP 3: BUILD DATAFRAMES ########################

# Now that data has been properly cleaned and converted to numbers, I want to
# decide which features should be included when training the algorithms. I
# decided that I didn't want to include 'education',  'occupation' 'relationship'
# because they would be covered by 'education-number', 'workclass', and
# 'marital-status'. Adding the former three could be redundant.
featuresDF = df1[[
    'normalized_age',
    'normalized_educationnum',
    'normalized_capitalgain',
    'normalized_capitalloss',
    'normalized_hoursperweek',
    'gender_label',
    'ogcountry_label',
    'wcPrivate',
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
    'maritalMarriedAfSpouse'
    ]]

labelDF = df1['salary_label']

feature_names = [
    'normalized_age',
    'normalized_educationnum',
    'normalized_capitalgain',
    'normalized_capitalloss',
    'normalized_hoursperweek',
    'gender_label',
    'ogcountry_label',
    'wcPrivate',
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
    'maritalMarriedAfSpouse'
    ]


#################### STEP 4: EXPLORE DATA VISUALLY ####################

plt.scatter(df1['salary_label'], df1['normalized_age'], alpha=0.1)
plt.xlabel('Target')
plt.ylabel('Age')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()

plt.scatter(df1['salary_label'], df1['normalized_educationnum'], alpha=0.1)
plt.xlabel('Target')
plt.ylabel('Education')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()

plt.scatter(df1['salary_label'], df1['normalized_capitalgain'], alpha=0.1)
plt.xlabel('Target')
plt.ylabel('CapGain')
ax = plt.subplot()
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['<=50K', '>50K'])
plt.show()

# From the next histogram, we see that education is the heaviest
# indicator of the label of salary.
labelcorr = featuresDF.corr()
labelcorr = labelcorr['salary_label']
labelcorr = labelcorr.sort_values(ascending=False)
print(labelcorr)
plt.hist(df1['normalized_educationnum'], bins=20, alpha=.4, normed=True)




# In[17]:


###################### STEP 5: K-NEAREST CLASSIFIER #####################

# First, we must find the best K index that gives us the highest accuracy.
# We previously created a class called NearestK that contains the method
# to search for the best K value to use.
trainknn_data, testknn_data, trainknn_labels, testknn_labels = train_test_split(featuresDF, labelDF, test_size = 0.2, random_state = 100)
knn_pairs = NearestK()
knn_pairs = knn_pairs.bestK(trainknn_data, testknn_data, trainknn_labels, testknn_labels)
knn_k = list(knn_pairs.keys())
knn_accuracy = list(knn_pairs.values())
knn_percent = [round(i*100, 2) for i in knn_accuracy]
max_accuracy = max(knn_percent)
best_k = knn_k[knn_accuracy.index(max(knn_accuracy))]
print('Best K is ' + str(best_k) + ' at accuracy of ' + str(max_accuracy) + '%')

# Let's graph the data and visualize how these accuracy rates look like,
# and highlight the best K fit with a red dot.
plt.plot(knn_k, knn_percent)
plt.scatter(best_k, max_accuracy, c='red', cmap='jet', s=20, label = ('Best K = ' + str(best_k)))
plt.xlabel('K Number')
plt.ylabel('Validation Accuracy (%)') 
plt.title('Above or Below 50K line?: best K values to predict household salary')
plt.legend(loc=4)
plt.show()

# The best K Nearest Neighbor is when k is equal to 13, giing us an accuracy
# of 83.84%.
knn_classifier = KNeighborsClassifier(n_neighbors = best_k)
knn_classifier.fit(testknn_data, np.ravel(testknn_labels))
knn_classifier.score(testknn_data, testknn_labels)

# Now that we've trained the labeled data on this classifier, let's study
# the precision, recall and F1 scores of this classifier to determine'
# how reliable it is.
knn_scores = []
predictknn_labels= knn_classifier.predict(testknn_data)
precisionknn = precision_score(testknn_labels, predictknn_labels, average='binary')
recallknn = recall_score(testknn_labels, predictknn_labels, average='binary')
f1knn = f1_score(testknn_labels, predictknn_labels, average='binary')
knn_scores.append(precisionknn)
knn_scores.append(recallknn)
knn_scores.append(f1knn)
scorenames = ['precision', 'recall', 'F1']
knn_score_names = list(zip(scorenames, knn_scores))
print("Scores for KNN are: " + str(knn_score_names))

# Discuss the results here, and run some more trainings with newly mixed features.




# In[48]:


################### STEP 6: DECISION FOREST CLASSIFIER ###################

# Now, let's train this labelled on a Decision Forest classifier. For this,
# we need to split up the data between training and testing sets differently
# by changing the random_state amount. When training on the decision forest
# classifier, we want to adjust the amount of trees in the forest as the 
# square root of the length of features. In this case, the square root of 25
# will be 5. We'll slightly change the variable names when spliting up data.
# For example, trainknn_data becomes traindf_data to reflect which classier
# we're training the datasets on.
traindf_data, testdf_data, traindf_labels, testdf_labels = train_test_split(featuresDF, labelDF, random_state=5) 
forest_classifier = RandomForestClassifier(random_state=5)
forest_classifier.fit(traindf_data, traindf_labels)
print('Accuracy of decision forest classifier is ' + str(forest_classifier.score(testdf_data, testdf_labels)*100))
important_coefficients = forest_classifier.feature_importances_

# PROBLEM: this doesn't really gage how columns that were set up in a range
# of binaries (these columns include race and workclass) influence against
# columns that didn't have to be split up.
names_coefficients = dict(zip(feature_names, important_coefficients))
print('\n', names_coefficients)




# In[56]:


################# STEP 7: LOGISTIC REGRESSION CLASSIFIER #################

# Let's now train the data on logistic regression. Split up the dataset
# between trained and tested data by making the test_size=0.2. Also note
# that we slightly change the variable names so that trainknn_data becomes
# trainlogr_data, and so on.
trainlogr_data, testlogr_data, trainlogr_labels, testlogr_labels = train_test_split(featuresDF, labelDF, test_size=0.2) 

# Transform the data using the StandardScaler() class. The Logistic Regression
# classifier needs the data to be in a standardized shape.
scaler = StandardScaler()
trainlogr_data = scaler.fit_transform(trainlogr_data)
testlogr_data = scaler.transform(testlogr_data)

# Create and train the model on the LogisticRegression class.
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(trainlogr_data, trainlogr_labels)

# Score the model on the train data.
train_logreg_model = log_reg_classifier.score(trainlogr_data, trainlogr_labels)
print('Train score is ' + str(round(train_logreg_model*100, 2)))

# Score the model on the test data.
test_logreg_model = log_reg_classifier.score(trainlogr_data, trainlogr_labels)
print('Test score is ' + str(round(test_logreg_model*100)))

