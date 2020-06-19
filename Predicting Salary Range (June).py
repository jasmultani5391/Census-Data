#!/usr/bin/env python
# coding: utf-8

# In[1]:


############ SPACE FOR FUNCTIONS + LIBRARIES ###########

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from numpy import array
from numpy import argmax

from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


class QualToQuant():
    # Using this class to clean up data. Qualitative to quantitative.
    ''' This method applies one-hot encoding across columns
    that have qualitative information and concat them within
    one dataframe.
    '''
    
    def onehotenc_vals(self, columns):
        df2 = pd.DataFrame(columns=[])
        for i in columns:
            a = pd.get_dummies(df1[i], prefix=i)
            df2 = pd.concat([df1, a],
                            axis=1
                           )
        return df2
                
    ''' The following method will allow us to normalize the range of
    datapoints between 0 and 1.
    '''
    def normalizer(self, data):
        float_list = []
        for i in data:
            age = float(i)
            float_list.append(age)
        minimum = min(float_list)
        maximum = max(float_list)
        normalized_list = []
        for i in float_list:
            normalized_list += [(i-minimum)
                                / (maximum-minimum)]
        return normalized_list
    
    ''' The following method is to convert the qualitative info of whether
    individual is male or female into a binary. This will be set up so
    that if the indivudal is a female, the information will be denoted
    as '1'.
    '''
    def gender_labeler(self, data):
        gender_list = []
        for i in data:
            if 'Female' in i:
                gender_list.append(1)
            else:
                gender_list.append(0)
        return gender_list

    ''' I am most interested in understanding whether being an immigrant
    vs. a native of the U.S. would influence the prediction. If the
    person is born in the U.S., the info will be denoted as '1'.
    '''
    def ogcountry_labeler(self, data):
        ogcountry_list = []
        for i in data:
            if 'United-States' in i:
                ogcountry_list.append(1)
            else:
                ogcountry_list.append(0)
        return ogcountry_list

    ''' The following method will label whether the individual makes less
    than or equal to $50,000 per year, or more. The former will be
    denoted as '0' and the latter as '1'. This will later be used as
    the label to train the algorithm to split up the indivudals into
    either classes.
    '''
    def salary_labeler(self, data):
        salary_list = []
        for i in data:
            if '<=50K' in i:
                salary_list.append(0)
            else:
                salary_list.append(1)
        return salary_list


class FilterGroups():
    # Organizing columns once one-hot encoding is done.
    
    '''This task is to filter the final correlation dicitionary
    into groups--specifically for the originally qualitative
    columns that I applied one-hot encoding for.
    '''
    def make_corr_groups(self, dictionary):
        maritalstatus_corr = {}
        relationship_corr = {}
        occupation_corr = {}
        workclass_corr = {}
        race_corr = {}
        for key, value in dictionary.items():
            if 'maritalstatus' in key:
                maritalstatus_corr[key] = value
            if 'relationship_' in key:
                relationship_corr[key] = value
            if 'occupation_' in key:
                occupation_corr[key] = value
            if 'workclass_' in key:
                workclass_corr[key] = value
            if 'race_' in key:
                race_corr[key] = value
        return maritalstatus_corr, relationship_corr,             occupation_corr, workclass_corr,             race_corr

    '''I noticed that after the one-hot encoding, the column
    names get really long and look annoying. So, with this
    function, I'll be able to split the names from "_ " and
    only use the description of the column group as new name.
    '''
    def separate_onehot(self, to_convert):
        desc_list = []
        for i in to_convert:
            if "_ " in i:
                i = i.split("_ ")
                sep_i = i[1]
                desc_list.append(sep_i)
            else:
                continue
        return desc_list

    
class NearestK():
    ''' The following method is to train the labeled data set 'i' number
    oftimes and to make a large dictionary where 'i' is the key, and
    accuracy of the classifier is the value. This will allow us to
    find the best K to use in our KNeighborsClassifier.
    '''
    def bestK(self, train_data,
              test_data, train_label,
              test_label):
        # First, set up an empty dictionary.
        k_dictionary = {}
        for i in range(1, 101, 1):
            classifier = KNeighborsClassifier(n_neighbors=i)
            classifier.fit(train_data, np.ravel(train_label))
            classifier_accuracy = classifier.score(test_data, test_label)
            k_dictionary[i] = classifier_accuracy
        return k_dictionary


# In[2]:


####################### STEP 1: DATA INTRODUCTION ######################

# Pull up the data from UCI Machine Learning Repository and set it up as
# a dataframe using pandas. I then want to use .describe() to get a
# first glance of any correlations that may pop up.
path = r'C:\Users\Jasmine\Desktop\Database\US Census 1994\rawdatatest.txt'
df1 = pd.read_csv(path,
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
                         'salaryrange'
                         ],
                  delimiter=","
                  )


# In[3]:


#################### STEP 2: DATA CLEANUP - EXPLORE ####################
# In order to understand how qualitative info can be switched to
# quantitative info, I first want to know the unique values in each
# column.

workclass_count = df1['workclass'].value_counts()
education_count = df1['education'].value_counts()
maritalstatus_count = df1['maritalstatus'].value_counts()
occupation_count = df1['occupation'].value_counts()
race_count = df1['race'].value_counts()
print(workclass_count,
      education_count,
      maritalstatus_count,
      occupation_count,
      race_count
      )


# In[4]:


# From this last print, I found out that there are 10 rows where there
# are indivudals who are without pay or have not worked. Because our goal
# is to understand which  the socioeconomic factors of those who make
# salary, I will have to drop these rows.
df1.workclass = df1['workclass'].replace('Without-pay',
                                         np.NaN,
                                         regex=True
                                         )
df1.workclass = df1['workclass'].replace('Never-worked',
                                         np.NaN,
                                         regex=True
                                         )
df1.dropna()

# From these last prints, I find out that 'workclass' and
# 'occupation' have datapoints with '?' being used.
# For now, I'll replace '?' with a 'NaN'.
df1.workclass = df1['workclass'].replace('[\?,]',
                                         np.NaN,
                                         regex=True
                                         )
df1.occupation = df1['occupation'].replace('[\?,]',
                                           np.NaN,
                                           regex=True
                                           )

# Print value count of workclass and occupation to make sure NaN
# decisions worked.
# print(df1['workclass'].value_counts(),
#      df1['occupation'].value_counts()
#     )


# In[5]:


##### STEP 2 (CONT): TURN QUALITATIVE COLUMNS INTO MATRIX OF BINARIES ######

# There are some columns with qualitative data that need to be turned into
# numbers. To avoid having a huge range of numbers, I'll make new columns as
# a range of binaries. Will use one hot encoding through "pd.get_dummies".
qualcol = ['workclass',
           'race',
           'occupation',
           'maritalstatus',
           'relationship'
           ]
df2 = pd.DataFrame(columns=[])
for i in qualcol:
    df2 = pd.concat([df2, pd.get_dummies(df1[i], prefix=i)],
                    axis=1
                   )
print(df2.columns)


# In[6]:


######################### STEP 2: FEATURE ENGINEERING ########################
# Now that data has been properly cleaned and converted to numbers, I want to
# decide which features should be included when training the algorithms.

# This is the label that we are trying to predict. More than 50K is 1.
# Less than 50K is 0.
labelDF = pd.DataFrame(columns=['salary_label'])


# In[7]:


##### STEP 2(CONT): NORMALIZE DATA THAT HAVE RANGE OF NUMBERS #####
norm_age = QualToQuant()
df2['norm_age'] = norm_age.normalizer(df1['age'])

norm_educationnum = QualToQuant()
df2['norm_edunum'] = norm_educationnum.normalizer(df1['educationnum'])

norm_capitalgain = QualToQuant()
df2['norm_capitalgain'] = norm_capitalgain.normalizer(df1['capitalgain'])

norm_capitalloss = QualToQuant()
df2['norm_capitalloss'] = norm_capitalloss.normalizer(df1['capitalloss'])

norm_hoursperweek = QualToQuant()
df2['norm_hoursperweek'] = norm_hoursperweek.normalizer(df1['hoursperweek'])

norm_fnlwgtint = QualToQuant()
df2['norm_fnlwgtint'] = norm_fnlwgtint.normalizer(df1['fnlwgt'])

#### STEP 2(CONT): Change columns that have two values into 0 or 1 ####
gender_label = QualToQuant()
df2['gender_label'] = gender_label.gender_labeler(df1['sex'])

salary_label = QualToQuant()
labelDF['salary_label'] = salary_label.salary_labeler(df1['salaryrange'])
df2['salary_label'] = salary_label.salary_labeler(df1['salaryrange'])

# For country origin, I split it up values up into American-born vs not
# American-born, to simplify things.
ogcountry_label = QualToQuant()
df2['ogcountry_label'] = ogcountry_label.ogcountry_labeler(df1['ogcountry'])
print(df2.head(0))


# In[8]:


# FeatDF contains all the original columns, and one-hot encoded columns, and
# normalized datapoints. I wanted to rename df2 as featDF for easier memory.
featDF = df2
# This completeDF will allow us to look at the original columns (dropped from
# featuresDF).
completeDF = df1[['age',
                  'workclass',
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
                  ]]


# In[9]:


#################### STEP 4: EXPLORE DATA VISUALLY ####################
edu_lvl_lbl = ['preschool',
               '1st-4th',
               '5th-6th',
               '7th-8th',
               '9th',
               '10th',
               '11th',
               '12th',
               'HS grad',
               'Some college',
               'Assoc. Voc.',
               'Assoc. Acad.',
               'Bachelors',
               'Masters',
               'Prof-school',
               'Doctorate'
               ]

# The following is a scatterplot that shows the age group spread
# between those who make more than 50K or less than 50K per year.
# I noticed that the datapoints are truncated at each sides for
# the >50K group, condensing the age frequency between 25-70 years old.
plt.scatter(df1['age'],
            df2['salary_label'],
            alpha=0.1
            )
ax = plt.subplot()
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(['<=50K', '>50K'])
ax.set_title('Spread of Individual\'s Age vs. Salary Label')
ax.set_xlabel('Age')
ax.set_ylabel('Salary Label')
plt.show()

# The following creates a new data frame sorted by education number (ascending),
# will make it easier to view the data if it's in order by education.
sort_by_edunum = df1.sort_values(by=['educationnum'])
plt.figure(figsize=(8, 3))
plt.scatter(sort_by_edunum['education'],
            df2['salary_label'],
            alpha=0.1
            )

ax = plt.subplot()
ax.set_xlabel([edu_lvl_lbl])
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(['<=50K', '>50K'])
ax.set_title('Spread of Individual\'s Education Level vs. Salary Label')
ax.set_xlabel('Education Level')
plt.xticks(rotation=55)
ax.set_ylabel('Salary Label')
plt.show()

# This scatterplot reveals if capitalgain per year has significantly helped
# push an individual's yearly salary above our threshold. I see a wider
# spread capitalgain amount for those who received a capitalgain thisy year.
# Unclear as to whether capitalgain means an extra side-job or tax benefits?
plt.scatter(df1['capitalgain'],
            df2['salary_label'],
            alpha=0.1)
ax = plt.subplot()
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(['<=50K', '>50K'])
ax.set_title('Amount of Capital Gain vs. Salary Label')
ax.set_xlabel('Capital Gain ($)')
ax.set_ylabel('Salary Label')
plt.show()

# This histogram visualizes their education distribution The bulk is
# made up by high school graduates. It is then followed by those who
# have some college education, and then those with bachelor's.
plt.figure(figsize=(8, 10))
plt.hist(sort_by_edunum['education'],
         bins=20,
         alpha=.4,
         normed=True
         )
ax = plt.subplot()
ax.set_title('Education Level Amongst Total Group')
ax.set_xlabel('Education Level')
plt.xticks(rotation=25)
ax.set_ylabel('Amount of People')


# In[10]:


# Pie chart to reveal education level distribution of total database.
total_edulvl_dist = dict(df1['education'].value_counts())
# print(total_edulvl_dist)
plt.figure(figsize=(8, 10))
edu_names = list(total_edulvl_dist.keys())
edu_freqs_total = list(total_edulvl_dist.values())
pie1 = plt.pie(edu_freqs_total,
               labels=edu_names,
               autopct='%0.1f%%',
               pctdistance=1.3,
               labeldistance=1.4,
               rotatelabels=True
               )
plt.legend(pie1[0],
           edu_names,
           loc="best",
           bbox_to_anchor=(1.5, 1)
           )
plt.axis('equal')
plt.title('Education Distribution of Total Individuals')
plt.show()

# Pie chart to reveal education level distribution of pool of people who
# make ABOVE 50K salary.
above_50_sort = df1.loc[df2['salary_label'] == 1]
above_50_byedu = dict(above_50_sort['education'].value_counts())

plt.figure(figsize=(15, 15))
edu_names_abv = list(above_50_byedu.keys())
edu_freqs_abv = list(above_50_byedu.values())
pie2 = plt.pie(edu_freqs_abv,
               labels=edu_names_abv,
               autopct='%0.1f%%',
               pctdistance=1.3,
               labeldistance=1.4,
               rotatelabels=True
               )
plt.legend(pie2[0],
           edu_names_abv,
           loc="best",
           bbox_to_anchor=(1.5, 1)
           )
plt.axis('equal')
plt.title('Education Distribution of >50K Individuals')
plt.show()

# Pie chart to reveal education level distribution of pool of people who
# make BELOW 50K salary
below_50_sort = df1.loc[df2['salary_label'] == 0]
below_50_byedu = dict(below_50_sort['education'].value_counts())

plt.figure(figsize=(8, 10))
edu_names_blw = list(below_50_byedu.keys())
edu_freqs_blw = list(below_50_byedu.values())
pie3 = plt.pie(edu_freqs_blw,
               labels=edu_names_blw,
               autopct='%0.1f%%',
               pctdistance=1.3,
               labeldistance=1.4,
               rotatelabels=True
               )
plt.legend(pie3[0],
           edu_names_blw,
           loc="best",
           bbox_to_anchor=(1.5, 1)
           )
plt.axis('equal')
plt.title('Education Distribution of <50K Individuals')
plt.show()


# In[11]:


# Two scatter plots, one for <50K label and another for >50k label to
# study the spread of age and hours worked per week. This could be
# helpful for someone who wants to narrow down the model and fit it
# only for individuals who are above legal working age or before
# retirement. It could also be interesting to see which individuals
# are relying on being part of a double-income family to contribute
# to their home salary.
sns.lmplot(x='age',
           y='hoursperweek',
           data=df1,
           fit_reg=True,  # Regression line
           hue='salaryrange',
           col='salaryrange',
           col_wrap=2,
           height=5,
           scatter_kws={'alpha': 0.3}
           )

# This overlays the two graphs from previous scatterplot into one.
sns.lmplot(x='age',
           y='hoursperweek',
           data=df1,
           fit_reg=False,  # Regression line
           hue='salaryrange',
           height=5,
           scatter_kws={'alpha': 0.3}
           )

# Scatterplot that studies spread of age vs capital gain and compares
# the spread between groups.
sns.lmplot(x='age',
           y='capitalgain',
           data=df1,
           fit_reg=True,  # Regression line
           hue='salaryrange',
           col='salaryrange',
           col_wrap=2,
           height=5,
           scatter_kws={'alpha': 0.3}
           )
# Scatterplot that studies spread of age vs education and compares
# the spread between groups. There's a clear line distinguishing
# level 8 (12th grade) in the >50K group.
sns.lmplot(x='age',
           y='educationnum',
           data=df1,
           fit_reg=True,  # Regression line
           hue='salaryrange',
           col='salaryrange',
           col_wrap=2,
           height=5,
           scatter_kws={'alpha': 0.3}
           )


# In[12]:


# Creating a heatmap to visualize which numeric columns correlate
# with salary label. # From the following printout, we see that
# education number, age, hours per week, and capital gain are roughly
# the highest correlators
corr = completeDF.corr()
min_color = 'white'
max_color = (0.03137254,
             0.18823529411,
             0.41960784313,
             1
             )

cmap = LinearSegmentedColormap.from_list("", [max_color,
                                              min_color,
                                              max_color
                                              ]
                                         )
fig, ax = plt.subplots(figsize=(7, 5))
fig = sns.heatmap(corr,
                  annot=True,
                  cmap=cmap,
                  xticklabels=corr.columns.values,
                  yticklabels=corr.columns.values,
                  cbar=True,
                  linewidths=.5,
                  annot_kws={"fontsize": 12},
                  ax=ax
                  )
plt.xticks(rotation=16)
fig.set_ylim(6.5, 0)  # First value should be (#of rows) + (.5).
fig.xaxis.set_tick_params(labelsize=10)
fig.yaxis.set_tick_params(labelsize=10)
plt.show()


# In[13]:


# However, the heatmap does not illuminate how qualitative features
# differentially effect correlation with the salary label. Here,
# I printed out the correlation values, which allows to see how the one-
# hot encoded columns compete against the numerical columns. With a quick
# glance, we see that being a married husband correlate with higher salary
# label.
featDF.describe()
values_corr = featDF.corr()
salary_corr = values_corr.iloc[-2]
salary_corr = salary_corr.sort_values(ascending=False)
print(salary_corr)
salary_corrdict = dict(salary_corr)

salary_corr_filter = FilterGroups()
make_corr_groups = salary_corr_filter.make_corr_groups(salary_corrdict)
maristat_corrs = make_corr_groups[0]
relationship_corrs = make_corr_groups[1]
occupation_corrs = make_corr_groups[2]
workclass_corrs = make_corr_groups[3]
race_corrs = make_corr_groups[4]

maristat_filter = FilterGroups()
maristat_desc = maristat_filter.separate_onehot(maristat_corrs)

relationship_filter = FilterGroups()
relationship_desc = relationship_filter.separate_onehot(relationship_corrs)

occupation_filter = FilterGroups()
occupation_desc = occupation_filter.separate_onehot(occupation_corrs)

workclass_filter = FilterGroups()
workclass_desc = workclass_filter.separate_onehot(workclass_corrs)

race_filter = FilterGroups()
race_desc = race_filter.separate_onehot(race_corrs)

# The following print gives colum description with correlation weight.
print(maristat_corrs,
      '\n', '\n',
      relationship_corrs,
      '\n', '\n',
      occupation_corrs,
      '\n', '\n',
      workclass_corrs,
      '\n', '\n',
      race_corrs
      )

print(maristat_desc,
      '\n', '\n',
      relationship_desc,
      '\n', '\n',
      occupation_desc,
      '\n', '\n',
      workclass_desc,
      '\n', '\n',
      race_desc
      )

data_qualcorrs = {'maritalstatus': list(maristat_corrs.values()),
                  'relationship': list(relationship_corrs.values()),
                  'occupation': list(occupation_corrs.values()),
                  'workclass': list(workclass_corrs.values()),
                  'race': list(race_corrs.values())
                  }

qualcol_corrgroup = pd.DataFrame.from_dict(data_qualcorrs,
                                           orient='index')

qualcol_corrgroupT = qualcol_corrgroup.T


# In[14]:


# Lineplot to reveal distribution of correlation weights.
# Correlation assessment follows the general rule:
# High correlation : +/- .50 <= x < 1.0
# Moderate correlation: +/- .30 <= x <= .49
# Low correlation: 0 < x <= +/- .29

# Graph ax1, which is correlation values for marital status.
f, ax1 = plt.subplots(nrows=1,
                      ncols=1,
                      figsize=(7, 3)
                      )
sns.lineplot(data=qualcol_corrgroupT['maritalstatus'])
maristat_xtick = ['',
                  'Married-civ-spouse',
                  'Married-AF-spouse',
                  'Married-spouse-absent',
                  'Widowed',
                  'Separated',
                  'Divorced',
                  'Never-married']
ax1.set_xticklabels(maristat_xtick)
ax1.set(xlabel='Marital Statuses',
        ylabel='Correlation Weight'
        )
ax1.set_title('>50K Correlation by Marital Status',
              fontsize=14
              )
plt.xticks(rotation=45)
y1 = qualcol_corrgroupT['maritalstatus']
x1 = range(0, len(y1), 1)
plt.plot(x1, y1, 'o')
plt.show()

# Following shows correlation weight for relationship.
f, ax2 = plt.subplots(nrows=1,
                      ncols=1,
                      figsize=(7, 3)
                      )
sns.lineplot(data=qualcol_corrgroupT['relationship'])
relationship_xtick = ['',
                      'Husband',
                      'Wife',
                      'Other-relative',
                      'Unmarried',
                      'Not-in-family',
                      'Own-child'
                      ]
ax2.set_xticklabels(relationship_xtick)
ax2.set(xlabel='Relationship',
        ylabel='Correlation Weight'
        )
ax2.set_title('>50K Correlation by Relationship',
              fontsize=14
              )
plt.xticks(rotation=45)
y2 = qualcol_corrgroupT['relationship']
x2 = range(0, len(y2), 1)
plt.plot(x2, y2, 'o')
plt.show()

# Following shows correlation weight for workclass.
f, ax3 = plt.subplots(nrows=1,
                      ncols=1,
                      figsize=(7, 3)
                      )
sns.lineplot(data=qualcol_corrgroupT['workclass'])
workclass_xtick = ['',
                   'Self-emp-inc',
                   'Federal-gov',
                   'Local-gov',
                   'Self-emp-not-inc',
                   'State-gov',
                   'Private'
                   ]
ax3.set_xticklabels(workclass_xtick)
ax3.set(xlabel='Workclass',
        ylabel='Correlation Weight'
        )
ax3.set_title('>50K Correlation by Workclass',
              fontsize=14
              )
plt.xticks(rotation=45)
y3 = qualcol_corrgroupT['workclass']
x3 = range(0, len(y3), 1)
plt.plot(x3, y3, 'o')
plt.show()

# Following shows correlation weight for race.
f, ax4 = plt.subplots(nrows=1,
                      ncols=1,
                      figsize=(7, 3)
                      )
sns.lineplot(data=qualcol_corrgroupT['race'])
race_xtick = ['',
              'White',
              'Asian-Pac-Islander',
              'Other',
              'Amer-Indian-Eskimo',
              'Black'
              ]
ax4.set_xticklabels(race_xtick)
ax4.set(xlabel='Race',
        ylabel='Correlation Weight'
        )
ax4.set_title('>50K Correlation by Race',
              fontsize=14
              )
plt.xticks(rotation=45)
y4 = qualcol_corrgroupT['race']
x4 = range(0, len(y4), 1)
plt.axis([-1, 6, -.100, .100])
plt.plot(x4, y4, 'o')
plt.show()


# In[ ]:


############ STEP 5: K-NEAREST CLASSIFIER - BASELINE #############
# Using the dataframe without any one-hot encoding applied as the
# baseline for our model. Our model should not be below this test score.


# In[17]:


###################### STEP 5: K-NEAREST CLASSIFIER #####################

# First, we must find the best K index that gives us the highest
# accuracy. We previously created a class called NearestK that contains
# the method to search for the best K value to use.
featDF = featDF.drop(['salary_label'],
                     axis=1
                     )
trnk_data, tstk_data, trnk_lbl, tstk_lbl = train_test_split(featDF,
                                                            labelDF,
                                                            test_size=0.2,
                                                            random_state=100
                                                            )
knn_pairs = NearestK()
knn_pairs = knn_pairs.bestK(trnk_data,
                            tstk_data,
                            trnk_lbl,
                            tstk_lbl
                            )

knn_k = list(knn_pairs.keys())
knn_accuracy = list(knn_pairs.values())
knn_percent = [round(i*100, 2) for i in knn_accuracy]
max_accuracy = max(knn_percent)
best_k = knn_k[knn_accuracy.index(max(knn_accuracy))]
print('Best K is '
      + str(best_k)
      + ' at accuracy of '
      + str(max_accuracy)
      + '%')

# Let's graph the data and visualize how these accuracy rates look like,
# and highlight the best K fit with a red dot.
plt.plot(knn_k, knn_percent)
plt.scatter(best_k,
            max_accuracy,
            c='red',
            cmap='jet',
            s=20,
            label=('Best K = ' + str(best_k)))
plt.xlabel('K Number')
plt.ylabel('Validation Accuracy (%)')
plt.title('Above or Below 50K line?: best K values to predict salary')
plt.legend(loc=4)
plt.show()


# In[24]:


# Build classifier with the best K previously calculated,
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(trnk_data, np.ravel(trnk_lbl))

# Score trained data results.
trainknn_score = knn_classifier.score(trnk_data, trnk_lbl)
print("Train score for KNN algorithm: {}".format(round(trainknn_score*100, 2)))

# Score test data results.
testknn_score = knn_classifier.score(tstk_data, tstk_lbl)
print("Test score for KNN algorithm: {}".format(round(testknn_score*100, 2)))

# Now that we've trained the labeled data on this classifier, let's study
# the precision, recall and F1 scores of this classifier to determine'
# how reliable it is.
knn_scores = []
predictknn_labels = knn_classifier.predict(tstk_data)
precisionknn = precision_score(tstk_lbl, predictknn_labels, average='binary')
recallknn = recall_score(tstk_lbl, predictknn_labels, average='binary')
f1knn = f1_score(tstk_lbl, predictknn_labels, average='binary')
knn_scores.append(precisionknn)
knn_scores.append(recallknn)
knn_scores.append(f1knn)
scorenames = ['precision', 'recall', 'F1']
knn_score_names = list(zip(scorenames, knn_scores))
print("Scores for KNN are: " + str(knn_score_names))


# In[21]:


################### STEP 6: DECISION FOREST CLASSIFIER ###################

# Now, let's train this labelled on a Decision Forest classifier. For
# this, we need to split up the data between training and testing sets
# differently by changing the random_state amount. When training on the
# decision forest classifier, we want to adjust the amount of trees in
# the forest as the square root of the length of features. In this case,
# the square root of 46 columns is 6.7, so I'll round up to 7.

trndfdata, tstdfdata, trndflbl, tstdflbl = train_test_split(featDF,
                                                            labelDF,
                                                            random_state=7)
forest_classifier = RandomForestClassifier(random_state=7)
forest_classifier.fit(trndfdata, trndflbl)

# Score based on trained data.
train_df_score = forest_classifier.score(trndfdata, trndflbl)
print('Train Score:' + str(round(train_df_score*100, 2)))

# Score based on test data.
test_df_score = forest_classifier.score(tstdfdata, tstdflbl)
print('Test Score:' + str(round(test_df_score*100, 2)))


# In[20]:


################# STEP 7: LOGISTIC REGRESSION CLASSIFIER #################

# Let's now train the data on logistic regression. Split up the dataset
# between trained and tested data by making the test_size=0.2.
trnlogdata, tstlogdata, trnloglbl, tstloglbl = train_test_split(featDF,
                                                                labelDF,
                                                                test_size=0.2)

# Transform the data using the StandardScaler() class. The Logistic Regression
# classifier needs the data to be in a standardized shape.
scaler = StandardScaler()
trnlogdata = scaler.fit_transform(trnlogdata)
tstlogdata = scaler.transform(tstlogdata)

# Create and train the model on the LogisticRegression class.
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(trnlogdata, trnloglbl)

# Score the model on the train data.
train_logreg_model = log_reg_classifier.score(trnlogdata, trnloglbl)
print('Train score is ' + str(round(train_logreg_model*100, 2)))

# Score the model on the test data.
test_logreg_model = log_reg_classifier.score(tstlogdata, tstloglbl)
print('Test score is ' + str(round(test_logreg_model*100, 2)))


# In[25]:


# Must pair this with the column names from featuresDF
important_coefficients = forest_classifier.feature_importances_
feature_columns = list(featDF.head(0))
importantcoef_columns = dict(zip(feature_columns, important_coefficients))
importantcoef_series = pd.Series(importantcoef_columns)
importantcoef_series = importantcoef_series.sort_values(ascending=False)
importantcoef_columns = dict(importantcoef_series)
print(importantcoef_series)
print(len(importantcoef_series))
corr_filter = FilterGroups()
df_corrs = corr_filter.make_corr_groups(importantcoef_columns)
# print(df_corrs)


# In[ ]:




