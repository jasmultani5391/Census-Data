# Census-Data: Predicting Salary Range

## Motivation
The goal of this supervised machine learning algorithm is to decide which socioeconomic factors will reliably predict whether an individual makes less than or more than $50,000 yearly. The dataset being used is found on UCI’s machine learning repository and was collected in 1994.

For context, $50,000 in 1994 is the rough equivalent to $87,000 in 2020. Calculations made on this 
<a href="https://www.in2013dollars.com/us/inflation/1994?amount=50000">website</a>.

## What to expect when opening the file
I have compiled each phase of building the algorithm, from preprocessing to analysis, into this one file. Annotation on steps are as following, and can be found embedded within the code:

 <ul style="list-style-type:disc">
 <li><b>Step 1:</b> Data introduction</li>
         <li><b>Step 2:</b> Data clean-up</li>
         <li><b>Step 3:</b> Creating dataframes by varying features</li>
         <li><b>Step 4:</b> Exploring data visually</li>
         <li><b>Step 5:</b> Training labeled data on K Nearest Classifier</li>
         <li><b>Step 6:</b> Training labeled data on Decision Forest Classifier</li>
         <li><b>Step 7:</b> Training labeled data on Logistic Regression Classifier</li>
      </ul>
      
## Code Style
<a href="https://docs.python.org/3.7/contents.html">Python version 3.7.4.</a>

## Package manager
<a href="https://repo.anaconda.com/">Anaconda</a> with the following <a href="https://www.youtube.com/watch?v=5mDYijMfSzs&t=255s">download tutorial.</a>

## IDE
<a href="https://jupyter.org/about">Jupyter Notebook</a>, which is downloaded when Anaconda is downloaded.

## Dataset downloaded from
<a href="https://archive.ics.uci.edu/ml/datasets/census+income">UCI Machine Learning Repository</a>

## Feature engineering from dataset

<ul style="list-style-type:disc">
         <li>Age</li>
         <li>Workclass</li>
         <li>Education</li>
         <li>Education Number (numerical ranking of education)
         </li>
         <li>Marital Status</li>
         <li>Occupation</li>
         <li>Relationship</li>
         <li>Race</li>
         <li>Sex</li>
         <li>Capital Gain</li>
         <li>Capital Loss</li>
         <li>Country of Origin</li>
         <li>Salary Label</li>
      </ul>

## Algorithms
 <ul style="list-style-type:disc">
         <li><a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">K Nearest Classifier</a></li>
         <li><a href="https://en.wikipedia.org/wiki/Random_forest">Decision Forest Classifier</a></li>
         <li><a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic Regression Classifier</a></li>
      </ul>
      
## License
MIT
      
