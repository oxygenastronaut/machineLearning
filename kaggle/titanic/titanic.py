# 10/09/17


# for predicting surviviors on the titantic using features of the passengers
# the data is taken from two csv files
# the output of the data should be a csv with a header. 
# the output will include the passenger number, 0 or 1 to indicate survival
# the higher the accuracy the better
# input files:
# training data: train.csv
# test data: test.csv

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Import the data frame
# 2. Visualize the data frame
# 3. Clean up and transform the data
# 4. Encode the data
# 5. Split Training and Testing sets
# 6. Fine tune the algorithim
# 7. Cross validate with KFold
# 8. Upload to kaggle

# 1. Importing the data frame

# csv files can be loaded into a dataframe by calling pd.read_csv(). 
# data.sample(3) can be used to see a sample of the data

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# 2. Visualize the data
# Visualizing is critical to understand underlying patterns within the data set

'''
# using seaborn as a plotting tool
sns.barplot(x = 'Embarked', y = 'Survived', hue = 'Sex', data = data_train, palette = 'muted')
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train, palette={"male": "blue", "female": "pink"}, 
				markers=["*", "o"], linestyles=["-", "--"])
plt.show()
'''
# the plots show that females, on average, had a better survival rate and the higher the class the higher the chance as well

# 3. Transforming the data into useful features
#print(data_train.Fare.describe())

# this function converts the random ages into age ranges
def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
	group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	categories = pd.cut(df.Age, bins, labels = group_names)
	df.Age = categories
	return df

def simplify_cabins(df):
	df.Cabin = df.Cabin.fillna('N')
	df.Cabin = df.Cabin.apply(lambda x: x[0])
	return df

def simplify_fares(df):
	df.Fare = df.Fare.fillna(-0.5)
	bins = (-1, 0, 8, 15, 31, 1000)
	group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
	categories = pd.cut(df.Fare, bins, labels = group_names)
	df.Fare = categories
	return df

def format_names(df):
	df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
	df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
	return df

def drop_features(df):
	return df.drop(['Ticket', 'Name', 'Embarked'], axis = 1)

def transform_features(df):
	df = simplify_ages(df)
	df = simplify_cabins(df)
	df = simplify_fares(df)
	df = format_names(df)
	df = drop_features(df)
	return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
#print(data_train.head())

# from here it is useful to plot the features against one another to see the effects
# however just for the sake of time, i'm not doing that. it just follows the same pattern
# as the last bar plots above

# 4. Encoding the data

'''
The last part of the preprocessing phase is to normalize labels. 
The LabelEncoder in Scikit-learn will convert each unique string value into a number, 
making out data more flexible for various algorithms.
'''

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

data_train, data_test = encode_features(data_train, data_test)
#print(data_train.head())

# 5. Splitting the training data

X_all = data_train.drop(['Survived', 'PassengerId'], axis = 1)
# all features except the ones we want to predict
y_all = data_train['Survived']
# the labels

num_test = 0.20
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, test_size = num_test, random_state = 23)
# random_state refers to the seed. being an int ensures that each run will have the same train test split

# 6. Fitting and tuning the algorithim

clf = RandomForestClassifier()
# parameter combinations to try
paramters = {'n_estimators': [4, 6, 9],
			'max_features': ['log2', 'sqrt', 'auto'],
			'criterion': ['entropy', 'gini'],
			'max_depth': [2, 3, 5, 10],
			'min_samples_split': [2, 3, 5],
			'min_samples_leaf': [1, 5, 8]}

# type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, paramters, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# fit the best algorithim to the data
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis = 1))

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('titantic-predictions.csv', index = False)
print(output.head())