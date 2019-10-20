import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import warnings

# ignoring the future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv('Data.csv')


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

'''
    - To take care of the missing data we use the imputer object
    - strategy='mean' is used because mean values will be stored for missing cells
    
    - fitting the imputer object to column 1 and 2 values as it contains missing data
    
    - labelEncoder is used for converting categorical data i.e values in form of words, to numerical value
        - Here "Country" column is in the form of words
        
    -   we need to index column number 0, because the countries does not
        have any relation with each other, but the labelEncoder
        converts them into numbers and create a problem such as
        0 < 1 < 2. In this way our model will mis understand data

    -   OneHotEncoder does is, it takes a column which has a categorical
        data, which has been label encoded, and then splits the column into
        multiple columns. The numbers are replaced by 1s and 0s, depending 
        on which column has what value
'''
imputer = SimpleImputer(strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
columntransform = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
X = columntransform.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)

'''
    - Splitting the data set into the Training set and Test set
      through training set we will predict test data
    - test_size = 0.2 means 20 percentage of total data is used for testing, and res
        80 percentage is used for training
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

