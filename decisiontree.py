"""
Decision Trees
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv('drug200.csv',delimiter=',')
print(my_data[0:5])


"""
Pre-processing
Using my_data as the Drug.csv data read by pandas, declare the following variables:

X as the Feature Matrix (data of my_data)
y as the response vector (target)
Remove the column containing the target name since it doesn't contain numeric values.
"""

X = my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
# print(X[0:5])


"""
As you may figure out, some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn Decision Trees 
do not handle categorical variables. But still we can convert these features to numerical values. pandas.get_dummies() 
Convert categorical variable into dummy/indicator variables.
"""


from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])


# print(X[0:5])

y = my_data["Drug"]
# print(y[0:11] )	

"""
Setting up the Decision Tree
We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.
"""

from sklearn.model_selection import train_test_split

"""
Now train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, 
and the random_state ensures that we obtain the same splits.
"""
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size=0.3,random_state = 2)


"""
Modeling
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
"""



drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# print(drugTree)


"""
fit the data with the training feature matrix X_trainset and training response vector y_trainset
"""

drugTree.fit(X_trainset,y_trainset)

"""
prediction
"""
predTree = drugTree.predict(X_testset)
# print (predTree [0:10])
# print (y_testset [0:10])

"""
Evaluation
Next, let's import metrics from sklearn and check the accuracy of our model.
"""


from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


print(drugTree.predict([[35,1,1,1,9.17]]))
















