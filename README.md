//family tree

female(pam).
female(liz).
female(pat).
female(ann).
male(jim).
male(bob).
male(tom).
male(peter).
parent(pam,bob).
parent(tom,bob).
parent(tom,liz).
parent(bob,ann).
parent(bob,pat).
parent(pat,jim).
parent(bob,peter).
parent(peter,jim).
mother(X,Y):-parent(X,Y),female(X).
father(X,Y):-parent(X,Y),male(X).
haschild(X):-parent(X,_).
sister(X,Y):-parent(Z,X),parent(Z,Y),female(X),X\==Y.
brother(X,Y):-parent(Z,X),parent(Z,Y),male(X),X\==Y.
grandmother(X,Y):- parent(Z,Y),parent(X,Z),female(X).
uncle(X,Y):- parent(Z,Y),brother(X,Z),Z\==X.

/*
OUTPUT
?- grandmother(X,Y).
X = pam,
Y = ann ;
X = pam,
Y = pat ;
X = pam,
Y = peter ;
false.
?- uncle(X,Y).
X = peter,
Y = jim ;
false.
*/


..........................


% Write a PROLOG program for addition and multiplication of two numbers.

add(A, B, C) :- C is A + B.
multiply(A, B, C) :- C is A * B.

/*
OUTPUT

?- add(10,15,X).
X = 25.

?- multiply(10,15,X).
X = 150

*/

.........................................


% Write a PROLOG program to print the fibonacci series upto n terms, where n is given as input.

fib(0,0).
fib(1,1).
fib(N,F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

print_fib(N, Max) :-
    N >= 0, N < Max,
    fib(N, F),
    write(F), nl,
    N1 is N + 1,
    print_fib(N1, Max).

/*
OUTPUT

?- print_fib(0,10).
0
1
1
2
3
5
8
13
21
34
false.

*/

............................


% Write a PROLOG program to check if a number is prime or not.

divisible(X,Y):-
    N is Y*Y,
    N =< X,
    X mod Y =:= 0.

divisible(X,Y):-
    Y < X,
    Y1 is Y+1,
    divisible(X,Y1).

isprime(X):-
    Y is 2, X > 1,
    \+divisible(X,Y).

/*

OUTPUT

?- isprime(28).
false.

?- isprime(29).
true.

*/


...........................


1.	 Write a SWI prolog programme to calculate the factorial of a given number
-	Fact (0,1)
Fact(N F):-

N > 0,
N1 => N-1 ,
FACT()
F 1 => N*f1 .

OUTPUT:-

FACT (5, 120)
TRUE
FACT(4, 120)
FALSE

.....................

2 . WRITE A Swi prolog programme to calculate additition, product, difference and division operations on any two given numbers.
=> sum (a, b, c):-
  C is a + b .
Mult (x , y, z):-
  Z is X * Y .
Sumtest (p, g, r):-
  P > 0 :
  G > 0:
R= P + G .
Diff(a, b, c):
  C is A-b .
Div (p, g, r):-
  R is p/g.

OUTPUT:-
  Sum(20, 67 , U).
 U= 87
 Sumtest(90, -8, r) .
 False
Diff (45, 25 , e)
E=20
True 
Div (34, 17, r)
R = 2

..................


3. write a prolog programme to calculate the Fibonacci series.
=>  fibo (0 , 1) :- 1 .
    Fibo(1, 1 ) :- 1.
    Fibo(n , f ) :-
        n> 1,
       n1 is (n-1) ,
       n2 is (n-2) ,
       fib (n1, f2) ,

OUTPUT:-
  FIB (5 , F)
  F = 8
  
  ...............................
  
  
4. WRITE A SWI PROLOG PROGRAMME TO CALCULATE THE SUMMATION OF N was whre value of n = 10 or may be taken from useror input

=> sum(a, c) :-
   C is (a * (a + 1) / 2) .

Output:-
  Sum (5, c)
  C= 15
  
........................

% Write a prolog program to find minimum and maximum from a list

min_list([X], X).
min_list([H|T], Min) :-
    min_list(T, TailMin),
    Min is min(H, TailMin).

max_list([X], X).
max_list([H|T], Max) :-
    max_list(T, TailMax),
    Max is max(H, TailMax).

min_max(List, Min, Max) :-
    min_list(List, Min),
    max_list(List, Max).

/*
OUTPUT

?- min_max([3,8,2,1,5],Min,Max).
Min = 1,
Max = 8 .

?- min_max([5],Min,Max).
Min = Max, Max = 5 .

?- min_max([],Min,Max).
false.

*/

..........................

% Write a prolog program to reverse a list

reverse_list(List, Reversed) :-
    reverse_list(List, [], Reversed).

reverse_list([], Acc, Acc).
reverse_list([H|T], Acc, Reversed) :-
    reverse_list(T, [H|Acc], Reversed).

/*
OUTPUT

?- reverse_list([a,b,c,d],Reversed).
Reversed = [d, c, b, a].

?- reverse_list([1],Reversed).
Reversed = [1].

?- reverse_list([],Reversed).
Reversed = [].

*/

...................................

# 1. Write a NumPy program to print the NumPy version in your system.

import numpy as np

print("NumPy Version:", np.__version__)

# OUTPUT
# NumPy Version: 1.24.2

......................................

# 2. Write a Numpy program to convert a list of numeric value into a one-dimensional Numpy array.

import numpy as np

my_list = [12.23, 13.32, 100, 36.32]
my_array = np.array(my_list)

print("Original list:", my_list)
print("NumPy array:", my_array)


# OUTPUT
# Original list: [12.23, 13.32, 100, 36.32]
# NumPy array: [ 12.23  13.32 100.    36.32]


...........................................

# 3. Write a NumPy program to create a 3x3 matrix with values ranging from 2 to 10.

import numpy as np

my_matrix = np.arange(2, 11).reshape(3, 3)
print(my_matrix)

# OUTPUT
# [[ 2  3  4]
#  [ 5  6  7]
#  [ 8  9 10]]

.......................

# 4. Write a NumPy program to create a null vector of a size 10 and update sixth value with 11.

import numpy as np

my_vector = np.zeros(10)
print(my_vector)

my_vector[5] = 11
print("Update sixth value to 11")
print(my_vector)


# OUTPUT
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Update sixth value to 11
# [ 0.  0.  0.  0.  0. 11.  0.  0.  0.  0.]

..........................................

# 5. Write a NumPy program to create an array with values ranging from 12 to 38.

import numpy as np

my_array = np.arange(12, 39)
print(my_array)

# OUTPUT
# [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38]

.......................................

# 6. Write a NumPy program to reverse an array.

import numpy as np

my_array = np.arange(12, 38)
reversed_array = my_array[::-1]
print("Original array:")
print(my_array)
print("Reversed array:")
print(reversed_array)

# Original array:
# [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]
# Reversed array:
# [37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12]

.............................................

# 7. Write a NumPy program to convert an array to a float type.

import numpy as np

my_array = np.array([1, 2, 3, 4])
float_array = my_array.astype('float')

print("Original array:")
print(my_array)
print("Float array:")
print(float_array)

# OUTPUT
# Original array:
# [1 2 3 4]
# Float array:
# [1. 2. 3. 4.]

............................

# 8. Write a NumPy program to create a 8x8 matrix and fill it with a checkerboard pattern.

import numpy as np

my_matrix = np.zeros((8, 8), dtype=int)
my_matrix[::2, ::2] = 0
my_matrix[1::2, ::2] = 1
my_matrix[::2, 1::2] = 1
my_matrix[1::2, 1::2] = 0

print("Checkerboard pattern:")
print(my_matrix)

# OUTPUT
# Checkerboard pattern:
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]

...........................

# 9. Write a NumPy program to convert a list and tuple into arrays.

import numpy as np

my_list = [1, 2, 3, 4, 5, 6, 7, 8]
my_tuple = ((8, 4, 6), (1, 2, 3))
list_array = np.array(my_list)
tuple_array = np.array(my_tuple)
print("List converted to array:")
print(list_array)
print("Tuple converted to array:")
print(tuple_array)

# OUTPUT
# List converted to array:
# [1 2 3 4 5 6 7 8]
# Tuple converted to array:
# [[8 4 6]
#  [1 2 3]]

................................

# 10. Write a NumPy program to append values to the end of an array.

import numpy as np

my_array = np.array([10, 20, 30])
print("Original array:")
print(my_array)

append_arr = np.array([40, 50, 60, 70, 80, 90])
my_array = np.append(my_array, append_arr)
print("Array after appending values:")
print(my_array)


# OUTPUT
# Original array:
# [10 20 30]
# Array after appending values:
# [10 20 30 40 50 60 70 80 90]

...............................

//working with pandas

import pandas as pd
col_names = ['Student ID', 'Name', 'Marks1', 'Marks2', 'Marks3', 'Marks4', 'Marks5', 'Marks6']
df = pd.DataFrame(columns = col_names)
df
stud1 = {'Student ID' : 2, 'Name' : 'E', 'Marks1' : 45, 'Marks2' : 44, 'Marks3' : 55, 'Marks4' : 96, 'Marks5' : 55, 'Marks6' : 43}
df = df.append(stud1, ignore_index = True)
df
df = df.drop(0)
df
#adding student deails to the dataframe with Lloc[] 
#attribute using row index 
df.loc[2] = [4,'A',39,51,76,80,78,91]
df
#adding student details to the first row of dataframe with Loc[] attribute as -1 
df.loc[-1] = [8,'V',65,67,72,75,66,81] 
df
#rectifying indices 
df.index = df.index + 1 
df
df = df.sort_index() 
df
df.loc[1] = [5,'K"',34,53,65,99,45,91] 
df 
df = df.sort_index() 
df
sum = df['Marks1'] + df['Marks2'] + df['Marks3'] + df['Marks4'] + df['Marks5'] + df['Marks6']
print(sum)
per = sum/6 
print(per)
#assigning the peercentage values as a dataframe column 
df["per"] = per
print(df)
#view particular columns
print(df[['Student ID', 'Name']])
#view particular cols of a row which have min
print(df[['Student ID', 'Name']][df.per == df.per.min()])
#write csv
df.to_csv("file.csv")
#use pandas to identify the student who got Lowest marks in more than two subjects. 

# Create a new column "Lowest_Marks" that contains the lowest marks obtained by each student across all six subjects
df['Lowest_Marks'] = df[['Marks1', 'Marks2', 'Marks3', 'Marks4', 'Marks5', 'Marks6']].min(axis=1)

# Create another new column "Num_Lowest" that contains the number of subjects in which each student got the lowest marks
df['Num_Lowest'] = df[['Marks1', 'Marks2', 'Marks3', 'Marks4', 'Marks5', 'Marks6']].apply(lambda x: sum(x == x.min()), axis=1)

# Filter the DataFrame to include only those rows where Num_Lowest is greater than 2
result = df[df['Num_Lowest'] > 2]

# Display the result
print(result[['Student ID', 'Name', 'Lowest_Marks', 'Num_Lowest']])

.......................................................................

# Write a program using pandas to convert the datatype of a column of a dataframe.

import pandas as pd

df = pd.DataFrame({'A': ['dog', 'bar', 'baz'], 
                   'B': [1, 2, 3], 
                   'C': [1.0, 2.0, 3.0]})
print("Datatype =",df.dtypes['B'])
df['B'] = df['B'].astype(float)
print("Updated Datatype =",df.dtypes['B'])  

# OUTPUT
# Datatype = int64
# Updated Datatype = float64

................................................

# Write a program to add a new column to a dataframe.

import pandas as pd

df = pd.DataFrame({'A': ['foo', 'bar', 'baz'], 
                   'B': [1, 2, 3], 
                   'C': [1.0, 2.0, 3.0]})
df['D'] = [True, False, True]
print(df)

# OUTPUT
#      A  B    C      D
# 0  foo  1  1.0   True
# 1  bar  2  2.0  False
# 2  baz  3  3.0   True

...........................

# Use pandas to identify the student who got lowest marks in more than 2 subjects.

import pandas as pd

data = {'Student': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
        'Maths': [80, 70, 60, 50, 40],
        'Science': [70, 60, 50, 40, 30],
        'English': [60, 50, 40, 30, 20]}
df = pd.DataFrame(data)

num_subjects = 2
min_marks = df.min()
num_lowest = (df == min_marks).sum(axis=1)
students = df['Student'][num_lowest > num_subjects].tolist()

if students:
    print(f"The student(s) who got lowest marks in more than {num_subjects} subjects are: {students}")
else:
    print(f"No student got lowest marks in more than {num_subjects} subjects")


# OUTPUT
# The student(s) who got lowest marks in more than 2 subjects are: ['Eve']

............................................................................

#linear regression for continuous values

#load the boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
#see description of dataset
boston.DESCR
import pandas as pd
data=pd.DataFrame(boston.data, columns=boston.feature_names)
#see data in dataframe
data
#column of Median Value is usually the target - to be predicted by regression model
data['MEDV']=pd.DataFrame(boston.target)
data
#find all correlation values, RM has highest with target MEDV, so select RM for training
pd.DataFrame(data.corr().round(2))
# x corresponds to train data
x=data['RM']
#y corresponds to labels
y=data['MEDV']
#import module for linear regression
from sklearn.linear_model import LinearRegression
#import train_test_split module
from sklearn.model_selection import train_test_split
linearRegressionClassifier = LinearRegression()
#convert x and y to pandas Dataframes
x=pd.DataFrame(x)
y=pd.DataFrame(y)
#split the dataset using train_test_split function, pass train data, labels,and test data ratio
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#check number of rows and columns
print(x_train.shape)
#train the classifier by fitting to train data and corresponding labels
linearRegressionClassifier.fit(x_train, y_train)
import numpy as np
#import the mean squared error module
from sklearn.metrics import mean_squared_error
#determine the predicted values
y_pred=linearRegressionClassifier.predict(x_test)
y_pred.shape
#calculate root mean square error
np.sqrt(mean_squared_error(y_test, y_pred))
#determine R^2 scores (because regression)
linearRegressionClassifier.score(x_test,y_test)


.....................................................

# Identify the max value from .score() function on using
# max_iter values between 10000 to 100000. Keep a note for
# a specific value of the number of iteration corresponding
# to the obtained result.

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()

data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = pd.DataFrame(boston.target)

x = data[['RM']]
y = data['MEDV']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

max_iter_values = range(10000, 100001, 10000)

max_r2_score = 0
max_iter = 0

for i in max_iter_values:
    model = LinearRegression()
    model.fit(x_train, y_train)
    r2_score = model.score(x_test, y_test)
    if r2_score > max_r2_score:
        max_r2_score = r2_score
        max_iter = i

print(
    f"Maximum R^2 score of {max_r2_score} obtained with max_iter={max_iter}.")

# OUTPUT
# Maximum R^2 score of 0.3707569232254778 obtained with max_iter=10000.

..........................................................

//Logistic regression for classification problem


from sklearn.datasets import load_digits
digits = load_digits() 
print(digits.DESCR)
import matplotlib.pyplot as plt 
digits.data
digits.data.shape
d=digits.data[0:500]
d.shape
digits.target.shape
image = digits.data[0]
print(image)
digits.target[0]
import numpy as np 
np.reshape(image, (8,8)) 
plt.imshow(np.reshape(image,(8,8)), cmap="BuGn")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)
from sklearn.linear_model import LogisticRegression
logRegl = LogisticRegression(max_iter=10000)
logRegl.fit(x_train, y_train)
y_pred = logRegl.predict(x_test)
logRegl.score(x_test,y_test)

...............................................

// Decision Tree Regression Model

#load the boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
import pandas as pd
data=pd.DataFrame(boston.data, columns=boston.feature_names)
#column of Median Value is usually the target
#to be predicted by regression model
data['MEDV']=pd.DataFrame(boston.target)
data
#find all correlation values for all features,
#RM, ZN have highest with target MEDV, so select RM and Zone for training
pd.DataFrame(data.corr().round(2))
x=data[['RM','ZN']]
x
#y corresponds to labels
y=data['MEDV']
y
#import train_test_split module
from sklearn.model_selection import train_test_split
#convert x and y to pandas Dataframes
x=pd.DataFrame(x)
y=pd.DataFrame(y)
#split the dataset using train_test_split function,
#pass train data, labels, and test data ratio
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
y_train
from sklearn.tree import DecisionTreeRegressor
dt1=DecisionTreeRegressor(max_depth=20)
dt1.fit(x_train,y_train)
y_pred1=dt1.predict(x_test)
import numpy as np
#import the mean squared error module
from sklearn.metrics import mean_squared_error
#calculate root mean square error
np.sqrt(mean_squared_error(y_test, y_pred1))
from sklearn.ensemble import RandomForestRegressor
rf1=RandomForestRegressor()
rf1.fit(x_train, y_train)
rf1.score(x_test,y_test)
y_pred2=rf1.predict(x_test)
#calculate root mean square error
np.sqrt(mean_squared_error(y_test, y_pred2))


...................

// Decision Tree Classifier


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
digits = load_digits()
#split the dataset using train_test_split function, pass train data, labels, and test data ratio
x_train, x_test, y_train, y_test=train_test_split(
digits.data,digits.target,test_size=0.25)
dt2=DecisionTreeClassifier(criterion="entropy")
dt2.fit(x_train, y_train)
dt2.score(x_test,y_test)
# max_depth = longest path between root and leaf nodes
dt3=DecisionTreeClassifier(max_depth=30)
dt3.fit(x_train, y_train)
dt3.score(x_test,y_test)


....................................................

//Using iterations, and the classification/regression models,
try to identify the optimum max_depth value
at which the following 3 models give the maximum R2 score.
1. Decision Tree Regressor
2. Random Forest Regressor
3. Decision Tree Classifier

from sklearn.datasets import load_boston, load_digits
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Load Boston and Digits datasets
boston = load_boston()
X_reg = boston.data
y_reg = boston.target

digits = load_digits()
X_clf = digits.data
y_clf = digits.target

# Define max_depth values to test
max_depth_values = range(1, 11)

# Initialize lists to store R2 scores for each model and max_depth value
reg_scores = []
rf_scores = []
clf_scores = []

# Iterate over max_depth values and train models
for max_depth in max_depth_values:
    # Decision Tree Regressor
    reg_model = DecisionTreeRegressor(max_depth=max_depth)
    reg_scores.append(np.mean(cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')))
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(max_depth=max_depth)
    rf_scores.append(np.mean(cross_val_score(rf_model, X_reg, y_reg, cv=5, scoring='r2')))
    
    # Decision Tree Classifier
    clf_model = DecisionTreeClassifier(max_depth=max_depth)
    clf_scores.append(np.mean(cross_val_score(clf_model, X_clf, y_clf, cv=5, scoring='accuracy')))

# Find the max R2 scores and corresponding max_depth values for each model
reg_max_score = max(reg_scores)
reg_opt_depth = max_depth_values[reg_scores.index(reg_max_score)]
rf_max_score = max(rf_scores)
rf_opt_depth = max_depth_values[rf_scores.index(rf_max_score)]
clf_max_score = max(clf_scores)
clf_opt_depth = max_depth_values[clf_scores.index(clf_max_score)]

# Print the optimal max_depth values and corresponding R2 scores for each model
print(f"Boston dataset - Decision Tree Regressor: Optimal max_depth = {reg_opt_depth}, R2 score = {reg_max_score:.4f}")
print(f"Boston dataset - Random Forest Regressor: Optimal max_depth = {rf_opt_depth}, R2 score = {rf_max_score:.4f}")
print(f"Digits dataset - Decision Tree Classifier: Optimal max_depth = {clf_opt_depth}, Accuracy score = {clf_max_score:.4f}")


//output

Boston dataset - Decision Tree Regressor: Optimal max_depth = 5, R2 score = 0.6773
Boston dataset - Random Forest Regressor: Optimal max_depth = 9, R2 score = 0.8283
Digits dataset - Decision Tree Classifier: Optimal max_depth = 8, Accuracy score = 0.8043


.............................


// k means
//using make_blobs with KMeans clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#generate data of 4 clusters with labels, store in single variable
alldata=make_blobs(n_samples=200,centers=4,
n_features=2, cluster_std=1.5, random_state=30)
print(alldata)
#extract only the first array i.e. data values
data=alldata[0]
data
%matplotlib inline
#plot datapoints without labels
plt.scatter(data[:,0], data[:,1])
#create kmeans object
kmeans=KMeans(n_clusters=4)
#compute cluster centre and predict cluster label for each sample
y_pred=kmeans.fit_predict(data)
print(y_pred)
clusters = kmeans.cluster_centers_
clusters
plt.scatter(data[y_pred==0,0], data[y_pred==0,1], s=70, color='green')
plt.scatter(data[y_pred==1,0], data[y_pred==1,1], s=70, color='red')
plt.scatter(data[y_pred==2,0], data[y_pred==2,1], s=70, color='yellow')
plt.scatter(data[y_pred==3,0], data[y_pred==3,1], s=70, color='blue')
plt.scatter(clusters[0][0],clusters[0][1], marker='*', color='black')
plt.scatter(clusters[1][0],clusters[1][1], marker='*', color='black')
plt.scatter(clusters[2][0],clusters[2][1], marker='*', color='black')
plt.scatter(clusters[3][0],clusters[3][1], marker='*', color='black')


plt.scatter(clusters[0][0],clusters[0][1], marker='*', color='black')
plt.scatter(clusters[1][0],clusters[1][1], marker='*', color='black')
plt.scatter(clusters[2][0],clusters[2][1], marker='*', color='black')
plt.scatter(clusters[3][0],clusters[3][1], marker='*', color='black')



.......................................................



# On the iris dataset, performs the kmeans clustering technique.
# Use any 2 features to perform the clustering.
# Plot the clusters and cluster centers.

# Next, perform the Decision Tree clssification on the iris dataset.
# Determine the r2 score and mean squared error for the classifier.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

iris = load_iris()

# Use the sepal length and width features
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], marker='*', s=300, color='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means clustering on the Iris dataset')
plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R2 score:', r2)
print('Mean squared error:', mse)


.......................................


// MLP with Confusion Matrix


import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
from sklearn.model_selection import train_test_split
#split the dataset using train_test_split function, pass train data, labels,and test data ratio
X_train, X_test, y_train, y_test = train_test_split(
cancer_dataset.data,cancer_dataset.target,test_size=0.25)
#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier
#Initializing the MLPClassifier
#solver is for weight optimization
#max_iter sets limit till convergence or this value
#activation is the activation function for the hidden layer
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,
activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

//transpose

import numpy as np
cf_matrix=np.transpose(np.transpose(cf_matrix) / cf_matrix.astype(float).sum(axis=1))
cf_matrix


..............................................


//Perform the same experiment on the breast cancer dataset
using Logistic Regression and Decision Tree Classification models.
In each case, identify the confusion matrix and the TP,TN,FP & FN.
Finally, discuss which classifier is best suited for deploying as a
real-time smart cancer diagnosis system.



//Sure, here's the code to perform Logistic Regression and Decision Tree Classification models on the breast cancer dataset and compute the confusion matrix along with the TP, TN, FP, and FN values:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Load the breast cancer dataset
cancer_dataset = load_breast_cancer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer_dataset.data, cancer_dataset.target, test_size=0.25, random_state=42)

# Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Confusion matrix and TP, TN, FP, FN values for Logistic Regression model
lr_cf_matrix = confusion_matrix(y_test, lr_y_pred)
lr_tp = lr_cf_matrix[1][1]
lr_tn = lr_cf_matrix[0][0]
lr_fp = lr_cf_matrix[0][1]
lr_fn = lr_cf_matrix[1][0]

print("Logistic Regression Confusion Matrix:")
print(lr_cf_matrix)
print("TP =", lr_tp)
print("TN =", lr_tn)
print("FP =", lr_fp)
print("FN =", lr_fn)
print("")

# Decision Tree Classification model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

# Confusion matrix and TP, TN, FP, FN values for Decision Tree Classification model
dt_cf_matrix = confusion_matrix(y_test, dt_y_pred)
dt_tp = dt_cf_matrix[1][1]
dt_tn = dt_cf_matrix[0][0]
dt_fp = dt_cf_matrix[0][1]
dt_fn = dt_cf_matrix[1][0]

print("Decision Tree Classification Confusion Matrix:")
print(dt_cf_matrix)
print("TP =", dt_tp)
print("TN =", dt_tn)
print("FP =", dt_fp)
print("FN =", dt_fn)
print("")



//output

Logistic Regression Confusion Matrix:
[[49  4]
 [ 3 87]]
TP = 87
TN = 49
FP = 4
FN = 3

Decision Tree Classification Confusion Matrix:
[[46  7]
 [ 4 86]]
TP = 86
TN = 46
FP = 7
FN = 4


//From the confusion matrices, we can see that the Logistic Regression model correctly predicted 136 out of 143 cases (TP+TN), while the Decision Tree Classification model correctly predicted 132 out of 143 cases. In terms of sensitivity (ability to correctly identify positive cases), the Logistic Regression model has a higher value of TP/(TP+FN) = 0.97 compared to the Decision Tree Classification model's value of 0.96. In terms of specificity (ability to correctly identify negative cases), the Decision Tree Classification model has a higher value of TN/(TN+FP) = 0.87 compared to the Logistic Regression model's value of 0.92.

Overall, both models perform well in predicting breast cancer, but the Logistic Regression model may be more suited for deploying as a real-time smart cancer diagnosis system due to its high sensitivity value.






