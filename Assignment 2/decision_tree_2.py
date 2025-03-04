#-------------------------------------------------------------------------
# AUTHOR: Johnny Liang
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

path = 'Assignment 2/assignment_2_data/'
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    dbTraining_transformed = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(path + ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    
    replace = {"Young" : 1, "Prepresbyopic" : 2, "Presbyopic" : 3, 
           "Myope" : 1, "Hypermetrope" : 2 , 
           "Reduced" : 1, "Normal" : 2, 
           "Yes" : 1, "No" : 2}

    # Transforms the original categorical training features to numbers and add it to dbTraining_transformed
    for row in range(len(dbTraining)):
        dbTraining_transformed.append([replace[feature] for feature in dbTraining[row]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    X = [x[:4] for x in dbTraining_transformed]
    Y = [x[4] for x in dbTraining_transformed]
    
    class_predicted = [] # records every prediction in the 10 tests
    class_actual = [] # records every actual value in the 10 tests

    # Loop your training and test tasks 10 times here
    for i in range (10):
       
       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       dbTest_transformed = []
       X_test = []
       Y_test = []
       
       with open(path + 'contact_lens_test.csv', 'r') as csvfile_test:
         reader_test = csv.reader(csvfile_test)
         for i, row in enumerate(reader_test):
             if i > 0: #skipping the header
                dbTest.append (row)
        
       # transforms dbTest features --> numeric 
       for row in range(len(dbTest)):       
         dbTest[row] = [replace[feature] for feature in dbTest[row]]

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           class_predicted.append(clf.predict([data[:4]])[0])

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           class_actual.append(data[4])

    # Find the average of this model during the 10 runs (training and test set)
    # --> add your Python code here
    class_actual_avg = sum(class_actual) / len(class_actual)
    class_predicted_avg = sum(class_predicted) / len(class_predicted)
    Y_avg = sum(Y) / len(Y)
    
    print("Training average", Y_avg)
    print("Test average", class_actual_avg)
    print("Test prediction average", class_predicted_avg)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    compare = []
    for i in range(len(class_actual)):
       compare.append(True if class_actual[i] == class_predicted[i] else False)
    
    print("Average accuracy: ", sum(compare) / len(compare))
    print()
