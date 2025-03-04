#-------------------------------------------------------------------------
# AUTHOR: Johnny Liang
# FILENAME: knn.py
# SPECIFICATION: Complete the Python program (knn.py) to read the file email_classification.csv
#    and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification
#    task. The dataset consists of email samples, where each sample includes the counts of 20
#    specific words (e.g., “agenda” or “prize”) representing their frequency of occurrence
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
test_results = []

#Reading the data in a csv file
with open('Assignment 2/assignment_2_data/email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# converts everything in the db to a numeric float value 
for i in range(len(db)):
    if db[i][20] == "spam":             # converts the ham/spam class to numeric 
        db[i][20] = 1
    else:
        db[i][20] = 0
    db[i] = [float(x) for x in db[i]]   # convert all data in db into floats 


#Loop your data to allow each instance to be your test set
for i in range(len(db)):
    
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    everything_else = db[:i] + db[i+1:]         # get all data except for the test sample for the training data 
    
    X = [row[:20] for row in everything_else]       # X train (excluding the test sample), transformed earlier 
    
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    
    Y = [row[20] for row in everything_else]        # training classes (excluding the test sample), transformed earlier 

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = db[i]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample[:20]])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    result = True if class_predicted == testSample[20] else False   # compare prediction
    
    test_results.append(result)
    print("Test case", i+1, ": ", result)   # prints out the results of this test case 


#Print the error rate
#--> add your Python code here
err_count = 0
for res in test_results:    # counts the total # of errors recorded 
    if res == False:
        err_count += 1
print()
print("ERROR RATE = ", err_count / len(test_results))   # print errir rate
