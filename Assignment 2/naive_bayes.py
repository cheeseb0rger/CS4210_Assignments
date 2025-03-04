#-------------------------------------------------------------------------
# AUTHOR: Johnny Liang
# FILENAME: naive_bayes.py
# SPECIFICATION: Complete the Python program (naïve_bayes.py) that will read the file
#      weather_training.csv (training set) and output the classification of each of the 10 instances from
#      the file weather_test (test set) if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db_train = []
db_test = []

with open('Assignment 2/assignment_2_data/weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db_train.append (row[1:])

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
table = {"Sunny": 1, "Overcast" : 2, "Rain" : 3,
         "Hot" : 1, "Mild" : 2, "Cool" : 3,
         "Normal" : 1, "High" : 2,
         "Strong" : 1, "Weak" : 2,
         "Yes" : 1, "No" : 2}

db_train = [[table[x] for x in instance] for instance in db_train]    # transform values of the training data using the lookup table

X = [instance[:4] for instance in db_train]

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [instance[4] for instance in db_train]

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
with open('Assignment 2/assignment_2_data/weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db_test.append (row[1:5])

#Printing the header os the solution
#--> add your Python code here
print("Day    Outlook   Temperature  Humidity  Wind    PlayTennis     Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i, test_sample in enumerate(db_test):
   test_sample_transformed = [table[x] for x in test_sample]   # transforms test sample into numeric
   output = clf.predict_proba([test_sample_transformed])[0]    # and makes prediciton using the transformed sample

   if (output[0] >= 0.75 or output[1] >= 0.75):  #  if the classification confidence is >= 0.75
      print(f"D100{i+1}" , end="\t")             # Day 
      print(test_sample[0] , end="    \t")       # Outlook
      print(test_sample[1] , end="\t")           # Temperature
      print(test_sample[2] , end="\t")           # Humidity
      print(test_sample[3] , end="\t")           # Wind
      print("Yes" if output[0] > output[1] else "No", end="\t\t")   # prediction 
      print(f"{max(output[0], output[1]):.3f}")  # confidence