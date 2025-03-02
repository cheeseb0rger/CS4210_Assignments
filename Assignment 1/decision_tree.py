#-------------------------------------------------------------------------
# AUTHOR: Johnn Liang
# FILENAME: decision_tree
# SPECIFICATION: decision tree algorithm 
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hrs 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here

replace = {"Young" : 1, "Prepresbyopic" : 2, "Presbyopic" : 3, 
           "Myope" : 1, "Hypermetrope" : 2 , 
           "Reduced" : 1, "Normal" : 2, 
           "Yes" : 1, "No" : 2}

for i in range(len(db)):
   db[i] = [replace[x] for x in db[i]]

# X = 
for row in db:
   X.append(row[:4])

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
# Y =
for row in db:
   Y.append(row[4])

#fitting the decision tree to the data

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()



#######################################################
# from scipy.stats import entropy

# def normalize(p):
#     total = sum(p)
#     return [x / total for x in p] if total != 0 else [0] * len(p)

# def find_ent(p, mult=1):
#   p = normalize(p)
#   ent = entropy(p, base=2)
#   print("Entropy:", ent * mult)



