import os
import pandas as pd
import torch as t

os.makedirs(os.path.join("..", 'data'), exist_ok = True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms, RoofType, Price
NA, NA, 127500
2, NA, 106000
4, Slate, 178100
NA, NA, 140000''')

data = pd.read_csv(data_file)
#print(data)

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
#print("Inputs: ", inputs)
#print("Targets: ", targets)

inputs = pd.get_dummies(inputs, dummy_na = True)
#print("Inputs: ", inputs)

inputs = inputs.fillna(inputs.mean())
#print(inputs)

x = t.tensor(inputs.to_numpy(dtype = float))
y = t.tensor(targets.to_numpy(dtype = float))
#print(x)
#print(y)

#Exercises

#1. Try loading datasets, e.g., Abalone from UCI Machine Learning Repository and inspect their properties. What fraction of them has missing values? What fraction of the variables are numerical, categorical, or text ?
abalone_file = os.path.join('..', 'data', 'abalone.data')
aba_data = pd.read_csv(abalone_file, names =["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"])

#print number of row and columns of table
print(aba_data.shape)
print(aba_data)
#Number of NaN or None values per column
print(aba_data.isnull().sum())
#Number of non NaN values, I'm suprised that there is no direct way to get the number of numeric or string values in the df ... 
print(aba_data.count())
#print converts df to np array however with object type since each row is of mixed dtype
aba_data_np = aba_data.to_numpy()
for i in range(aba_data_np.shape[1]):
    #Check whether there are strings in the column thus categorical data not smooth but whatever
    print(f"It is {all(True if not isinstance(num, str) else False for num in aba_data_np[:, i])} that we have strings in this column")

#2. Try indexing and selecting data columns by name rather than by column number. The pandas documentation on indexing has furhter details on how to do this.
print(aba_data.loc[:, "Sex"])
#Can just check the pandas documentary for functions

#3. How large a dataset do you think you could load this way ? What might be the limitations?
#Pandas are in-memory datastructures thus if the array is bigger than the memory it becomes quite unhandy. Panda has some specific datatypes to decrease the space in memory. Probably not the best way to save data if there is a lot of textual data.

#4. How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?
#Question is whether the categorical data is important for what you are trying to predict. Does it have any causal influence on the output you want to have? In the extreme case where each category label is unique it does not have any predictive power and thus should be excluded. If you have a lot of categorical data, it might be helpful beforehand to see if it in any way correlates with something you want to measure through boxplots or smth else.
