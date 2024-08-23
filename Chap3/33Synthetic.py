import random
import time
import torch as t
import utils 

class SyntheticRegressionData(utils.HyperParameters):
    def __init__(self, w, b, noise = 0.01, num_train = 1000, num_val = 1000, batch_size = 32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = t.randn(n, len(w))
        noise = t.randn(n, 1) * noise
        self.Y = t.matmul(self.X, w.reshape((-1, 1))) + b + noise

    def get_dataloader_1(self, train):
        if train:
            indices = list(range(0, self.num_train))
            random.shuffle(indices)   
        else:
            indices = list(range(self.num_train, self.num_train+self.num_val))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = t.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.Y[batch_indices]
    
    def train_dataloader(self):
        return self.get_dataloader_2(train = True)

    def val_dataloader(self):
        return self.get_dataloader_2(train = False)
    
    def get_tensorloader(self, tensors, train, indices = slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = t.utils.data.TensorDataset(*tensors)
        return t.utils.data.DataLoader(dataset, self.batch_size, shuffle = train)
    
    def get_dataloader_2(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.Y), train, i)
        


data = SyntheticRegressionData(w = t.tensor([-2, -3.4]), b = 4.2)

print(data.num_train)
print("Features: ", data.X[0], '\nlabel: ', data.Y[0])

X, Y = next(iter(data.train_dataloader()))
print("X shape: ", X.shape, "\nY shape: ", Y.shape)

print(len(data.train_dataloader()))

#Exercises
#1. What will happen if the number of examples cannot be divided by the batch size. How would you change this behaviour by specifying a different argument by using the framework's API?
# If the examples cannot be divided by the batchsize, then it will just leave out last samples. This could be fixed by defining the iterator differently such that it tries to get a batch with batch_size but if its not possible it will get the largest possible. Or only cycling through the range where the batch_size is specified. Since in each iteration we pick the indices randomly this should also be fine. Depending on how specialized the pipeline is having the same batchsize might be more efficient.  

#2. Suppose that we want to generate a huge dataset, where both size of the parameter vector w and the number of examples num_examples are large.
#   1. What happens if we cannot hold all data in memory?
#   2. How would you shuffle that data if it is held on disk? Your task is to design an efficient algorithm that does not require too many random reads or writes. 
#1. Then the RAM will be overloaded and one will have to fix it otherwise data cannot be moved. Sampling the data into smaller structures seems the easiest way. 
#2. Not the most efficient but the Fisher Yates shuffling algorithm works (time complexity can be reduced using Knut's method):

def FisherYatesShuffle(arr):
    output = []
    visited = [False] * len(arr)
    for i in range(len(arr)):
        j = random.randint(0, len(arr) - 1)
        while visited[j]:
            j = random.randint(0, len(arr) - 1)
        output.append(arr[j])
        visited[j] = True
    return output 

def KnutShuffle(arr):
    output = []
    while len(arr) != 0:
        j = random.randint(0, len(arr)-1)
        output.append(arr[j])
        arr[j], arr[-1] = arr[-1], arr[j]
        arr.pop()
    return output

arr = ["A", "B", "C", "D", "E", "F"]
random.seed(333)
start = time.time()
shuffled_array = FisherYatesShuffle(arr)
end = time.time()
print(shuffled_array, end-start) 
start = time.time()
shuffled_arr = KnutShuffle(arr)
end = time.time()
print(shuffled_arr, end-start)

#3. Implement a data generator that produces new data on the fly, every time the iterator is called.
class SyntheticIterData(HyperParameters):
    def __init__(self, w, b, noise = 0.01, batch_size = 32):
        super().__init__()
        self.save_hyperparameters()
        self.X = t.randn(batch_size, len(w))
        noise = t.randn(n, 1) * noise
        self.Y = t.matmul(self.X, w.reshape((-1,1)) + b + noise)
    
    def train_dataloader(self):
        return self.get_dataloader(train = True)

    
    def get_dataloader(self, train, generate_new):
        if generate_new:
            new_X = t.randn(batch_size, len(w))
            new_Y = t.matmul(new_X, self.w.reshape((-1, 1)) + self.b + self.noise)
            self.X = t.cat((self.X, new_x), 0)
            self.Y = t.cat((self.Y, new_Y), 0)

        #Only training with half the data; can easily be changed, only problem if no new data is generated then in the beginning it will not take any data since 16 is too small for step size 32 
        if train:
            indices = list(range(0, len(self.X)/2)) 
        
        else:
            indices = list(range(len(self.X)/2, len(self.X))) 

        for i in range(0, len(indices), self.batch_size):
            btach_indices = t.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices, self.Y[batch_indices]
 
#4. How would you design a random data generator that generates the same data each time it is called?
# Specify a random.seed(), then the data or order is reproducible.        

