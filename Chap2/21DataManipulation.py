import torch as t

#creates tensor with 12 elements from 0 to 11
x = t.arange(12, dtype= t.float32)
y = t.arange(12, dtype= t.float32)
print(x)

#returns number of elements in x
print(x.numel())
# returns size of tensor
print(x.shape)

#changes size of tensor to designated one, first row than column, -1 can be used to infer size
x = x.reshape(3, 4)
y = y.reshape(-1, 4)
print(x)
print(y)

#print tensor with zeros or ones in it in designated size
z = t.zeros((2, 3, 4))
a = t.ones((2, 3, 4))
print(z)
print(a)

#print tensor with numbers picked from a random distribution, here gaussian distribution mean = 0 sd = 1
b = t.randn(3,4)
print(b)

#can create tensor out of list of data
c = t.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(c)

#Select first axis 0 and so on, standard indexing practice
print(x[-1], x[1:3])
x[1, 2] = 17
print(x)
x[:2, :] = 12
print(x)

#Most standard fuctions are included in pytorch and can be applied element wise f.e. exp
print(t.exp(x))

#Standard arithmetic operators work between tensors if they have the same size
print(x + y, x - y, x * y, x / y, x **y)

#Concatenating tensors is possible just have to specify along which axis and has to have same size, along the other axes
print(t.cat((x,y), dim=0), t.cat((x,y), dim=1))

# == can be used to compare tensors and get a binary tensor with 0 and 1 (False or True) values
print(x == y)

#Broadcasting works if tensors have equal size, are of by at most one dimension or one dimension is 1. The tensor will be expanded along this dimension such that both tensor have the same size
d = t.arange(3).reshape((3, 1))
e = t.arange(2).reshape((1, 2))
print((d+e).shape)

#Pytorch allocates a adress in memory for every newly calculated tensor. To save memory by indexing the whole tensor from before f[:] the same memory can be allocated 
before = id(e)
e = d + e
print(id(e) == before)

f = t.zeros_like(e)
print(f, id(f))
f[:] = d + e
print(f, id(f))

before = id(f)
f += e
print(id(f) == before)

#Changing from numpy to torch is easily done by 
g = f.numpy()
h = t.from_numpy(g)
print(type(g), type(h))

#Can use pythons build in functions to to convert a size-1 tensor to other python scalars

i = t.tensor([3.5])
print(i, i.item(), float(i), int(i))

#Exercises:
#1. Run the code in this section. Change the conditional statement X == Y to X < Y or X > Y, and then see what kind of tensor you get.
#2. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected ?

#1.
print(x == y, x > y)
#Element-wise comparison
#2.
d = t.arange(3).reshape((1, 1, 3))
e = t.arange(2).reshape((1, 2, 1))
print(d, e)
print(d+e)
#Same behaviour as expected
