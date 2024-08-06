import torch as t

x = t.tensor(3.0)
y = t.tensor(2.0)

print(x + y, x * y, x / y, x ** y)

z = t.arange(3, dtype = t.float32)
print(z, z[2])
print("Dimensionality: ", len(z), z.shape)

A = t.arange(6, dtype = t.float32).reshape(2,3)
print(A, A.T)

B = t.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]], dtype = t.float32)
print(B == B.T)

C = t.arange(24, dtype = t.float32).reshape(2, 3, 4)
print(C)

A_clone = A.clone()
print(A, A + A_clone, A * A_clone)

print(x + A, x * A)

#Sums over all axis unless specified
print("A: ",A.sum(), A.shape, A.sum(axis=0).shape)
print(A.sum(axis=[0,1]) == A.sum())

print(t.mean(A))
print(t.mean(A) == A.sum() / A.numel())

#one can keep the dimensions when summing
A_sum = A.sum(axis=1, keepdims=True)
print(A_sum.shape)

#Get percentual contribution where each row sums to 1
print(A / A_sum)

#Row wise summation (axis = 0 is the axis we sum over)
print(A.cumsum(axis=0))

d = t.ones(3, dtype = t.float32)
print(z, d, t.dot(z, d), t.dot(z, d) == t.sum(z * d))

#Matrix vector multiplication -> is also matrix matrix multiplication if we extend dimensions of vector
print(B.shape, z.shape, t.mv(B, z), B@z) 

E = t.ones(3,4, dtype = t.float32)

print(t.mm(B, E), B@E, (B@E).shape)

#Norm root of sum squares (l2-norm)
print(t.norm(z))

#Norm sum of abs values (l1-norm)
print(t.abs(z).sum())

#Frobenius norm is the l2-norm of matrices
print(t.norm(t.ones((4,9))))

#Exercises
#1. Prove that the transpose of the transpose of a matrix is the matrix itself: A_T_T = A
#If we use the transpose of a matrix a_i_j = b_j_i, apply transpose to b so we get b_j_i = c_i_j == a_i_j. Thus transposing twice gives the same matrix. The transpose is its own reverse linear map.

#2. Given two matrices A and B, show that sum and transpotison commute A_T+B_T = (A+B)_T
#Assume that both matrices have same shape. Then we have A = a_ij and B = b_ij. A_T + B_T = a_ji + b_ji = c_ji. (a_ij+b_ij)_T = (c_ij)_T = c_ji which shows that they are the same

#3. Given any square matrix A, is A + A_T always symmetric ? Can you prove the result by using only the results of the previous two exercises ?
# A + A_T = A_T_T + A_T = (A_T + A)_T = (A + A_T)_T; since the sum is equal to its tranpose A is symmetric

#4. We defined the tensor X of shape (2,3,4) in this section. What is the output of len(X)? Write your answer without implemnting any code, then check you anser using code.
#Answer: 3 since we have three dimensions -> len gives I guess the length of the first dimension which is not the size if the vector space which I have assumed 
print(len(C), len(E), len(t.tensor([]))) 

#5. For a tensor X of arbitrary shape, does len(X) always correspond to the length of a certain axis of X? What is that axis?
#It corresponds to the length of the first axis. 

#6 Run A/A.sum(axis=1) and see what happens. Can you analyze the results ?
#Prediction: We will get a matrix which gives the ratio between a_ij and the sum of all elements.
print(A, A.sum(axis=1))
#This gives an error message. Since we sum over axis 1, we get a tensor with shape (2) for A.sum and one of shape (2,3), these are not broadcastable since we did not keepdim = True. Forgot that sum does not sum here over the whole matrix

#7. When traveling between two points in downtown Manhatten, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets ? Can you travel diagonally?
# I find the question confusing? Ah its a grid lol... Then just sum up the distancei n y and x coordinate. If you can fly, you can take l2-norm. 

#8. Consider a tensor of shape (2,3,4). What are the shapes of the summation outputs along axis 0, 1 and 2?
# Prediction: axis = 0 shape = (3,4); axis = 1 shape = (2,4); axis = 2 shape = (2,3) -> Correct
print(C.sum(axis=0).shape, C.sum(axis=1).shape, C.sum(axis=2).shape)

#9. Feed a tensor with three or more axis to linalg.norm function and observe its output. What does this function compute for tensors of arbitraty shape?
print(t.norm(C), t.linalg.norm(C))
# same as taking the root over all values squared

#10. Consider three large matrices, say A element of R2^10x2^16, B elemt of R2^16x2^5 and C elemt of R2^5x2^14, initialized with Gaussian random variables. You want to compute the product ABC. Is there any difference in memory footprint and speed, depending on whether you compute (AB)C or A(BC). Why?
#Memory: If we take (AB) then we will get a tensor in R2^10x2^5 compared to (BC) in R2^16x2^14 which will take 2^15 orders of magnitude more memory. So first method for memory is better
#Speed: If we take (AB) compute 2^10x2^5 entries with 2^16 summation and multiplications. For (BC) we compute 2^16x2^14 entries with 2^5 summations and multiplications.
# Then we will perfrom (AB)C which will have 2^10*2^14 entries with 2^5 summations and multiplications. While for A(BC) we also have 2^10*2^14 entries with 2^16 summations and multiplications. The amount of summations and multiplications stay the same while the amount of entries in the second methods is way bigger.

#11. Consider three large matrices, say A element of R2^10x2^16, B element of R2^16x2^5 and C element of R2^5x2^16. Is there any difference in speed depending on whether you compute AB or AC_T? Why? What changes if you intialize C = B_T without cloning memory? Why?
#Since C_T includes one more step, I would expect it to take longer to compute. It probably assigns a new memory space to all values.
#If u initiliaze it through B, then no new memory will be allocated but just the indexing will change for C which would likely not make a big difference in speed

#12. Consider three matrices, say A, B, C element of R100x200. Construct a tensor with three axes by stacking [A, B, C]. What is the dimensionality? Slice out the second coordinate of the thrid axis to recover B. Check that your answer is correct.
A = t.arange(20000).reshape(100, 200)
B = t.arange(20000).reshape(100, 200)
C = t.arange(20000).reshape(100, 200)
ABC = t.stack((A, B, C))
print(B.sum() == ABC[2, :, :].sum())

 
