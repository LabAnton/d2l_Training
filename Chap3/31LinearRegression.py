import math
import time
import numpy as np
import torch as t
import matplotlib.pyplot as plt

n = 10000  
a = t.ones(n)
b = t.ones(n)
c = t.ones(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time()-t:.5f} sec')

t = time.time()
d = a + b
print(f'{time.time()-t:.5f} sec')

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)

x = np.arange(-7, 7, 0.01)

params = [(0,1), (0,2), (3,1)]
for param in params:
    plt.plot(x, normal(x, param[0], param[1]), label = f"{param}")
plt.legend(loc = "upper left")
plt.show()

#Exercises:
#1. Assume that we have some data x_1,...x_n element of R. Our goal is to find a constant b such that sum_{i}(x_i-b)^2 is minimized.
#   1. Find an analytic solution for the optimal value of b.
#   2. How does this problem and its solution relate to the normal distribution?
#   3. What if we change the loss from sum_{i}(x-b^2) to sum_{i}|x_i-b|? Can you find the optimal solution for b?
#1. df(x)/dx = d/dx sum_{i}(x_i-b)^2 = 2 * sum_{i}(x_i-b)
# df/dx = 0 -> b = 1/n * sum_{i}x_i
#2. Data is picked from a linear function with an offset. However due to noise there is an error in how well the data is drawn. The log-likelihood is sum_{i}(y_i-x_i-b)^2. The negative log-likelihood basically just looks at the average distance between y and x with and constant offset b.
#3. Analytically not solvable since the standard minimization would give n=0. However the problem is very similiar to the earlier one since taking the squared is similiar to the abs value in that we get a positive number just a smaller one. So the optimal solution for b would probably be the average of the data.

#2. Prove that the affine functions that can be expressed by x^Tw+b are equivalent to linear functions on (x,1).
#Unsure what they mean by (x,1) is this supposed to mean (x,b)? If b =0 then we obv. have a linear map since it goes through the origin. If they mean that there is a mapping between the affine function and a linear one then one just has to substract b to get a linear map.

#3. Assume that you want to find quadratic functions of x, i.e. f(x) = b + sum_{i} (w_i * x_i) + sum_{j<=i} w_ij * x_i * x_j. How would you formulate this in a deep network?
#Include nodes where there is a permutation of the data without indexical copies and multiply by a weight vector with the same length.

#4. Recall that one of the conditions for the linear regression problem to be solvable was that the design matrix X^T*X has full rank.
#   1. What happens if this is not the case?
#   2. How would you fix it? What happens if you add a small amount of coordinate-wise independent Gaussian noise to all entries of X?
#   3. What is the expected value of the design matrix X^T*X in this case?
#   4. What happens with stochastic gradient descent when X^T*X does not have full rank?
#1. If X has not full rank then taking the inverse of X^T X is not possible and thus there is no optimal solution for w.
#2. Check which column is a linear dependence of two other ones and either delete it or look whether another linearly independent vector can be substituted for this direction. Adding independent Gaussian noise, will lead to X becoming full rank.
#3. It is the covariance matrix of the data. The covariance matrix is always atleast one rank below X^T * X.
#4. It might not converge onto a specific optimum but run around in different direction since there is more than one minima.

#5. Assume that the noise model governing the additive noise epsilon (e) is the exponential distribution. That is p(e) = 1/2 exp(-|e|).
#   1. Write out the negative log-likelihood of the data under the model -logP(y|X).
#   2. Can you find a closed form solution?
#   3. Suggest a minibatch stochastic gradient descent algorithm to solve this problem. What could possible go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?
#1. p(y|x) = 1/2 exp(|y-w^T*x-b|) -> -log(p(y|x)) = sum_{i}|y_i-w^T*x_i-b| + log(2)*n
#2. The expression can be written in linear algebra. so I do not see any problem
#3. Derivative dp(y|x)/dw = 0 -> 0 = sum_{i} sgn(y_i-w^T*x_i+b)*x_i; the gradient flips its sign and thus there is not optimal point. Maybe instead of sgn use a sigmoidal function ? In the limit we will have the same behaviour however without any jumps.

#6. Assume that we want to design a neural network with two layer by composing two linear layers. That is, the output of the first layer becomes the input of the second layer. Why would such a naive composition not work?
# Because we could just represent it as one linear function; one dimensional example w_2(w_1x+b_1)+b_2 =w_2w_1x+b_1w_2+b_2 = wx+b 

#7. What happens if you want to use regression for realistic price estimation of houses or stock prices ?
#   1. Show that the additive Gaussian noise assumption is not appropriate. Hint: can we have negative prices? What about fluctuations?
#   2. Why would regression to the logarithm of the price be much better, i.e. y = log(price)?
#   3. What do you need to worry about when dealing with pennystock, i.e., stock with very low prices? Hint: can you trade at all possible prices? Why is this a bigger problem for cheap stocks? 
#1. A house cannot have negative prices. The line has to start at the origin since no stock costs nothing. The fluctuations of the housing and stock market are not gaussian distributed i.e it is not noise that changes the housing prices. The mean of gaussians is stable over time while for the market it moves.
#2. It depends on how housing prices are distributed however if we presume and exponential distribution of prices than the log of the price would be a straight line which is easier to fit.
#3. Small fluctuations will influence your fit drastically.

#8.Suppose we want to use regression to estimate the number of apples sold in a grocery store.
#   1. What are the problems with a Gaussian addtive noise model? Hint: you are selling apples not oil.
#   2. The poisson distribution captures distribtuions over counts. It is given by p(k|l) = l^k exp(-l)/k!. Here l is the rate function and k is the number of events you see. Prove that l is the expected value of counts k.
#   3. Design a loss function associated with the Poisson distribution.
#   4. Design a loss function for estimating log(l) instead.
#1. You can not sell half an apple or one fourth.
#2. E[k] = sum_{k=1} k*l^k*exp(-l)/k! = exp(-l)*l*sum_{i=1}l^(k-1)/(k-1)! = l
#3. log(P(y|y_hat=w^Tx-b)) = sum y_i * log(y_hat) - y_hat - log(y_i!) ~ sum y_i * log(y_hat) - y_i
#4. log(P(y|l=exp(y_hat))= sum y_hat * y_i + exp(-y_i) - log(y_i!) ~ sum y_hat * y_i + exp(-y_i)

