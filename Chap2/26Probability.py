import numpy as np 
import torch as t
import random
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt

num_tosses = 100    
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print ("heads, tails", [heads, tails])
fair_probs = t.tensor([0.5, 0.5])
print(Multinomial(100, fair_probs).sample())
print(Multinomial(10000, fair_probs).sample()/10000)

counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims = True)
estimates = estimates.numpy()

plt.plot(estimates[:, 0], label = "P(coin = heads)")
plt.plot(estimates[:, 1], label = "P(coin = tails)")
plt.legend()
plt.show()

#Exercises
#1. Give an example where observing more data can reduce the amount of uncertainty about the outcome to an arbitrarily low level.
# Fake coin which is not 50-50. With more observation it will become clear whether the coin is fair or not. If uncertainty = 0% is meant, then whether a rock will fall or something physics related.

#2. Give an example where observing more data will only reduce the amount of uncerainty to a point and then no further. Explain why this is the case and where you expect this point to occur. 
# Epistemic uncertainty is hard because it will most likely always reduce. An example would be if the underlying probability distribution is randomly changing with more repeats. If aleatoric uncertainty is meant, take the coint example 

#3. We empeirically demonstrated convergence to the mean for the toss of a coin. Calculate the variance of the estimate of the probability that we see a head after drawing n samples ?
#   1. How does the variance scale with the number of observations?
#   2. Use Chebyshevs inequality to bound the deviation from the expectation.
#   3. How does it relate to the central limit theorem?    
# Let X_i = 0 for tails and 1 for heads
# The expectation of heads is E[Heads] = n * p = 0.5 * n. Thus the variance is, since every event is independent, we can just add up every single event:  Var[Head] = n * p - n * p^2 = n * 0.25 
#   1. With increasing number of observation the variance scales as 0.25 with the number of observations.
#   2. F.e. take n = 10 and k = 2**0.5 then we have a probability of 50% of finding an value in the interval (5 - 2**0.5 * 2.5, 5 + 2**0.5 * 2.5)     
#   3. If we pick a large n, then the interval, where most of the probability mass will be, will shrink in comparison to all possibilities.

#4. Assume that we draw m samples x_i from a probability distribution with zero mean and unit variance. Compute the averages z_m = m^-1 sum_{i=1 to m} x_i. Can we apply Chebyshevs inequality for every z_m independently? Why not?
#The average is the same as the expected value which would be the integral over a gaussian with mean 0 (assuming continous data) which would be 0. For discrete data you have to calculate it directly. For small sample sizes Chebyshev's inequality might not hold. Otherwise I see now reason since the mean and std can be calculated without any problem.

#5. Given two events with proability P(A) and P(B), compute upper and lower bounds on P(A U B) and P(A N B) (N meaning intersection). Hint: graph the situation using a Venn diagram.
# For P(A U B)= P(A) + P(B) - P(A N B) has higher bound and lower bound of 100% if A and B are exhaustive of the sample set S.  
# For P(A N B) = P(A U B) - P(A) - P(B) has a higher bound of close to a 100% if A and B would be basically the same event and lower bound of 0% if A and B are mutually exclusive events.  

#6. Assume that we have a sequence of random variables say A, B, C, where B only depends on A, and C only depends on B, can you simplify the joint probability P(A,B,C)? Hint: This is a Markov chain.
#P(A, B, C) = P(C|B) * P(B|A) * P(A), all other than A are conditional on a previous state. 

#7. In section 2.6.5, assume that the outcomes of the two tests are not independent. In particular assume that either test on its own has a false positive rate of 10% and a false negative rate of 1%. That is assume that P(D=1|H=0) = 0.1 and that P(D=0|H=1) = 0.01. Moreover, assume that for H = 1 (infected) the test outcomes are conditionally independent, i.e., that P(D_1,D_2|H=1) = P(D_1|H=1)P(D_2|H=1) but that for healthy patients the outcomes are coupled via P(D_1=D_2=1|H=0)=0.02.
#   1. Work out the joint probability table for D_1 and D_2, given H=0 based on the information you have so far.
#   2. Derive the probability that the patient is diseased (H=1) after one test returns positive. You can assume the same baseline probability P(H=1)=0.0015 as before.
#   3. Derive the probability that the patient is diseased (H=1) after both tests return positive.
#1. done on paper 3 are not given for healthy patients 
#2. Answer is 0.01465
#3. Answer is 0.06857

#8. Assume that you are an asset manager for an investment bank and you have a choice of stocks s_i to invest in. Your portfolio needs to add up to 1 with weights a_i for each stock. The stocks have an average return of y=E_{s~P}[s] and covariance Sum = Cov_{s~P}[s].
#   1. Compute the expected return for a given portfolio alpha.
#   2. If you wanted to maximize the return of the portfolio, how should you choose your investment?
#   3. Compute the variance of the portfolio.
#   4. Formulate an optimization problem of maximizing the return while keeping the variance constrained to an upper bound. 
#Intuitively I expect all stocks to be independent of each other. That would mean all non-diagonal positions are 0. 
#1.Independent case:  The expected return for the portfolio is just the sum over each individual expected return times the weights.
# Dependent case: multiply alpha^T * Sum * alpha
#2.Independent case: Set the weight for the stock with highest expected value to 1
#Dependent case: Maximize alpha^T * Sum ^ alpha
#3.See equation 2.6.16 
#4.If each stock has an upper bound on its variance, then we can just pick the portfolio that under these constraints makes the maximum profit. That means we optimize the profits under the constraint of a specific risk.
