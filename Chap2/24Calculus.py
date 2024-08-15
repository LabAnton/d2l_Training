import numpy as np 
import math
import matplotlib.pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x

def t(x):
    return 2 * x - 3
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')

#Dont know why they do it this complicated ...

a = np.arange(0, 3, 0.1)

plt.plot(a, t(a), "r")
plt.plot(a, f(a), "b")
plt.show()

#Exercises
#1.So far we took the rules for derivatives for granted. Using the definition and limits prove the properties for (i) f(x) = c, (ii) f(x) = x**n, (iii) f(x) = e**x and (iv) f(x) = log(x)
# (i) f'(x) = lim h->0 (c-c)/h = 0 (ii) f'(x) = lim h->0 ((x+h)**n-x**n)/h use induction case n=1: lim h->0 (x+h-x)/h = 1 n: f'(x) = nx**(n-1) N=n+1: f'(x) = lim h->0 ((x+h)**(N)-x**N)/h = lim h->0 ((x+h)**n*(x+h)-x**n*x)/h = lim h->0 (x*((x+h)*n-x*n)+h*(x+h)**n)/h = n*x**n+x**n = N*x**n (iii) f'(x) = lim h->0 (e**(x+h)-e**x)/h = lim h->0 e**x * (h-1)/h = lim h->0 e**x - 1/h = e**x (iv) f'(x) = lim h->0 (log(x+h)-log(x))/h = lim h->0 log((x+h)/x)/h = log(1+h/x)/h


