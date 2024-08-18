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

#1. So far we took the rules for derivatives for granted. Using the definition and limits prove the properties for (i) f(x) = c, (ii) f(x) = x**n, (iii) f(x) = e**x and (iv) f(x) = log(x)
# (i) f'(x) = lim h->0 (c-c)/h = 0 (ii) f'(x) = lim h->0 ((x+h)**n-x**n)/h use induction case n=1: lim h->0 (x+h-x)/h = 1 n: f'(x) = nx**(n-1) N=n+1: f'(x) = lim h->0 ((x+h)**(N)-x**N)/h = lim h->0 ((x+h)**n*(x+h)-x**n*x)/h = lim h->0 (x*((x+h)*n-x*n)+h*(x+h)**n)/h = n*x**n+x**n = N*x**n (iii) f'(x) = lim h->0 (e**(x+h)-e**x)/h = lim h->0 e**x * (h-1)/h = lim h->0 e**x - 1/h = e**x (iv) f'(x) = lim h->0 (log(x+h)-log(x))/h = lim h->0 log((x+h)/x)/h = lim h->0 log(1+h/x)/h = lim h->0 log(1+t)/tx = 1/x lim h->0 log((1+t)^(1/t)) -> 1/x log(e) -> depending on the logarithm we get 1/(x* ln(b)), where b is the base of the log

#2. In the same vein, prove the product, sum, and quotient rule from first principles.
#(i) (f(x)+g(x))' = lim h->0 (f(x+h)+g(x+h)-f(x)-g(x))/h = lim h->0 (f(x+h)-f(x))/h + (g(x+h)-g(x))/h = f'(x)+g'(x) (ii) (f(x)*g(x))' = lim h->0 (f(x+h)*g(x+h)-f(x)*g(x))/h = lim h->0 (f(x+h)*g(x+h)-g(x)f(x)+g(x)*f(x+h)-g(x)*f(x+h))/h = lim h->0 (g(x)*(f(x+h)-f(x)))/h + (f(x+h)*(g(x+h)*g(x)))/h = g(x)*f'(x)+ f(x)*g'(x) (iii) (f(x)/g(x))' = lim h->0 (f(x+h)/g(x+h)-f(x)/g(x))/h = lim h->0 1/h( (f(x+h)*g(x))/(g(x+h)*g(x)) - (f(x)*g(x+h))/(g(x)+g(x+h)) + (f(x)*g(x))/g(x)^2 - (f(x)*g(x))/g(x)^2) = lim h->0 1/h (((f(x+h)-f(x))*g(x))/g(x)^2 - ((g(x+h)-g(x))*f(x))/g(x)^2) = (f'(x)g(x)-f(x)g'(x))/g(x)^2

#3. Prove that the constant multiple rule follows as a special case of the product rule.
#(C*f(x))' = C'*f(x)+C*f'(x) = C*f'(x)

#4. Calculate the derivative of f(x) = x^x
#(x^x)' = (e^(x*ln(x)))' = e^(x*ln(x)) * (x*ln(x))' = x^x * (ln(x)+1)

#5. What does it mean that f'(x)=0 for some x? Give an example of a function f and a location x for which this might hold.
#Minima, Maxima or plateau. Exemplary functions are x^2 or x^3

#6. Plot the function g(x) = x^3-1/x and plot its tangent line at x=1.
def g(x): 
    return x ** 3 - 1/x
def tg(x):
    return 4 * x - 4

plt.plot(a, g(a), "r")
plt.plot(a, tg(a), "b")
plt.show()

#7 Find the gradient of the function f(x) = 3x_1^2+5e^x_2
# Derivative df(x)/dx_1 = 6x_1; df(x)/d_x2 = 5e^x_2 -> can be rewritten as gradient vector

#8 What is the gradient of the function f(x) = ||x||_2? What happens for x=0?
#Assume vector x then: df(x)/dx_i = ((sum_{from i=1 to n} x_i^2)^0.5)' = (sum_{from i=1 to n} x_i^2)^(-0.5) * x_i

#9 Can you write out the chain rule for the case where u = f(x,y,z) and x=x(a,b), y=(a,b) and z=z(a,b)?
#du/da = df/dx dx/da + df/dy dy/da + df/dz dz/da; du/db = df/dx dx/db + df/dy dy/db + df/dz dz/db -> total derivative would be the sum of both.

#10 Given a function f(x) that is invertible, compute the derivative of its inverse f^-1(x). Here we have that f^-1(f(x)) = x and conversely f(f^-1(y)) = y. Hint: Use these properties in your derivation.
# Use Chain rule (f^-1(f(x)))' = f^-1'(f(x)) * f'(x) => 1 = f^-1(y) * f'(x) => 1/f'(x) = f^-1(y)

