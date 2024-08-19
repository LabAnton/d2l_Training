import torch as t
import math
import matplotlib.pyplot as plt

x = t.arange(4.0)
print(x)

x.requires_grad_(True)
print('Gradient empty: ', x.grad)

y = 2 * t.dot(x, x)
print('Output of function: ', y)

y.backward()
print("Gradient of y at each x: ", x.grad)
print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x 
u = y.detach() 
z = u * x

z.sum().backward()
print(u)
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b 
    return c

a = t.randn(size = (), requires_grad = True)
d = f(a)
d.backward()

print('d and a :', d, a, a.grad)
print(a.grad == d/a)

#Exercises
#1. Why is the second derivative much more expensive to compute than the first derivative ?
#Take f.e. a function f:R^N -> R then the jacobian has a len of n while the heassian (second derivative) has a size of nxn. The hessian scales badly with bigger matrices.

#2. After running the function for backpropagation, immediately run it again and see what happens. Investigate.

x.grad.zero_()
print("x: ", x)
r = x * x
print(r)
r.backward(gradient = t.ones(len(r)), retain_graph = True)
print(x.grad)
r.backward(gradient = t.ones(len(r)))
print(x.grad)

#It gives the error that the gradient is supposed to be saved somewhere otherwise it will get deleted. Then it add them

#3. In the control flow example where we calculate the derivative of d with respect to a, what would happen if we changed the variable a to a random vector or matrix? At this point, the result of the calculation f(a) is no longer a scalar. What happens to the result? How do we analyze this?
#The result would also be a vector/matrix. We would have to pass to backward an argument for v^T such that it calculates the jacobian properly. The result is v^T * J which is basically a weighted sum across our partial derivatives.

#4 Let f(x) = sin(x). Plot the graph of f and of its derivative f'. Do not exploit the fact that f'(x) = cos(x) but rather use automatic differentiation to get the result.

x = t.arange(0, 2*math.pi, 0.1, requires_grad = True)
sin = t.sin(x)
print(x)
print(sin)
sin.backward(gradient = t.ones(len(x)))
cos = x.grad.numpy()
x = x.detach().numpy()
sin = sin.detach().numpy()
plt.plot(x, sin, "r")
plt.plot(x, cos, "g")
plt.show()

#5. Let f(x) = ((log(x^2) * sin(x)) + 1/x. Write out a dependency graph tracing results from x to f(x).
# Start with one node x that transverses into three nodes sin(x), x^2 and 1/x. x^2 transverses to log(x^2) then connects with sin(x) and finally with 1/x. 

#6. Use the chain rule to compute the derivative df/dx of the aforementioned function, placing each term on the dependency graph that you constructed previously.
#Done on paper

#7. Given the graph and the intermediate derivative results, you have a number of option when computing the gradient. Evaluate the result once starting from x to f and once from f tracing back to x. The path from x to f is commonly known as forward differentiaion, whereas the path from f to x is known as backward differentiation.
#Depending on the graph forward differentiation or backward differentiation could be better. Since the graph will look different for different differentiations, one will probably be more efficient than the other. Backward prop has the advantage that it is easy to work your way from the result to the beginning, while for forward prop you have to know which nodes to transverse to save compute etc.

#8. When might you want to use forward and when backward, differentiation? Hint: consider the amount of intermediate data needed, the ability to parallelize steps, and the size of the matrices and vectors involved.
#Above statement -> It depends on the sizes of the matrixes. If the space you are mapping to is way smaller than the space you are coming from then backward propagation has to do less multiplications. If its the other way around then forward propagation has to do way less computations.
