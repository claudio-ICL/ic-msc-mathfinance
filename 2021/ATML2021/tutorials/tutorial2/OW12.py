"""
The following script defines functions peculiar
to the induction argument in the case of
Exercise 1 in Problem Set 2 (Fri 27 November 2020)
"""
import sympy
print("\nCall to OW12.py: __name__ = {}".format(__name__))
def phi(x):
    return (1-e_rho)*(s0 - theta*x + theta*x0)
def expected_price(s, x, v):
    return phi(x) + e_rho*s + (theta + gamma)*v
def expected_squared_price(s, x, v):
    return (sigma**2)*(s0**2) + (expected_price(s, x, v))**2
def last_expected_executioncost(s, x):
    return x*(s+alpha*x)
def declare_symbols():
    s, x, v = sympy.symbols('s x v')
    x0, s0 = sympy.symbols('x0 s0', constant=True)
    alpha, theta, gamma, rho, sigma = sympy.symbols('alpha theta gamma rho sigma', constant=True)
    e_rho = sympy.symbols('e_rho', constant=True) #e_rho is meant to represent sympy.exp(-rho)
    e = sympy.exp(1)
    c_1t, c_2t, c_3t, c_4t, c_5t, c_6t = \
            sympy.symbols('c_1t c_2t c_3t c_4t c_5t c_6t', constant=True)
    c_1t1, c_2t1, c_3t1, c_4t1, c_5t1, c_6t1 = \
            sympy.symbols('c_1t1 c_2t1 c_3t1 c_4t1 c_5t1 c_6t1', constant=True)
    return locals()

globals().update(declare_symbols())
