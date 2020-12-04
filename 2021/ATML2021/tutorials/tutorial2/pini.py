"""
The following script defines functions 
peculiar to the induction argument in the case of 
'Permanent price impact with no information'
"""
import sympy
print("\nCall to ex1.py: __name__ = {}".format(__name__))
def expected_price(s, x, v):
    return s + theta*v
def expected_squared_price(s, x, v):
    return (sigma**2)*(s0**2) + (expected_price(s, x, v))**2
def last_expected_executioncost(s, x):
    return s*x + theta*x**2
def declare_symbols():
    s, x, v = sympy.symbols('s x v')
    x0, s0 = sympy.symbols('x0 s0', constant=True)
    theta, sigma = sympy.symbols('theta sigma', constant=True)
    alpha = theta
    e = sympy.exp(1)
    c_1t, c_2t, c_3t, c_4t, c_5t, c_6t = \
            sympy.symbols('c_1t c_2t c_3t c_4t c_5t c_6t', constant=True)
    c_1t1, c_2t1, c_3t1, c_4t1, c_5t1, c_6t1 = \
            sympy.symbols('c_1t1 c_2t1 c_3t1 c_4t1 c_5t1 c_6t1', constant=True)
    return locals()

globals().update(declare_symbols())
