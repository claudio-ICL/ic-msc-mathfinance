"""
The following script defines functions used in the solution to 
Exercise 1 in Problem Set 2 (Fri 27 November 2020)
"""

import sympy
print("\nCall to ex1.py: __name__ = {}".format(__name__))
def phi(x):
    return (1-e_rho)*(s0 - theta*x + theta*x0)
def expected_price(s, x, v):
    return phi(x) + e_rho*s + (theta + gamma)*v
def expected_squared_price(s, x, v):
    return (sigma**2)*(s0**2) + (expected_price(s, x, v))**2
def Cstar_t(s, x):
    return c_1t*(x**2) + c_2t*(s**2) + c_3t*(s*x) + c_4t*x + c_5t*s + c_6t
def G_t(s, x, v):
    return  v*(s + alpha*v) \
            + c_1t1*(x-v)**2 \
            + c_2t1*expected_squared_price(s, x, v) \
            + c_3t1*(x-v)*expected_price(s, x, v) \
            + c_4t1*(x-v) \
            + c_5t1*expected_price(s, x, v) \
            + c_6t1
def delta_Xstar_t1(s, x):
    solutionset = sympy.solveset(sympy.diff(G_t(s, x, v), v), v)
    solutionlist = list(solutionset)
    if not len(solutionlist)==1:
        print("WARNING: len(solutionlist)={}". format(len(solutionlist)))
    return -solutionlist[0]
def Gstar_t(s, x):
    return G_t(s, x, -delta_Xstar_t1(s, x))
def get_coeff_recursion_rules():
    s, x = sympy.symbols('s x')
    expanded_Gstar_t = sympy.expand(Gstar_t(s, x))
    recursion_rules={}
    recursion_rules[c_1t] = expanded_Gstar_t.coeff(x, 2)
    recursion_rules[c_2t] = expanded_Gstar_t.coeff(s, 2)
    recursion_rules[c_3t] = expanded_Gstar_t.coeff(s*x)
    recursion_rules[c_4t] = expanded_Gstar_t.coeff(x) \
            - s*recursion_rules[c_3t]
    recursion_rules[c_5t] = expanded_Gstar_t.coeff(s) \
            - x*recursion_rules[c_3t]
    recursion_rules[c_6t] =  expanded_Gstar_t.coeff(x, 0)
    return recursion_rules
def evaluate_coefficients(t=0, T=10):
    """
    last evaluation amounts to t=T-1
    """
    assert t<T
    lastvals = [alpha, 0, 1, 0, 0, 0]
    if (t<T-1) and ('recursion_rules' not in globals().keys()):
        recursion_rules = get_coeff_recursion_rules()
    c_t = {key: val for (key, val) in zip(recursion_rules.keys(), lastvals)}
    c_t1 = {}
    while t<T-1:
        for key in c_t.keys():
            c_t1[globals()[str(key)+'1']] = c_t[key]
        for key in c_t.keys():
            c_t[key] = recursion_rules[key].subs(c_t1)
        T-=1
    return c_t
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
if __name__=='__main__':
    globals().update(declare_symbols())




