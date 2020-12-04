import sympy
def Cstar_t(s, x):
    return c_1t*(x**2) + c_2t*(s**2) + c_3t*(s*x) \
            + c_4t*x + c_5t*s + c_6t
def G_t(s, x, v):
    return  v*(s + theta*v) \
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
    recursion_rules = {}
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
    lastvals = [alpha, 0, 1, 0, 0, 0] #These are the values of c_1t ... c_6t when t=T-1
    if 'recursion_rules' not in globals().keys():
        global recursion_rules
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
