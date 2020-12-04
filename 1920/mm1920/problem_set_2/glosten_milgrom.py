import numpy as np
class GlostenMilgrom:
    def __init__(self):
        print("GlostenMilgrom Constructor")
    def print_last(self):
        print("theta_t = {:1.4f}; a_t ={:1.2f}; b_t ={:1.2f}".format(
            self.thetas[-1], self.quotes[-1][0], self.quotes[-1][1]))
    def print_param(self):
        print("pi={:1.4f};    nu_H = {:1.2f}; nu_L = {:1.2f}; ". format(self.pi, self.nu_H, self.nu_L))
    def set_param(self, pi=0.5, nu_H=3.0, nu_L=1.0,):
        self.pi = pi
        self.nu_H = nu_H
        self.nu_L = nu_L
        self.delta_nu = self.nu_H - self.nu_L
        self.thetas = []
        self.quotes=[[nu_H, nu_L]]
        try:
            self.trade_directions = iter(self.directions)
        except AttributeError:
            pass
    def store_directions(self,directions):
        self.infer_from_directions(directions)
        directions = list(directions)
        directions.insert(0,0) # for index consistency d_0 = 0, i.e. no trades at the initial time
        self.directions = directions  
        self.trade_directions = iter(self.directions)
    def store_empirical_quotes(self, ask, bid):
        self.empirical_ask = np.array(ask)
        self.empirical_bid = np.array(bid)
    def produce_all_quotes(self):
        [self.update() for d in self.directions]
    def store_quoted_prices(self):
        a, b = [], []
        for p in self.quotes:
            a.append(p[0])
            b.append(p[1])
        self.ask_price=np.array(a, dtype=np.float)
        self.bid_price=np.array(b, dtype=np.float)
    def update(self, draw_random_sign=False, true_price='nu_H'):
        self.quotes.append(next(self.new_quotes()))
        try:
            self.thetas.append(next(self.new_theta(draw_random_sign, true_price)))
        except StopIteration:
            print('No new theta')
    def new_theta(self,draw_random_sign=False,true_price='nu_H' ):
        while True:
            if draw_random_sign:
                d=self.generate_directions(true_price)
            else:
                try:
                    d=next(self.trade_directions)
                except StopIteration:
                    print("No more trades")
                    break
            if self.thetas==[]:
                yield 0.5
            else:
                old_theta = self.thetas[-1]
                yield (1.0+self.pi*d)*old_theta/(1.0+(2.0*old_theta-1.0)*self.pi*d)
    def expected_price(self):
        if self.thetas == []:
            return (self.nu_H+self.nu_L)/2.0
        else:
            theta=self.thetas[-1]
            return theta*self.nu_H + (1-theta)*self.nu_L
    def new_quotes(self,):
        while True:
            midprice = self.expected_price() 
            if self.thetas==[]:
                half_spread = (self.pi/2.0)*self.delta_nu
            else:
                theta = self.thetas[-1]
                half_spread = 2*self.pi*theta*(1-theta)*self.delta_nu/(1.0+self.pi*(2*theta-1.0))
            yield [midprice + eps*half_spread for eps in [1,-1]]
    def infer_from_directions(self,directions):
        directions = list(directions)
        true_price= ['nu_H', 'nu_L'][int(directions.count(1)<=directions.count(-1))]
        prob_1 = float(directions.count(1))/len(directions)
        if true_price=='nu_H':
            pi=2*prob_1-1.0
        elif true_price=='nu_L':
            pi=1.0-2*prob_1
        self.empirical_pi = pi    
        self.empirical_true_price = true_price
    def generate_directions(self,true_price='nu_H', size=1, store=False):
        if true_price=='nu_H':
            prob_1 = (1.0+self.pi)/2.0
        elif true_price=='nu_L':
            prob_1 = (1.0-self.pi)/2.0
        probs = [1-prob_1, prob_1]
        directions=np.random.choice([-1,1], p=probs, size=size)
        if store:
            self.store_directions(directions)
        else:
           return np.squeeze(directions)
