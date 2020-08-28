import numpy as np
from math import log, exp
from scipy.stats import poisson
from scipy.special import gamma, gammaln
import random

class WeibullCount:
    
    @staticmethod
    def _faster_percentile_function_lookup(cache, total_array_length):
        return lambda x: cache[int(x*total_array_length)]
        
    def _faster_percentile_function(self, precision = 2):
        ##We don't need to double loop through the input values,
        ##and our cumulative sum. We can build a cache that is
        ##split up so we get O1 lookup of values down to any
        ##precision i.e. cache[0.45*precision] = 4
        
        ##This is a valuable speed up where we are querying
        ##a large number of values or where our cumulative sum
        ##is large. The gain is fairly marginal though as most of
        ##the processing time is spent looping through the queried
        ##values not querying for values.
        
        dist = self.dist()
        
        total_array_length = 100 ** precision
        cumulative = [
            int(round(i, 0))
            for i
            in np.cumsum(dist) * total_array_length
        ]
        cumulative.append(total_array_length) ##Add so we get correct size of final chunk
        cache = {}
        curr_prob = 0
        for i in range(total_array_length):
            if i > cumulative[curr_prob]:
                curr_prob +=1
            cache[i] = curr_prob
        return WeibullCount._faster_percentile_function_lookup(cache, total_array_length)
        ##We return a function that queries list rather than list
    
    @staticmethod
    def test():
        ##If shape param is 1 then we should get the same value
        ##as the poisson
        w = WeibullCount(3, 1)
        prob = w.pmf([1])[0]
        return (round(prob, 2) == round(poisson.pmf(1,3), 2))
    
    def inner_func(self, j, m):
        return exp(gammaln(self.shape*(j-m)+1) - gammaln(j-m+1))
    
    def outer_func(self, j, n, alpha):
        return ((-1)**(j+n) * (self.rate * self.time ** self.shape)**j*alpha)/gamma(self.shape*j+1)
    
    def __init__(self, rate, shape, precision = 20, outcomes = 10, time = 1):
        self.precision = precision
        self.outcomes = outcomes
        self.time = time
        self.rate = rate
        self.shape = shape
        self.cache = np.zeros((outcomes, precision))
        return
    
    def dist(self):
        results = np.array([
            sum(self.outer_func(
                np.array(list(range(e, e+self.precision))),
                e,
                self._alpha(e)
            )) 
            for e 
            in range(self.outcomes)
        ])
        for i, j in enumerate(results):
            if j < 0:
                results[i] = np.inf
        return results
    
    def cdf(self, vals):
        dist = self.dist()
        cumulative = np.cumsum(dist)
        return [
            cumulative[i]
            for i
            in vals
        ]
    
    def ppf(self, vals, precision = 2, speed = False):
        if speed == True:
            query_func = self._faster_percentile_function(precision)
            return [query_func(i) for i in vals]
        
        else:
            dist = self.dist()
            cumulative = np.cumsum(dist)
            result = []
            for i in vals:
                pos = None
                for j, k in enumerate(cumulative):
                    if k > i:
                        pos=j
                        break
                if pos == None: ##0 is Falsy, we check for None explictly
                    result.append(j)
                else:
                    result.append(pos)
            return result
    
    def rvs(self, size = 10, speed = False):
        random = np.random.uniform(0, 1, size)
        return self.ppf(random, speed=speed)

    def pmf(self, vals):
        dist = self.dist()
        return [
            dist[i] 
            for i 
            in vals
        ]
    
    def logpmf(self, vals):
        dist = self.dist()
        return [
            log(dist[i])
            for i 
            in vals
        ]
    
    def _alpha(self, n):
        if n == 0:
            vals = np.array([
                self.inner_func(i, 0) 
                for i 
                in range(self.precision)
            ])
            self.cache[n] = vals
            return vals
        else:
            buf = np.zeros(self.precision)
            for i, j in enumerate(range(n, n+self.precision)):
                new_vals = np.array([
                    self.inner_func(j, i) 
                    for i 
                    in range(n-1, j)
                ])
                last = self.cache[n-1][:len(new_vals)]
                buf[i] = np.dot(last, new_vals)
            self.cache[n] = buf
            return buf
