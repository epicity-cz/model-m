# -*- coding: utf-8 -*-


from engine import BaseEngine
from graph_gen import RandomSingleGraphGenerator

import scipy.stats as stats
from bisect import bisect_left
from itertools import accumulate
import random
import sys
import time



# =============================================================================
# model = SEIRSNetworkModel(G       =G_normal, 
#                           beta    =0.155, 
#                           sigma   =1/5.2, 
#                           gamma   =1/12.39, 
#                           mu_I    =0.0004,
#                           mu_0    =0, 
#                           nu      =0, 
#                           xi      =0,
#                           p       =0.5,
#                           Q       =G_quarantine, 
#                           beta_D  =0.155, 
#                           sigma_D =1/5.2, 
#                           gamma_D =1/12.39, 
#                           mu_D    =0.0004,
#                           theta_E =0, 
#                           theta_I =0, 
#                           phi_E   =0, 
#                           phi_I   =0, 
#                           psi_E   =1.0, 
#                           psi_I   =1.0,
#                           q       =0.5,
#                           initI   =numNodes/100, 
#                           initE   =0, 
#                           initD_E =0, 
#                           initD_I =0, 
#                           initR   =0, 
#                           initF   =0)
# 
# =============================================================================



# constants for Models

MAGIC_SEED_BY_PETRA = 42

# β: rate of transmission (transmissions per S-I contact per time)
# σ: rate of progression (inverse of incubation period)
# γ: rate of recovery (inverse of infectious period)
# ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
# μI: rate of mortality from the disease (deaths per infectious individual per time)
BETA = 0.155
SIGMA = 1/5.2
GAMMA = 1/12.39
XI = 0
MJU_I = 0.0004
MJU_ALL = 0.00003 # 11 ppl per 1000 ppl per year ... 3 ppl per day per 100k ppl

# probability of distant contact, 1 means only distant contacts, 0 means only close contacts
# in deterministic model, there are only distant contacts
# the ration of p / (1-p) tells the distribution of contacts between distant and close
# average number of contacts is per day, actual values are sampled from normal distribution (truncated at LTRUNC and HTRUNC)


PROB_FOR_GRAPH = 0.5

SUSCEPTIBLE = 0
SUSCEPTIBLE_S = 1
EXPOSED = 2
INFECTIOUS_N = 3
INFECTIOUS_A = 4
INFECTIOUS_S = 5
INFECTIOUS_D = 6
RECOVERED_D = 7
RECOVERED_U = 8
DEAD_D = 9
DEAD_U = 10

NO_OF_STATES = 11
       
class Person:
    def __init__ (self, id, degree, sex, age, init_state=SUSCEPTIBLE):
        self.state = init_state
        self.time_of_state = 0
        self.id = id
        self.degree = degree
        self.sex = sec
        self.age = age
        
    def get_id():
        return self.id
    
    def get_state(self):
        return self.state

    def set_state(self, s=SUSCEPTIBLE):
        old_state = self.state
        self.state = s
#        print (self.id, ':', SEIRSNAMES[old_state], '->', SEIRSNAMES[s])
        
    def prob_change_state(self, probs):
        su = sum(probs)
        if su < 1 :
            probs[self.state] = 1 - su
        cum_p   = list(accumulate(probs))
        value = random.random()
        new_state = bisect_left(cum_p, value)
        if new_state != self.state :
            self.set_state(new_state)

    
# model working with SEIRS dynamic and simple graph of contacts
    
   
class NoGraphSEIRShModel(NoSEIRSModel):
    def __init__ (self, graph, T_time=TIME_OF_SIMULATION, number_of_people=NUMBER_OF_PEOPLE, 
                 number_of_infected=NUMBER_OF_INFECTED, prob=PROB_FOR_DETERMINISTIC, 
                 beta=BETA, sigma=SIGMA, gamma=GAMMA, xi=XI, mju_i=MJU_I, 
                 random_seed=MAGIC_SEED_BY_PETRA):


        if random_seed:
            random.seed(random_seed)

        self.G = graph

        self.N = number_of_people
        self.Ni = number_of_infected
        self.p = prob
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.xi = xi
        self.mju_i = mju_i
        self.T = T_time
        self.People = []
        self.t = 0
        
        inf_idx = random.sample(range(self.N), self.Ni)
        
        for p in range(self.N):
#            get data from nodes
            new_person = SEIRSPerson(p)
            if p in inf_idx:
                new_person.set_state(INFECTIOUS)
            else:
                new_person.set_state(SUSCEPTIBLE)
            self.People.append(new_person)
        self.p = PROB_FOR_GRAPH


# compute S->E probabilty by estimate and sampling
    def s_to_e_estimate(self, p):
        distant_part = self.Ni / self.N
        if self.G.degree(p.id) == 0:
            close_part = 0
        else:
            nbrs  =  [ n for n in self.G[p.id]  ] 
            inf_nbrs = [ n for n in nbrs if p.state == INFECTIOUS ]
            if len(inf_nbrs) == 0:
                close_part = 0
            else: 
                s = 0
                for i in inf_nbrs:
                    s += self.G.edges[(p.id,i)]['weight']
                close_part = s / len(inf_nbrs)    

        return self.beta * (self.p * distant_part + (1-self.p ) * close_part)


    def run_iteration(self):
#        print ('This is run_iteration: ', self.t)
        for p in self.People:
# set probabilites of transmision from p.state to a new p.state
            probs = [0, 0, 0, 0, 0]
            if p.state == SUSCEPTIBLE:
#                print('S->E: ', self.s_to_e_sampling(p), self.s_to_e_estimate(p))
                probs[EXPOSED] = self.s_to_e_estimate(p)
            elif p.state == EXPOSED:
                probs[INFECTIOUS] = self.sigma
            elif p.state == INFECTIOUS:
                probs[RECOVERED] = self.gamma
                probs[FATAL] = self.mju_i
            elif p.state == RECOVERED:
                probs[SUSCEPTIBLE] = self.xi
            p.prob_change_state(probs)    

    def do_statistics(self):
        print(self.t, end = ' ')
        stat = [0, 0, 0, 0, 0]
        for e in self.People:
            stat[e.state] += 1
        for i in stat:
            print (i, end = ' ')
        print()
# this is ugly!
        self.Ni = stat[INFECTIOUS]

    def run(self):
        self.do_statistics()
        for self.t in range(1, self.T + 1):
            self.do_statistics()
            self.run_iteration()
             

if __name__ == "__main__":
#    m = NoSEIRSModel(100, 100, 30)
#    m.run()

    N_ppl = 100000
    T_iter = 300
    N_inf = N_ppl // 100
    
    print('Doing the graph', N_ppl) 
    gg =  RandomSingleGraphGenerator(N_ppl)
    g = gg.as_one_graph()
    print('Doing the model')       
    m = NoGraphSEIRShModel(g, T_time=T_iter, number_of_people=N_ppl, number_of_infected=N_inf)
    print('Running')        
    time1 = time.time()
    m.run()
    e = int(time.time() - time1)
    print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), '  Bye')        