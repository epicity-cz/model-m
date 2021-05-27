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
TIME_OF_SIMULATION = 30
NUMBER_OF_PEOPLE = 10
NUMBER_OF_INFECTED = 3

# constants for Person
HEALTHY = 0
INFECTED = 1
TIME_OF_INFECTION = 3
TRANS_RATE = 0.8


# constants for SEIRPerson
SUSCEPTIBLE = 0
EXPOSED = 1
INFECTIOUS = 2
RECOVERED = 3
FATAL = 4

SEIRSNAMES = [ 'S', 'E', 'I', 'R', 'F' ]

# β: rate of transmission (transmissions per S-I contact per time)
# σ: rate of progression (inverse of incubation period)
# γ: rate of recovery (inverse of infectious period)
# ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
# μI: rate of mortality from the disease (deaths per infectious individual per time)
BETA = 0.355
SIGMA = 1/5.2
GAMMA = 1/12.39
XI = 0
MJU_I = 0.0004
MJU_ALL = 0.00003 # 11 ppl per 1000 ppl per year ... 3 ppl per day per 100k ppl

# probability of distant contact, 1 means only distant contacts, 0 means only close contacts
# in deterministic model, there are only distant contacts
# the ration of p / (1-p) tells the distribution of contacts between distant and close
# average number of contacts is per day, actual values are sampled from normal distribution (truncated at LTRUNC and HTRUNC)

PROB_FOR_DETERMINISTIC = 1  
PROB_FOR_GRAPH = 0.5
AVG_CONTACTS = 30
VAR_CONTACTS = 10
LTRUNC = 0
HTRUNC = 1000

class Person():
    def __init__ (self, id, init_state=HEALTHY):
        self.state = init_state
        self.time_of_state = 0
        self.id = id

    def infect(self, by_whom):
        self.state = INFECTED
        self.time_of_state = 0
        if by_whom == -1:
            print(self.id, " infected initially")
        else:
            print(self.id, " infected by", by_whom)

    def heal(self):
        self.state = HEALTHY
        self.time_of_state = 0
        print(self.id, " is healed")

    def stay_infected(self):
        self.time_of_state += 1
        if self.time_of_state == TIME_OF_INFECTION:
            self.heal()

    def stay_healthy(self):
        self.time_of_state += 1

         
class SEIRSPerson(Person):
    def __init__ (self, id, init_state=SUSCEPTIBLE):
        self.state = init_state
        self.time_of_state = 0
        self.id = id
        
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
        total_p = cum_p[-1]
        value = random.random() * total_p
        new_state = bisect_left(cum_p, value)
        if new_state != self.state :
            self.set_state(new_state)


class NoModel(BaseEngine):

    def __init__(self, T_time=TIME_OF_SIMULATION, number_of_people=NUMBER_OF_PEOPLE, number_of_infected=NUMBER_OF_INFECTED, avg_contacts=AVG_CONTACTS, avg_trans=TRANS_RATE,
                 random_seed=42):

        if random_seed:
            random.seed(random_seed)

        self.N = number_of_people
        self.Ni = number_of_infected
        self.contacts_per_day = avg_contacts
        self.transmission_rate = avg_trans
        self.T = T_time


        self.People = []
        inf_idx = random.sample(range(self.N), self.Ni)
        for p in range(self.N):
            new_person = Person(p)
            if p in inf_idx:
                new_person.infect(-1)
            else:
                new_person.heal()
            self.People.append(new_person)

    def is_it_transmission(self, a, b):
        # ignore everything, just toss a coin with transmission_rate if b is infected
        if b.state == HEALTHY:
            return False
        else:
            if random.random() < self.transmission_rate:
                return True
            else:
                return False
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
            
    def run_iteration(self):
        for p in self.People:
            if p.state == HEALTHY:
                contacts = random.sample(range(self.N), self.contacts_per_day)
                for c in contacts:
                    if self.is_it_transmission(p, self.People[c]):
                        p.infect(c)
            if p.state == INFECTED:
                p.stay_infected()

    def run(self):
        self.do_statistics()
        for self.t in range(1, self.T + 1):
            self.do_statistics()
            self.run_iteration()

# model working with SEIRS dynamic and deterministic sampling from the whole population

class NoSEIRSModel(NoModel):
    def __init__(self, T_time=TIME_OF_SIMULATION, number_of_people=NUMBER_OF_PEOPLE, 
                 number_of_infected=NUMBER_OF_INFECTED, prob=PROB_FOR_DETERMINISTIC, 
                 beta=BETA, sigma=SIGMA, gamma=GAMMA, xi=XI, mju_i=MJU_I, 
                 random_seed=MAGIC_SEED_BY_PETRA):

        if random_seed:
            random.seed(random_seed)

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
            new_person = SEIRSPerson(p)
            if p in inf_idx:
                new_person.set_state(INFECTIOUS)
            else:
                new_person.set_state(SUSCEPTIBLE)
            self.People.append(new_person)
 
    def run_iteration(self):
        inf_p = [ x for x in self.People if x.get_state() == INFECTIOUS ]
        self.Ni = len(inf_p)
        
        for p in self.People:
            probs = [0, 0, 0, 0, 0]
            if p.state == SUSCEPTIBLE:
                probs[EXPOSED] = self.beta * self.Ni / self.N
#                print (self.beta * self.Ni / self.N)
            if p.state == EXPOSED:
                probs[INFECTIOUS] = self.sigma
            if p.state == INFECTIOUS:
                probs[RECOVERED] = self.gamma
                probs[FATAL] = self.mju_i
            if p.state == RECOVERED:
                probs[SUSCEPTIBLE] = self.xi
            p.prob_change_state(probs)    
    
# model working with SEIRS dynamic and simple graph of contacts
    
def dirty_choice2(limit, number, exclude):
    ll = list(range(limit))
    ll.remove(exclude)
    return(random.choices(ll,number))

def dirty_choice(limit, number, exclude):
    ll = []
    for i in range(number):
        x = int(limit * random.random())
        if x == exclude:
# this is dirty, IMPROVE!
            x = int(limit * random.random())
        ll.append(x)    
    return (ll)
    
class NoGraphSEIRShModel(NoSEIRSModel):
    def __init__ (self, graph, **kwargs):
        super().__init__(**kwargs)
        self.G = graph
#        print('This is ngsm Init')
        # THIS IS HACk, CHNGE!!!        
        self.p = PROB_FOR_GRAPH

    def rand_no_of_contacts(self):
 # random           
        x = stats.truncnorm.rvs((LTRUNC - AVG_CONTACTS) / VAR_CONTACTS, 
                               (HTRUNC - AVG_CONTACTS) / VAR_CONTACTS, 
                               loc=AVG_CONTACTS, scale=VAR_CONTACTS)
        return(int(round(x)))

# compute S->E probabilty by sampling the graph 
    def s_to_e_sampling(self, p):
        num_of_contacts = self.rand_no_of_contacts()
        num_of_distant_contacts = int(round(num_of_contacts * self.p))
        num_of_close_contacts = num_of_contacts - num_of_distant_contacts                
#                print ('MEANWHILE IN THE S=>E', p.id, num_of_contacts, num_of_distant_contacts, num_of_close_contacts)               
# sample close contacts in a proper way   
        nbrs  =  [ n for n in self.G[p.id] ] 
        if nbrs == [] :
            close_contacts = []
        else : 
            close_contacts = random.choices(nbrs, k=num_of_close_contacts)
# sample distant contacts in a dirty way   
        distant_contacts = dirty_choice(self.N, num_of_distant_contacts, p.id)                            
# computing probability  of infection by 1 - product of (1-beta) for sampled contacts
        prod = 1;                
        for contact in close_contacts + distant_contacts:
            if self.People[contact].state == INFECTIOUS:
                prod *= (1 - self.beta)                
        return 1- prod

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