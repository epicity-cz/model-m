# auxiliary graph for testing
# not supported at the moment

# from graphviz import Graph
from graph_gen import GraphGenerator
import networkx as nx
import scipy.stats as stats
import pandas as pd


class RomeoAndJuliet(GraphGenerator):
    layer_probs = [1] * 15
    layer_names = ['not in Verona', 'family', 'not in Verona', 'not in Verona',
                   'not in Verona', 'not in Verona', 'not in Verona', 'not in Verona',
                   'not in Verona', 'not in Verona', 'friends', 'work', 'work to clients', 'public transport', 'shops and events']

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.G.graph['layer_probs'] = self.layer_probs

        # One-man chorus
        self.G.add_node(0, label='Chorus', sex=0, age=61)
        # Romeo and Juliet
        self.G.add_node(1, label='Romeo', sex=0, age=18)
        self.G.add_node(2, label='Juliet', sex=1, age=13)
        self.G.add_edge(1, 2, type=1)

        # House of Montague
        self.G.add_node(3, label='Lord Montague', sex=0, age=58)
        self.G.add_node(4, label='Lady Montague', sex=1, age=40)
        self.G.add_edge(3, 4, type=1)
        self.G.add_edge(3, 1, type=1)
        self.G.add_edge(4, 1, type=1)
        self.G.add_edge(3, 4, type=10)
        self.G.add_edge(3, 1, type=10)
        self.G.add_edge(4, 1, type=10)
        self.G.add_node(5, label='Benvolio', sex=0, age=17)
        self.G.add_edge(1, 5, type=1)
        self.G.add_edge(3, 5, type=1)
        self.G.add_edge(4, 5, type=1)

        # House of Capulet
        self.G.add_node(6, label='Lord Capulet', sex=0, age=50)
        self.G.add_node(7, label='Lady Capulet', sex=1, age=35)
        self.G.add_edge(2, 6, type=1)
        self.G.add_edge(2, 7, type=1)
        self.G.add_edge(6, 7, type=1)
        self.G.add_edge(2, 6, type=10)
        self.G.add_edge(2, 7, type=10)
        self.G.add_edge(6, 7, type=10)
        self.G.add_node(8, label='Tybalt', sex=0, age=17)
        self.G.add_edge(8, 2, type=1)
        self.G.add_edge(8, 6, type=1)
        self.G.add_edge(8, 7, type=1)
        self.G.add_edge(8, 2, type=10)
        self.G.add_edge(8, 6, type=10)
        self.G.add_edge(8, 7, type=10)

        # House of Prince of Verona
        self.G.add_node(9, label='Prince Escalus', sex=0, age=60)
        self.G.add_node(10, label='Paris', sex=0, age=24)
        self.G.add_node(11, label='Mercutio', sex=0, age=20)
        self.G.add_edge(9, 10, type=1)
        self.G.add_edge(9, 10, type=10)
        self.G.add_edge(9, 11, type=1)
        self.G.add_edge(10, 11, type=1)
        self.G.add_edge(1, 11, type=10)

        # Servants
        self.G.add_node(12, label='Nurse', sex=1, age=28)
        self.G.add_edge(2, 12, type=11)
        self.G.add_edge(2, 12, type=10)
        self.G.add_edge(2, 6, type=10)
        self.G.add_edge(2, 7, type=10)
        self.G.add_edge(2, 8, type=10)
        self.G.add_node(13, label='Peter', sex=0, age=60)
        self.G.add_edge(13, 12, type=11)
        self.G.add_edge(13, 12, type=10)
        self.G.add_edge(13, 6, type=10)
        self.G.add_edge(13, 7, type=10)
        self.G.add_edge(13, 1, type=10)
        self.G.add_edge(13, 8, type=10)
        self.G.add_node(14, label='Balthasar', sex=0, age=23)
        self.G.add_edge(1, 14, type=11)
        self.G.add_edge(1, 14, type=10)
        self.G.add_edge(1, 14, type=10)
        self.G.add_edge(14, 3, type=10)
        self.G.add_edge(14, 4, type=10)
        self.G.add_node(15, label='Abram', sex=0, age=68)
        self.G.add_edge(3, 15, type=11)
        self.G.add_edge(4, 15, type=11)
        self.G.add_edge(3, 15, type=10)
        self.G.add_edge(4, 15, type=10)
        self.G.add_edge(1, 15, type=10)
        self.G.add_edge(14, 15, type=10)
        self.G.add_node(16, label='Gregory', sex=0, age=34)
        self.G.add_node(17, label='Sampson', sex=0, age=36)
        self.G.add_edge(6, 16, type=11)
        self.G.add_edge(7, 16, type=11)
        self.G.add_edge(8, 16, type=11)
        self.G.add_edge(8, 17, type=11)
        self.G.add_edge(6, 17, type=11)
        self.G.add_edge(7, 17, type=11)
        self.G.add_edge(8, 16, type=10)
        self.G.add_edge(8, 17, type=10)
        self.G.add_node(18, label='Page', sex=0, age=11)
        self.G.add_edge(18, 10, type=11)
        self.G.add_edge(18, 10, type=10)

        # Friars and Merchants
        self.G.add_node(19, label='Friar Lawrence', sex=0, age=68)
        self.G.add_node(20, label='Friar John', sex=0, age=42)
        self.G.add_edge(19, 20, type=11)
        self.G.add_edge(19, 1, type=12)
        self.G.add_edge(19, 2, type=12)
        self.G.add_edge(19, 20, type=12)
        self.G.add_node(21, label='Apothacary', sex=0, age=75)
        self.G.add_edge(21, 1, type=14)
        self.G.add_edge(21, 2, type=14)

        # Former love interest of Romeo
        self.G.add_node(22, label='Rosaline', sex=1, age=16)
        self.G.add_edge(1, 22, type=10)

        # Fairy Quenn Mab visits Romeo in a dream
        self.G.add_node(23, label='Queen Mab', sex=1, age=20)
        self.G.add_edge(1, 23, type=13)

        # self.Grandpa Capulet
        self.G.add_node(24, label='Old Capulet', sex=0, age=82)
        self.G.add_edge(24, 2, type=1)
        self.G.add_edge(24, 6, type=1)
        self.G.add_edge(24, 7, type=1)
        self.G.add_edge(24, 8, type=1)

        # Capulets Servants
        self.G.add_node(25, label='Anthony', sex=0, age=38)
        self.G.add_node(26, label='Potpan', sex=0, age=35)
        self.G.add_node(27, label='Servant 1', sex=0, age=49)
        self.G.add_node(28, label='Servant 2', sex=0, age=31)
        self.G.add_edge(6, 25, type=11)
        self.G.add_edge(6, 26, type=11)
        self.G.add_edge(6, 27, type=11)
        self.G.add_edge(6, 28, type=11)
        self.G.add_edge(7, 25, type=11)
        self.G.add_edge(7, 26, type=11)
        self.G.add_edge(7, 27, type=11)
        self.G.add_edge(7, 28, type=11)

        # Petruchio is a ghost at Capulet party
        self.G.add_node(29, label='Ghost Petruchio', sex=0, age=27)
        self.G.add_edge(2, 29, type=13)

        # Valentine is Mercutio brother at a party
        self.G.add_node(30, label='Valentine', sex=0, age=26)
        self.G.add_edge(9, 30, type=1)
        self.G.add_edge(10, 30, type=1)
        self.G.add_edge(11, 30, type=1)

        # Watchmen at a fight
        self.G.add_node(31, label='Watchmen 1', sex=0, age=29)
        self.G.add_node(32, label='Watchmen 2', sex=0, age=25)
        self.G.add_node(33, label='Watchmen 3', sex=0, age=30)
        self.G.add_edge(31, 32, type=11)
        self.G.add_edge(31, 33, type=11)
        self.G.add_edge(33, 32, type=11)

        # Musicians at a party
        self.G.add_node(34, label='Musician 1', sex=0, age=39)
        self.G.add_node(35, label='Musician 2', sex=0, age=31)
        self.G.add_node(36, label='Musician 3', sex=0, age=49)
        self.G.add_edge(34, 35, type=11)
        self.G.add_edge(34, 36, type=11)
        self.G.add_edge(35, 36, type=11)

        # party at Capulet - is now a complete subgraph
        self.G.add_edge(1, 2, type=14)
        self.G.add_edge(1, 5, type=14)
        self.G.add_edge(1, 6, type=14)
        self.G.add_edge(1, 7, type=14)
        self.G.add_edge(1, 8, type=14)
        self.G.add_edge(1, 9, type=14)
        self.G.add_edge(1, 10, type=14)
        self.G.add_edge(1, 11, type=14)
        self.G.add_edge(1, 16, type=14)
        self.G.add_edge(1, 17, type=14)
        self.G.add_edge(1, 22, type=14)
        self.G.add_edge(1, 25, type=14)
        self.G.add_edge(1, 26, type=14)
        self.G.add_edge(1, 27, type=14)
        self.G.add_edge(1, 28, type=14)
        self.G.add_edge(1, 29, type=14)
        self.G.add_edge(1, 34, type=14)
        self.G.add_edge(1, 35, type=14)
        self.G.add_edge(1, 36, type=14)

        self.G.add_edge(2, 5, type=14)
        self.G.add_edge(2, 6, type=14)
        self.G.add_edge(2, 7, type=14)
        self.G.add_edge(2, 8, type=14)
        self.G.add_edge(2, 9, type=14)
        self.G.add_edge(2, 10, type=14)
        self.G.add_edge(2, 11, type=14)
        self.G.add_edge(2, 16, type=14)
        self.G.add_edge(2, 17, type=14)
        self.G.add_edge(2, 22, type=14)
        self.G.add_edge(2, 25, type=14)
        self.G.add_edge(2, 26, type=14)
        self.G.add_edge(2, 27, type=14)
        self.G.add_edge(2, 28, type=14)
        self.G.add_edge(2, 29, type=14)
        self.G.add_edge(2, 34, type=14)
        self.G.add_edge(2, 35, type=14)
        self.G.add_edge(2, 36, type=14)

        self.G.add_edge(5, 6, type=14)
        self.G.add_edge(5, 7, type=14)
        self.G.add_edge(5, 8, type=14)
        self.G.add_edge(5, 9, type=14)
        self.G.add_edge(5, 10, type=14)
        self.G.add_edge(5, 11, type=14)
        self.G.add_edge(5, 16, type=14)
        self.G.add_edge(5, 17, type=14)
        self.G.add_edge(5, 22, type=14)
        self.G.add_edge(5, 25, type=14)
        self.G.add_edge(5, 26, type=14)
        self.G.add_edge(5, 27, type=14)
        self.G.add_edge(5, 28, type=14)
        self.G.add_edge(5, 29, type=14)
        self.G.add_edge(5, 34, type=14)
        self.G.add_edge(5, 35, type=14)
        self.G.add_edge(5, 36, type=14)

        self.G.add_edge(6, 7, type=14)
        self.G.add_edge(6, 8, type=14)
        self.G.add_edge(6, 8, type=14)
        self.G.add_edge(6, 10, type=14)
        self.G.add_edge(6, 11, type=14)
        self.G.add_edge(6, 16, type=14)
        self.G.add_edge(6, 17, type=14)
        self.G.add_edge(6, 22, type=14)
        self.G.add_edge(6, 25, type=14)
        self.G.add_edge(6, 26, type=14)
        self.G.add_edge(6, 27, type=14)
        self.G.add_edge(6, 28, type=14)
        self.G.add_edge(6, 29, type=14)
        self.G.add_edge(6, 34, type=14)
        self.G.add_edge(6, 35, type=14)
        self.G.add_edge(6, 36, type=14)

        self.G.add_edge(7, 8, type=14)
        self.G.add_edge(7, 9, type=14)
        self.G.add_edge(7, 10, type=14)
        self.G.add_edge(7, 11, type=14)
        self.G.add_edge(7, 16, type=14)
        self.G.add_edge(7, 17, type=14)
        self.G.add_edge(7, 22, type=14)
        self.G.add_edge(7, 25, type=14)
        self.G.add_edge(7, 26, type=14)
        self.G.add_edge(7, 27, type=14)
        self.G.add_edge(7, 28, type=14)
        self.G.add_edge(7, 29, type=14)
        self.G.add_edge(7, 34, type=14)
        self.G.add_edge(7, 35, type=14)
        self.G.add_edge(7, 36, type=14)

        self.G.add_edge(8, 9, type=14)
        self.G.add_edge(8, 10, type=14)
        self.G.add_edge(8, 11, type=14)
        self.G.add_edge(8, 16, type=14)
        self.G.add_edge(8, 17, type=14)
        self.G.add_edge(8, 22, type=14)
        self.G.add_edge(8, 25, type=14)
        self.G.add_edge(8, 26, type=14)
        self.G.add_edge(8, 27, type=14)
        self.G.add_edge(8, 28, type=14)
        self.G.add_edge(8, 29, type=14)
        self.G.add_edge(8, 34, type=14)
        self.G.add_edge(8, 35, type=14)
        self.G.add_edge(8, 36, type=14)

        self.G.add_edge(9, 10, type=14)
        self.G.add_edge(9, 11, type=14)
        self.G.add_edge(9, 16, type=14)
        self.G.add_edge(9, 17, type=14)
        self.G.add_edge(9, 22, type=14)
        self.G.add_edge(9, 25, type=14)
        self.G.add_edge(9, 26, type=14)
        self.G.add_edge(9, 27, type=14)
        self.G.add_edge(9, 28, type=14)
        self.G.add_edge(9, 29, type=14)
        self.G.add_edge(9, 34, type=14)
        self.G.add_edge(9, 35, type=14)
        self.G.add_edge(9, 36, type=14)

        self.G.add_edge(10, 11, type=14)
        self.G.add_edge(10, 16, type=14)
        self.G.add_edge(10, 17, type=14)
        self.G.add_edge(10, 22, type=14)
        self.G.add_edge(10, 25, type=14)
        self.G.add_edge(10, 26, type=14)
        self.G.add_edge(10, 27, type=14)
        self.G.add_edge(10, 28, type=14)
        self.G.add_edge(10, 29, type=14)
        self.G.add_edge(10, 34, type=14)
        self.G.add_edge(10, 35, type=14)
        self.G.add_edge(10, 36, type=14)

        self.G.add_edge(11, 16, type=14)
        self.G.add_edge(11, 17, type=14)
        self.G.add_edge(11, 22, type=14)
        self.G.add_edge(11, 25, type=14)
        self.G.add_edge(11, 26, type=14)
        self.G.add_edge(11, 27, type=14)
        self.G.add_edge(11, 28, type=14)
        self.G.add_edge(11, 29, type=14)
        self.G.add_edge(11, 34, type=14)
        self.G.add_edge(11, 35, type=14)
        self.G.add_edge(11, 36, type=14)

        self.G.add_edge(16, 17, type=14)
        self.G.add_edge(16, 22, type=14)
        self.G.add_edge(16, 25, type=14)
        self.G.add_edge(16, 26, type=14)
        self.G.add_edge(16, 27, type=14)
        self.G.add_edge(16, 28, type=14)
        self.G.add_edge(16, 29, type=14)
        self.G.add_edge(16, 34, type=14)
        self.G.add_edge(16, 35, type=14)
        self.G.add_edge(16, 36, type=14)

        self.G.add_edge(17, 22, type=14)
        self.G.add_edge(17, 25, type=14)
        self.G.add_edge(17, 26, type=14)
        self.G.add_edge(17, 27, type=14)
        self.G.add_edge(17, 28, type=14)
        self.G.add_edge(17, 29, type=14)
        self.G.add_edge(17, 34, type=14)
        self.G.add_edge(17, 35, type=14)
        self.G.add_edge(17, 36, type=14)

        self.G.add_edge(22, 25, type=14)
        self.G.add_edge(22, 26, type=14)
        self.G.add_edge(22, 27, type=14)
        self.G.add_edge(22, 28, type=14)
        self.G.add_edge(22, 29, type=14)
        self.G.add_edge(22, 34, type=14)
        self.G.add_edge(22, 35, type=14)
        self.G.add_edge(22, 36, type=14)

        self.G.add_edge(25, 26, type=14)
        self.G.add_edge(25, 27, type=14)
        self.G.add_edge(25, 28, type=14)
        self.G.add_edge(25, 29, type=14)
        self.G.add_edge(25, 34, type=14)
        self.G.add_edge(25, 35, type=14)
        self.G.add_edge(25, 36, type=14)

        self.G.add_edge(26, 27, type=14)
        self.G.add_edge(26, 28, type=14)
        self.G.add_edge(26, 29, type=14)
        self.G.add_edge(26, 34, type=14)
        self.G.add_edge(26, 35, type=14)
        self.G.add_edge(26, 36, type=14)

        self.G.add_edge(27, 28, type=14)
        self.G.add_edge(27, 29, type=14)
        self.G.add_edge(27, 34, type=14)
        self.G.add_edge(27, 35, type=14)
        self.G.add_edge(27, 36, type=14)

        self.G.add_edge(28, 29, type=14)
        self.G.add_edge(28, 34, type=14)
        self.G.add_edge(28, 35, type=14)
        self.G.add_edge(28, 36, type=14)

        self.G.add_edge(29, 34, type=14)
        self.G.add_edge(29, 35, type=14)
        self.G.add_edge(29, 36, type=14)

        self.G.add_edge(34, 35, type=14)
        self.G.add_edge(34, 36, type=14)

        self.G.add_edge(35, 36, type=14)

        # weights from trunc norm for now
        lower, upper = 0, 1
        mu, sigma = 0.7, 0.3
        for (u, v, k, d) in self.G.edges(data=True, keys=True):
            self.G[u][v][k]['weight'] = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    def write_to_csv(self, prefix='raj'):
        df = nx.to_pandas_edgelist(self.G)
        df.columns = ['vertex1', 'vertex2', 'type', 'weight']
        df.to_csv(prefix+'-edges.csv', index=False)
#        nx.write_edgelist(self.G, e_name, data = ['type', 'weight'])
        n = list(self.G.nodes(data=True))
   #     ids, labels, sexes, ages = [ i[0], i[1]['label'], i[1]['sex'], i[1]['age'] for  = i in list(self.G.nodes(data=True)) ]
        nn = [(i[0], i[1]['label'],  i[1]['sex'], i[1]['age']) for i in n]
        df = pd.DataFrame(nn)
        df.columns = ['type', 'label', 'sex', 'age']
        df.to_csv(prefix+'-nodes.csv', index=False)
        nnn = [(i, self.layer_names[i], self.layer_probs[i])
               for i in range(len(self.layer_names))]
        df = pd.DataFrame(nnn)
        df.columns = ['id', 'name', 'weight']
        df.to_csv(prefix+'-etypes.csv', index=False)
