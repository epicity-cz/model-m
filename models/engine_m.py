import pandas as pd
import numpy as np
import numpy_indexed as npi
import scipy as scipy
import scipy.integrate
import networkx as nx
import time
from operator import itemgetter

from history_utils import TimeSeries, TransitionHistory
from engine_sequential import SequentialEngine
# from extended_network_model import STATES as s
from sparse_utils import prop_of_row


class STATES():
    # TODO get rid of this
    S = 0
    S_s = 1
    E = 2
    I_n = 3
    I_a = 4
    I_s = 5
    I_ds = 6
    J_s = 11
    J_n = 12
    E_d = 13
    I_da = 14
    I_dn = 15
    J_ds = 16
    J_dn = 17
    R_d = 7
    R_u = 8
    D_d = 9
    D_u = 10

    pass


def _searchsorted2d(a, b):
    m, n = a.shape
    max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * np.arange(a.shape[0])[:, None]
    p = np.searchsorted((a+r).ravel(), (b+r).ravel())
    return p - n*np.arange(m)


class EngineM(SequentialEngine):
    """ be the final one for model-m
        + operates on multigraph
        + iterates in days
    """

        

    def update_graph(self, new_G):
        """ create adjacency matrix for G """
        self.G = new_G  # just for backward compability, TODO: remove G from everywhere and replace by graph
        self.graph = new_G
        self.num_nodes = self.graph.num_nodes
#        print(f"DBD: graph udpate {self.graph}")

    def node_degrees(self, Amat):
        raise NotImplementedError("We use the graph directly, not matrix.")

    def prob_of_contact(self, source_states, source_candidate_states, dest_states, dest_candidate_states, beta, beta_in_family):

        assert type(dest_states) == list and type(source_states) == list

        # 1. select active edges
        # candidates for active edges are edges between source_candidate_states and dest_candidate_states
        # source (the one who is going to be infected)
        # dest (the one who can offer infection)

        s = time.time()
        source_candidate_flags = self.memberships[source_candidate_states, :, :].reshape(
            len(source_candidate_states), self.num_nodes).sum(axis=0)
        # source_candidate_indices = source_candidate_flags.nonzero()[0]

        dest_candidate_flags = self.memberships[dest_candidate_states, :, :].reshape(
            len(dest_candidate_states), self.num_nodes).sum(axis=0)
        # dest_candidate_indices = dest_candidate_flags.nonzero()[0]
        e = time.time()
        print("Create flags", e-s)

        s = time.time()
        possibly_active_edges, possibly_active_edges_dirs = self.graph.get_edges(
            source_candidate_flags,
            dest_candidate_flags
        )
        num_possibly_active_edges = len(possibly_active_edges)
        e = time.time()
        print("Select possibly active", e-s)

        if self.t == 6:
            print("DBG SUPERSPREAD possibly active edges ", num_possibly_active_edges)

        s = time.time()
        if num_possibly_active_edges == 0:
            return np.zeros((self.num_nodes, 1))

        # for each possibly active edge flip coin
        r = np.random.rand(num_possibly_active_edges)
        e = time.time()
        print("Random numbers: ", e-s)
        s = time.time()
        # edges probs
        p = self.graph.get_edges_probs(possibly_active_edges)
        e = time.time()
        print("Get intensities:", e-s)

        s = time.time()
        active_indices = (r < p).nonzero()[0]
        num_active_edges = len(active_indices)
        if num_active_edges == 0:
            return np.zeros((self.num_nodes, 1))
            
        if self.t == 6:
            print("DBG SUPERSPREAD really active edges ", num_active_edges)


        # select active edges
        # if num_active_edges == 1:
        #     active_edges = [possibly_active_edges[active_indices[0]]]
        #     active_edges_dirs = [possibly_active_edges_dirs[active_indices[0]]]
        # else:
        #     active_edges = list(itemgetter(*active_indices)
        #                         (possibly_active_edges))
        #     active_edges_dirs = list(itemgetter(
        #         *active_indices)(possibly_active_edges_dirs))
        active_edges = possibly_active_edges[active_indices]
        active_edges_dirs = possibly_active_edges_dirs[active_indices]
        e = time.time()
        print("Select active edges:", e-s)

        s = time.time()
        # get source and dest nodes for active edges
        # source and dest met today, dest is possibly infectious, source was possibly infected
        source_nodes, dest_nodes = self.graph.get_edges_nodes(
            active_edges, active_edges_dirs)
        # add to contact_history (infectious node goes first!!!)
        contact_indices = list(zip(dest_nodes, source_nodes, active_edges))
        self.contact_history.append(contact_indices)

        
        if self.t == 6:
            contact_numbers = {} 
            for e in active_edges:
                layer_number = self.graph.get_layer_for_edge(e)
                contact_numbers[layer_number] = contact_numbers.get(layer_number, 0) + 1
            print("DBG contact numbers ", 
                  ", ".join([f"{layer_type}:{number}" for layer_type, number in contact_numbers.items()])
            ) 

        # print("Potkali se u kolina:", contact_indices)
        print("Todays contact num:",  len(contact_indices))
        e = time.time()
        print("Archive active edges:", e-s)

        # restrict the selection to only relevant states
        # (ie. candidates can be both E and I, relevant are only I)
        # candidates are those who will be possibly relevant in future
        s = time.time()
        dest_flags = self.memberships[dest_states, :, :].reshape(
            len(dest_states), self.num_nodes).sum(axis=0)
        source_flags = self.memberships[source_states, :, :].reshape(
            len(source_states), self.num_nodes).sum(axis=0)

        relevant_edges, relevant_edges_dirs = self.graph.get_edges(
            source_flags, dest_flags)
        if len(relevant_edges) == 0:
            return np.zeros((self.num_nodes, 1))

        # get intersection
        active_relevant_edges = np.intersect1d(active_edges, relevant_edges)

        if len(active_relevant_edges) == 0:
            return np.zeros((self.num_nodes, 1))

        # print(active_relevant_edges.shape)
        # print(relevant_edges.shape)
        # print((active_relevant_edges[:, np.newaxis] == relevant_edges).shape)
        try:
            # this causes exceptin, but only sometimes ???
            # where_indices = (
            #    active_relevant_edges[:, np.newaxis] == relevant_edges).nonzero()[1]
            # lets try npi instead
            where_indices = npi.indices(relevant_edges, active_relevant_edges)
        except AttributeError as e:
            print(e)
            print("Lucky we are ...")
            print(active_relevant_edges.shape)
            np.save("active_relevant_edges", active_relevant_edges)
            print(relevant_edges.shape)
            np.save("relevant_edges", relevant_edges)
            print(active_relevant_edges[:, np.newaxis] == relevant_edges)
            exit()
        #        print(where_indices, len(where_indices))
        # always one index! (sources and dest must be disjunct)
        active_relevant_edges_dirs = relevant_edges_dirs[where_indices]
        e = time.time()
        print("Get relevant active edges:", e-s)

        s = time.time()
        intensities = self.graph.get_edges_intensities(
            active_relevant_edges).reshape(-1, 1)
        relevant_sources, relevant_dests = self.graph.get_edges_nodes(
            active_relevant_edges, active_relevant_edges_dirs)

        is_family_edge = self.graph.is_family_edge(
            active_relevant_edges).reshape(-1, 1)
        """
        pro experiment s hospodama jenom: 
        """
        ## TODO list of no-masks layers into config
        # get rid of this mess
        #        if self.t <= 111:
        if True:
            is_class_edge = self.graph.is_class_edge(
                active_relevant_edges).reshape(-1, 1)
            is_pub_edge = self.graph.is_pub_edge(
                active_relevant_edges).reshape(-1, 1)
            is_super_edge = self.graph.is_super_edge(
                active_relevant_edges).reshape(-1, 1)
            is_family_edge = np.logical_or.reduce((is_family_edge, 
                                                   is_class_edge,
                                                   is_super_edge,
                                                   is_pub_edge))
        """
        else:
            is_class_edge = self.graph.is_class_edge(
                active_relevant_edges).reshape(-1, 1)
            is_family_edge = np.logical_or(is_family_edge, is_class_edge)
        """

        #        assert len(relevant_sources) == len(set(relevant_sources))
        # TOD beta - b_ij
        # new beta depands on the one who is going to be infected
        #        b_intensities = beta[relevant_sources]
        #        b_f_intensities = beta_in_family[relevant_sources]

        # reduce asymptomatic
        is_A = (
            self.memberships[STATES.I_a][relevant_dests] +
            self.memberships[STATES.I_n][relevant_dests] +
            self.memberships[STATES.I_dn][relevant_dests] +
            self.memberships[STATES.I_da][relevant_dests]
        )
        b_original_intensities = (
            beta_in_family[relevant_sources] * (1 - is_A) +
            self.beta_A_in_family[relevant_sources] * is_A
        )
        b_reduced_intensities = (
            beta[relevant_sources] * (1 - is_A) +
            self.beta_A[relevant_sources] * is_A
        )

        b_intensities = (
            b_original_intensities * is_family_edge +
            b_reduced_intensities * (1 - is_family_edge)
        )

        # print(beta[relevant_sources].shape)
        # print(is_family_edge)
        # print(b_intensities.shape)
        # exit()

        assert b_intensities.shape == intensities.shape
        # print(b_intensitites.shape, intensities.shape,
        #      prob_of_no_infection.shape)
        relevant_sources_unique, unique_indices = np.unique(
            relevant_sources, return_inverse=True)
        no_infection = (1 - b_intensities * intensities).ravel()

        res = np.ones(len(relevant_sources_unique), dtype='float32')
        for i in range(len(unique_indices)):
            res[unique_indices[i]] = res[unique_indices[i]]*no_infection[i]
        prob_of_no_infection = res
        # prob_of_no_infection2 = np.fromiter((np.prod(no_infection, where=(relevant_sources==v).T)
        #                         for v in relevant_sources_unique), dtype='float32')

        result = np.zeros(self.num_nodes)
        result[relevant_sources_unique] = 1 - prob_of_no_infection
        e = time.time()
        print("Comp probability", e-s)
        return result.reshape(self.num_nodes, 1)
