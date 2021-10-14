import torch
import numpy as np
import itertools
from abc import ABCMeta, abstractmethod


class ParameterizedSCM():
    __metaclass__ = ABCMeta
    def __init__(self, adj):
        alpha = 'XYZWABCDEFGHIJKLMNOPQRSTUV'
        self.i2n = lambda x: alpha[x]
        self.V = []
        self.S = {}
        self.U = {}
        self.graph = {}
        for V in range(len(adj)):
            pa_V = list(np.where(adj[:,V])[0]) # assumes binary adjacency with row causes column
            self.graph.update({V: pa_V})
            V_name = self.i2n(V)
            self.V.append(V_name)
            U_V = lambda bs: torch.rand(bs,1) # uniform [0,1)
                    # torch.bernoulli(.25 * torch.ones(bs,1)) # bernoulli {0,1}
            self.U.update({V_name: U_V})
        self.topologicalSort()

    def print_graph(self):
        print('The NCM models the following graph:')
        for k in self.graph:
            print(f'{[self.i2n(x) for x in self.graph[k]]} --> {self.i2n(k)}')

    def indices_to_names(self, indices):
        return [self.i2n(x) for x in indices]

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * len(self.V)
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(len(self.V)):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        self.topology = list(reversed(stack))

    def compute_marginals(self, samples, doX=-1, Xi=-1, debug=False):
        pred_marginals = {}
        N = len(self.V)
        for ind_d in range(N):
            vals = []
            for val in [0, 1]:
                domains = [[0, 1]] * (N - 1)
                domains.insert(ind_d, [val])
                combinations = np.stack([x for x in itertools.product(*domains)])
                p_comb = []
                for ind, c in enumerate(combinations):
                    # print(f'{ind}:\t{c}')
                    c = torch.tensor(c,dtype=torch.float).unsqueeze(0)
                    pC = self.forward(c, torch.tensor([doX]*samples).unsqueeze(1), Xi, samples, debug)
                    # print(f'mean(p(c)) = {pC}')
                    p_comb.append(pC)
                # print(f'Sum = {sum(p_comb)}\t [{p_comb}]')
                vals.append(sum(p_comb).item())
            pred_marginals.update({ind_d: vals})
        # print(f'Marginals =\n\t{pred_marginals}')
        if debug:
            import pdb; pdb.set_trace()
        return pred_marginals

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def forward(self, v, doX, Xi, samples, debug=False):
        pass