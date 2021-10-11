import torch
import numpy as np
import itertools


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()

        # define a simple MLP neural net
        self.net = []
        hs = [input_size] + hidden_sizes + [output_size]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        print(f'Layer Sizes: {str([l.weight.shape if isinstance(l, torch.nn.Linear) else None for l in self.net])}')
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class NCM():
    def __init__(self, adj):
        alpha = 'XYZW'
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
            print(f'Variable {V_name} := ')
            self.S.update({V_name:
                               MLP(len(pa_V)+1, 1, [10, 10, 10])
                           })
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

    def params(self):
        return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]

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

    def forward(self, v, doX, Xi, samples, debug=False):

        # eine Perle der Codegeschichte - consistency check pure torch
        Consistency = torch.ones((samples, 1))
        Consistency = torch.where(doX >= 0,
                                  torch.where(torch.tile(v[:,0].unsqueeze(1),(samples,1)) == doX, 1., 0.), Consistency)

        pVs = []
        for V in self.topology:

            pa_V = self.graph[V]

            V_arg = torch.cat((*[torch.tile(v[:,pa].unsqueeze(1),(samples,1)) for pa in pa_V],self.U[self.i2n(V)](samples)),axis=1)

            pV = torch.sigmoid(self.S[self.i2n(V)](V_arg))
            pV = pV * v[:, V].unsqueeze(1) + (torch.ones_like(v[:, V]) - v[:, V]).unsqueeze(1) * (1 - pV)

            # the intervention checking might do an extra run for the intervened node, not rly important tho
            # furthermore, it is not validated yet - and rn we mainly check for interventions on X, where we change
            # X,Y relationships
            pV = torch.where(torch.tensor(Xi) == torch.tensor(V), torch.where(doX >= 0, torch.ones((samples, 1)), pV), pV)

            pVs.append(pV)

        pV = torch.cat((Consistency, *pVs),axis=1)

        agg = lambda t: torch.mean(torch.prod(t,axis=1))

        if debug:#all(doX != -1*torch.ones_like(doX)):#debug:
            import pdb; pdb.set_trace()

        ret = agg(pV)

        if torch.isnan(ret):
            import pdb; pdb.set_trace()

        return ret

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
