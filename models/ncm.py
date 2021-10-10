import torch
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()

        # bias = False
        # modules = [
        #     torch.nn.Linear(input_size, hidden_size, bias=bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_size, hidden_size, bias=bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_size, output_size, bias=bias),
        # ]
        # self.net = torch.nn.Sequential(*modules)

        # define a simple MLP neural net
        self.net = []
        hs = [input_size] + hidden_sizes + [output_size]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        print(f'Layer Sizes: {str([l.weight.shape if isinstance(l, torch.nn.Linear) else None for l in self.net])}')
        self.net = nn.Sequential(*self.net)


    def forward(self, input):
        return self.net(input)

# class NCMChain():
#     def __init__(self, adj):
#         alpha = 'XYZW'
#         self.i2n = lambda x: alpha[x]
#         self.V = []
#         self.S = {}
#         self.U = {}
#         for V in range(len(adj)):
#             pa_V = list(np.where(adj[:,V])[0]) # assumes binary adjacency with row causes column
#             V_name = self.i2n(V)
#             self.V.append(V_name)
#             print(f'Variable {V_name} := ')
#             self.S.update({V_name:
#                                MLP(len(pa_V)+1, 1, 5)
#                            })
#             U_V = lambda bs: torch.rand(bs,1) # uniform [0,1)
#             self.U.update({V_name: U_V})
#
#     def params(self):
#         return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]
#
#     def sampling(self, n_samples):
#         # TODO: generalize this to automatically follow the topology
#         order = ['X', 'Y', 'Z', 'W']
#
#         x = self.S['X'](self.U['X'](n_samples))
#         y = self.S['Y'](torch.cat((x,self.U['Y'](n_samples)),axis=1))
#         z = self.S['Z'](torch.cat((y,self.U['Z'](n_samples)),axis=1))
#         w = self.S['W'](torch.cat((z,self.U['W'](n_samples)),axis=1))
#         return torch.cat((x,y,z,w),axis=1)
#
#     def forward(self, v, samples=1000, doX=None, debug=False):
#         # TODO: generalize this to automatically follow the topology
#         bs, ds = v.shape
#
#         if doX is not None:
#             pV0 = torch.tile(doX,(samples,1))
#         else:
#             pV0 = torch.sigmoid(self.S['X'](self.U['X'](samples))) # X
#             pV0 = pV0 * v[:,0].unsqueeze(1) + (torch.ones_like(v[:,0])-v[:,0]).unsqueeze(1) * (1-pV0)
#
#         if doX is None or v[:,0] == doX: # consistency assumption cannot have p(..., x=0 | do(x=1)) since that is zero
#
#             pV1 = torch.sigmoid(self.S['Y'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),self.U['Y'](samples)),axis=1))) # Y
#             pV1 = pV1 * v[:,1].unsqueeze(1) + (torch.ones_like(v[:,1])-v[:,1]).unsqueeze(1) * (1-pV1)
#
#             pV2 = torch.sigmoid(self.S['Z'](torch.cat((torch.tile(v[:,1].unsqueeze(1),(samples,1)),self.U['Z'](samples)),axis=1))) # Z
#             pV2 = pV2 * v[:,2].unsqueeze(1) + (torch.ones_like(v[:,2])-v[:,2]).unsqueeze(1) * (1-pV2)
#
#             pV3 = torch.sigmoid(self.S['W'](torch.cat((torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['W'](samples)),axis=1))) # W
#             pV3 = pV3 * v[:,3].unsqueeze(1) + (torch.ones_like(v[:,3])-v[:,3]).unsqueeze(1) * (1-pV3)
#
#             if doX is not None:
#                 pV = torch.cat((pV1, pV2, pV3), axis=1)
#             else:
#                 pV = torch.cat((pV0, pV1, pV2, pV3),axis=1)
#
#             agg = lambda t: torch.mean(torch.prod(t,axis=1))
#
#             if debug:
#                 import pdb; pdb.set_trace()
#
#             return agg(pV)
#
#         else:
#             if debug:
#                 import pdb; pdb.set_trace()
#
#             return torch.zeros((1,))
#
#     def compute_marginals(self, samples=1000, doX=None, debug=False):
#         pred_marginals = {}
#         N = len(self.V)
#         for ind_d in range(N):
#             vals = []
#             for val in [0, 1]:
#                 domains = [[0, 1]] * (N - 1)
#                 domains.insert(ind_d, [val])
#                 combinations = np.stack([x for x in itertools.product(*domains)])
#                 p_comb = []
#                 for ind, c in enumerate(combinations):
#                     # print(f'{ind}:\t{c}')
#                     c = torch.tensor(c,dtype=torch.float).unsqueeze(0)
#                     pC = self.forward(c, samples, doX, debug)
#                     # print(f'mean(p(c)) = {pC}')
#                     p_comb.append(pC)
#                 # print(f'Sum = {sum(p_comb)}\t [{p_comb}]')
#                 vals.append(sum(p_comb).item())
#             pred_marginals.update({ind_d: vals})
#         # print(f'Marginals =\n\t{pred_marginals}')
#         if debug:
#             import pdb; pdb.set_trace()
#         return pred_marginals


class NCMConfounder():
    def __init__(self, adj):
        alpha = 'XYZW'
        self.i2n = lambda x: alpha[x]
        self.V = []
        self.S = {}
        self.U = {}
        for V in range(len(adj)):
            pa_V = list(np.where(adj[:,V])[0]) # assumes binary adjacency with row causes column
            V_name = self.i2n(V)
            self.V.append(V_name)
            print(f'Variable {V_name} := ')
            self.S.update({V_name:
                               MLP(len(pa_V)+1, 1, [10, 10, 10])
                               #MADE(len(pa_V) + 1, [10, 10, 10], len(pa_V) + 1)
                           })
            U_V = lambda bs: torch.rand(bs,1) # uniform [0,1)
                    # torch.bernoulli(.25 * torch.ones(bs,1)) # bernoulli {0,1}
            self.U.update({V_name: U_V})

    def params(self):
        return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]

    # def sampling(self, n_samples):
    #     # TODO: generalize this to automatically follow the topology
    #     order = ['X', 'Y', 'Z', 'W']
    #
    #     x = self.S['X'](self.U['X'](n_samples))
    #     y = self.S['Y'](torch.cat((x,self.U['Y'](n_samples)),axis=1))
    #     z = self.S['Z'](torch.cat((y,self.U['Z'](n_samples)),axis=1))
    #     w = self.S['W'](torch.cat((z,self.U['W'](n_samples)),axis=1))
    #     return torch.cat((x,y,z,w),axis=1)

    def forward(self, v, doX, samples, debug=False):
        # TODO: generalize this to automatically follow the topology
        bs, ds = v.shape

        # eine Perle der Codegeschichte - consistency check pure torch
        Consistency = torch.ones((samples, 1))
        Consistency = torch.where(doX >= 0,
                                  torch.where(torch.tile(v[:,0].unsqueeze(1),(samples,1)) == doX, 1., 0.), Consistency)

        pV2 = torch.sigmoid(self.S['Z'](self.U['Z'](samples))) # Z
        pV2 = pV2 * v[:,2].unsqueeze(1) + (torch.ones_like(v[:,2])-v[:,2]).unsqueeze(1) * (1-pV2)

        # because of torch.where, doX = -1 means no intervention
        pV0 = torch.where(doX >= 0, torch.ones((samples, 1)), torch.sigmoid(self.S['X'](torch.cat((torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['X'](samples)),axis=1))))
        pV0 = torch.where(doX >= 0, pV0, pV0 * v[:, 0].unsqueeze(1) + (torch.ones_like(v[:, 0]) - v[:, 0]).unsqueeze(1) * (1 - pV0))

        pV3 = torch.sigmoid(self.S['W'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),self.U['W'](samples)),axis=1))) # W
        pV3 = pV3 * v[:,3].unsqueeze(1) + (torch.ones_like(v[:,3])-v[:,3]).unsqueeze(1) * (1-pV3)

        pV1 = torch.sigmoid(self.S['Y'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['Y'](samples)),axis=1))) # Y
        pV1 = pV1 * v[:,1].unsqueeze(1) + (torch.ones_like(v[:,1])-v[:,1]).unsqueeze(1) * (1-pV1)

        if any(torch.isnan(pV0)) or any(torch.isnan(pV1)) or any(torch.isnan(pV2)) or any(torch.isnan(pV3)):
            import pdb; pdb.set_trace()

        pV = torch.cat((Consistency, pV0, pV1, pV2, pV3),axis=1)

        agg = lambda t: torch.mean(torch.prod(t,axis=1))

        if debug:#all(doX != -1*torch.ones_like(doX)):#debug:
            import pdb; pdb.set_trace()

        ret = agg(pV)

        if torch.isnan(ret):
            import pdb; pdb.set_trace()

        return ret

    def compute_marginals(self, samples, doX=-1, debug=False):
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
                    pC = self.forward(c, torch.tensor([doX]*samples).unsqueeze(1), samples, debug)
                    # print(f'mean(p(c)) = {pC}')
                    p_comb.append(pC)
                # print(f'Sum = {sum(p_comb)}\t [{p_comb}]')
                vals.append(sum(p_comb).item())
            pred_marginals.update({ind_d: vals})
        # print(f'Marginals =\n\t{pred_marginals}')
        if debug:
            import pdb; pdb.set_trace()
        return pred_marginals
