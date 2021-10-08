import torch
import numpy as np
import itertools

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        bias = False
        modules = [
            torch.nn.Linear(input_size, hidden_size, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=bias),
        ]
        self.net = torch.nn.Sequential(*modules)

        print(f'Layer Sizes: {str([l.weight.shape if isinstance(l, torch.nn.Linear) else None for l in modules])}')

    def forward(self, input):
        return self.net(input)

class NCMChain():
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
            self.S.update({V_name: MLP(len(pa_V)+1, 1, 5)})
            U_V = lambda bs: torch.rand(bs,1) # uniform [0,1)
            self.U.update({V_name: U_V})

    def params(self):
        return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]

    def sampling(self, n_samples):
        # TODO: generalize this to automatically follow the topology
        order = ['X', 'Y', 'Z', 'W']

        x = self.S['X'](self.U['X'](n_samples))
        y = self.S['Y'](torch.cat((x,self.U['Y'](n_samples)),axis=1))
        z = self.S['Z'](torch.cat((y,self.U['Z'](n_samples)),axis=1))
        w = self.S['W'](torch.cat((z,self.U['W'](n_samples)),axis=1))
        return torch.cat((x,y,z,w),axis=1)

    def forward(self, v, samples=1000, doX=None, debug=False):
        # TODO: generalize this to automatically follow the topology
        bs, ds = v.shape

        if doX is not None:
            pV0 = torch.tile(doX,(samples,1))
        else:
            pV0 = torch.sigmoid(self.S['X'](self.U['X'](samples))) # X
            pV0 = pV0 * v[:,0].unsqueeze(1) + (torch.ones_like(v[:,0])-v[:,0]).unsqueeze(1) * (1-pV0)

        if doX is None or v[:,0] == doX: # consistency assumption cannot have p(..., x=0 | do(x=1)) since that is zero

            pV1 = torch.sigmoid(self.S['Y'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),self.U['Y'](samples)),axis=1))) # Y
            pV1 = pV1 * v[:,1].unsqueeze(1) + (torch.ones_like(v[:,1])-v[:,1]).unsqueeze(1) * (1-pV1)

            pV2 = torch.sigmoid(self.S['Z'](torch.cat((torch.tile(v[:,1].unsqueeze(1),(samples,1)),self.U['Z'](samples)),axis=1))) # Z
            pV2 = pV2 * v[:,2].unsqueeze(1) + (torch.ones_like(v[:,2])-v[:,2]).unsqueeze(1) * (1-pV2)

            pV3 = torch.sigmoid(self.S['W'](torch.cat((torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['W'](samples)),axis=1))) # W
            pV3 = pV3 * v[:,3].unsqueeze(1) + (torch.ones_like(v[:,3])-v[:,3]).unsqueeze(1) * (1-pV3)

            if doX is not None:
                pV = torch.cat((pV1, pV2, pV3), axis=1)
            else:
                pV = torch.cat((pV0, pV1, pV2, pV3),axis=1)

            agg = lambda t: torch.mean(torch.prod(t,axis=1))

            if debug:
                import pdb; pdb.set_trace()

            return agg(pV)

        else:
            if debug:
                import pdb; pdb.set_trace()

            return torch.zeros((1,))

    def compute_marginals(self, samples=1000, doX=None, debug=False):
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
                    pC = self.forward(c, samples, doX, debug)
                    # print(f'mean(p(c)) = {pC}')
                    p_comb.append(pC)
                # print(f'Sum = {sum(p_comb)}\t [{p_comb}]')
                vals.append(sum(p_comb).item())
            pred_marginals.update({ind_d: vals})
        # print(f'Marginals =\n\t{pred_marginals}')
        if debug:
            import pdb; pdb.set_trace()
        return pred_marginals


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
            self.S.update({V_name: MLP(len(pa_V)+1, 1, 5)})
            U_V = lambda bs: torch.rand(bs,1) # uniform [0,1)
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

    def forward(self, v, samples=1000, doX=None, debug=False):
        # TODO: generalize this to automatically follow the topology
        bs, ds = v.shape

        if doX is None or v[:,0] == doX: # consistency assumption cannot have p(..., x=0 | do(x=1)) since that is zero

            pV2 = torch.sigmoid(self.S['Z'](self.U['Z'](samples))) # Z
            pV2 = pV2 * v[:,2].unsqueeze(1) + (torch.ones_like(v[:,2])-v[:,2]).unsqueeze(1) * (1-pV2)

            if doX is not None:
                pV0 = torch.tile(doX, (samples, 1))
            else:
                pV0 = torch.sigmoid(self.S['X'](torch.cat((torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['X'](samples)),axis=1)))  # X
                pV0 = pV0 * v[:, 0].unsqueeze(1) + (torch.ones_like(v[:, 0]) - v[:, 0]).unsqueeze(1) * (1 - pV0)

            pV3 = torch.sigmoid(self.S['W'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),self.U['W'](samples)),axis=1))) # W
            pV3 = pV3 * v[:,3].unsqueeze(1) + (torch.ones_like(v[:,3])-v[:,3]).unsqueeze(1) * (1-pV3)

            pV1 = torch.sigmoid(self.S['Y'](torch.cat((torch.tile(v[:,0].unsqueeze(1),(samples,1)),torch.tile(v[:,2].unsqueeze(1),(samples,1)),self.U['Y'](samples)),axis=1))) # Y
            pV1 = pV1 * v[:,1].unsqueeze(1) + (torch.ones_like(v[:,1])-v[:,1]).unsqueeze(1) * (1-pV1)

            if doX is not None:
                pV = torch.cat((pV1, pV2, pV3), axis=1)
            else:
                pV = torch.cat((pV0, pV1, pV2, pV3),axis=1)

            agg = lambda t: torch.mean(torch.prod(t,axis=1))

            if debug:
                import pdb; pdb.set_trace()

            return agg(pV)

        else:
            if debug:
                import pdb; pdb.set_trace()

            return torch.zeros((1,))

    def compute_marginals(self, samples=1000, doX=None, debug=False):
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
                    pC = self.forward(c, samples, doX, debug)
                    # print(f'mean(p(c)) = {pC}')
                    p_comb.append(pC)
                # print(f'Sum = {sum(p_comb)}\t [{p_comb}]')
                vals.append(sum(p_comb).item())
            pred_marginals.update({ind_d: vals})
        # print(f'Marginals =\n\t{pred_marginals}')
        if debug:
            import pdb; pdb.set_trace()
        return pred_marginals
