import torch, itertools
import numpy as np
from spn.algorithms.layerwise.layers import Product, Sum
from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.algorithms.layerwise.utils import SamplingContext
from .parameterizedSCM import ParameterizedSCM
#from models.EinsumNetwork import Graph, EinsumNetwork

class SPN(torch.nn.Module):
    # layered sum-product network
    def __init__(self, D, C, K, C2=None):
        super().__init__()

        assert D%K == 0
        self.C2 = C2

        # Normal leaf layer, output shape: [N=?, D=2, C=5, R=1]
        self.leaf1 = RatNormal(in_features=D, out_channels=C)

        # Product layer, output shape: [N=?, D=1, C=5, R=1]
        self.p1 = Product(in_features=D, cardinality=K)

        # Sum layer, root node, output shape:  [N=?, D=1, C=1, R=1]
        self.s1 = Sum(in_channels=C, in_features=int(D/K), out_channels=1)

        if C2 is not None:
            self.s1 = Sum(in_channels=C, in_features=int(D/K), out_channels=C2)
            self.leaf2 = RatNormal(in_features=D, out_channels=C)
            self.p2 = Product(in_features=D, cardinality=K)
            self.s2 = Sum(in_channels=C, in_features=int(D/K), out_channels=C2)
            self.pc = Product(in_features=2, cardinality=2)
            self.sc = Sum(in_channels=C2, in_features=1, out_channels=1)

        #print(f'Layer Sizes: {str([self.leaf, self.p, self.s])}')

    def forward(self, x):
        # Forward bottom up
        if self.C2 is None:
            x = self.leaf1(x)
            x = self.p1(x)
            xc = self.s1(x)
        else:
            x1 = self.leaf1(x)
            x2 = self.leaf2(x)
            x1 = self.p1(x1)
            x2 = self.p2(x2)
            x1 = self.s1(x1)
            x2 = self.s2(x2)
            xc = torch.cat((x1,x2), axis=1)
            xc = self.pc(xc)
            xc = self.sc(xc)

        return xc

    def sample(self, n=100):
        # Sample top down
        ctx = self.s.sample(n=n, context=SamplingContext(n=n))
        ctx = self.p.sample(context=ctx)
        samples = self.leaf.sample(context=ctx)
        return samples

class TNCM(ParameterizedSCM):
    """
    Tractable NCM, based on SPN for tractable (fast/efficient) inference.
    """

    def __init__(self, adj, spn_type="EinSum", scale=False):
        super(TNCM, self).__init__(adj)
        self.spn_type = spn_type
        for V in self.graph:
            V_name = self.i2n(V)
            pa_V = self.graph[V]
            #print(f'Variable {V_name} := ')

            # if self.spn_type == "EinSum":
            #     einet = EinsumNetwork.EinsumNetwork(
            #         Graph.random_binary_trees(num_var=len(pa_V) + 2, depth=1,
            #                                   num_repetitions=3),
            #         EinsumNetwork.Args(
            #             num_classes=1,
            #             num_input_distributions=5,
            #             exponential_family=EinsumNetwork.CategoricalArray,
            #             exponential_family_args={'K': 2},
            #             num_sums=2,
            #             num_var=len(pa_V) + 2,
            #             online_em_frequency=1,
            #             online_em_stepsize=0.05)
            #     )
            #     einet.initialize()
            #     model = einet#lambda x: EinsumNetwork.log_likelihoods(einet.forward(x))
            # else:
            C = 30#len(pa_V)+1+scale #int((len(pa_V)+1)+10) if scale else 30
            model = SPN(D=len(pa_V)+1, C=C, K=len(pa_V)+1)

            self.S.update({V_name: model})

    def params(self):
        return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]

    def forward(self, v, doX, Xi, samples, debug=False):

        # eine Perle der Codegeschichte - consistency check pure torch
        Consistency = torch.ones((samples, 1))
        Consistency = torch.where(doX >= 0,
                                  torch.where(v[:,0] == doX, 1., 0.), Consistency)

        pVs = []
        for V in self.topology:

            pa_V = self.graph[V]

            V_arg = torch.cat((*[torch.tile(v[:,pa].unsqueeze(1),(samples,1)) for pa in pa_V],self.U[self.i2n(V)](samples)),axis=1)

            pV = torch.minimum(torch.exp(self.S[self.i2n(V)](V_arg)).reshape(samples, 1), torch.ones(samples, 1))
            pV = pV * v[:, V].unsqueeze(1) + (torch.ones_like(v[:, V]) - v[:, V]).unsqueeze(1) * (1 - pV)

            # debug, if values are not bounded
            # if any(pV > 1):
            #     import pdb; pdb.set_trace()

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
                    #c = torch.tile(c, (samples, 1))
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