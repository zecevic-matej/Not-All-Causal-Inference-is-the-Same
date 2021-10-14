import torch
from .parameterizedSCM import ParameterizedSCM


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
        #print(f'Layer Sizes: {str([l.weight.shape if isinstance(l, torch.nn.Linear) else None for l in self.net])}')
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class NCM(ParameterizedSCM):
    """
    Classical NCM, own implementation following the Xia et al. 2021 paper.
    """

    def __init__(self, adj, scale=False):
        super(NCM, self).__init__(adj)
        for V in self.graph:
            V_name = self.i2n(V)
            pa_V = self.graph[V]
            #print(f'Variable {V_name} := ')
            hs = [10,10,10]#[2*scale for _ in range(3)] #[10*int(len(pa_V)+1) for _ in range(3)] if scale else [10,10,10]
            self.S.update({V_name:
                               MLP(len(pa_V)+1, 1, hs)
                           })

    def params(self):
        return [item for sublist in [list(f.parameters()) for f in self.S.values()] for item in sublist]

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