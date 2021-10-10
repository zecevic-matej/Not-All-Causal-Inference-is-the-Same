import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class SCM():
    __metaclass__ = ABCMeta
    def __init__(self, U_params=None):
        if U_params is None:
            U_params = np.round(np.random.uniform(0.1, 0.9, 4),decimals=1)
        print(f'U params, [U_X, U_Y, U_Z, U_W]={U_params}')
        self.U_X = lambda s: np.random.binomial(1,U_params[0],s)
        self.U_Y = lambda s: np.random.binomial(1,U_params[1],s)
        self.U_Z = lambda s: np.random.binomial(1,U_params[2],s)
        self.U_W = lambda s: np.random.binomial(1,U_params[3],s)
        self.X = self.Y = self.Z = self.W = None
        self.x = self.y = self.z = self.w = None
        self.l1 = self.l2 = None
        self.U_params = U_params

    @abstractmethod
    def topological_computation(self, n_samples, doX=None):
        ''' To override, return df with x,y,z,w'''
        pass

    def sample(self, n_samples, doX=None):
        df = self.topological_computation(n_samples, doX)
        if doX is not None:
            self.l2 = df
            print(f'Generated Interventional Data (L2) with {n_samples} samples where do(X={doX}).')
        else:
            self.l1 = df
            print(f'Generated Observational Data (L1) with {n_samples} samples.')

    def ate(self, n_samples, debug=False):
        # p(Y=1|do(X=1)) - p(Y=1|do(X=0))
        self.sample(n_samples, doX=np.ones_like(n_samples))
        l2x1 = self.l2
        self.sample(n_samples, doX=np.zeros_like(n_samples))
        l2x0 = self.l2
        w = 1/n_samples
        ate = l2x1['y'].sum()*w - l2x0['y'].sum()*w
        print(f'Computed empirical ATE p(Y=1|do(X=1)) - p(Y=1|do(X=0)) = {ate:.2f} for {n_samples} samples.')
        self.l2 = None
        if debug:
            import pdb; pdb.set_trace()
        return ate, l2x1, l2x0

class ChainSCM(SCM):
    # X -> Y -> Z -> W
    def __init__(self, U_params=None):
        super(ChainSCM, self).__init__(U_params)
        self.name = "ChainSCM"
        self.X = lambda u: u
        self.Y = lambda u, x: np.logical_and(u, x).astype(np.int64)
        self.Z = lambda u, y: np.logical_and(u, y).astype(np.int64)
        self.W = lambda u, z: np.logical_and(u, z).astype(np.int64)
        self.adj = np.array([[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 0]],dtype=float)

    def topological_computation(self, n_samples, doX=None):
        if doX is not None:
            self.x = doX
        else:
            self.x = self.X(self.U_X(n_samples))
        self.y = self.Y(self.U_Y(n_samples), self.x)
        self.z = self.Z(self.U_Z(n_samples), self.y)
        self.w = self.W(self.U_W(n_samples), self.z)
        df = pd.DataFrame.from_dict({'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})
        return df

class ColliderSCM(SCM):
    # Y -> Z <- X <- W
    def __init__(self, U_params=None):
        super(ColliderSCM, self).__init__(U_params)
        self.name = "ColliderSCM"
        self.X = lambda u, w: np.logical_and(u, w).astype(np.int64)
        self.Y = lambda u: u
        self.Z = lambda u, y, x: np.logical_or(np.logical_and(u, y), x).astype(np.int64)
        self.W = lambda u: u
        self.adj = np.array([[0, 0, 1, 0],[0, 0, 1, 0],[0, 0, 0, 0],[1, 0, 0, 0]],dtype=float)

    def topological_computation(self, n_samples, doX=None):
        self.w = self.W(self.U_W(n_samples))
        if doX is not None:
            self.x = doX
        else:
            self.x = self.X(self.U_X(n_samples), self.w)
        self.y = self.Y(self.U_Y(n_samples))
        self.z = self.Z(self.U_Z(n_samples), self.y, self.x)
        df = pd.DataFrame.from_dict({'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})
        return df

class ConfounderSCM(SCM):
    #  Y <- Z -> X -> {W,Y}
    def __init__(self, U_params=None):
        super(ConfounderSCM, self).__init__(U_params)
        self.name = "ConfounderSCM"
        self.X = lambda u, z: np.logical_xor(u, z).astype(np.int64) # this right one violates positivity assumption, since Z=1 implies X=1 ALWAYS! np.logical_or(u, z).astype(np.int64)
        self.Y = lambda u, z, x: np.logical_xor(np.logical_and(u, x), np.logical_and(u, z)).astype(np.int64)
        self.Z = lambda u: u
        self.W = lambda u, x: np.logical_and(u, x).astype(np.int64)
        self.adj = np.array([[0, 1, 0, 1],[0, 0, 0, 0],[1, 1, 0, 0],[0, 0, 0, 0]],dtype=float)

    def topological_computation(self, n_samples, doX=None):
        self.z = self.Z(self.U_Z(n_samples))
        if doX is not None:
            self.x = doX
        else:
            self.x = self.X(self.U_X(n_samples), self.z)
        self.y = self.Y(self.U_Y(n_samples), self.z, self.x)
        self.w = self.W(self.U_W(n_samples), self.x)
        df = pd.DataFrame.from_dict({'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})
        return df

class BackdoorSCM(SCM):
    # Y -> Z <- X -> W -> Y
    def __init__(self, U_params=None):
        super(BackdoorSCM, self).__init__(U_params)
        self.name = "BackdoorSCM"
        self.X = lambda u: u
        self.Y = lambda u, w: np.logical_xor(u, w).astype(np.int64)
        self.Z = lambda u, y, x: np.logical_or(np.logical_and(u, y), x).astype(np.int64)
        self.W = lambda u, x: np.logical_or(u, x).astype(np.int64)
        self.adj = np.array([[0, 0, 1, 1],[0, 0, 1, 0],[0, 0, 0, 0],[0, 1, 0, 0]],dtype=float)

    def topological_computation(self, n_samples, doX=None):
        if doX is not None:
            self.x = doX
        else:
            self.x = self.X(self.U_X(n_samples))
        self.w = self.W(self.U_W(n_samples), self.x)
        self.y = self.Y(self.U_Y(n_samples), self.w)
        self.z = self.Z(self.U_Z(n_samples), self.y, self.x)
        df = pd.DataFrame.from_dict({'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})
        return df