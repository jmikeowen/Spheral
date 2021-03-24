from math import *
from NodeGeneratorBase import *
from Spheral import Vector1d, Tensor1d, SymTensor1d

#-------------------------------------------------------------------------------
# Provide a 1D generator consistent with the 2D/3D conventions in
# GenerateNodeDistribution{2d,3d}.py
#-------------------------------------------------------------------------------
class GenerateNodeDistribution1d(NodeGeneratorBase):

    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    def __init__(self, n, rho,
                 xmin,
                 xmax,
                 nNodePerh = 2.01,
                 offset = 0.0,
                 rejecter = None):
        assert xmax > xmin
        assert n > 0

        # If the user provided a constant for rho, then use the constantRho
        # class to provide this value.
        if type(rho) == type(1.0):
            self.rho = ConstantRho(rho)
        else:
            self.rho = rho

        # Generate the basic state
        dx = (xmax - xmin)/n
        assert dx > 0.0
        h0 = nNodePerh*dx
        self.x = [xmin + (i + 0.5)*dx for i in xrange(n)]
        self.m = [self.rho(xi)*dx for xi in self.x]
        self.H = [SymTensor1d(1.0/h0) for i in xrange(n)]
        
        # Apply any rejection
        if rejecter:
            self.x, self.y, self.m, self.H = rejecter(self.x,
                                                      self.y,
                                                      self.m,
                                                      self.H)

        # Have the base class break up the serial node distribution
        # for parallel cases.
        NodeGeneratorBase.__init__(self, True,
                                   self.x, self.m, self.H)

        return

    #---------------------------------------------------------------------------
    # Get the position for the given node index.
    #---------------------------------------------------------------------------
    def localPosition(self, i):
        assert i >= 0 and i < len(self.x)
        return Vector1d(self.x[i])

    #---------------------------------------------------------------------------
    # Get the mass for the given node index.
    #---------------------------------------------------------------------------
    def localMass(self, i):
        assert i >= 0 and i < len(self.m)
        return self.m[i]

    #---------------------------------------------------------------------------
    # Get the mass density for the given node index.
    #---------------------------------------------------------------------------
    def localMassDensity(self, i):
        assert i >= 0 and i < len(self.x)
        return self.rho(self.localPosition(i))

    #---------------------------------------------------------------------------
    # Get the H tensor for the given node index.
    #---------------------------------------------------------------------------
    def localHtensor(self, i):
        assert i >= 0 and i < len(self.H)
        return self.H[i]

