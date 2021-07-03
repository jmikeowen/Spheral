from math import *

from NodeGeneratorBase import *

from Spheral import Vector1d, Vector2d, Vector3d, Tensor1d, Tensor2d, \
    SymTensor1d, SymTensor2d, CylindricalBoundary, rotationMatrix2d, testPointInBox2d
from SpheralTestUtilities import fuzzyEqual

#-------------------------------------------------------------------------------
# Estimate the mass of a volume given the bounding coordinates
#-------------------------------------------------------------------------------
def computeRhoAndMass1d(p0, p1, pi, rhofunc):
    rho0 = rhofunc(p0)
    rho1 = rhofunc(p1)
    rhoi = rhofunc(pi)
    V1 = (p1 - pi).magnitude()
    V2 = (p0 - pi).magnitude()
    Vi = V1 + V2
    mi = 0.5*(V1 * (rhoi + rho0)
              + V2 * (rhoi + rho1))
    rhoi = mi / Vi
    return rhoi, mi
    
def computeRhoAndMass2d(p00, p10, p11, p01, pi, rhofunc):
    rho00 = rhofunc(p00)
    rho10 = rhofunc(p10)
    rho11 = rhofunc(p11)
    rho01 = rhofunc(p01)
    rhoi = rhofunc(pi)
    V1 = (p00 - pi).cross(p10 - pi).magnitude()  # actually 2x
    V2 = (p10 - pi).cross(p11 - pi).magnitude()
    V3 = (p11 - pi).cross(p01 - pi).magnitude()
    V4 = (p01 - pi).cross(p00 - pi).magnitude()
    Vi = 0.5*(V1 + V2 + V3 + V4)
    mi = (V1 * (rhoi + rho00 + rho10) +
          V2 * (rhoi + rho10 + rho11) +
          V3 * (rhoi + rho11 + rho01) +
          V4 * (rhoi + rho01 + rho00))/6.0
    rhoi = mi/Vi
    stuff = (rho00, rho10, rho11, rho01, rhofunc(pi))
    rhomin = min(stuff)
    rhomax = max(stuff)
    return rhoi, mi

def computeRhoAndMass3d(p000, p100, p110, p010, p001, p101, p111, p011, pi, rhofunc):
    # base square
    # then roof square
    '''
    110 -- 010
    |       |
    100 -- 000

    111 -- 011
    |       |
    101 -- 001
    '''
    p = [p000, p100, p110, p010, p001, p101, p111, p011, pi]
    rhos = []
    for x in p:
        rhos.append(rhofunc(x))
    base = []
    base.append((p[1]-p[0]).cross(p[3]-p[0]))
    base.append((p[4]-p[0]).cross(p[1]-p[0]))
    base.append((p[5]-p[4]).cross(p[7]-p[4]))
    base.append((p[3]-p[2]).cross(p[6]-p[2]))
    base.append((p[2]-p[1]).cross(p[6]-p[2]))
    base.append((p[4]-p[0]).cross(p[3]-p[0]))
    bm = []
    for b in base:
        bm.append(b.magnitude())
    height = []
    height.append(abs((pi-p[0]).dot(base[0]/bm[0])))
    height.append(abs((pi-p[0]).dot(base[1]/bm[1])))
    height.append(abs((pi-p[4]).dot(base[2]/bm[2])))
    height.append(abs((pi-p[3]).dot(base[3]/bm[3])))
    height.append(abs((pi-p[1]).dot(base[4]/bm[4])))
    height.append(abs((pi-p[0]).dot(base[5]/bm[5])))
    vs = []
    for i in range(len(bm)):
        vs.append(1.0/3.0*bm[i]*height[i])
    mi = (vs[0]*(rhos[0]+rhos[1]+rhos[2]+rhos[3]+rhos[8])+
          vs[1]*(rhos[0]+rhos[1]+rhos[4]+rhos[5]+rhos[8])+
          vs[2]*(rhos[4]+rhos[5]+rhos[6]+rhos[7]+rhos[8])+
          vs[3]*(rhos[2]+rhos[3]+rhos[6]+rhos[7]+rhos[8])+
          vs[4]*(rhos[1]+rhos[2]+rhos[5]+rhos[6]+rhos[8])+
          vs[5]*(rhos[0]+rhos[3]+rhos[4]+rhos[7]+rhos[8]))/5.0
    vi = 0
    for v in vs:
        vi += v
    rhoi = mi/vi
    rhomin = min(rhos)
    rhomax = max(rhos)
    return rhoi, mi

#-------------------------------------------------------------------------------
# Specialized node generator to generate nodes ratioing from a 1D slab surface.
#-------------------------------------------------------------------------------
class GenerateRatioSlab1d(NodeGeneratorBase):
    
    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    def __init__(self,
                 dxSurface, xratio,
                 rho,
                 xmin,
                 xmax,
                 nNodePerh = 2.01,
                 SPH = False,
                 flipx = False):
        
        assert dxSurface > 0.0
        assert xratio > 0.0
        assert xmin < xmax
        assert nNodePerh > 0.0
        
        # If the user provided a constant for rho, then use the constantRho
        # class to provide this value.
        if type(rho) == type(1.0):
            self.rhofunc = ConstantRho(rho)
        else:
            self.rhofunc = rho

        self.x, self.m, self.H, self.rho = [], [], [], []

        # Decide the actual ratios we're going to use to arrive at an integer number of radial bins.
        def adjustRatio(drStart, drRatio, rmin, rmax):
            if abs(drRatio - 1.0) > 1e-4:
                neff = max(1, int(log(1.0 - (rmax - rmin)*(1.0 - drRatio)/drStart)/log(drRatio) + 0.5))
                drStart = (rmax - rmin)*(1.0 - drRatio)/(1.0 - drRatio**neff)
            else:
                neff = max(1, int((rmax - rmin)/drStart + 0.5))
                drStart = (rmax - rmin)/neff
            return drStart, neff
        dxSurface, nxeff = adjustRatio(dxSurface, xratio, xmin, xmax)
        print "Adjusting initial spacing to (%g) in order to create integer numbers of bins (%i) to edges." % (dxSurface, nxeff)

        def flipcoord(xi, x0, x1):
            return x0 + x1 - xi

        # Work our way in from the x surface.
        x1 = xmax
        dx = dxSurface
        while x1 > xmin + 0.1*dx:
            x0 = max(xmin, x1 - dx)
            xi = 0.5*(x0 + x1)
            hx = nNodePerh*dx
            
            xi0 = x0
            xi1 = x1
            if flipx:
                xi = flipcoord(xi, xmin, xmax)
                xi0 = flipcoord(xi0, xmin, xmax)
                xi1 = flipcoord(xi1, xmin, xmax)
                
            self.x.append(xi)
            rhoi, mi = computeRhoAndMass1d(Vector1d(xi0), Vector1d(xi1),
                                           Vector1d(xi),
                                           self.rhofunc)
            self.m.append(mi)
            self.rho.append(rhoi)
            self.H.append(SymTensor1d(1.0/hx))
            
            x1 = x0
            dx *= xratio
            
        # Have the base class break up the serial node distribution
        # for parallel cases.
        NodeGeneratorBase.__init__(self, True,
                                   self.x, self.m, self.H, self.rho)
        return


    #---------------------------------------------------------------------------
    # Get the position for the given node index.
    #---------------------------------------------------------------------------
    def localPosition(self, i):
        return Vector1d(self.x[i])
    
    #---------------------------------------------------------------------------
    # Get the mass for the given node index.
    #---------------------------------------------------------------------------
    def localMass(self, i):
        return self.m[i]
    
    #---------------------------------------------------------------------------
    # Get the mass density for the given node index.
    #---------------------------------------------------------------------------
    def localMassDensity(self, i):
        return self.rho[i]
    
    #---------------------------------------------------------------------------
    # Get the H tensor for the given node index.
    #---------------------------------------------------------------------------
    def localHtensor(self, i):
        assert i >= 0 and i < len(self.H)
        return self.H[i]

#-------------------------------------------------------------------------------
# Specialized node generator to generate nodes ratioing from a 1D slab surface.
#-------------------------------------------------------------------------------
class GenerateRatioSlab2d(NodeGeneratorBase):
    
    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    def __init__(self,
                 dxSurface, xratio,
                 dySurface, yratio,
                 rho,
                 xmin,
                 xmax,
                 nNodePerh = 2.01,
                 SPH = False,
                 flipx = False,
                 flipy = False):
        
        assert dxSurface > 0.0
        assert xratio > 0.0
        assert dySurface > 0.0
        assert yratio > 0.0
        assert xmin[0] < xmax[0]
        assert xmin[1] < xmax[1]
        assert nNodePerh > 0.0
        
        # If the user provided a constant for rho, then use the constantRho
        # class to provide this value.
        if type(rho) == type(1.0):
            self.rhofunc = ConstantRho(rho)
        else:
            self.rhofunc = rho

        self.x, self.y, self.m, self.H, self.rho = [], [], [], [], []

        # Decide the actual ratios we're going to use to arrive at an integer number of radial bins.
        def adjustRatio(drStart, drRatio, rmin, rmax):
            if abs(drRatio - 1.0) > 1e-4:
                neff = max(1, int(log(1.0 - (rmax - rmin)*(1.0 - drRatio)/drStart)/log(drRatio) + 0.5))
                drStart = (rmax - rmin)*(1.0 - drRatio)/(1.0 - drRatio**neff)
            else:
                neff = max(1, int((rmax - rmin)/drStart + 0.5))
                drStart = (rmax - rmin)/neff
            return drStart, neff
        dxSurface, nxeff = adjustRatio(dxSurface, xratio, xmin[0], xmax[0])
        dySurface, nyeff = adjustRatio(dySurface, yratio, xmin[1], xmax[1])
        print "Adjusting initial spacing to (%g, %g) in order to create integer numbers of bins (%i, %i) to edges." % (dxSurface, dySurface, nxeff, nyeff)

        def flipcoord(xi, x0, x1):
            return x0 + x1 - xi

        # Work our way in from the y surface.
        y1 = xmax[1]
        dy = dySurface
        while y1 > xmin[1] + 0.1*dy:
            y0 = max(xmin[1], y1 - dy)
            yi = 0.5*(y0 + y1)
            hy = nNodePerh*dy

            yi0 = y0
            yi1 = y1
            if flipy:
                yi = flipcoord(yi, xmin[1], xmax[1])
                yi0 = flipcoord(yi0, xmin[1], xmax[1])
                yi1 = flipcoord(yi1, xmin[1], xmax[1])

            # Work our way in from the x surface.
            x1 = xmax[0]
            dx = dxSurface
            while x1 > xmin[0] + 0.1*dx:
                x0 = max(xmin[0], x1 - dx)
                xi = 0.5*(x0 + x1)
                hx = nNodePerh*dx

                xi0 = x0
                xi1 = x1
                if flipx:
                    xi = flipcoord(xi, xmin[0], xmax[0])
                    xi0 = flipcoord(xi0, xmin[0], xmax[0])
                    xi1 = flipcoord(xi1, xmin[0], xmax[0])

                self.x.append(xi)
                self.y.append(yi)
                rhoi, mi = computeRhoAndMass2d(Vector2d(xi0, yi0), Vector2d(xi1, yi0),
                                               Vector2d(xi1, yi1), Vector2d(xi0, yi1),
                                               Vector2d(xi, yi),
                                               self.rhofunc)
                self.m.append(mi)
                self.rho.append(rhoi)
                self.H.append(SymTensor2d(1.0/hx, 0.0, 0.0, 1.0/hy))
                
                x1 = x0
                dx *= xratio

            y1 = y0
            dy *= yratio

        # # Make sure the total mass is what we intend it to be, by applying
        # # a multiplier to the particle masses.
        # M0 = 0.0
        # for m in self.m:
        #     M0 += m
        # assert M0 > 0.0
        # M1 = (xmax[0] - xmin[0]) * (xmax[1] - xmin[1]) * rho
        # massCorrection = M1/M0
        # for i in xrange(len(self.m)):
        #     self.m[i] *= massCorrection
        # print "Applied a mass correction of %f to ensure total mass is %f." % (massCorrection, M1)

        # Have the base class break up the serial node distribution
        # for parallel cases.
        NodeGeneratorBase.__init__(self, True,
                                   self.x, self.y, self.m, self.H, self.rho)
        return


    #---------------------------------------------------------------------------
    # Get the position for the given node index.
    #---------------------------------------------------------------------------
    def localPosition(self, i):
        assert len(self.x) == len(self.y)
        return Vector2d(self.x[i], self.y[i])
    
    #---------------------------------------------------------------------------
    # Get the mass for the given node index.
    #---------------------------------------------------------------------------
    def localMass(self, i):
        return self.m[i]
    
    #---------------------------------------------------------------------------
    # Get the mass density for the given node index.
    #---------------------------------------------------------------------------
    def localMassDensity(self, i):
        return self.rho[i]
    
    #---------------------------------------------------------------------------
    # Get the H tensor for the given node index.
    #---------------------------------------------------------------------------
    def localHtensor(self, i):
        assert i >= 0 and i < len(self.H)
        return self.H[i]
    

 #-------------------------------------------------------------------------------
# Specialized node generator to generate nodes ratioing from a 1D slab surface.
#-------------------------------------------------------------------------------
class GenerateRatioSlab3d(NodeGeneratorBase):
    
    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    def __init__(self,
                 dxSurface, xratio,
                 dySurface, yratio,
                 dzSurface, zratio,
                 rho,
                 xmin,
                 xmax,
                 nNodePerh = 2.01,
                 SPH = False,
                 flipx = False,
                 flipy = False,
                 flipz = False):
        
        assert dxSurface > 0.0
        assert xratio > 0.0
        assert dySurface > 0.0
        assert yratio > 0.0
        assert dzSurface > 0.0
        assert zratio > 0.0
        assert xmin[0] < xmax[0]
        assert xmin[1] < xmax[1]
        assert xmin[2] < xmax[2]
        assert nNodePerh > 0.0
        
        # If the user provided a constant for rho, then use the constantRho
        # class to provide this value.
        if type(rho) == type(1.0):
            self.rhofunc = ConstantRho(rho)
        else:
            self.rhofunc = rho

        self.x, self.y, self.z, self.m, self.H, self.rho = [], [], [], [], [], []

        # Decide the actual ratios we're going to use to arrive at an integer number of radial bins.
        def adjustRatio(drStart, drRatio, rmin, rmax):
            if abs(drRatio - 1.0) > 1e-4:
                neff = max(1, int(log(1.0 - (rmax - rmin)*(1.0 - drRatio)/drStart)/log(drRatio) + 0.5))
                drStart = (rmax - rmin)*(1.0 - drRatio)/(1.0 - drRatio**neff)
            else:
                neff = max(1, int((rmax - rmin)/drStart + 0.5))
                drStart = (rmax - rmin)/neff
            return drStart, neff
        dxSurface, nxeff = adjustRatio(dxSurface, xratio, xmin[0], xmax[0])
        dySurface, nyeff = adjustRatio(dySurface, yratio, xmin[1], xmax[1])
        dzSurface, nzeff = adjustRatio(dzSurface, zratio, xmin[2], xmax[2])
        print("Adjusting initial spacing to (%g, %g, %g) in order to create integer numbers of bins (%i, %i, %i) to edges." % 
            (dxSurface, dySurface, dzSurface, nxeff, nyeff, nzeff))

        def flipcoord(xi, x0, x1):
            return x0 + x1 - xi

        # Work our way in from the z surface.
        z1 = xmax[2]
        dz = dzSurface
        while z1 > xmin[2] + 0.1*dz:
            z0 = max(xmin[2],z1-dz)
            zi = 0.5*(z0+z1)
            hz = nNodePerh*dz
            
            zi0 = z0
            zi1 = z1
            if flipz:
                zi = flipcoord(zi, xmin[2], xmax[2])
                zi0 = flipcoord(zi0, xmin[2], xmax[2])
                zi1 = flipcoord(zi1, xmin[2], xmax[2])
            # Work our way in from the y surface.
            y1 = xmax[1]
            dy = dySurface
            while y1 > xmin[1] + 0.1*dy:
                y0 = max(xmin[1], y1 - dy)
                yi = 0.5*(y0 + y1)
                hy = nNodePerh*dy

                yi0 = y0
                yi1 = y1
                if flipy:
                    yi = flipcoord(yi, xmin[1], xmax[1])
                    yi0 = flipcoord(yi0, xmin[1], xmax[1])
                    yi1 = flipcoord(yi1, xmin[1], xmax[1])

                # Work our way in from the x surface.
                x1 = xmax[0]
                dx = dxSurface
                while x1 > xmin[0] + 0.1*dx:
                    x0 = max(xmin[0], x1 - dx)
                    xi = 0.5*(x0 + x1)
                    hx = nNodePerh*dx

                    xi0 = x0
                    xi1 = x1
                    if flipx:
                        xi = flipcoord(xi, xmin[0], xmax[0])
                        xi0 = flipcoord(xi0, xmin[0], xmax[0])
                        xi1 = flipcoord(xi1, xmin[0], xmax[0])

                    self.x.append(xi)
                    self.y.append(yi)
                    self.z.append(zi)
                    rhoi, mi = computeRhoAndMass3d(Vector3d(xi0,yi0,zi0),
                                                   Vector3d(xi1,yi0,zi0),
                                                   Vector3d(xi1,yi1,zi0),
                                                   Vector3d(xi0,yi1,zi0),
                                                   Vector3d(xi0,yi0,zi1),
                                                   Vector3d(xi1,yi0,zi1),
                                                   Vector3d(xi1,yi1,zi1),
                                                   Vector3d(xi0,yi1,zi1),
                                                   Vector3d(xi,yi,zi),
                                                   self.rhofunc)
                    self.m.append(mi)
                    self.rho.append(rhoi)
                    self.H.append(SymTensor3d(1.0/hx, 0.0, 0.0, 0.0, 1.0/hy, 0.0, 0.0,0.0,1.0/hz))
                    
                    x1 = x0
                    dx *= xratio

                y1 = y0
                dy *= yratio
            
            z1 = z0
            dz *= zratio


        # # Make sure the total mass is what we intend it to be, by applying
        # # a multiplier to the particle masses.
        # M0 = 0.0
        # for m in self.m:
        #     M0 += m
        # assert M0 > 0.0
        # M1 = (xmax[0] - xmin[0]) * (xmax[1] - xmin[1]) * rho
        # massCorrection = M1/M0
        # for i in xrange(len(self.m)):
        #     self.m[i] *= massCorrection
        # print "Applied a mass correction of %f to ensure total mass is %f." % (massCorrection, M1)

        # Have the base class break up the serial node distribution
        # for parallel cases.
        NodeGeneratorBase.__init__(self, True,
                                   self.x, self.y, self.z, self.m, self.H, self.rho)
        return


    #---------------------------------------------------------------------------
    # Get the position for the given node index.
    #---------------------------------------------------------------------------
    def localPosition(self, i):
        assert len(self.x) == len(self.y) == len(self.z)
        return Vector3d(self.x[i], self.y[i], self.z[i])
    
    #---------------------------------------------------------------------------
    # Get the mass for the given node index.
    #---------------------------------------------------------------------------
    def localMass(self, i):
        return self.m[i]
    
    #---------------------------------------------------------------------------
    # Get the mass density for the given node index.
    #---------------------------------------------------------------------------
    def localMassDensity(self, i):
        return self.rho[i]
    
    #---------------------------------------------------------------------------
    # Get the H tensor for the given node index.
    #---------------------------------------------------------------------------
    def localHtensor(self, i):
        assert i >= 0 and i < len(self.H)
        return self.H[i]
