# This bit of code solves the equation for hydrostatic equilibrium to return
# a density profile function to be used in various NodeGenerators

from math import *

#-------------------------------------------------------------------------------
# 3-D solvers
#-------------------------------------------------------------------------------
class HydroSolveConstantTemp3D():
    def __init__(self,
                 rho0,
                 rMax,
                 temp,
                 eos,
                 units,
                 y0=0,
                 nbins=1000):

        self.y0     = y0
        self.rMax   = rMax
        self.nbins  = nbins
        self.rho0   = rho0
        self.eos    = eos
        self.soln   = []
        
        r   = self.rMax
        rho = self.rho0
        dr  = self.rMax/self.nbins
        y   = self.y0
        dy  = 0
        
        while(r>0):
            e       = eos.specificThermalEnergy(rho,temp)
            K       = eos.bulkModulus(rho,e)
            dy      = dr*(2.0/rho*y*y - 2.0/r*y - units.G/K*4.0*pi*pow(rho,3.0))
            self.soln.append([r,rho])
            y       = y + dy
            rho     = rho - y*dr
            r       = r - dr

        self.soln.sort()

    def __call__(self,r):
        rho = 0
        for i in xrange(len(self.soln)):
            if(self.soln[i][0] > r):
                if(i>0):
                    f1  = self.soln[i][1]
                    f0  = self.soln[i-1][1]
                    r1  = self.soln[i][0]
                    r0  = self.soln[i-1][0]
                    rho = (f1-f0)*(r-r1)/(r1-r0)+f1
                else:
                    rho = self.soln[0][1]
                break
        return rho

#-------------------------------------------------------------------------------
# 2-D solvers
#-------------------------------------------------------------------------------
class HydroSolveConstantTemp2D():
    def __init__(self,
                 rho0,
                 rMax,
                 temp,
                 eos,
                 units,
                 y0=0,
                 nbins=1000):
        
        self.y0     = y0
        self.rMax   = rMax
        self.nbins  = nbins
        self.rho0   = rho0
        self.eos    = eos
        self.soln   = []
        
        r   = self.rMax
        rho = self.rho0
        dr  = self.rMax/self.nbins
        y   = self.y0
        dy  = 0
        
        while(r>0):
            e       = eos.specificThermalEnergy(rho,temp)
            K       = eos.bulkModulus(rho,e)
            dy      = dr*(2.0/rho*y*y - 2.0/r*y - units.G/K*2.0*pi*pow(rho,3.0)/r)
            self.soln.append([r,rho])
            y       = y + dy
            rho     = rho - y*dr
            r       = r - dr
        
        self.soln.sort()
    
    def __call__(self,r):
        rho = 0
        for i in xrange(len(self.soln)):
            if(self.soln[i][0] > r):
                if(i>0):
                    f1  = self.soln[i][1]
                    f0  = self.soln[i-1][1]
                    r1  = self.soln[i][0]
                    r0  = self.soln[i-1][0]
                    rho = (f1-f0)*(r-r1)/(r1-r0)+f1

                else:
                    rho = self.soln[0][1]
                break
        return rho


