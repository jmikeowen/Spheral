#-------------------------------------------------------------------------------
# JohnsonCookDamageFactories -- implements a number of factory functions for
#   the JohnsonCookDamage model. Primarily these fill in the D1 and D2 fields
#   appropriately.
#   Specific factories provided are:
#     JohnsonCookDamageConstant
#     JohnsonCookDamageGaussian
#     JohnsonCookDamageWeibull
#-------------------------------------------------------------------------------
from math import *
import numpy as np
import mpi

from spheralDimensions import spheralDimensions

#-------------------------------------------------------------------------------
# Generic method (not used from outside)
#-------------------------------------------------------------------------------
def __JohnsonCookDamageFactory(nodeList,
                               D1,
                               D2,
                               D3,
                               D4,
                               D5,
                               epsilondot0,
                               Tcrit,
                               sigmamax,
                               efailmin,
                               domainIndependent,
                               D1method,
                               D2method):
                             
    # What dimension are we?
    assert nodeList.__class__.__name__ in ("SolidNodeList1d", "SolidNodeList2d", "SolidNodeList3d")
    ndim = int(nodeList.__class__.__name__[-2:-1])
    assert ndim in spheralDimensions()

    # Import the approprite bits of Spheral.
    exec("""
from SpheralCompiledPackages import JohnsonCookDamage%(ndim)sd as JohnsonCookDamage
from SpheralCompiledPackages import ScalarField%(ndim)sd as ScalarField
from SpheralCompiledPackages import DataBase%(ndim)sd as DataBase
from SpheralCompiledPackages import nodeOrdering%(ndim)sd as nodeOrdering
from SpheralCompiledPackages import mortonOrderIndices%(ndim)sd as mortonOrderIndices
""" % {"ndim" : ndim})

    # Prepare the fields for D1 and D2
    fD1 = ScalarField("D1", nodeList, D1)
    fD2 = ScalarField("D2", nodeList, D2)

    # Now fill in the D1 and D2 fields using "D1method" and "D2method" to generate the values.
    # Are we generating domain-independent?
    if domainIndependent:

        # Assign a unique ordering to the nodes so we can step through them
        # in a domain independent manner.
        db = DataBase()
        db.appendNodeList(nodeList)
        keyList = mortonOrderIndices(db);
        orderingList = nodeOrdering(keyList);
        assert orderingList.numFields == 1
        ordering = orderingList[0]
        nglobal = max(0, ordering.max() + 1)

        # Reverse lookup in the ordering.
        order2local = [ordering[i] for i in xrange(nodeList.numInternalNodes)]

        # Generate random numbers in batches.
        iglobal = 0
        nbatch = max(1, nglobal/mpi.procs)
        for iproc in xrange(mpi.procs):
            if iproc == mpi.procs - 1:
                nlocal = nglobal - iglobal
            else:
                nlocal = nbatch
            D1vals = D1method(nlocal)
            D2vals = D2method(nlocal)
            for i in xrange(nlocal):
                try:
                    j = order2local.index(iglobal + i)
                    fD1[j] = D1vals[i]
                    fD2[j] = D2vals[i]
                except ValueError:
                    pass
            iglobal += nlocal

    else:

        # In the non-domain independent case we can generate more quickly in parallel.
        D1vals = D1method(nodeList.numInternalNodes)
        D2vals = D2method(nodeList.numInternalNodes)
        for i in xrange(nodeList.numInternalNodes):
            fD1[i] = vals[i]
            fD2[i] = vals[i]

    # Now we build and and return the damage model.
    return JohnsonCookDamage(nodeList,
                             fD1,
                             fD2,
                             D3,
                             D4,
                             D5,
                             epsilondot0,
                             Tcrit,
                             sigmamax,
                             efailmin)


#-------------------------------------------------------------------------------
# Constant D1 and D2.
#-------------------------------------------------------------------------------
def JohnsonCookDamageConstant(nodeList,
                              D1,
                              D2,
                              D3,
                              D4,
                              D5,
                              epsilondot0,
                              Tcrit,
                              sigmamax,
                              efailmin):

    # Our methods to fill in numbers.
    def D1vals(n):
        return np.full(n, D1)
    def D2vals(n):
        return np.full(n, D2)

    # The generic method does all the work.
    return __JohnsonCookDamageFactory(nodeList,
                                      D1,
                                      D2,
                                      D3,
                                      D4,
                                      D5,
                                      epsilondot0,
                                      Tcrit,
                                      sigmamax,
                                      efailmin,
                                      False,
                                      D1vals,
                                      D2vals)

#-------------------------------------------------------------------------------
# Gaussian distribute D1 and D2.
#-------------------------------------------------------------------------------
def JohnsonCookDamageGaussian(nodeList,
                              D1,
                              D2,
                              D3,
                              D4,
                              D5,
                              epsilondot0,
                              Tcrit,
                              sigmamax,
                              efailmin,
                              sigmaD1,
                              sigmaD2,
                              seed = None,
                              domainIndependent = True):

    # Initialize the random number generator.
    if domainIndependent:
        np.random.seed(seed % 2**32)
    else:
        np.random.seed(((seed + mpi.rank)*(seed + mpi.rank + 1)/2 + mpi.rank) % 2**32)

    # Our methods to fill in numbers.
    def D1vals(n):
        if sigmaD1 != 0.0:
            return np.random.normal(D1, sigmaD1, n)
        else:
            return np.full(n, D1)
    def D2vals(n):
        if sigmaD2 != 0.0:
            return np.random.normal(D2, sigmaD2, n)
        else:
            return np.full(n, D2)

    # The generic method does all the work.
    return __JohnsonCookDamageFactory(nodeList,
                                      D1,
                                      D2,
                                      D3,
                                      D4,
                                      D5,
                                      epsilondot0,
                                      Tcrit,
                                      sigmamax,
                                      efailmin,
                                      domainIndependent,
                                      D1vals,
                                      D2vals)

#-------------------------------------------------------------------------------
# Weibull distribute D1 and D2.
#-------------------------------------------------------------------------------
def JohnsonCookDamageWeibull(nodeList,
                             D1,
                             D2,
                             D3,
                             D4,
                             D5,
                             epsilondot0,
                             Tcrit,
                             sigmamax,
                             efailmin,
                             aD1,
                             bD1,
                             eps0D1,
                             aD2,
                             bD2,
                             eps0D2,
                             seed = None,
                             domainIndependent = True):

    # Initialize the random number generator.
    if domainIndependent:
        np.random.seed(seed % 2**32)
    else:
        np.random.seed(((seed + mpi.rank)*(seed + mpi.rank + 1)/2 + mpi.rank) % 2**32)

    # Our methods to fill in numbers.
    def D1vals(n):
        if aD1 != 0.0:
            return aD1*np.random.weibull(bD1, n) + eps0D1
        else:
            return np.full(n, D1)
    def D2vals(n):
        if aD2 != 0.0:
            return aD2*np.random.weibull(bD2, n) + eps0D2
        else:
            return np.full(n, D2)

    # The generic method does all the work.
    return __JohnsonCookDamageFactory(nodeList,
                                      D1,
                                      D2,
                                      D3,
                                      D4,
                                      D5,
                                      epsilondot0,
                                      Tcrit,
                                      sigmamax,
                                      efailmin,
                                      domainIndependent,
                                      D1vals,
                                      D2vals)
