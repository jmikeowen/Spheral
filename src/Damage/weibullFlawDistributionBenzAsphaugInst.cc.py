text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Geometry/Dimension.hh"
#include "Damage/weibullFlawDistributionBenzAsphaug.cc"

template Spheral::Field<Spheral::Dim< %(ndim)s >, std::vector<double> > 
Spheral::weibullFlawDistributionBenzAsphaug<Spheral::Dim< %(ndim)s > >(double,
                                                                       const double,
                                                                       const unsigned,
                                                                       const double,
                                                                       const double,
                                                                       const Spheral::FluidNodeList<Spheral::Dim< %(ndim)s > >&,
                                                                       const int,
                                                                       const int,
                                                                       const Spheral::Field<Spheral::Dim<%(ndim)s>, int>&);
"""
