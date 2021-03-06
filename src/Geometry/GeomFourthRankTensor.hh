//---------------------------------Spheral++----------------------------------//
// GeomFourthRankTensor -- Geometric Tensor (rank 4) Class.
//
// This is a very simple, limited functionality rank 4 tensor.  Assumes
// the thing is dimensioned nDim x nDim x nDim x nDim.
//
// Created by JMO, Sun Oct 11 22:10:34 PDT 2015
//----------------------------------------------------------------------------//
#ifndef __Spheral_GeomFourthRankTensor_hh__
#define __Spheral_GeomFourthRankTensor_hh__

#include <iostream>

#include "Geometry/RankNTensor.hh"
#include "Geometry/GeomFourthRankTensor_fwd.hh"

namespace Spheral {

template<int nDim>
class GeomFourthRankTensor : public RankNTensor<nDim, 4, GeomFourthRankTensor<nDim> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename RankNTensor<nDim, 4, GeomFourthRankTensor>::size_type size_type;
  static const size_type numElements;

  // Useful static member data.
  static const GeomFourthRankTensor zero;

  // Constructors.
  GeomFourthRankTensor();
  explicit GeomFourthRankTensor(const double val);
  GeomFourthRankTensor(const GeomFourthRankTensor& rhs);

  // Destructor.
  ~GeomFourthRankTensor();

  // Assignment.
  GeomFourthRankTensor& operator=(const GeomFourthRankTensor& rhs);
  GeomFourthRankTensor& operator=(const double rhs);

  // Access the elements by indicies.
  double operator()(const size_type i, const size_type j, const size_type k, const size_type m) const;
  double& operator()(const size_type i, const size_type j, const size_type k, const size_type m);

private:
  //--------------------------- Private Interface ---------------------------//
  using RankNTensor<nDim, 4, GeomFourthRankTensor>::mElements;
};

template<int nDims> const typename GeomFourthRankTensor<nDims>::size_type GeomFourthRankTensor<nDims>::numElements = calcNumNRankElements<nDims, 4>();
template<int nDims> const GeomFourthRankTensor<nDims> GeomFourthRankTensor<nDims>::zero = GeomFourthRankTensor<nDims>(0.0);

}

#include "GeomFourthRankTensorInline.hh"

#endif
