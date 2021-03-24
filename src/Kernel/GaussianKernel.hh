//---------------------------------Spheral++----------------------------------//
// GaussianKernel -- The gaussian interpolation kernel.
//
// Volume normalization:
// 1-D:  1/sqrt(Pi)
// 2-D:  1/Pi
// 3-D:  1/Pi^(3/2)
//
// Kernel extent:  3.0
//
// Created by JMO, Wed Dec  1 14:38:51 PST 1999
//----------------------------------------------------------------------------//

#ifndef __Spheral_GaussianKernel_hh__
#define __Spheral_GaussianKernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class GaussianKernel: public Kernel<Dimension, GaussianKernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  GaussianKernel(const double extent);
  ~GaussianKernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the second derivative for a given normalized distance or position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

};

}

#include "GaussianKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class GaussianKernel;
}

#endif
