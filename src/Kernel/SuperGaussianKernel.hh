//---------------------------------Spheral++----------------------------------//
// SuperGaussianKernel -- The Super gaussian interpolation kernel, ala Monaghan.
//
// Volume normalization:
// 1-D:  A = 1/sqrt(Pi)
// 2-D:  A = 1/Pi
// 3-D:  A = 1/Pi^(3/2)
//
// Kernel extent:  3.0
//
// Created by JMO, Wed Dec  1 22:38:21 PST 1999
//----------------------------------------------------------------------------//

#ifndef __Spheral_SuperGaussianKernel_hh__
#define __Spheral_SuperGaussianKernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class SuperGaussianKernel: public Kernel<Dimension, SuperGaussianKernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  SuperGaussianKernel();
  ~SuperGaussianKernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

private:
  //--------------------------- Private Interface ---------------------------//
  static const double mKW;
  static const double mKGW;

};

}

#include "SuperGaussianKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class SuperGaussianKernel;
}

#endif
