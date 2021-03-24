//---------------------------------Spheral++----------------------------------//
// SincKernel -- The sinc interpolation kernel: W = sin(pi*eta)/(pi*eta).
//
// Created by JMO, Mon Jan  6 22:42:01 PST 2003
//----------------------------------------------------------------------------//
#ifndef __Spheral_SincKernel_hh__
#define __Spheral_SincKernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class SincKernel: public Kernel<Dimension, SincKernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  SincKernel(const double extent);
  ~SincKernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the second derivative for a given normalized distance or position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

};

}

#include "SincKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class SincKernel;
}

#endif
