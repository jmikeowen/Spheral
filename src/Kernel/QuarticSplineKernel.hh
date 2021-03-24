//---------------------------------Spheral++----------------------------------//
// QuarticSplineKernel -- A quartic spline, as described in
// Belytschko et al., Computational Methods in Applied Mathematics and Engineering
// 1996, 139, 3-47.
//
// Kernel extent: 2.0
//
// Created by JMO, Wed Jan  8 22:45:10 PST 2003
//----------------------------------------------------------------------------//
#ifndef __Spheral_QuarticSplineKernel_hh__
#define __Spheral_QuarticSplineKernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class QuarticSplineKernel: public Kernel<Dimension, QuarticSplineKernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  QuarticSplineKernel();
  ~QuarticSplineKernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the second derivative value for a given normalized distance or
  // position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

};

}

#include "QuarticSplineKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class QuarticSplineKernel;
}

#endif
