//---------------------------------Spheral++----------------------------------//
// WendlandC2Kernel -- .
//
//----------------------------------------------------------------------------//
#ifndef __Spheral_WendlandC2Kernel_hh__
#define __Spheral_WendlandC2Kernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class WendlandC2Kernel: public Kernel<Dimension, WendlandC2Kernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  WendlandC2Kernel();
  ~WendlandC2Kernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the second derivative value for a given normalized distance or
  // position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

};

}

#include "WendlandC2KernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class WendlandC2Kernel;
}

#endif
