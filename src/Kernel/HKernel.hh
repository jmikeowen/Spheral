//---------------------------------Spheral++----------------------------------//
// HKernel -- A kernel modifier for use by the ideal H methods of the NodeList.
//
// Based and generalized from the counting kernel proposed in
// Thacker, Tittley, Pearce, Couchman, & Thomas 2000, MNRAS, 319, 619.
//
// Volume normalizations:
// 1-D:  A = 1.0
// 2-D:  A = 1.0
// 3-D:  A = 1.0
//
// The idea is to take an existing kernel which is going to be used in a 
// computation, and give a modified version as:
//               / 1.0,   for eta \in [0.0, 0.75*kernelExtent]
// WH(W, eta) = <
//               \ 0.5*(1 + cos(4.0*pi/kernelExtent * eta)),
//                     for eta \in [0.75*kernelExtent, kernelExtent]
//
// Created by J. Michael Owen, Thu Apr 22 16:00:39 2004
//----------------------------------------------------------------------------//
#ifndef __Spheral_HKernel_hh__
#define __Spheral_HKernel_hh__

#include "Kernel.hh"

namespace Spheral {

template<typename Dimension>
class HKernel: public Kernel<Dimension, HKernel<Dimension> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructors, destructors.
  explicit HKernel(double extent = 2.0);
  ~HKernel();

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(double etaMagnitude, const double Hdet) const;

  // Return the second derivative value for a given normalized distance or
  // position.
  double grad2Value(double etaMagnitude, const double Hdet) const;

  // Clone the extent of the given kernel.
  void matchKernel(const TableKernel<Dimension>& W);
  void matchKernelExtent(double extent);

  // Return the volume integral constant for use in the ideal H calculation.
  double Ns() const;

private:
  //--------------------------- Private Interface ---------------------------//
  double mInterval14;
  double mInterval34;
  double mPeriod;
  double mOffset;

  // The normalization constant appropriate for this kernel, calculated as a service
  // for the ideal H calucation.
  double mNs;
};

}

#include "HKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class HKernel;
}

#endif
