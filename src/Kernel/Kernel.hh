//---------------------------------Spheral++----------------------------------//
// Kernel -- The interpolation kernel for use in smoothed field estimates.
//
// Created by JMO, Thu Jul 29 19:43:35 PDT 1999
//----------------------------------------------------------------------------//
#ifndef __Spheral_Kernel_hh__
#define __Spheral_Kernel_hh__

namespace Spheral {

template<typename Dimension, typename Descendant>
class Kernel {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  // Cast as the descendent type.
  Descendant& asDescendant() const;

  // Constructors, destructors.
  Kernel();
  ~Kernel();

  // Assignment.
  Kernel& operator=(const Kernel& rhs);

  //======================================================================
  // Return the kernel weight for a given normalized distance or position.
  double operator()(const Vector& eta, const SymTensor& H) const;
  double operator()(const Vector& eta, const Scalar& Hdet) const;
  double operator()(const double& etaMagnitude, const SymTensor& H) const;
  double operator()(const double& etaMagnitude, const Scalar& Hdet) const;

  //======================================================================
  // Return the gradient value for a given normalized distance or position.
  double grad(const Vector& eta, const SymTensor& H) const;
  double grad(const Vector& eta, const Scalar& Hdet) const;
  double grad(const double& etaMagnitude, const SymTensor& H) const;
  double grad(const double& etaMagnitude, const Scalar& Hdet) const;

  //======================================================================
  // Return the second derivative of the kernel for a given normalized distance
  //  or position.
  double grad2(const Vector& eta, const SymTensor& H) const;
  double grad2(const Vector& eta, const Scalar& Hdet) const;
  double grad2(const double& etaMagnitude, const SymTensor& H) const;
  double grad2(const double& etaMagnitude, const Scalar& Hdet) const;

  //======================================================================
  // Return the gradient with respect to h for a given normalized distance
  // or position.
  double gradh(const Vector& eta, const SymTensor& H) const;
  double gradh(const Vector& eta, const Scalar& Hdet) const;
  double gradh(const double& etaMagnitude, const SymTensor& H) const;
  double gradh(const double& etaMagnitude, const Scalar& Hdet) const;

  //======================================================================
  // Get the volume normalization constant.
  double volumeNormalization() const;

  // Get the extent of the kernel (the cutoff distance in eta over which the
  // kerel is non-zero.
  double kernelExtent() const;

  // We also require that all kernels provide their inflection point, i.e., the
  // point at which their gradient maxes out and starts rolling over.
  double inflectionPoint() const;

  // Call the descendent Kernel implementations to get the real values.
  // All Kernels are required to define the "kernelValue", "gradValue",
  // and "grad2Value" methods, with the same call signatures 
  // as these functions.
  double kernelValue(double etaMagnitude, const double Hdet) const;
  double gradValue(double etaMagnitude, const double Hdet) const;
  double grad2Value(double etaMagnitude, const double Hdet) const;

  // Compute the gradient with respect to h, which we can do in terms of
  // the already provided gradient method.
  double gradhValue(double etaMagnitude, const double Hdet) const;

  // Test if the Kernel is in a valid state.
  virtual bool valid() const;

protected:
  //--------------------------- Protected Interface ---------------------------//
  // Descendant Kernel classes are allowed (in fact required!) to set the 
  // volume normalization.
  void setVolumeNormalization(double volumeNormalization);
  void setKernelExtent(double extent);
  void setInflectionPoint(double x);

// private:
//   //--------------------------- Private Interface ---------------------------//
  double mVolumeNormalization;
  double mKernelExtent;
  double mInflectionPoint;

};

}

#include "KernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension, typename Descendant> class Kernel;
}

#endif
