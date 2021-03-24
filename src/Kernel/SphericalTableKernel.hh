//---------------------------------Spheral++----------------------------------//
// SphericalTableKernel
//
// Take a 3D Kernel and build a specialized 1D tabulated version appropriate
// for use with the spherical SPH algorithm described in
// Omang, M., Børve, S., & Trulsen, J. (2006). SPH in spherical and cylindrical coordinates.
// Journal of Computational Physics, 213(1), 391412. https://doi.org/10.1016/j.jcp.2005.08.023
//
// Created by JMO, Wed Dec  2 16:41:20 PST 2020
//----------------------------------------------------------------------------//
#ifndef __Spheral_SphericalTableKernel_hh__
#define __Spheral_SphericalTableKernel_hh__

#include "Kernel.hh"
#include "TableKernel.hh"
#include "Utilities/BiQuadraticInterpolator.hh"
#include "Geometry/Dimension.hh"

namespace Spheral {

class SphericalTableKernel {

public:
  //--------------------------- Public Interface ---------------------------//
  using InterpolatorType = BiQuadraticInterpolator;
  using Scalar = Dim<1>::Scalar;
  using Vector = Dim<1>::Vector;
  using Tensor = Dim<1>::Tensor;
  using SymTensor = Dim<1>::SymTensor;

  // Constructor.
  // Takes a normal 3D TableKernel and constructs the integral form appropriate
  // for 1D spherical coordinates.
  SphericalTableKernel(const TableKernel<Dim<3>>& kernel);
  SphericalTableKernel(const SphericalTableKernel& rhs);

  // Destructor.
  virtual ~SphericalTableKernel();

  // Assignment.
  SphericalTableKernel& operator=(const SphericalTableKernel& rhs);

  // These methods taking a Vector eta and Vector position are the special methods
  // allowing this kernel to implement the asymmetric sampling as a function of r.
  // Arguments:
  //  etaj : Vector normalized coordinate: etaj = H*posj
  //  etai : Vector normalized coordinate: etai = H*posi
  //  Hdet  : Determinant of the H tensor used to compute eta
  double operator()(const Vector& etaj, const Vector& etai, const Scalar Hdeti) const;
  Vector grad(const Vector& etaj, const Vector& etai, const Scalar Hdeti) const;
  std::pair<double, Vector> kernelAndGradValue(const Vector& etaj, const Vector& etai, const Scalar Hdeti) const;

  // Return the kernel weight for a given normalized distance or position.
  double kernelValue(const double etaMagnitude, const double Hdet) const;

  // Return the gradient value for a given normalized distance or position.
  double gradValue(const double etaMagnitude, const double Hdet) const;

  // Return the second derivative value for a given normalized distance or position.
  double grad2Value(const double etaMagnitude, const double Hdet) const;

  // Access our internal data.
  const InterpolatorType& Winterpolator() const;
  const TableKernel<Dim<3>>& kernel() const;
  Scalar etamax() const;

  // Test if the kernel is currently valid.
  virtual bool valid() const;

private:
  //--------------------------- Private Interface ---------------------------//
  // Data for the kernel tabulation.
  InterpolatorType mInterp;
  TableKernel<Dim<3>> mKernel;
  Scalar metamax;
};

}

#include "SphericalTableKernelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  class SphericalTableKernel;
}

#endif

