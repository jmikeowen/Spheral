#include "Utilities/SpheralFunctions.hh"  // sgn
#include "Utilities/FastMath.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Lookup the kernel for (rj/h, ri/h) = (etaj, etai)
//------------------------------------------------------------------------------
inline
double
SphericalTableKernel::operator()(const Dim<1>::Vector& etaj,
                                 const Dim<1>::Vector& etai,
                                 const Dim<1>::Scalar  Hdeti) const {
  REQUIRE(Hdeti >= 0.0);
  const auto ei = std::max(1e-10, etai[0]);
  const auto ej = std::max(1e-10, etaj[0]);
  CHECK(ei > 0.0);
  CHECK(ej > 0.0);
  const auto min_bound = std::abs(ej - ei);
  if (min_bound > metamax) return 0.0;
  const auto max_bound = std::min(metamax, ei + ej);
  return 2.0*M_PI/(ei*ej)*FastMath::cube(Hdeti)*mInterp(Dim<2>::Vector(min_bound, max_bound));
}

//------------------------------------------------------------------------------
// Lookup the grad kernel for (rj/h, ri/h) = (etaj, etai)
//------------------------------------------------------------------------------
inline
Dim<1>::Vector
SphericalTableKernel::grad(const Dim<1>::Vector& etaj,
                           const Dim<1>::Vector& etai,
                           const Dim<1>::Scalar  Hdeti) const {
  REQUIRE(Hdeti >= 0.0);
  const auto ei = std::max(1e-10, etai[0]);
  const auto ej = std::max(1e-10, etaj[0]);
  CHECK(ei > 0.0);
  CHECK(ej > 0.0);
  const auto min_bound = std::abs(ej - ei);
  if (min_bound > metamax) return 0.0;
  const auto max_bound = std::min(metamax, ei + ej);
  const auto A = (ei + ej >= metamax ?
                  0.0 :
                  max_bound*mKernel.kernelValue(max_bound, Hdeti));
  const auto B = (ej - ei)*mKernel.kernelValue(min_bound, Hdeti);
  return Vector(2.0*M_PI/(ei*ej)*FastMath::cube(Hdeti)*(A - B - Hdeti/ej*mInterp(Dim<2>::Vector(min_bound, max_bound))));
}

//------------------------------------------------------------------------------
// Simultaneously lookup (W,  grad W) for (rj/h, ri/h) = (etaj, etai)
//------------------------------------------------------------------------------
inline
std::pair<double, Dim<1>::Vector>
SphericalTableKernel::kernelAndGradValue(const Dim<1>::Vector& etaj,
                                         const Dim<1>::Vector& etai,
                                         const Dim<1>::Scalar  Hdeti) const {
  REQUIRE(Hdeti >= 0.0);
  const auto ei = std::max(1e-10, etai[0]);
  const auto ej = std::max(1e-10, etaj[0]);
  CHECK(ei > 0.0);
  CHECK(ej > 0.0);
  const auto min_bound = std::abs(ej - ei);
  if (min_bound > metamax) return std::make_pair(0.0, 0.0);
  const auto max_bound = std::min(metamax, ei + ej);
  const auto A = (ei + ej >= metamax ?
                  0.0 :
                  max_bound*mKernel.kernelValue(max_bound, Hdeti));
  const auto B = (ej - ei)*mKernel.kernelValue(min_bound, Hdeti);
  const auto interpVal = mInterp(Dim<2>::Vector(min_bound, max_bound));
  const auto pre = 2.0*M_PI/(ei*ej)*FastMath::cube(Hdeti);
  const auto Wval = pre*interpVal;
  const auto gradWval = pre*(A - B - Hdeti/ej*interpVal);
  return std::make_pair(Wval, Vector(gradWval));
}

//------------------------------------------------------------------------------
// kernelValue, gradValue, grad2Value -- pass through to the base TableKernel
//------------------------------------------------------------------------------
inline
double 
SphericalTableKernel::kernelValue(const double etaMagnitude, const double Hdet) const {
  return mKernel.kernelValue(etaMagnitude, Hdet);
}

inline
double 
SphericalTableKernel::gradValue(const double etaMagnitude, const double Hdet) const {
  return mKernel.gradValue(etaMagnitude, Hdet);
}

inline
typename Dim<1>::Scalar
SphericalTableKernel::grad2Value(const double etaMagnitude, const double Hdet) const {
  return mKernel.grad2Value(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Data accessors
//------------------------------------------------------------------------------
inline
const typename SphericalTableKernel::InterpolatorType&
SphericalTableKernel::Winterpolator() const {
  return mInterp;
}

inline
const TableKernel<Dim<3>>&
SphericalTableKernel::kernel() const {
  return mKernel;
}

inline
double
SphericalTableKernel::etamax() const {
  return metamax;
}

}
