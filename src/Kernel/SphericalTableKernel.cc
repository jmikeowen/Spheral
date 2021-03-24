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

#include "SphericalTableKernel.hh"
#include "Utilities/simpsonsIntegration.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

namespace {  // anonymous

//------------------------------------------------------------------------------
// Functor for doing our volume integration of the kernel  
//------------------------------------------------------------------------------
struct W3S1Func {
  const TableKernel<Dim<3>>& mW;
  double metaMax;
  W3S1Func(const TableKernel<Dim<3>>& W): mW(W), metaMax(mW.kernelExtent()) {}

  // Define a local nested functor we'll use to do the volume integral for
  // an (etaj, etai) lookup.
  struct VolFunc {
    const TableKernel<Dim<3>>& mW;
    VolFunc(const TableKernel<Dim<3>>& W): mW(W) {}
    double operator()(const double eta) const {
      return eta * mW(eta, 1.0);
    }
  };

  // Now the lookup based on (etaj, etai) -- the (rprime/h, r/h) from the paper
  double operator()(const Dim<2>::Vector& bounds) const {
    const auto low = bounds[0];
    const auto high = bounds[1];
    if (low >= high) return 0.0;
    return simpsonsIntegration<VolFunc, double, double>(VolFunc(mW),
                                                        low,
                                                        high,
                                                        std::max(size_t(1000), mW.numPoints()));
  }
};

}            // anonymous

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
SphericalTableKernel::SphericalTableKernel(const TableKernel<Dim<3>>& kernel):
  mInterp(Dim<2>::Vector(0.0, 0.0),
          Dim<2>::Vector(kernel.kernelExtent(), kernel.kernelExtent()),
          std::max(size_t(200), kernel.numPoints()),
          std::max(size_t(200), kernel.numPoints()),
          W3S1Func(kernel)),
  mKernel(kernel),
  metamax(kernel.kernelExtent()) {
}

//------------------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------------------
SphericalTableKernel::SphericalTableKernel(const SphericalTableKernel& rhs):
  mInterp(rhs.mInterp),
  mKernel(rhs.mKernel),
  metamax(rhs.metamax) {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
SphericalTableKernel::~SphericalTableKernel() {
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
SphericalTableKernel&
SphericalTableKernel::operator=(const SphericalTableKernel& rhs) {
  if (this != &rhs) {
    mInterp = rhs.mInterp;
    mKernel = rhs.mKernel;
    metamax = rhs.metamax;
  }
  return *this;
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
bool
SphericalTableKernel::valid() const {
  return true;
}

}
