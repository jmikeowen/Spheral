//---------------------------------Spheral++----------------------------------//
// TableKernel -- Build an interpolation kernel using interpolation between
// tabulated points.
//
// Created by JMO, Mon Jun 19 21:06:28 PDT 2000
//----------------------------------------------------------------------------//
#include "Eigen/Dense"

#include "TableKernel.hh"

#include "Utilities/SpheralFunctions.hh"
#include "Utilities/bisectSearch.hh"
#include "Utilities/simpsonsIntegration.hh"
#include "Utilities/safeInv.hh"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

namespace Spheral {

//------------------------------------------------------------------------------
// Sum the Kernel values for the given stepsize.
//------------------------------------------------------------------------------
inline
double
sumKernelValues(const TableKernel<Dim<1> >& W,
                const double deta) {
  REQUIRE(deta > 0);
  double result = 0.0;
  double etax = deta;
  while (etax < W.kernelExtent()) {
    result += 2.0*std::abs(W.gradValue(etax, 1.0));
    etax += deta;
  }
  return result;
}

inline
double
sumKernelValues(const TableKernel<Dim<2> >& W,
                const double deta) {
  REQUIRE(deta > 0);
  typedef Dim<2>::Vector Vector;
  double result = 0.0;
  double etay = 0.0;
  while (etay < W.kernelExtent()) {
    double etax = 0.0;
    while (etax < W.kernelExtent()) {
      const Vector eta(etax, etay);
      double dresult = std::abs(W.gradValue(eta.magnitude(), 1.0));
      if (distinctlyGreaterThan(etax, 0.0)) dresult *= 2.0;
      if (distinctlyGreaterThan(etay, 0.0)) dresult *= 2.0;
      if (fuzzyEqual(eta.magnitude(), 0.0)) dresult *= 0.0;
      result += dresult;
      etax += deta;
    }
    etay += deta;
  }
  return sqrt(result);
}

inline
double
sumKernelValues(const TableKernel<Dim<3> >& W,
                const double deta) {
  REQUIRE(deta > 0);
  typedef Dim<3>::Vector Vector;
  double result = 0.0;
  double etaz = 0.0;
  while (etaz < W.kernelExtent()) {
    double etay = 0.0;
    while (etay < W.kernelExtent()) {
      double etax = 0.0;
      while (etax < W.kernelExtent()) {
        const Vector eta(etax, etay, etaz);
        CHECK(eta >= 0.0);
        double dresult = std::abs(W.gradValue(eta.magnitude(), 1.0));
        if (distinctlyGreaterThan(etax, 0.0)) dresult *= 2.0;
        if (distinctlyGreaterThan(etay, 0.0)) dresult *= 2.0;
        if (distinctlyGreaterThan(etaz, 0.0)) dresult *= 2.0;
        if (fuzzyEqual(eta.magnitude(), 0.0)) dresult *= 0.0;
        result += dresult;
        etax += deta;
      }
      etay += deta;
    }
    etaz += deta;
  }
  return FastMath::CubeRootHalley2(result);
}

// Special hacked version to allow for running 1-D stacks of nodes in 3-D.
// This is way ugly and tricky -- DON'T EMULATE THIS KIND OF EXAMPLE!
template<typename KernelType>
inline
double
sumKernelValuesAs1D(const KernelType& W,
                    const double deta) {
  REQUIRE(deta > 0);
  double result = 0.0;
  double etax = deta;
  while (etax < W.kernelExtent()) {
    result += 2.0*std::abs(W.gradValue(etax, 1.0));
    etax += deta;
  }
  return result;
}

//------------------------------------------------------------------------------
// Compute the (f1,f2) integrals relation for the given zeta = r/h 
// (RZ corrections).
//------------------------------------------------------------------------------
template<typename KernelType>
class volfunc {
  const KernelType& W;
public:
  volfunc(const KernelType& W): W(W) {}
  double operator()(const double eta) const {
    return W.kernelValue(eta, 1.0);
  }
};

template<typename KernelType>
class f1func {
  const KernelType& W;
  double zeta;
public:
  f1func(const KernelType& W, const double zeta): W(W), zeta(zeta) {}
  double operator()(const double eta) const {
    return abs(safeInvVar(zeta)*eta)*W.kernelValue(abs(zeta - eta), 1.0);
  }
};


template<typename KernelType>
class f2func {
  const KernelType& W;
  double zeta;
public:
  f2func(const KernelType& W, const double zeta): W(W), zeta(zeta) {}
  double operator()(const double eta) const {
    return safeInvVar(zeta*zeta)*eta*abs(eta)*W.kernelValue(abs(zeta - eta), 1.0);
  }
};


template<typename KernelType>
class gradf1func {
  const KernelType& W;
  double zeta;
public:
  gradf1func(const KernelType& W, const double zeta): W(W), zeta(zeta) {}
  double operator()(const double eta) const {
    const double Wu = W.kernelValue(abs(zeta - eta), 1.0);
    const double gWu = W.gradValue(abs(zeta - eta), 1.0);
    const double gf1inv = safeInvVar(zeta)*abs(eta)*gWu - safeInvVar(zeta*zeta)*abs(eta)*Wu;
    if (eta < 0.0) {
      return -gf1inv;
    } else {
      return gf1inv;
    }
  }
};


template<typename KernelType>
double
f1Integral(const KernelType& W,
           const double zeta,
           const unsigned numbins) {
  const double etaMax = W.kernelExtent();
  CHECK(zeta <= etaMax);
  return safeInvVar(simpsonsIntegration<f1func<KernelType>, double, double>(f1func<KernelType>(W, zeta), 
                                                                            zeta - etaMax, 
                                                                            zeta + etaMax,
                                                                            numbins));
}

template<typename KernelType>
double
f2Integral(const KernelType& W,
           const double zeta,
           const unsigned numbins) {
  const double etaMax = W.kernelExtent();
  CHECK(zeta <= etaMax);
  return safeInvVar(simpsonsIntegration<f2func<KernelType>, double, double>(f2func<KernelType>(W, zeta), 
                                                                            zeta - etaMax, 
                                                                            zeta + etaMax,
                                                                            numbins));
}

template<typename KernelType>
double
gradf1Integral(const KernelType& W,
               const double zeta,
               const unsigned numbins) {
  const double etaMax = W.kernelExtent();
  CHECK(zeta <= etaMax);
  return simpsonsIntegration<gradf1func<KernelType>, double, double>(gradf1func<KernelType>(W, zeta), 
                                                                     zeta - etaMax, 
                                                                     zeta + etaMax,
                                                                     numbins);
}

//------------------------------------------------------------------------------
// Construct from a kernel.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename KernelType>
TableKernel<Dimension>::TableKernel(const KernelType& kernel,
                                    const unsigned numPoints,
                                    const double hmult):
  Kernel<Dimension, TableKernel<Dimension> >(),
  mInterp(),
  mGradInterp(),
  mGrad2Interp(),
  mNumPoints(numPoints),
  mNperhValues(),
  mWsumValues(),
  mMinNperh(0.25),
  mMaxNperh(10.0),
  mf1Interp(),
  mf2Interp() {

  // Pre-conditions.
  VERIFY(numPoints > 0);
  VERIFY(hmult > 0.0);

  // Set the volume normalization and kernel extent.
  this->setVolumeNormalization(1.0); // (kernel.volumeNormalization() / Dimension::pownu(hmult));  // We now build this into the tabular kernel values.
  this->setKernelExtent(hmult * kernel.kernelExtent());
  this->setInflectionPoint(hmult * kernel.inflectionPoint());

  // Set the number of points and table step size.
  const double etamax = this->kernelExtent();
  const double stepSize = etamax/(numPoints - 1);
  CHECK(stepSize > 0.0);

  // Set the fitting coefficients.
  // Note that we will go ahead and fold the normalization constants in here, 
  // so we don't have to multiply by them in the value lookups.
  std::vector<double> kernelValues(numPoints), gradValues(numPoints), grad2Values(numPoints);
  const double correction = 1.0/Dimension::pownu(hmult);
  const double deta = stepSize/hmult;
  for (auto i = 0u; i < numPoints; ++i) {
    CHECK(i*stepSize >= 0.0);
    kernelValues[i] = correction*kernel(i*deta, 1.0);
    gradValues[i] = correction*kernel.grad(i*deta, 1.0);
    grad2Values[i] = correction*kernel.grad2(i*deta, 1.0);
  }
  mInterp.initialize(0.0, etamax, kernelValues);
  mGradInterp.initialize(0.0, etamax, gradValues);
  mGrad2Interp.initialize(0.0, etamax, grad2Values);

  // If we're a 2D kernel we set the RZ correction information.
  if (Dimension::nDim == 2) {
    std::vector<double> f1Values(numPoints), f2Values(numPoints);
    const auto K1d = 0.5/simpsonsIntegration<volfunc<TableKernel<Dimension>>, double, double>(volfunc<TableKernel<Dimension>>(*this), 0.0, etamax, numPoints);
    for (auto i = 0u; i < numPoints; ++i) {
      CHECK(i*stepSize >= 0.0);
      const double zeta = i*stepSize;
      f1Values[i] = f1Integral(*this, zeta, numPoints)/K1d;
      f2Values[i] = f2Integral(*this, zeta, numPoints)/K1d;
      // gradf1Values[i] = -f1Values[i]*f1Values[i]*gradf1Integral(*this, zeta, numPoints)*K1d;
    }
    mf1Interp.initialize(0.0, etamax, f1Values);
    mf2Interp.initialize(0.0, etamax, f2Values);
  }

  // Set the table of n per h values.
  this->setNperhValues();

  // That should be it, so we should have left the kernel in a valid state.
  ENSURE(valid());
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
TableKernel<Dimension>::
~TableKernel() {
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
template<typename Dimension>
TableKernel<Dimension>&
TableKernel<Dimension>::
operator=(const TableKernel<Dimension>& rhs) {
  if (this != &rhs) {
    Kernel<Dimension, TableKernel<Dimension> >::operator=(rhs);
    mInterp = rhs.mInterp;
    mNumPoints = rhs.mNumPoints;
    mNperhValues = rhs.mNperhValues;
    mWsumValues =  rhs.mWsumValues;
    mMinNperh = rhs.mMinNperh;
    mMaxNperh = rhs.mMaxNperh;
    mf1Interp = rhs.mf1Interp;
    mf2Interp = rhs.mf2Interp;
  }
  return *this;
}

//------------------------------------------------------------------------------
// Determine the number of nodes per smoothing scale implied by the given
// sum of kernel values.
//------------------------------------------------------------------------------
template<typename Dimension>
double
TableKernel<Dimension>::
equivalentNodesPerSmoothingScale(const double Wsum) const {

  // Find the lower bound in the tabulated Wsum's bracketing the input
  // value.
  const int lb = bisectSearch(mWsumValues, Wsum);
  CHECK((lb >= -1) and (lb <= int(mWsumValues.size()) - 1));
  const int ub = lb + 1;
  const int n = int(mNumPoints);
  CHECK((lb == -1 and Wsum <= mWsumValues[0]) ||
        (ub == n and Wsum >= mWsumValues[n - 1]) ||
        (Wsum >= mWsumValues[lb] and Wsum <= mWsumValues[ub]));

  // Now interpolate for the corresponding nodes per h (within bounds);
  double result;
  if (lb == -1) {
    result = mNperhValues[0];
  } else if (ub == n) {
    result = mNperhValues[n - 1];
  } else {
    result = std::min(mNperhValues[ub],
                      std::max(mNperhValues[lb],
                               mNperhValues[lb] +
                               (Wsum - mWsumValues[lb])/
                               (mWsumValues[ub] - mWsumValues[lb])*
                               (mNperhValues[ub] - mNperhValues[lb])));
    ENSURE(result >= mNperhValues[lb] and result <= mNperhValues[ub]);
  }
  return result;
}

//------------------------------------------------------------------------------
// Determine the effective Wsum we would expect for the given n per h.
//------------------------------------------------------------------------------
template<typename Dimension>
double
TableKernel<Dimension>::
equivalentWsum(const double nPerh) const {

  // Find the lower bound in the tabulated n per h's bracketing the input
  // value.
  const int lb = bisectSearch(mNperhValues, nPerh);
  CHECK((lb >= -1) and (lb <= int(mNperhValues.size()) - 1));
  const int ub = lb + 1;
  const int n = int(mNumPoints);
  CHECK((lb == -1 and nPerh <= mNperhValues[0]) ||
        (ub == n and nPerh >= mNperhValues[n - 1]) ||
        (nPerh >= mNperhValues[lb] and nPerh <= mNperhValues[ub]));

  // Now interpolate for the corresponding Wsum.
  double result;
  if (lb == -1) {
    result = mWsumValues[0];
  } else if (ub == n) {
    result = mWsumValues[n - 1];
  } else {
    result = std::min(mWsumValues[ub], 
                      std::max(mWsumValues[lb],
                               mWsumValues[lb] +
                               (nPerh - mNperhValues[lb])/
                               (mNperhValues[ub] - mNperhValues[lb])*
                               (mWsumValues[ub] - mWsumValues[lb])));
    ENSURE(result >= mWsumValues[lb] and result <= mWsumValues[ub]);
  }
  return result;
}

//------------------------------------------------------------------------------
// Initialize the Nperh values.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TableKernel<Dimension>::
setNperhValues(const bool scaleTo1D) {
  REQUIRE(mMinNperh > 0.0);
  REQUIRE(mMaxNperh > mMinNperh);
  REQUIRE(mNumPoints > 1);
  REQUIRE(this->kernelExtent() > 0.0);

  // Size the Nperh array.
  mWsumValues = vector<double>(mNumPoints);
  mNperhValues = vector<double>(mNumPoints);

  // For the allowed range of n per h, sum up the kernel values.
  const double dnperh = (mMaxNperh - mMinNperh)/(mNumPoints - 1u);
  for (auto i = 0u; i < mNumPoints; ++i) {
    const double nperh = mMinNperh + i*dnperh;
    CHECK(nperh >= mMinNperh and nperh <= mMaxNperh);
    const double deta = 1.0/nperh;
    mNperhValues[i] = nperh;
    if (scaleTo1D) {
      mWsumValues[i] = sumKernelValuesAs1D(*this, deta);
    } else {
      mWsumValues[i] = sumKernelValues(*this, deta);
    }
  }

  // Post-conditions.
  BEGIN_CONTRACT_SCOPE
  ENSURE(mWsumValues.size() == mNumPoints);
  ENSURE(mNperhValues.size() == mNumPoints);
  for (auto i = 0u; i < mNumPoints - 1; ++i) {
    ENSURE(mWsumValues[i] <= mWsumValues[i + 1]);
    ENSURE(mNperhValues[i] <= mNperhValues[i + 1]);
  }
  END_CONTRACT_SCOPE

}

//------------------------------------------------------------------------------
// Determine if the kernel is in a valid, ready to use state.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
TableKernel<Dimension>::
valid() const {
  return (Kernel<Dimension, TableKernel<Dimension>>::valid() and
          mNumPoints > 0);
}

}
