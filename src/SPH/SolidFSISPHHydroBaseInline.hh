namespace Spheral {

//------------------------------------------------------------------------------
// set/get the generalized density exponent (Monaghan 1992)
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
SolidFSISPHHydroBase<Dimension>::
surfaceForceCoefficient(double x) {
  mSurfaceForceCoefficient = x;
}


template<typename Dimension>
inline
double
SolidFSISPHHydroBase<Dimension>::
surfaceForceCoefficient() const {
  return mSurfaceForceCoefficient;
}

template<typename Dimension>
inline
void
SolidFSISPHHydroBase<Dimension>::
densityStabilizationCoefficient(double x) {
  mDensityStabilizationCoefficient = x;
}

template<typename Dimension>
inline
double
SolidFSISPHHydroBase<Dimension>::
densityStabilizationCoefficient() const {
  return mDensityStabilizationCoefficient;
}

template<typename Dimension>
inline
void
SolidFSISPHHydroBase<Dimension>::
densityDiffusionCoefficient(double x) {
  mDensityDiffusionCoefficient = x;
}

template<typename Dimension>
inline
double
SolidFSISPHHydroBase<Dimension>::
densityDiffusionCoefficient() const {
  return mDensityDiffusionCoefficient;
}

template<typename Dimension>
inline
void
SolidFSISPHHydroBase<Dimension>::
specificThermalEnergyDiffusionCoefficient(double x) {
  mSpecificThermalEnergyDiffusionCoefficient = x;
}

template<typename Dimension>
inline
double
SolidFSISPHHydroBase<Dimension>::
specificThermalEnergyDiffusionCoefficient() const {
  return mSpecificThermalEnergyDiffusionCoefficient;
}

//------------------------------------------------------------------------------
// swtich to turn on density sum for different nodeLists
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
SolidFSISPHHydroBase<Dimension>::
sumDensityNodeLists(std::vector<int> x) {
  mSumDensityNodeLists = x;
}


template<typename Dimension>
inline
std::vector<int>
SolidFSISPHHydroBase<Dimension>::
sumDensityNodeLists() const {
  return mSumDensityNodeLists;
}

}
