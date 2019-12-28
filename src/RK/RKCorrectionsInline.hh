namespace Spheral {

//------------------------------------------------------------------------------
// WR
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const ReproducingKernel<Dimension>&
RKCorrections<Dimension>::
WR(const RKOrder order) const {
  const auto itr = mWR.find(order);
  VERIFY2(itr != mWR.end(),
          "RKCorrections::WR error: attempt to access for unknown correction " << order);
  return itr->second;
}

//------------------------------------------------------------------------------
// corrections
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const FieldList<Dimension, std::vector<double>>&
RKCorrections<Dimension>::
corrections(const RKOrder order) const {
  const auto itr = mCorrections.find(order);
  VERIFY2(itr != mCorrections.end(),
          "RKCorrections::corrections error: attempt to access for unknown correction " << order);
  return itr->second;
}

}
