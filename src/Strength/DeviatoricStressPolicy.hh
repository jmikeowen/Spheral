//---------------------------------Spheral++----------------------------------//
// DeviatoricStressPolicy.
//
// Specialized version of the IncrementFieldList policy, with some criteria for
// zeroing out the deviatoric stress in special cases.
//
// Created by JMO, Mon Feb  6 11:34:57 PST 2006
//----------------------------------------------------------------------------//
#ifndef __Spheral_DeviatoricStress_hh__
#define __Spheral_DeviatoricStress_hh__

#include "DataBase/IncrementFieldList.hh"

#include <string>

namespace Spheral {

template<typename Dimension>
class DeviatoricStressPolicy: public IncrementFieldList<Dimension, typename Dimension::SymTensor> {
public:
  //--------------------------- Public Interface ---------------------------//
  // Useful typedefs
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  typedef typename UpdatePolicyBase<Dimension>::KeyType KeyType;

  // Constructors, destructor.
  DeviatoricStressPolicy();
  virtual ~DeviatoricStressPolicy();
  
  // Overload the methods describing how to update Fields.
  virtual void update(const KeyType& key,
                      State<Dimension>& state,
                      StateDerivatives<Dimension>& derivs,
                      const double multiplier,
                      const double t,
                      const double dt);

  // Equivalence.
  virtual bool operator==(const UpdatePolicyBase<Dimension>& rhs) const;

private:
  //--------------------------- Private Interface ---------------------------//
  DeviatoricStressPolicy(const DeviatoricStressPolicy& rhs);
  DeviatoricStressPolicy& operator=(const DeviatoricStressPolicy& rhs);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class DeviatoricStressPolicy;
}

#endif
