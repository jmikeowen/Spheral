//---------------------------------Spheral++----------------------------------//
// GeomVector -- Geometric Vector Class.
//
// Created by JMO, Thu Apr 15 17:39:47 PDT 1999
// Modified by:
//   July 31, 99: JMO, removing MTL based version due to problems compiling
//                MTL code with gcc2.95.
//   2004-08-19:  JMO, switch to using boost::ublas, and our vector class is
//                just an interface.
//   2004-08-23:  JMO, ublas is still too slow, so going to primitive C 
//                internal data types in accordance with suggestions from
//                Brian White
//----------------------------------------------------------------------------//
#ifndef __Spheral_GeomVector_default_hh__
#define __Spheral_GeomVector_default_hh__

#include "RAJA/RAJA.hpp"

#include "Geometry/GeomVector_fwd.hh"
#include "Geometry/GeomTensor_fwd.hh"
#include "Geometry/GeomSymmetricTensor_fwd.hh"
#include "GeomVectorBase_default.hh"

#include <iostream>
#include "Eigen/Dense"

namespace Spheral {

template<int nDim>
class GeomVector: public GeomVectorBase<nDim> {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef const double* const_iterator;
  typedef double* iterator;
  typedef unsigned size_type;
  typedef Eigen::Matrix<double, nDim, 1> EigenType;

  // Useful static member data.
  static const size_type nDimensions;
  static const size_type numElements;
  static const GeomVector zero;
  static const GeomVector one;

  // Constructors.
  RAJA_HOST_DEVICE explicit GeomVector(const double x = 0.0,
                      const double y = 0.0,
                      const double z = 0.0);
  GeomVector(const GeomVector& vec);
  template<typename Derived> explicit GeomVector(const Eigen::MatrixBase<Derived>& vec);

  // Destructor.
  RAJA_HOST_DEVICE ~GeomVector() = default;

  // Assignment.
  GeomVector& operator=(const GeomVector<nDim>& vec);
  GeomVector& operator=(const double val);
  template<typename Derived> GeomVector& operator=(const Eigen::MatrixBase<Derived>& vec);

  // Allow the elements by indicies.
  double operator()(size_type index) const;
  double& operator()(size_type index);

  // More C++ style indexing.
  RAJA_HOST_DEVICE double operator[](size_type index) const;
  RAJA_HOST_DEVICE double& operator[](size_type index);

  // Access the individual elements by (x, y, z) notation.
  double x() const;
  double y() const;
  double z() const;
  void x(const double val);
  void y(const double val);
  void z(const double val);

  // Iterator access to the raw data.
  RAJA_HOST_DEVICE iterator begin();
  RAJA_HOST_DEVICE iterator end();

  RAJA_HOST_DEVICE const_iterator begin() const;
  RAJA_HOST_DEVICE const_iterator end() const;

  // Zero the vector.
  void Zero();

  // Mathematical operators.
  GeomVector operator-() const;

  GeomVector operator+(const GeomVector& vec) const;
  GeomVector operator-(const GeomVector& vec) const;
  GeomVector operator*(const double val) const;
  GeomVector operator/(const double val) const;

  GeomVector& operator+=(const GeomVector& vec);
  GeomVector& operator-=(const GeomVector& vec);

  template<typename Derived> GeomVector& operator+=(const Eigen::MatrixBase<Derived>& vec);
  template<typename Derived> GeomVector& operator-=(const Eigen::MatrixBase<Derived>& vec);

  GeomVector& operator*=(const double val);
  GeomVector& operator/=(const double val);

  int compare(const GeomVector& vec) const;
  int compare(const double val) const;

  bool operator==(const GeomVector& vec) const;
  bool operator!=(const GeomVector& vec) const;
  bool operator<(const GeomVector& vec) const;
  bool operator>(const GeomVector& vec) const;
  bool operator<=(const GeomVector& vec) const;
  bool operator>=(const GeomVector& vec) const;

  bool operator==(const double val) const;
  bool operator!=(const double val) const;
  bool operator<(const double val) const;
  bool operator>(const double val) const;
  bool operator<=(const double val) const;
  bool operator>=(const double val) const;

  double dot(const GeomVector& vec) const;
  GeomVector<3> cross(const GeomVector& vec) const;
  GeomTensor<nDim> dyad(const GeomVector& rhs) const;
  GeomSymmetricTensor<nDim> selfdyad() const;
  GeomTensor<nDim> operator*(const GeomVector& vec) const;

  GeomVector unitVector() const;

  double magnitude() const;
  double magnitude2() const;
  double minElement() const;
  double maxElement() const;
  double maxAbsElement() const;
  double sumElements() const;
  
  //  Convert to an Eigen Vector
  EigenType eigen() const;
};

// Declare explicit specializations.
template<> GeomVector<1>::GeomVector(const double, const double, const double);
template<> GeomVector<2>::GeomVector(const double, const double, const double);
template<> GeomVector<3>::GeomVector(const double, const double, const double);

template<> GeomVector<1>& GeomVector<1>::operator=(const GeomVector<1>& vec);
template<> GeomVector<2>& GeomVector<2>::operator=(const GeomVector<2>& vec);
template<> GeomVector<3>& GeomVector<3>::operator=(const GeomVector<3>& vec);

template<> GeomVector<1>& GeomVector<1>::operator=(const double val);
template<> GeomVector<2>& GeomVector<2>::operator=(const double val);
template<> GeomVector<3>& GeomVector<3>::operator=(const double val);

// template<> GeomVector<1>& GeomVector<1>::operator=(const GeomVector<1>::EigenType& vec);
// template<> GeomVector<2>& GeomVector<2>::operator=(const GeomVector<2>::EigenType& vec);
// template<> GeomVector<3>& GeomVector<3>::operator=(const GeomVector<3>::EigenType& vec);

template<> double GeomVector<1>::y() const;
template<> double GeomVector<1>::z() const;
template<> double GeomVector<2>::z() const;

template<> void GeomVector<1>::y(const double val);
template<> void GeomVector<1>::z(const double val);
template<> void GeomVector<2>::z(const double val);

template<> void GeomVector<1>::Zero();
template<> void GeomVector<2>::Zero();
template<> void GeomVector<3>::Zero();

template<> GeomVector<1> GeomVector<1>::operator-() const;
template<> GeomVector<2> GeomVector<2>::operator-() const;
template<> GeomVector<3> GeomVector<3>::operator-() const;

template<> GeomVector<1>& GeomVector<1>::operator+=(const GeomVector<1>& vec);
template<> GeomVector<2>& GeomVector<2>::operator+=(const GeomVector<2>& vec);
template<> GeomVector<3>& GeomVector<3>::operator+=(const GeomVector<3>& vec);

template<> GeomVector<1>& GeomVector<1>::operator-=(const GeomVector<1>& vec);
template<> GeomVector<2>& GeomVector<2>::operator-=(const GeomVector<2>& vec);
template<> GeomVector<3>& GeomVector<3>::operator-=(const GeomVector<3>& vec);

// template<> GeomVector<1>& GeomVector<1>::operator+=(const GeomVector<1>::EigenType& vec);
// template<> GeomVector<2>& GeomVector<2>::operator+=(const GeomVector<2>::EigenType& vec);
// template<> GeomVector<3>& GeomVector<3>::operator+=(const GeomVector<3>::EigenType& vec);

// template<> GeomVector<1>& GeomVector<1>::operator-=(const GeomVector<1>::EigenType& vec);
// template<> GeomVector<2>& GeomVector<2>::operator-=(const GeomVector<2>::EigenType& vec);
// template<> GeomVector<3>& GeomVector<3>::operator-=(const GeomVector<3>::EigenType& vec);

#if defined(_OPENMP) && _OPENMP >= 201107
#pragma omp declare reduction(vecadd : GeomVector<1> : omp_out += omp_in ) initializer( omp_priv = GeomVector<1>(0.0,0.0,0.0) )
#pragma omp declare reduction(vecdif : GeomVector<1> : omp_out -= omp_in ) initializer( omp_priv = GeomVector<1>(0.0,0.0,0.0) )
#pragma omp declare reduction(vecadd : GeomVector<2> : omp_out += omp_in ) initializer( omp_priv = GeomVector<2>(0.0,0.0,0.0) )
#pragma omp declare reduction(vecdif : GeomVector<2> : omp_out -= omp_in ) initializer( omp_priv = GeomVector<2>(0.0,0.0,0.0) )
#pragma omp declare reduction(vecadd : GeomVector<3> : omp_out += omp_in ) initializer( omp_priv = GeomVector<3>(0.0,0.0,0.0) )
#pragma omp declare reduction(vecdif : GeomVector<3> : omp_out -= omp_in ) initializer( omp_priv = GeomVector<3>(0.0,0.0,0.0) )
#endif

template<> GeomVector<1>& GeomVector<1>::operator*=(const double val);
template<> GeomVector<2>& GeomVector<2>::operator*=(const double val);
template<> GeomVector<3>& GeomVector<3>::operator*=(const double val);

template<> GeomVector<1>& GeomVector<1>::operator/=(const double val);
template<> GeomVector<2>& GeomVector<2>::operator/=(const double val);
template<> GeomVector<3>& GeomVector<3>::operator/=(const double val);

template<> int GeomVector<1>::compare(const GeomVector<1>& vec) const;
template<> int GeomVector<2>::compare(const GeomVector<2>& vec) const;
template<> int GeomVector<3>::compare(const GeomVector<3>& vec) const;

template<> int GeomVector<1>::compare(const double val) const;
template<> int GeomVector<2>::compare(const double val) const;
template<> int GeomVector<3>::compare(const double val) const;

template<> bool GeomVector<1>::operator==(const GeomVector<1>& vec) const;
template<> bool GeomVector<2>::operator==(const GeomVector<2>& vec) const;
template<> bool GeomVector<3>::operator==(const GeomVector<3>& vec) const;

template<> bool GeomVector<1>::operator==(const double val) const;
template<> bool GeomVector<2>::operator==(const double val) const;
template<> bool GeomVector<3>::operator==(const double val) const;

template<> double GeomVector<1>::dot(const GeomVector<1>& vec) const;
template<> double GeomVector<2>::dot(const GeomVector<2>& vec) const;
template<> double GeomVector<3>::dot(const GeomVector<3>& vec) const;

template<> GeomVector<3> GeomVector<1>::cross(const GeomVector<1>& vec) const;
template<> GeomVector<3> GeomVector<2>::cross(const GeomVector<2>& vec) const;
template<> GeomVector<3> GeomVector<3>::cross(const GeomVector<3>& vec) const;

template<> GeomTensor<1> GeomVector<1>::dyad(const GeomVector<1>& rhs) const;
template<> GeomTensor<2> GeomVector<2>::dyad(const GeomVector<2>& rhs) const;
template<> GeomTensor<3> GeomVector<3>::dyad(const GeomVector<3>& rhs) const;

template<> GeomSymmetricTensor<1> GeomVector<1>::selfdyad() const;
template<> GeomSymmetricTensor<2> GeomVector<2>::selfdyad() const;
template<> GeomSymmetricTensor<3> GeomVector<3>::selfdyad() const;

template<> double GeomVector<1>::magnitude() const;
template<> double GeomVector<2>::magnitude() const;
template<> double GeomVector<3>::magnitude() const;

template<> double GeomVector<1>::magnitude2() const;
template<> double GeomVector<2>::magnitude2() const;
template<> double GeomVector<3>::magnitude2() const;

template<> double GeomVector<1>::minElement() const;
template<> double GeomVector<2>::minElement() const;
template<> double GeomVector<3>::minElement() const;

template<> double GeomVector<1>::maxElement() const;
template<> double GeomVector<2>::maxElement() const;
template<> double GeomVector<3>::maxElement() const;

template<> double GeomVector<1>::maxAbsElement() const;
template<> double GeomVector<2>::maxAbsElement() const;
template<> double GeomVector<3>::maxAbsElement() const;

template<> double GeomVector<1>::sumElements() const;
template<> double GeomVector<2>::sumElements() const;
template<> double GeomVector<3>::sumElements() const;

// template<> GeomVector<1>::EigenType GeomVector<1>::eigen() const;
// template<> GeomVector<2>::EigenType GeomVector<2>::eigen() const;
// template<> GeomVector<3>::EigenType GeomVector<3>::eigen() const;

// Forward declare the global functions.
template<int nDim> GeomVector<nDim> elementWiseMin(const GeomVector<nDim>& lhs,
                                                   const GeomVector<nDim>& rhs);
template<int nDim> GeomVector<nDim> elementWiseMax(const GeomVector<nDim>& lhs,
                                                   const GeomVector<nDim>& rhs);

template<> GeomVector<1> elementWiseMin(const GeomVector<1>& lhs,
                                        const GeomVector<1>& rhs);
template<> GeomVector<2> elementWiseMin(const GeomVector<2>& lhs,
                                        const GeomVector<2>& rhs);
template<> GeomVector<3> elementWiseMin(const GeomVector<3>& lhs,
                                        const GeomVector<3>& rhs);

template<> GeomVector<1> elementWiseMax(const GeomVector<1>& lhs,
                                        const GeomVector<1>& rhs);
template<> GeomVector<2> elementWiseMax(const GeomVector<2>& lhs,
                                        const GeomVector<2>& rhs);
template<> GeomVector<3> elementWiseMax(const GeomVector<3>& lhs,
                                        const GeomVector<3>& rhs);

template<int nDim> std::istream& operator>>(std::istream& is, GeomVector<nDim>& vec);
template<int nDim> std::ostream& operator<<(std::ostream& os, const GeomVector<nDim>& vec);

}

#include "GeomVectorInline_default.hh"

#endif

