//---------------------------------Spheral++----------------------------------//
// DataTypeTraits.  A simple trait class to define the number of independent
// elements in the given DataType.
//
// Created by J. Michael Owen, Thu Oct 18 16:36:21 PDT 2001
//----------------------------------------------------------------------------//
#ifndef DataTypeTraits_HH
#define DataTypeTraits_HH

#include <stdint.h>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include "Geometry/Dimension.hh"
#include "Geometry/polyclipper.hh"
#include "RegisterMPIDataTypes.hh"
#include "Utilities/DomainNode.hh"
#include "RK/RKCorrectionParams.hh"
#include "RK/RKCoefficients.hh"
#include "RAJA/RAJA.hpp"

#ifdef USE_MPI
extern "C" {
#include "mpi.h"
}
#endif

namespace Spheral {

template<typename DataType> struct DataTypeTraits;

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<bool> {
  typedef bool ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static bool zero() { return false; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_C_BOOL; }
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<char> {
  typedef char ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static int zero() { return '\0'; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_CHAR; }
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<int> {
  typedef int ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static int zero() { return 0; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_INT; }
#endif
};

#if __APPLE__
//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<size_t> {
  typedef size_t ElementType;
  static bool fixedSize() { return true; }
  static size_t numElements(const ElementType& x) { return 1; }
  static size_t zero() { return 0U; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_UNSIGNED; }
#endif
};
#endif

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<uint32_t> {
  typedef uint32_t ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static uint32_t zero() { return 0UL; }
#ifdef USE_MPI
#ifdef MPI_UINT32_T
  static MPI_Datatype MpiDataType() { return MPI_UINT32_T; }
#else
  static MPI_Datatype MpiDataType() { return MPI_UNSIGNED; }
#endif
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<uint64_t> {
  typedef uint64_t ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static uint64_t zero() { return 0ULL; }
#ifdef USE_MPI
#ifdef MPI_UINT64_T
  static MPI_Datatype MpiDataType() { return MPI_UINT64_T; }
#else
  static MPI_Datatype MpiDataType() { return MPI_UNSIGNED_LONG_LONG; }
#endif
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<float> {
  typedef float ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static float zero() { return 0.0; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_FLOAT; }
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<double> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const ElementType&) { return 1; }
  static double zero() { return 0.0; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return MPI_DOUBLE; }
#endif
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<std::string> {
  typedef std::string ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const ElementType& x) { return x.size(); }
  static ElementType zero() { return ""; }
};

//------------------------------------------------------------------------------
template<typename Value>
struct DataTypeTraits<std::vector<Value> > {
  typedef std::vector<Value> ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const std::vector<Value>& x) { return x.size(); }
  static std::vector<Value> zero() { return std::vector<Value>(); }
};

//------------------------------------------------------------------------------
template<typename Value>
struct DataTypeTraits<std::tuple<Value, Value, Value> > {
  typedef Value ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const std::tuple<Value, Value, Value>&) { return 3; }
  static std::tuple<Value, Value, Value> zero() { return std::make_tuple(Value(), Value(), Value()); }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return DataTypeTraits<Value>::MpiDataType(); }
#endif
};

//------------------------------------------------------------------------------
template<typename Value>
struct DataTypeTraits<std::tuple<Value, Value, Value, Value> > {
  typedef std::tuple<Value, Value, Value, Value> ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const std::tuple<Value, Value, Value, Value>&) { return 4; }
  static std::tuple<Value, Value, Value, Value> zero() { return std::make_tuple(Value(), Value(), Value(), Value()); }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return DataTypeTraits<Value>::MpiDataType(); }
#endif
};

//------------------------------------------------------------------------------
template<typename Value>
struct DataTypeTraits<std::tuple<Value, Value, Value, Value, Value> > {
  typedef std::tuple<Value, Value, Value, Value, Value> ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const std::tuple<Value, Value, Value, Value, Value>&) { return 5; }
  static std::tuple<Value, Value, Value, Value, Value> zero() { return std::make_tuple(Value(), Value(), Value(), Value(), Value()); }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return DataTypeTraits<Value>::MpiDataType(); }
#endif
};

//------------------------------------------------------------------------------
template<typename Value1, typename Value2>
struct DataTypeTraits<std::pair<Value1, Value2> > {
  typedef std::pair<Value1, Value2> ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const std::pair<Value1, Value2>&) { return 2; }
  static std::pair<Value1, Value2> zero() { return std::make_pair(Value1(), Value2()); }
};

//------------------------------------------------------------------------------
template<typename Value1, typename Value2, typename Hash>
struct DataTypeTraits<std::unordered_map<Value1, Value2, Hash> > {
  typedef std::pair<Value1, Value2> ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const std::unordered_map<Value1, Value2, Hash>& x) { return x.size(); }
  static std::unordered_map<Value1, Value2, Hash> zero() { return std::unordered_map<Value1, Value2, Hash>(); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<Dim<1>::Vector> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::Vector&) { return Dim<1>::Vector::numElements; }
  static Dim<1>::Vector zero() { return Dim<1>::Vector::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Vector1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::Vector3d> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::Vector3d&) { return Dim<1>::Vector3d::numElements; }
  static Dim<1>::Vector3d zero() { return Dim<1>::Vector3d::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Vector3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::Tensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::Tensor&) { return Dim<1>::Tensor::numElements; }
  static Dim<1>::Tensor zero() { return Dim<1>::Tensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Tensor1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::SymTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::SymTensor&) { return Dim<1>::SymTensor::numElements; }
  static Dim<1>::SymTensor zero() { return Dim<1>::SymTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_SymTensor1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::ThirdRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::ThirdRankTensor&) { return Dim<1>::ThirdRankTensor::numElements; }
  static Dim<1>::ThirdRankTensor zero() { return Dim<1>::ThirdRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_ThirdRankTensor1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::FourthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::FourthRankTensor&) { return Dim<1>::FourthRankTensor::numElements; }
  static Dim<1>::FourthRankTensor zero() { return Dim<1>::FourthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FourthRankTensor1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::FifthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<1>::FifthRankTensor&) { return Dim<1>::FifthRankTensor::numElements; }
  static Dim<1>::FifthRankTensor zero() { return Dim<1>::FifthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FifthRankTensor1d; }
#endif
};

template<>
struct DataTypeTraits<Dim<1>::FacetedVolume> {
  typedef double ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const Dim<1>::FacetedVolume) { return 0; }
  static Dim<1>::FacetedVolume zero() { return Dim<1>::FacetedVolume(); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<Dim<2>::Vector> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::Vector&) { return Dim<2>::Vector::numElements; }
  static Dim<2>::Vector zero() { return Dim<2>::Vector::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Vector2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::Tensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::Tensor&) { return Dim<2>::Tensor::numElements; }
  static Dim<2>::Tensor zero() { return Dim<2>::Tensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Tensor2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::SymTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::SymTensor&) { return Dim<2>::SymTensor::numElements; }
  static Dim<2>::SymTensor zero() { return Dim<2>::SymTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_SymTensor2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::ThirdRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::ThirdRankTensor&) { return Dim<2>::ThirdRankTensor::numElements; }
  static Dim<2>::ThirdRankTensor zero() { return Dim<2>::ThirdRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_ThirdRankTensor2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::FourthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::FourthRankTensor&) { return Dim<2>::FourthRankTensor::numElements; }
  static Dim<2>::FourthRankTensor zero() { return Dim<2>::FourthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FourthRankTensor2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::FifthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<2>::FifthRankTensor&) { return Dim<2>::FifthRankTensor::numElements; }
  static Dim<2>::FifthRankTensor zero() { return Dim<2>::FifthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FifthRankTensor2d; }
#endif
};

template<>
struct DataTypeTraits<Dim<2>::FacetedVolume> {
  typedef double ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const Dim<2>::FacetedVolume) { return 0; }
  static Dim<2>::FacetedVolume zero() { return Dim<2>::FacetedVolume(); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<Dim<3>::Vector> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::Vector&) { return Dim<3>::Vector::numElements; }
  static Dim<3>::Vector zero() { return Dim<3>::Vector::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Vector3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::Tensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::Tensor&) { return Dim<3>::Tensor::numElements; }
  static Dim<3>::Tensor zero() { return Dim<3>::Tensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_Tensor3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::SymTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::SymTensor&) { return Dim<3>::SymTensor::numElements; }
  static Dim<3>::SymTensor zero() { return Dim<3>::SymTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_SymTensor3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::ThirdRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::ThirdRankTensor&) { return Dim<3>::ThirdRankTensor::numElements; }
  static Dim<3>::ThirdRankTensor zero() { return Dim<3>::ThirdRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_ThirdRankTensor3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::FourthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::FourthRankTensor&) { return Dim<3>::FourthRankTensor::numElements; }
  static Dim<3>::FourthRankTensor zero() { return Dim<3>::FourthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FourthRankTensor3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::FifthRankTensor> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const Dim<3>::FifthRankTensor&) { return Dim<3>::FifthRankTensor::numElements; }
  static Dim<3>::FifthRankTensor zero() { return Dim<3>::FifthRankTensor::zero; }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return RegisterMPIDataTypes::instance().MPI_FifthRankTensor3d; }
#endif
};

template<>
struct DataTypeTraits<Dim<3>::FacetedVolume> {
  typedef double ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const Dim<3>::FacetedVolume) { return 0; }
  static Dim<3>::FacetedVolume zero() { return Dim<3>::FacetedVolume(); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<PolyClipper::Vertex2d> {
  typedef PolyClipper::Vertex2d ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const ElementType&) { return (DataTypeTraits<Dim<2>::Vector>::numElements(Dim<2>::Vector::zero) + 4); }
  static ElementType zero() { return PolyClipper::Vertex2d(); }
};

template<>
struct DataTypeTraits<PolyClipper::Vertex3d> {
  typedef PolyClipper::Vertex3d ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const ElementType& x) { return (DataTypeTraits<Dim<3>::Vector>::numElements(Dim<3>::Vector::zero) +
                                                         x.neighbors.size() +
                                                         2); }
  static ElementType zero() { return PolyClipper::Vertex3d(); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<PolyClipper::Plane2d> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const PolyClipper::Plane2d&) { return 6; }
  static PolyClipper::Plane2d zero() { return PolyClipper::Plane2d(); }
};

template<>
struct DataTypeTraits<PolyClipper::Plane3d> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const PolyClipper::Plane3d&) { return 8; }
  static PolyClipper::Plane3d zero() { return PolyClipper::Plane3d(); }
};

//------------------------------------------------------------------------------
template<int ndim>
struct DataTypeTraits<DomainNode<Dim<ndim>>> {
  typedef double ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const DomainNode<Dim<ndim>>&) { return DomainNode<Dim<ndim>>::packSize(); }
  static DomainNode<Dim<ndim>> zero() { return DomainNode<Dim<ndim>>({0, 0, 0, 0, 0, 0.0, Dim<ndim>::Vector::zero}); }
};

//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<RKOrder> {
  typedef int ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const RKOrder&) { return 1; }
  static RKOrder zero() { return RKOrder::ZerothOrder; }
};

//------------------------------------------------------------------------------
template<int ndim>
struct DataTypeTraits<RKCoefficients<Dim<ndim>>> {
  typedef double ElementType;
  static bool fixedSize() { return false; }
  static int numElements(const RKCoefficients<Dim<ndim>>& x) { return x.size() + 1; }
  static RKCoefficients<Dim<ndim>> zero() { return RKCoefficients<Dim<ndim>>(); }
};

template<typename POLICY, typename T>
struct DataTypeTraits<RAJA::ReduceSum<POLICY, T>> {
  typedef T ElementType;
  static bool fixedSize() { return DataTypeTraits<T>::fixedSize(); }
  static int numElements(const RAJA::ReduceSum<POLICY, T>&) { return DataTypeTraits<T>::numElements(T()); }
  static T zero() { return DataTypeTraits<T>::zero(); }
#ifdef USE_MPI
  static MPI_Datatype MpiDataType() { return DataTypeTraits<T>::MpiDataType; }
#endif
};

}

#else

namespace Spheral {
  template<typename DataType> struct DataTypeTraits;
}

#endif
