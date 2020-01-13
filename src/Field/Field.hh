//---------------------------------Spheral++----------------------------------//
// Field -- Provide a field of a type (Scalar, Vector, Tensor) over the nodes
//          in a NodeList.
//
// This version of the Field is based on standard constructs like the STL
// vector.  This will certainly be slower at run time than the Blitz Array
// class.
//
// Created by JMO, Thu Jun 10 23:26:50 PDT 1999
//----------------------------------------------------------------------------//
#ifndef __Spheral_Field_hh__
#define __Spheral_Field_hh__

#include "FieldBase.hh"

#include <string>
#include <vector>

#ifdef USE_UVM
#include "uvm_allocator.hh"
#endif

namespace Spheral {

template<typename Dimension> class NodeIteratorBase;
template<typename Dimension> class CoarseNodeIterator;
template<typename Dimension> class RefineNodeIterator;
template<typename Dimension> class NodeList;
template<typename Dimension> class TableKernel;

#ifdef USE_UVM
template<typename DataType>
using DataAllocator = typename uvm_allocator::UVMAllocator<DataType>;
#else
template<typename DataType>
using DataAllocator = std::allocator<DataType>;
#endif

template<typename Dimension, typename DataType>
class Field: 
    public FieldBase<Dimension> {
   
public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  
  typedef typename FieldBase<Dimension>::FieldName FieldName;
  typedef DataType FieldDataType;
  typedef DataType value_type;      // STL compatibility.

  typedef typename std::vector<DataType,DataAllocator<DataType>>::iterator iterator;
  typedef typename std::vector<DataType,DataAllocator<DataType>>::const_iterator const_iterator;

  // Constructors.
  explicit Field(FieldName name);
  Field(FieldName name, const Field& field);
  Field(FieldName name,
        const NodeList<Dimension>& nodeList);
  Field(FieldName name,
        const NodeList<Dimension>& nodeList,
        DataType value);
  Field(FieldName name,
        const NodeList<Dimension>& nodeList, 
        const std::vector<DataType,DataAllocator<DataType>>& array);
  Field(const NodeList<Dimension>& nodeList, const Field& field);
  Field(const Field& field);
  virtual std::shared_ptr<FieldBase<Dimension> > clone() const;

  // Destructor.
  virtual ~Field();

  // Assignment operator.
  virtual FieldBase<Dimension>& operator=(const FieldBase<Dimension>& rhs);
  Field& operator=(const Field& rhs);
  Field& operator=(const std::vector<DataType,DataAllocator<DataType>>& rhs);
  Field& operator=(const DataType& rhs);

  // Required method to test equivalence with a FieldBase.
  virtual bool operator==(const FieldBase<Dimension>& rhs) const;

  // Element access.
  DataType& operator()(int index);
  const DataType& operator()(int index) const;

  DataType& operator()(const NodeIteratorBase<Dimension>& itr);
  const DataType& operator()(const NodeIteratorBase<Dimension>& itr) const;

  DataType& at(int index);
  const DataType& at(int index) const;

  // The number of elements in the field.
  unsigned numElements() const;
  unsigned numInternalElements() const;
  unsigned numGhostElements() const;
  virtual unsigned size() const;

  // Zero out the field elements.
  virtual void Zero();

  // Methods to apply limits to Field data members.
  void applyMin(const DataType& dataMin);
  void applyMax(const DataType& dataMax);

  void applyScalarMin(const double dataMin);
  void applyScalarMax(const double dataMax);

  // Standard field additive operators.
  Field operator+(const Field& rhs) const;
  Field operator-(const Field& rhs) const;

  Field& operator+=(const Field& rhs);
  Field& operator-=(const Field& rhs);

  Field operator+(const DataType& rhs) const;
  Field operator-(const DataType& rhs) const;

  Field& operator+=(const DataType& rhs);
  Field& operator-=(const DataType& rhs);

//   // Multiplication of two fields, possibly by another DataType.
//   template<typename OtherDataType>
//   Field<Dimension, typename CombineTypes<DataType, OtherDataType>::ProductType>
//   operator*(const Field<Dimension, OtherDataType>& rhs) const;

//   template<typename OtherDataType>
//   Field<Dimension, typename CombineTypes<DataType, OtherDataType>::ProductType>
//   operator*(const OtherDataType& rhs) const;

  Field<Dimension, DataType>& operator*=(const Field<Dimension, Scalar>& rhs);
  Field<Dimension, DataType>& operator*=(const Scalar& rhs);

  // Division.  Only meaningful when dividing by a scalar field.
  Field<Dimension, DataType> operator/(const Field<Dimension, Scalar>& rhs) const;
  Field<Dimension, DataType>& operator/=(const Field<Dimension, Scalar>& rhs);
       
  Field<Dimension, DataType> operator/(const Scalar& rhs) const;
  Field<Dimension, DataType>& operator/=(const Scalar& rhs);

  // Some useful reduction operations.
  DataType sumElements() const;
  DataType min() const;
  DataType max() const;

  // Some useful reduction operations (local versions -- no MPI reductions)
  DataType localSumElements() const;
  DataType localMin() const;
  DataType localMax() const;

  // Comparison operators (Field-Field element wise).
  bool operator==(const Field& rhs) const;
  bool operator!=(const Field& rhs) const;
  bool operator>(const Field& rhs) const;
  bool operator<(const Field& rhs) const;
  bool operator>=(const Field& rhs) const;
  bool operator<=(const Field& rhs) const;

  // Comparison operators (Field-value element wise).
  bool operator==(const DataType& rhs) const;
  bool operator!=(const DataType& rhs) const;
  bool operator>(const DataType& rhs) const;
  bool operator<(const DataType& rhs) const;
  bool operator>=(const DataType& rhs) const;
  bool operator<=(const DataType& rhs) const;

//   // Interpolate from this Field onto the given position.  Assumes that the
//   // neighbor initializations have already been performed for the given
//   // position!
//   DataType operator()(const Vector& r,
//                       const TableKernel<Dimension>& W) const;

//   // Interpolate from this Field onto a new Field defined at the positions
//   // of the given NodeList.
//   Field<Dimension, DataType>
//   sampleField(const NodeList<Dimension>& splatNodeList,
//               const TableKernel<Dimension>& W) const;

//   // Conservatively splat values from this Field onto a new Field defined
//   // at the positions of the given NodeList, using the MASH formalism.
//   Field<Dimension, DataType>
//   splatToFieldMash(const NodeList<Dimension>& splatNodeList,
//                    const TableKernel<Dimension>& W) const;

  // Test if this Field is in a valid, internally consistent state.
  bool valid() const;

  // Provide the standard iterator methods over the field.
  iterator begin();
  iterator end();
  iterator internalBegin();
  iterator internalEnd();
  iterator ghostBegin();
  iterator ghostEnd();

  const_iterator begin() const;
  const_iterator end() const;
  const_iterator internalBegin() const;
  const_iterator internalEnd() const;
  const_iterator ghostBegin() const;
  const_iterator ghostEnd() const;

  // Index operator.
  DataType& operator[](const unsigned int index);
  const DataType& operator[](const unsigned int index) const;

  // Required functions from FieldBase
  virtual void setNodeList(const NodeList<Dimension>& nodeList);
  virtual void resizeField(unsigned size);
  virtual void resizeFieldInternal(unsigned size, unsigned oldFirstGhostNode);
  virtual void resizeFieldGhost(unsigned size);
  virtual void deleteElement(int nodeID);
  virtual void deleteElements(const std::vector<int>& nodeIDs);
  virtual std::vector<char> packValues(const std::vector<int>& nodeIDs) const;
  virtual void unpackValues(const std::vector<int>& nodeIDs,
                            const std::vector<char>& buffer);
  virtual void copyElements(const std::vector<int>& fromIndices,
                            const std::vector<int>& toIndices);
  virtual bool fixedSizeDataType() const override;
  virtual bool numValsInDataType() const override;
  virtual int sizeofDataType() const override;

  // Methods to use the iostream methods converting a Field to/from a string.
  std::string string(const int precision = 20) const;
  void string(const std::string& s);

  // Provide std::vector copies of the data.  This is mostly useful for the
  // python interface.
  std::vector<DataType> internalValues() const;
  std::vector<DataType> ghostValues() const;
  std::vector<DataType> allValues() const;

private:
  //--------------------------- Private Interface ---------------------------//
  // Private Data
//  std::vector<DataType,std::allocator<DataType> > mDataArray;
  std::vector<DataType, DataAllocator<DataType>> mDataArray;
  bool mValid;

  // No default constructor.
  Field();
};

}

#include "FieldInline.hh"

#else

// Forward declare the Field class.
namespace Spheral {
  template<typename Dimension, typename DataType> class Field;
}

#endif
