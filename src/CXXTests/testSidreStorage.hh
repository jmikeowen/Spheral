//---------------------------------Spheral++----------------------------------//
// testSidreStorage -- Unit tests for SidreDataCollection
//
//
// Created by Mikhail Zakharchanka, 2021
//----------------------------------------------------------------------------//


#include "Field/Field.hh"
#include "Utilities/SidreDataCollection.hh"
#include "Geometry/Dimension.hh"
#include "axom/sidre.hpp"

#include "gtest/gtest.h"

using arithmeticTypes = ::testing::Types<char, int, size_t, uint32_t, uint64_t, float, double>;

template <typename T>
class SidreDataCollectionTest : public ::testing::Test
{
  public:
    Spheral::SidreDataCollection myData;
    int n = 100;
    T* rawSidreData;

    Spheral::Field<Spheral::Dim<1>, T> makeField()
    {
      Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
      Spheral::Field<Spheral::Dim<1>, T> testField("test field", makeNodeList);
      for (int i = 0; i < n; ++i)
        testField[i] = i;
      return testField;
    }

    void allocRawSidreData(const Spheral::Field<Spheral::Dim<1>, T>& testField)
    {
      rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
    }
};

template <typename T>
class SidreDataCollectionTestVector : public ::testing::Test
{
  public:
    Spheral::SidreDataCollection myData;
    int n = 10;
    T* rawSidreData;

    Spheral::Field<Spheral::Dim<1>, std::vector<T>> makeField()
    {
      Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
      Spheral::Field<Spheral::Dim<1>, std::vector<T>> testField("test field", makeNodeList);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; j++)
          testField[i].emplace_back(i);
      return testField;
    }

    void allocRawSidreData(const Spheral::Field<Spheral::Dim<1>, std::vector<T>>& testField)
    {
      rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
    }
};

template <typename T>
class SidreDataCollectionTestTupleThree : public ::testing::Test
{
  public:
    Spheral::SidreDataCollection myData;
    int n = 5;
    T* rawSidreData;

    Spheral::Field<Spheral::Dim<1>, std::tuple<T, T, T>> makeField()
    {
      Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
      Spheral::Field<Spheral::Dim<1>, std::tuple<T, T, T>> testField("test field", makeNodeList);
      for (int i = 0; i < n; ++i)
        testField[i] = std::make_tuple(i, i, i);
      return testField;
    }

    void allocRawSidreData(const Spheral::Field<Spheral::Dim<1>, std::tuple<T, T, T>>& testField)
    {
      rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
    }
};

TYPED_TEST_SUITE(SidreDataCollectionTestTupleThree, arithmeticTypes);
TYPED_TEST_SUITE(SidreDataCollectionTest, arithmeticTypes);
TYPED_TEST_SUITE(SidreDataCollectionTestVector, arithmeticTypes);

//
//
// Possible New Test Org vvvvv
//
//

template<int Dim, typename T>
struct SpheralTypeInfo {
  using DIM_T = Spheral::Dim<Dim>;
  using SPHERAL_T = T;
}; 

template<typename T_DATA>
using SpheralTestField = Spheral::Field<typename T_DATA::DIM_T, typename T_DATA::SPHERAL_T> ;
template<typename T_DATA>
using SpheralTestNodeList = Spheral::NodeList<typename T_DATA::DIM_T> ;

#define INIT_FIELD( _dim, _type)          \
template<typename T_DATA,                 \
         typename std::enable_if<         \
           std::is_same<                  \
             T_DATA,                      \
             SpheralTypeInfo<_dim, _type> \
           >::value                       \
         >::type* = nullptr>              \
void initField(SpheralTestField<T_DATA>& testField, size_t n)

#define TEST_RAW_DATA( _dim, _type)                     \
template<typename T_DATA,                               \
         typename std::enable_if<                       \
           std::is_same<                                \
             T_DATA,                                    \
             SpheralTypeInfo<_dim, _type>               \
           >::value                                     \
         >::type* = nullptr>                            \
void testSidreData(size_t n,                            \
                   SpheralTestField<T_DATA>& testField, \
                   typename Spheral::DataTypeTraits<_type>::AxomType* rawSidreData)

#define INIT_VECTOR_FIELD( _dim) \
template<typename T_DATA,        \
         typename T>             \
void initField(Spheral::Field<typename T_DATA::DIM_T, std::vector<T> >& testField, size_t n)

#define TEST_VECTOR_RAW_DATA( _dim)                                                    \
template<typename T_DATA, typename T>                                                  \
void testSidreData(size_t n,                                                           \
                   Spheral::Field<typename T_DATA::DIM_T, std::vector<T> >& testField, \
                   typename Spheral::DataTypeTraits<T>::AxomType* rawSidreData)

// ------------------------------------------------------------
// Dim 1 : std::vector<T>
// ------------------------------------------------------------
INIT_VECTOR_FIELD(1)
{
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; j++)
      testField[i].emplace_back((T)i);
}

TEST_VECTOR_RAW_DATA(1)
{
  for (size_t i = 0; i < n; ++i)
    EXPECT_EQ(testField[0][i], rawSidreData[i]);
}

// ------------------------------------------------------------
// Dim 1 : String
// ------------------------------------------------------------
INIT_FIELD(1, std::string)
{
  for (size_t i = 0; i < n; ++i)
    testField[i] = "This is a test string: " + std::to_string(i) + "\n";
}

TEST_RAW_DATA(1, std::string)
{
  for (size_t i = 0; i < n; ++i)
    EXPECT_EQ(testField[0][i], rawSidreData[i]);
}

// ------------------------------------------------------------
// Dim 1 : Vector
// ------------------------------------------------------------
INIT_FIELD(1, Spheral::Dim<1>::Vector)
{
  for (size_t i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::Vector(i);
}

TEST_RAW_DATA(1, Spheral::Dim<1>::Vector)
{
  for (size_t i = 0; i < n; ++i)
    EXPECT_EQ(testField[i].x(), rawSidreData[i]);
}

// ------------------------------------------------------------
// Dim 1 : Vector3d
// ------------------------------------------------------------
INIT_FIELD(1, Spheral::Dim<1>::Vector3d)
{
  for (size_t i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::Vector3d(i, i + 1, i + 2);
}

TEST_RAW_DATA(1, Spheral::Dim<1>::Vector3d)
{
  for (size_t i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].x(), rawSidreData[i * 3 + 0]);
    EXPECT_EQ(testField[i].y(), rawSidreData[i * 3 + 1]);
    EXPECT_EQ(testField[i].z(), rawSidreData[i * 3 + 2]);
  }
}

using SpheralTypes = ::testing::Types<
  SpheralTypeInfo<1, std::string>,
  SpheralTypeInfo<1, std::vector<char>>,
  SpheralTypeInfo<1, std::vector<double>>,
  SpheralTypeInfo<1, std::vector<int>>,
  SpheralTypeInfo<1, std::vector<size_t>>,
  SpheralTypeInfo<1, std::vector<uint32_t>>,
  SpheralTypeInfo<1, Spheral::Dim<1>::Vector>,
  SpheralTypeInfo<1, Spheral::Dim<1>::Vector3d>//,
  //SpheralTypeInfo<1, Spheral::Dim<1>::Tensor>,
  //SpheralTypeInfo<1, Spheral::Dim<1>::SymTensor>,
  //SpheralTypeInfo<1, Spheral::Dim<1>::ThirdRankTensor>,
  //SpheralTypeInfo<1, Spheral::Dim<1>::FourthRankTensor>,
  //SpheralTypeInfo<1, Spheral::Dim<1>::FifthRankTensor>
  // .... for Dim 2 and 3 as well
>;

template <typename T>
class SidreDataCollectionTestNew : public ::testing::Test {};
TYPED_TEST_SUITE(SidreDataCollectionTestNew, SpheralTypes);
