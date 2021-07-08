#include "testSidreStorage.hh"

// Should only need this single test
TYPED_TEST(SidreDataCollectionTestNew, SidreAllocViewTest)
{
  Spheral::SidreDataCollection myData;
  size_t n = 10;

  SpheralTestNodeList<TypeParam> makeNodeList("test bed", n, 0);
  SpheralTestField<TypeParam> testField("test field", makeNodeList);

  initField<TypeParam>(testField, n);

  auto rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  testSidreData<TypeParam>(n, testField, rawSidreData);
}


TYPED_TEST(SidreDataCollectionTest, scalar)
{
  auto testField = this->makeField();

  this->allocRawSidreData(testField);
  
  for (int i = 0; i < this->n; ++i)
    EXPECT_EQ(testField[i], this->rawSidreData[i]);
}

TEST(SidreDataCollectionTestDim1, string)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, std::string> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = "This is a test string: " + std::to_string(i) + "\n";
  
  char *rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[0][i], rawSidreData[i]);
}

TYPED_TEST(SidreDataCollectionTestVector, vector)
{
  auto testField = this->makeField();

  this->allocRawSidreData(testField);

  for (int i = 0; i < this->n; ++i)
    EXPECT_EQ(testField[0][i], this->rawSidreData[i]);
}

TYPED_TEST(SidreDataCollectionTestTupleThree, tuple)
{
  auto testField = this->makeField();

  this->allocRawSidreData(testField);

  EXPECT_EQ(std::get<0>(testField[0]), this->rawSidreData[0]);
  EXPECT_EQ(std::get<1>(testField[0]), this->rawSidreData[1]);
  EXPECT_EQ(std::get<2>(testField[0]), this->rawSidreData[2]);
}

//--------------------------------------------------------------

//TEST(SidreDataCollectionTest, Dim1Vector)
//{
//  Spheral::SidreDataCollection myData;
//  int n = 10;
//
//  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
//  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::Vector> testField("test field", makeNodeList);
//  for (int i = 0; i < n; ++i)
//    testField[i] = Spheral::Dim<1>::Vector(i);
//  
//  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
//
//  for (int i = 0; i < n; ++i)
//    EXPECT_EQ(testField[i].x(), rawSidreData[i]);
//}

//TEST(SidreDataCollectionTest, Dim1Vector3d)
//{
//  Spheral::SidreDataCollection myData;
//  int n = 10;
//
//  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
//  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::Vector3d> testField("test field", makeNodeList);
//  for (int i = 0; i < n; ++i)
//    testField[i] = Spheral::Dim<1>::Vector3d(i, i + 1, i + 2);
//
//  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
//
//  for (int i = 0; i < n; ++i)
//  {
//    EXPECT_EQ(testField[i].x(), rawSidreData[i * 3 + 0]);
//    EXPECT_EQ(testField[i].y(), rawSidreData[i * 3 + 1]);
//    EXPECT_EQ(testField[i].z(), rawSidreData[i * 3 + 2]);
//  }
//}

TEST(SidreDataCollectionTest, Dim1Tensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::Tensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::Tensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[i].xx(), rawSidreData[i]);
}

TEST(SidreDataCollectionTest, Dim1SymTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::SymTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::SymTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[i].xx(), rawSidreData[i]);
}

TEST(SidreDataCollectionTest, Dim1ThirdRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::ThirdRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::ThirdRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[i], rawSidreData[i]);
}

TEST(SidreDataCollectionTest, Dim1FourthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::FourthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::FourthRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[i], rawSidreData[i]);
}

TEST(SidreDataCollectionTest, Dim1FifthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<1>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<1>, Spheral::Dim<1>::FifthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<1>::FifthRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(testField[i], rawSidreData[i]);
}

//--------------------------------------------------------------

TEST(SidreDataCollectionTest, Dim2Vector)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::Vector> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::Vector(i, i);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].x(), rawSidreData[i * 2 + 0]);
    EXPECT_EQ(testField[i].y(), rawSidreData[i * 2 + 1]);
  }
}

TEST(SidreDataCollectionTest, Dim2Tensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::Tensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::Tensor(i, i + 1, i + 2, i + 3);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].xx(), rawSidreData[i * 4 + 0]);
    EXPECT_EQ(testField[i].xy(), rawSidreData[i * 4 + 1]);
    EXPECT_EQ(testField[i].yx(), rawSidreData[i * 4 + 2]);
    EXPECT_EQ(testField[i].yy(), rawSidreData[i * 4 + 3]);
  }
}

TEST(SidreDataCollectionTest, Dim2SymTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::SymTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::SymTensor(i, i + 1, i + 1, i);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();
  
  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].xx(), rawSidreData[i * 3 + 0]);
    EXPECT_EQ(testField[i].xy(), rawSidreData[i * 3 + 1]);
    EXPECT_EQ(testField[i].yy(), rawSidreData[i * 3 + 2]);
  }
}

TEST(SidreDataCollectionTest, Dim2ThirdRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::ThirdRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::ThirdRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<2>::ThirdRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<2>::ThirdRankTensor>::numElements(testField[0])) + j]);
}

TEST(SidreDataCollectionTest, Dim2FourthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::FourthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::FourthRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<2>::FourthRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<2>::FourthRankTensor>::numElements(testField[0])) + j]);
}

TEST(SidreDataCollectionTest, Dim2FifthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<2>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<2>, Spheral::Dim<2>::FifthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<2>::FifthRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<2>::FifthRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<2>::FifthRankTensor>::numElements(testField[0])) + j]);
}

//--------------------------------------------------------------

TEST(SidreDataCollectionTest, Dim3Vector)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::Vector> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::Vector(i, i, i);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].x(), rawSidreData[i * 3 + 0]);
    EXPECT_EQ(testField[i].y(), rawSidreData[i * 3 + 1]);
    EXPECT_EQ(testField[i].z(), rawSidreData[i * 3 + 2]);
  }
}

TEST(SidreDataCollectionTest, Dim3Tensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::Tensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::Tensor(i, i + 1, i + 2,
                                           i, i + 1, i + 2,
                                           i, i + 1, i + 2);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].xx(), rawSidreData[i * 9 + 0]);
    EXPECT_EQ(testField[i].xy(), rawSidreData[i * 9 + 1]);
    EXPECT_EQ(testField[i].xz(), rawSidreData[i * 9 + 2]);
    EXPECT_EQ(testField[i].yx(), rawSidreData[i * 9 + 3]);
    EXPECT_EQ(testField[i].yy(), rawSidreData[i * 9 + 4]);
    EXPECT_EQ(testField[i].yz(), rawSidreData[i * 9 + 5]);
    EXPECT_EQ(testField[i].zx(), rawSidreData[i * 9 + 6]);
    EXPECT_EQ(testField[i].zy(), rawSidreData[i * 9 + 7]);
    EXPECT_EQ(testField[i].zz(), rawSidreData[i * 9 + 8]);
  }
}

TEST(SidreDataCollectionTest, Dim3SymTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::SymTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::SymTensor(i, i + 1, i + 2,
                                              i + 1, i, i + 1,
                                              i + 2, i + 1, i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
  {
    EXPECT_EQ(testField[i].xx(), rawSidreData[i * 6 + 0]);
    EXPECT_EQ(testField[i].xy(), rawSidreData[i * 6 + 1]);
    EXPECT_EQ(testField[i].xz(), rawSidreData[i * 6 + 2]);
    EXPECT_EQ(testField[i].yy(), rawSidreData[i * 6 + 3]);
    EXPECT_EQ(testField[i].yz(), rawSidreData[i * 6 + 4]);
    EXPECT_EQ(testField[i].zz(), rawSidreData[i * 6 + 5]);
  }
}

TEST(SidreDataCollectionTest, Dim3ThirdRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::ThirdRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::ThirdRankTensor(i);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<3>::ThirdRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<3>::ThirdRankTensor>::numElements(testField[0])) + j]);
}

TEST(SidreDataCollectionTest, Dim3FourthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::FourthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::FourthRankTensor(i);

  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<3>::FourthRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<3>::FourthRankTensor>::numElements(testField[0])) + j]);
}

TEST(SidreDataCollectionTest, Dim3FifthRankTensor)
{
  Spheral::SidreDataCollection myData;
  int n = 10;

  Spheral::NodeList<Spheral::Dim<3>> makeNodeList("test bed", n, 0);
  Spheral::Field<Spheral::Dim<3>, Spheral::Dim<3>::FifthRankTensor> testField("test field", makeNodeList);
  for (int i = 0; i < n; ++i)
    testField[i] = Spheral::Dim<3>::FifthRankTensor(i);
  
  double* rawSidreData = myData.alloc_view("SidreTest", testField)->getData();

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < Spheral::DataTypeTraits<Spheral::Dim<3>::FifthRankTensor>::numElements(testField[0]); ++j)
      EXPECT_EQ(testField[i][j], rawSidreData[(i * Spheral::DataTypeTraits<Spheral::Dim<3>::FifthRankTensor>::numElements(testField[0])) + j]);
}

