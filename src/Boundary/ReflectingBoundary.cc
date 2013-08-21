//---------------------------------Spheral++----------------------------------//
// ReflectingBoundary -- Apply a Reflecting boundary condition to Spheral++
// Fields.
//
// Created by JMO, Wed Feb 16 21:01:02 PST 2000
//----------------------------------------------------------------------------//

#include "ReflectingBoundary.hh"
#include "Geometry/GeomPlane.hh"
#include "Geometry/innerProduct.hh"
#include "Field/Field.hh"
#include "Utilities/DBC.hh"
#include "FileIO/FileIO.hh"
#include "Utilities/planarReflectingOperator.hh"
#include "Mesh/Mesh.hh"

using namespace std;

namespace Spheral {
namespace BoundarySpace {

using NodeSpace::NodeList;
using FieldSpace::Field;
using FieldSpace::FieldList;
using DataBaseSpace::DataBase;
using MeshSpace::Mesh;
using Geometry::innerProduct;

//------------------------------------------------------------------------------
// Empty constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
ReflectingBoundary<Dimension>::ReflectingBoundary():
  PlanarBoundary<Dimension>() {
}

//------------------------------------------------------------------------------
// Construct with the given plane.
//------------------------------------------------------------------------------
template<typename Dimension>
ReflectingBoundary<Dimension>::
ReflectingBoundary(const GeomPlane<Dimension>& plane):
  PlanarBoundary<Dimension>(plane, plane) {

  // Once the plane has been set, construct the reflection operator.
  mReflectOperator = planarReflectingOperator(plane);

  // Once we're done the boundary condition should be in a valid state.
  ENSURE(valid());
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
ReflectingBoundary<Dimension>::~ReflectingBoundary() {
}

//------------------------------------------------------------------------------
// Apply the ghost boundary condition to fields of different DataTypes.
//------------------------------------------------------------------------------
// Specialization for int fields, just perform a copy.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, int>& field) const {

  REQUIRE(valid());

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
    field(*ghostItr) = field(*controlItr);
  }
}

// Specialization for scalar fields, just perform a copy.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, typename Dimension::Scalar>& field) const {

  REQUIRE(valid());

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
    field(*ghostItr) = field(*controlItr);
  }
}

// Specialization for Vector fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, typename Dimension::Vector>& field) const {

  REQUIRE(valid());

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
    field(*ghostItr) = reflectOperator()*field(*controlItr);
  }
}

// Specialization for Tensor fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, typename Dimension::Tensor>& field) const {

  REQUIRE(valid());

  // Duh!  The inverse of a reflection operator is itself.
//   // Get the inverse of the reflection operator, which is just the transpose.
//   Tensor inverseReflectOperator = reflectOperator().Transpose();

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
//     field(*ghostItr) = inverseReflectOperator*field(*controlItr)*reflectOperator();
    field(*ghostItr) = reflectOperator()*(field(*controlItr)*reflectOperator());
  }
}

// Specialization for symmetric tensors.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, typename Dimension::SymTensor>& field) const {

  REQUIRE(valid());

  // Duh!  The inverse of a reflection operator is itself.
//   // Get the inverse of the reflection operator, which is just the transpose.
//   Tensor inverseReflectOperator = reflectOperator().Transpose();

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
//     field(*ghostItr) = (inverseReflectOperator*field(*controlItr)*reflectOperator()).Symmetric();
    field(*ghostItr) = (reflectOperator()*(field(*controlItr)*reflectOperator())).Symmetric();
  }
}

// Specialization for ThirdRankTensor fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, typename Dimension::ThirdRankTensor>& field) const {

  REQUIRE(valid());

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
    field(*ghostItr) = innerProduct<Dimension>(reflectOperator(), innerProduct<Dimension>(field(*controlItr), reflectOperator()));
  }
}

// Specialization for vector<scalar> fields, just perform a copy.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
applyGhostBoundary(Field<Dimension, std::vector<typename Dimension::Scalar> >& field) const {

  REQUIRE(valid());

  // Apply the boundary condition to all the ghost node values.
  const NodeList<Dimension>& nodeList = field.nodeList();
  CHECK(this->controlNodes(nodeList).size() == this->ghostNodes(nodeList).size());
  vector<int>::const_iterator controlItr = this->controlBegin(nodeList);
  vector<int>::const_iterator ghostItr = this->ghostBegin(nodeList);
  for (; controlItr < this->controlEnd(nodeList); ++controlItr, ++ghostItr) {
    CHECK(ghostItr < this->ghostEnd(nodeList));
    CHECK(*controlItr >= 0 && *controlItr < nodeList.numNodes());
    CHECK(*ghostItr >= nodeList.firstGhostNode() && *ghostItr < nodeList.numNodes());
    field(*ghostItr) = field(*controlItr);
  }
}

//------------------------------------------------------------------------------
// Enforce the boundary condition on the set of nodes in violation of the 
// boundary.
//------------------------------------------------------------------------------
// Specialization for int fields.  A no-op.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, int>& field) const {
  REQUIRE(valid());
}

// Specialization for scalar fields.  A no-op.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, typename Dimension::Scalar>& field) const {
  REQUIRE(valid());
}

// Specialization for vector fields.  Apply the reflection operator.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, typename Dimension::Vector>& field) const {
  REQUIRE(valid());

  const NodeList<Dimension>& nodeList = field.nodeList();
  for (vector<int>::const_iterator itr = this->violationBegin(nodeList);
       itr < this->violationEnd(nodeList); 
       ++itr) {
    CHECK(*itr >= 0 && *itr < nodeList.numInternalNodes());
    field(*itr) = reflectOperator()*field(*itr);
  }
}

// Specialization for tensor fields.  Apply the reflection operator.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, typename Dimension::Tensor>& field) const {
  REQUIRE(valid());

  const NodeList<Dimension>& nodeList = field.nodeList();
  for (vector<int>::const_iterator itr = this->violationBegin(nodeList);
       itr < this->violationEnd(nodeList); 
       ++itr) {
    CHECK(*itr >= 0 && *itr < nodeList.numInternalNodes());
    field(*itr) = reflectOperator()*field(*itr)*reflectOperator();
  }
}

// Specialization for tensor fields.  Apply the reflection operator.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, typename Dimension::SymTensor>& field) const {
  REQUIRE(valid());

  const NodeList<Dimension>& nodeList = field.nodeList();
  for (vector<int>::const_iterator itr = this->violationBegin(nodeList);
       itr < this->violationEnd(nodeList); 
       ++itr) {
    CHECK(*itr >= 0 && *itr < nodeList.numInternalNodes());
    field(*itr) = (reflectOperator()*field(*itr)*reflectOperator()).Symmetric();
  }
}

// Specialization for third rank tensor fields.  Apply the reflection operator.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(Field<Dimension, typename Dimension::ThirdRankTensor>& field) const {
  REQUIRE(valid());

  const NodeList<Dimension>& nodeList = field.nodeList();
  for (vector<int>::const_iterator itr = this->violationBegin(nodeList);
       itr < this->violationEnd(nodeList); 
       ++itr) {
    CHECK(*itr >= 0 && *itr < nodeList.numInternalNodes());
    field(*itr) = innerProduct<Dimension>(reflectOperator(), innerProduct<Dimension>(field(*itr), reflectOperator()));
  }
}

//------------------------------------------------------------------------------
// Enforce the boundary condition on fields on faces of a tessellation.
// We assume these fields have been summed, and therefore the application of the
// boundary should complete the sum as though there's appropriate stuff on the
// other side of the reflecting plane.
//------------------------------------------------------------------------------
// Specialization for int fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<int>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] *= 2;
  }
}

// Specialization for Scalar fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<typename Dimension::Scalar>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] *= 2.0;
  }
}

// Specialization for Vector fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<typename Dimension::Vector>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] += mReflectOperator*faceField[*itr];
  }
}

// Specialization for Tensor fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<typename Dimension::Tensor>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] += mReflectOperator*faceField[*itr]*mReflectOperator;
  }
}

// Specialization for SymTensor fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<typename Dimension::SymTensor>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] += (mReflectOperator*faceField[*itr]*mReflectOperator).Symmetric();
  }
}

// Specialization for ThirdRankTensor fields.
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
enforceBoundary(vector<typename Dimension::ThirdRankTensor>& faceField,
                const Mesh<Dimension>& mesh) const {
  REQUIRE(faceField.size() == mesh.numFaces());
  const GeomPlane<Dimension>& plane = this->enterPlane();
  const vector<unsigned> faceIDs = this->facesOnPlane(mesh, plane, 1.0e-6);
  for (vector<unsigned>::const_iterator itr = faceIDs.begin();
       itr != faceIDs.end();
       ++itr) {
    CHECK(*itr < faceField.size());
    faceField[*itr] += innerProduct<Dimension>(mReflectOperator, innerProduct<Dimension>(faceField[*itr], mReflectOperator));
  }
}

//------------------------------------------------------------------------------
// Dump state.
//------------------------------------------------------------------------------
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
dumpState(FileIOSpace::FileIO& file,
          const std::string& pathName) const {

  // Call the ancestor class.
  PlanarBoundary<Dimension>::dumpState(file, pathName);

  file.write(reflectOperator(), pathName + "/reflectOperator");
}

//------------------------------------------------------------------------------
// Restore state.
//------------------------------------------------------------------------------
template<typename Dimension>
void
ReflectingBoundary<Dimension>::
restoreState(const FileIOSpace::FileIO& file,
             const std::string& pathName) {

  // Call the ancestor class.
  PlanarBoundary<Dimension>::restoreState(file, pathName);

  file.read(mReflectOperator, pathName + "/reflectOperator");
}

//------------------------------------------------------------------------------
// Test if the reflecting boundary is minimally valid.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
ReflectingBoundary<Dimension>::valid() const {
  return (reflectOperator().Determinant() != 0.0 &&
          PlanarBoundary<Dimension>::valid());
}

}
}
