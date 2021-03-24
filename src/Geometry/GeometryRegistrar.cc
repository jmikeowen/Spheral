//------------------------------------------------------------------------------
// GeometryRegistrar
//
// A singleton class for maintaining global information about the type of
// coordinate Spheral is running in.
//
// Created by JMO, Thu Mar 11 16:41:33 PST 2021
//------------------------------------------------------------------------------

#include "Geometry/GeometryRegistrar.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Instance
//------------------------------------------------------------------------------
inline
GeometryRegistrar&
GeometryRegistrar::instance() {
  if (mInstancePtr == 0) mInstancePtr = new GeometryRegistrar;
  CHECK(mInstancePtr != 0);
  return *mInstancePtr;
}

//------------------------------------------------------------------------------
// Constructor (private)
//------------------------------------------------------------------------------
GeometryRegistrar::GeometryRegistrar(){
}

//------------------------------------------------------------------------------
// Copy constructor (private)
//------------------------------------------------------------------------------
GeometryRegistrar::GeometryRegistrar(const GeometryRegistrar&) {
}

//------------------------------------------------------------------------------
// Assignment (private)
//------------------------------------------------------------------------------
GeometryRegistrar&
GeometryRegistrar::operator=(const GeometryRegistrar&) {
  return *mInstancePtr;
}

//------------------------------------------------------------------------------
// Destructor (private)
//------------------------------------------------------------------------------
GeometryRegistrar::~GeometryRegistrar() {
}

//------------------------------------------------------------------------------
// Initializations
//------------------------------------------------------------------------------
GeometryRegistrar* GeometryRegistrar::mInstancePtr = 0;
CoordinateType GeometryRegistrar::mCoords = CoordinateType::Cartesian;
}
