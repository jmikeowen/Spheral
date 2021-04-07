//---------------------------------Spheral++----------------------------------//
// GeomVectorBase -- provides the dimension dependent storage for GeomVector.
//
// Created by JMO, Wed Nov 10 14:46:51 2004
//----------------------------------------------------------------------------//
#ifndef __Spheral__GeomVectorBase__
#define __Spheral__GeomVectorBase__

namespace Spheral {

template<int nDim> class GeomVectorBase {};

template<>
class GeomVectorBase<1> {
 public:
  RAJA_HOST_DEVICE GeomVectorBase(const double x):
    mx(x) {}
 protected:
  double mx;
};

template<>
class GeomVectorBase<2> {
 public:
  RAJA_HOST_DEVICE GeomVectorBase(const double x,
                 const double y):
    mx(x),
    my(y) {}
 protected:
  double mx;
  double my;
  double mz;
};

template<>
class GeomVectorBase<3> {
 public:
  RAJA_HOST_DEVICE GeomVectorBase(const double x,
                 const double y,
                 const double z):
    mx(x),
    my(y),
    mz(z) {}
 protected:
  double mx;
  double my;
  double mz;
};

}

#else

namespace Spheral {
  template<int nDim> class GeomVectorBase;
}

#endif
