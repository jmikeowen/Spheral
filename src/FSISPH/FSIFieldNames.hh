//---------------------------------Spheral++----------------------------------//
// FSIFieldNames -- Fields names used in FSI package
//----------------------------------------------------------------------------//
#ifndef _Spheral_FSIFieldNames_
#define _Spheral_FSIFieldNames_

#include <string>

namespace Spheral {

struct FSIFieldNames {
  static const std::string interfaceNormals;
};

}

#else

namespace Spheral {
  struct FSIFieldNames;
}

#endif
