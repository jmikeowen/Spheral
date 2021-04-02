#ifndef __SPH_STRING__
#define __SPH_STRING__

#include "RAJA/RAJA.hpp"

namespace Spheral
{
  static constexpr size_t SPHERAL_STRING_CAPACITY = 64;

  class SPHString {
  public:
    char data[SPHERAL_STRING_CAPACITY]; 
    RAJA_HOST_DEVICE SPHString(const char* d) {
      for(size_t i = 0; i < SPHERAL_STRING_CAPACITY; i++) {
        data[i] = d[i];
        if (d[i] == '\0') break;
      }
    } 

    RAJA_HOST_DEVICE bool operator != (const SPHString& rhs) {
      for(size_t i = 0; i < SPHERAL_STRING_CAPACITY; i++) {
        if (data[i] != rhs.data[i]) return false;
        if (data[i] == '\0' && rhs.data[i] == '\0') return true;
      }
      return false;
    }
  };

}


#endif //  __SPH_STRING__

