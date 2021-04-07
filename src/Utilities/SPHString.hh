#ifndef __SPH_STRING__
#define __SPH_STRING__

#include <cassert>
#include "RAJA/RAJA.hpp"

namespace Spheral
{
  static constexpr size_t SPHERAL_STRING_CAPACITY = 128;

  class SPHString {
  public:
    
    char data[SPHERAL_STRING_CAPACITY]; 
    size_t size = 0;

    RAJA_HOST_DEVICE SPHString(){};
    RAJA_HOST_DEVICE constexpr SPHString& operator=(SPHString& rhs)=default;

    SPHString operator=(std::string rhs) {
      assert(rhs.size() > 127);
      for(size_t i = 0; i < rhs.size(); i++) {
        data[i] = rhs[i];
      }
      data[rhs.size()] = '\0';
      return *this;
    }
    SPHString& operator=(std::string& rhs) {
      assert(rhs.size() > 127);
      for(size_t i = 0; i < rhs.size(); i++) {
        data[i] = rhs[i];
      }
      data[rhs.size()] = '\0';
      return *this;
    }

    RAJA_HOST_DEVICE SPHString(const char* d) {
      for(size_t i = 0; i < SPHERAL_STRING_CAPACITY; i++) {
        data[i] = d[i];
        if (d[i] == '\0'){
          size = i;
          break;
        }
      }
    } 

    RAJA_HOST_DEVICE bool operator == (const SPHString& rhs) {
      for(size_t i = 0; i < SPHERAL_STRING_CAPACITY; i++) {
        if (data[i] != rhs.data[i]) return false;
        if (data[i] == '\0' && rhs.data[i] == '\0') return true;
      }
      return false;
    }

    RAJA_HOST_DEVICE bool operator != (const SPHString& rhs) {
      return !(*this == rhs);
    }

    friend SPHString operator+(const SPHString& lhs, const SPHString& rhs); 
    
    //friend std::ostream& operator<<(std::ostream& os, const SPHString& rhs);
  };

  //SPHString operator+(const SPHString& lhs, const SPHString& rhs) {
  //  SPHString result(lhs);
  //  size_t begin = result.size;
  //  for (size_t i = 0; i < rhs.size; i++) {
  //    result.data[begin + i] = rhs.data[i];
  //    if (rhs.data[i] == '\0') {
  //      result.size += rhs.size;
  //      break;
  //    }
  //  }
  //  return result;
  //}

  //std::ostream& operator<<(std::ostream& os, const SPHString& rhs)
  //{
  //  os << rhs.data;
  //  return os;
  //}

}


#endif //  __SPH_STRING__

