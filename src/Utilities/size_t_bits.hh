// Kind of hard-wired, so may have to extend this on new platforms.

#include <stdint.h>
#if (SIZE_MAX == 0xFFFF)
  #define SIZE_T_BITS 16
#elif (SIZE_MAX == 0xFFFFFFFF)
  #define SIZE_T_BITS 32
#elif (SIZE_MAX == 0xFFFFFFFFFFFFFFFF)
  #define SIZE_T_BITS 64
#else
  #error TBD code SIZE_T_BITS
#endif
