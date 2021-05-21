//#include "Field/Field.hh"

#include "LvArray/Array.hpp"
#include "LvArray/ChaiBuffer.hpp"

using EXEC_POL = RAJA::cuda_exec<256>;
using HOST_POL = RAJA::seq_exec;

template<typename T>
using Array1D = LvArray::Array< T, 1, camp::idx_seq<0>, std::ptrdiff_t, LvArray::ChaiBuffer >;

template<typename T>
using Array1DView = LvArray::ArrayView< T, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer >;

int main() {

  constexpr int N = 10;

  using ValueType = double;

  Array1D<ValueType> data(10);
  for (int i = 0; i < N; i++) {
    data.emplace_back(0.0);
  }
  
  const Array1DView<ValueType>& data_view(data);

  RAJA::RangeSegment range(0, data_view.size());
  RAJA::forall<HOST_POL>(range,
    [=](unsigned int kk) {
      data_view[kk]++;
  });

  //data.move(LvArray::MemorySpace::cuda);

  RAJA::forall<EXEC_POL>(range, 
    [=] RAJA_HOST_DEVICE (int kk) {
      data_view[kk]++;
  });

  //data.move(LvArray::MemorySpace::host);

  bool correctness = true;
  RAJA::forall<HOST_POL>(range,
    [=, &correctness] (int kk) {
      if (data_view[kk] != 2.0) correctness = false;
  });

  if (correctness)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";

  return EXIT_SUCCESS;
}
