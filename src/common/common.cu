/*!
 * Copyright 2018-2022 XGBoost contributors
 */
#include "common.h"

namespace xgboost {
namespace common {

void SetDevice(std::int32_t device) {
  if (device >= 0) {
    dh::safe_cuda(cudaSetDevice(device));
  }
}

int AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    std::cout <<"abs0" << std::endl;
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
    std::cout <<"abs1" << std::endl;
  } catch (const dmlc::Error &e) {
    std::cout <<"abs2" << std::endl;
    cudaGetLastError();  // reset error.
    std::cout << e.what() << std::endl;
    return 0;
  }
  std::cout <<"abs3" << std::endl;
  return n_visgpus;
}

}  // namespace common
}  // namespace xgboost
