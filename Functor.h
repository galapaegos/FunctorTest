#pragma once

#include "ParameterContainer.h"

class Functor {
  public:
    __host__ __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const { return 0.0; }
};
