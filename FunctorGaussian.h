#pragma once

#include <cmath>

#include "Functor.h"
#include "ParameterContainer.h"

class FunctorGaussian : public Functor {
  public:
    FunctorGaussian(fptype sigma, fptype mean) {}

    //__device__ fptype operator()(fptype *evt, ParameterContainer &pc
    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int id   = pc.getObservable(0);
        fptype x = evt[id];

        fptype mean = pc.getParameter(0);
        fptype sigma = pc.getParameter(1);

        pc.incrementIndex(1, 2, 0, 1, 1);

        return exp(-0.5 * (x - mean) * (x - mean) / sigma * sigma);
    }

  protected:
};
