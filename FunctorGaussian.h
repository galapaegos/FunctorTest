#pragma once

#include <cmath>

#include "Functor.h"
#include "ParameterContainer.h"

class FunctorGaussian : public Functor {
  public:
    __host__ __device__ FunctorGaussian(fptype sigma, fptype mean) : m_mean(mean), m_sigma(sigma) {}

    //__device__ fptype operator()(fptype *evt, ParameterContainer &pc
    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int id   = pc.getObservable(0);
        fptype x = evt[id];

        pc.incrementIndex(1, 2, 0, 1, 1);

        return exp(-0.5 * (x - m_mean) * (x - m_mean) / m_sigma * m_sigma);
    }

  protected:
    fptype m_sigma;
    fptype m_mean;
};
