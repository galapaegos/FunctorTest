#pragma once

#include <cmath>

#include "Functor.h"
#include "ParameterContainer.h"

struct FunctorArgus : public Functor {
  public:
    __device__ FunctorArgus(fptype m0, fptype slope, fptype power) : m_m0(m0), m_slope(slope), m_power(power) {}

    // This is argus_upper
    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int id   = pc.getObservable(0);
        fptype x = evt[id];

        pc.incrementIndex(1, 3, 0, 1, 1);

        fptype t = x / m_m0;

        if(t >= 1.0)
            return 0.0;

        t = 1 - t * t;

        return x * pow(t, m_power) * exp(m_slope * t);
    }

  protected:
    fptype m_m0;
    fptype m_slope;
    fptype m_power;
};
