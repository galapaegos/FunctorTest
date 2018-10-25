#pragma once

#include <cmath>

#include "Functor.h"
#include "ParameterContainer.h"

class FunctorBreitWigner : public Functor {
  public:
    FunctorBreitWigner(fptype mean, fptype gamma, fptype rootPi) : m_mean(mean), m_gamma(gamma), m_rootPi(rootPi) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int id   = pc.getObservable(0);
        fptype x = evt[id];

        pc.incrementIndex(1, 2, 0, 1, 1);

        return (m_gamma / ((x - m_mean) * (x - m_mean) + m_gamma * m_gamma / 4.0)) / (2.0 * m_rootPi);
    }

  protected:
    fptype m_gamma;
    fptype m_rootPi;
    fptype m_mean;
};
