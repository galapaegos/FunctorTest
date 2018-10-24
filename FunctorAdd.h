#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

template <typename T, typename U> class FunctorAdd : public Functor {
  public:
    __host__ __device__ FunctorAdd(T f1, U f2) : functor1(f1), functor2(f2) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int numParameters  = pc.getNumParameters();
        fptype ret         = 0;
        fptype totalWeight = 0;

        ParameterContainer pci = pc;

        pci.incrementIndex();

        // Functor1
        {
            fptype weight = pc.getParameter(0);
            totalWeight += weight;

            fptype norm = pci.getNormalization(0);

            fptype curr = functor1(evt, pci);

            ret += weight * curr * norm;
        }

        // resource our container structure
        pc = pci;

        fptype normFactor = pc.getNormalization(0);

        fptype last = functor2(evt, pc);
        ret += (1.0 - totalWeight) * last * normFactor;

        return ret;
    }

  protected:
    T functor1;
    U functor2;
};
