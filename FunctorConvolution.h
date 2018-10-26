#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

__constant__ fptype *dev_modWorkSpace[100];
__constant__ fptype *dev_resWorkSpace[100];

template <typename T, typename U> class FunctorProduct : public Functor {
  public:
    __device__ FunctorProduct(T f1, U f2) : functor1(f1), functor2(f2) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int id = pc.getObservable(0);

        fptype ret = 0;

        fptype loBound     = pc.getConstant(0);
        fptype hiBound     = pc.getConstant(1);
        fptype step        = pc.getConstant(2);
        int workSpaceIndex = pc.getConstant(3);

        fptype x0 = evt[id];

        int numbins = (hiBound - loBound) / step + 0.5);

        fptype lowerBoundOffset = loBound / step;
        lowerBoundOffset -= floor(lowerBoundOffset);

        int offsetInBins = floor(x0 / step - lowerBoundOffset);

        for(int i = 0; i < numbins; i++) {
            ret += dev_modWorkSpace[workSpaceIndex][i] * dev_resWorkSpace[workSpaceIndex][i + offset - offsetInBins];
        }

        pc.incrementIndex(1, 0, 4, 1, 1);

        ret *= pc.getNormalization(0);

        pc.incrementIndex();

        ret *= pc.getNormalization(0);

        pc.incrementIndex();

        return ret;
    }

  protected:
    T functor1;
    U functor2;
};
