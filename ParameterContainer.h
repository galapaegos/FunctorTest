#pragma once

#include <cuda_runtime.h>

#include "Definitions.h"

struct ParameterContainer {
    __device__ ParameterContainer() {
        parameters     = d_parameters;
        constants      = d_constants;
        observables    = d_observables;
        normalizations = d_normalizations;
    }

    __device__ ParameterContainer(const ParameterContainer &pc) : ParameterContainer() {
        parameterIdx  = pc.parameterIdx;
        constantIdx   = pc.constantIdx;
        observableIdx = pc.observableIdx;
        normalIdx     = pc.normalIdx;
        funcIdx       = pc.funcIdx;
    }

    fptype *parameters;
    fptype *constants;
    fptype *observables;
    fptype *normalizations;

    int parameterIdx{0};
    int constantIdx{0};
    int observableIdx{0};
    int normalIdx{0};

    int funcIdx{0};

    inline __device__ fptype getParameter(const int i) { return parameters[parameterIdx + i + 1]; }
    inline __device__ fptype getConstant(const int i) { return constants[constantIdx + i + 1]; }
    inline __device__ fptype getObservable(const int i) { return observables[observableIdx + i + 1]; }
    inline __device__ fptype getNormalization(const int i) { return normalizations[normalIdx + i + 1]; }

    inline __device__ int getNumParameters() { return (int)parameters[parameterIdx]; }
    inline __device__ int getNumConstants() { return (int)constants[constantIdx]; }
    inline __device__ int getNumObservables() { return (int)observables[observableIdx]; }
    inline __device__ int getNumNormalizations() { return (int)normalizations[normalIdx]; }

    inline __device__ void
    incrementIndex(const int funcs, const int params, const int cons, const int obs, const int norms) {
        funcIdx += funcs;
        parameterIdx += params + 1;
        constantIdx += cons + 1;
        observableIdx += obs + 1;
        normalIdx += norms + 1;
    }

    inline __device__ void incrementIndex() {
        parameterIdx += getNumParameters() + 1;
        constantIdx += getNumConstants() + 1;
        observableIdx += getNumObservables() + 1;
        normalIdx += getNumNormalizations() + 1;
        funcIdx++;
    }
};
