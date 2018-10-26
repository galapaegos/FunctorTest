#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>

#include <thrust/functional.h>

#include <thrust/transform_reduce.h>

#include "Definitions.h"
#include "Functor.h"
#include "ParameterContainer.h"

template <typename T> class MetricTaker1 : public thrust::unary_function<thrust::tuple<int, fptype *, int>, fptype> {
  public:
    __device__ MetricTaker1(T f) : pdf(f) {}

    __device__ fptype operator()(thrust::tuple<int, fptype *, int> t) const {
        ParameterContainer pc;

        int eventIndex       = thrust::get<0>(t);
        int eventSize        = thrust::get<2>(t);
        fptype *eventAddress = thrust::get<1>(t) + (eventIndex * abs(eventSize));

        fptype ret = pdf(eventAddress, pc);

        return ret;
    }

    T pdf;
};

template <typename T> MetricTaker1<T> MetricTaker(T t) { return MetricTaker1<T>(t); }
