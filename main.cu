#include <algorithm>
#include <chrono>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

#include "MetricTaker.cuh"
#include "FunctorAdd.h"
#include "FunctorGaussian.h"

int main() {
    thrust::host_vector<fptype> src_vector;

    int length = 10;

    // Reserve a bunch of points
    src_vector.resize(length);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.1, 2.0);

    // Populate a normal distribution into the source vector
    for(auto i : src_vector)
        src_vector[i] = d(gen);

    for(auto i : src_vector)
        std::cout << src_vector[i] << "\n";

    thrust::device_vector<fptype> dev_vector(src_vector);

    // Unwrapping top down
    auto gauss1 = FunctorGaussian(1.0, 2.0);
    auto gauss2 = FunctorGaussian(3.0, 1.5);

    auto add_pdf = FunctorAdd(gauss1, gauss2);

    auto mt = MetricTaker(add_pdf);


    thrust::constant_iterator<int> eventSize(1);
    //thrust::constant_iterator<fptype *> arrayAddress(thrust::raw_pointer_cast(src_vector.data()));
    thrust::constant_iterator<fptype *> arrayAddress(thrust::raw_pointer_cast(dev_vector.data()));
    thrust::counting_iterator<int> eventIndex(0);

    // Run a transform_reduce
    fptype results = thrust::transform_reduce(
	thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
        thrust::make_zip_iterator(thrust::make_tuple(eventIndex + length, arrayAddress, eventSize)),
        mt,
        0.0,
        thrust::plus<fptype>());

    return 0;
}
