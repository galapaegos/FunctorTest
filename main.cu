#include <algorithm>
#include <chrono>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

#include "MetricTaker.cuh"
#include "FunctorAdd.h"
#include "FunctorArgus.h"
#include "FunctorBreitWigner.h"
#include "FunctorGaussian.h"
#include "FunctorMapped.h"
#include "FunctorProduct.h"

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
    auto gauss3 = FunctorGaussian(4.0, 2.5);

    auto argus1 = FunctorArgus(1.0, 4.0, 3);
    auto argus2 = FunctorArgus(2.0, 8.0, 4);

    auto bw1 = FunctorBreitWigner(1.0, 5.0, 3.14);
    auto bw2 = FunctorBreitWigner(2.0, 5.0, 3.14159);

    auto add_pdf1 = FunctorAdd(gauss1, bw1, argus1);
    auto add_pdf2 = FunctorAdd(gauss2, argus2);

    auto prod_pdf1 = FunctorProduct(add_pdf1, argus1);
    auto prod_pdf2 = FunctorProduct(gauss1, argus1);
    auto prod_pdf3 = FunctorProduct(add_pdf1, add_pdf2);
    auto prod_pdf4 = FunctorProduct(prod_pdf1, add_pdf2);

    auto mapping1 = FunctorMapped(prod_pdf1, bw1, bw2, prod_pdf2);
    auto mapping2 = FunctorMapped(gauss1, bw1, argus1);
    auto mapping3 = FunctorMapped(gauss1, bw1, argus1, add_pdf2);

    auto mt = MetricTaker(gauss1);
    auto mt1 = MetricTaker(mapping3);
    auto mt2 = MetricTaker(prod_pdf3);

    thrust::constant_iterator<int> eventSize(1);
    // thrust::constant_iterator<fptype *> arrayAddress(thrust::raw_pointer_cast(src_vector.data()));
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
