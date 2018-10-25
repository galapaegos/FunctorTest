#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

template <typename T = std::nullptr_t,
          typename U = std::nullptr_t,
          typename V = std::nullptr_t,
          typename W = std::nullptr_t,
          typename X = std::nullptr_t,
          typename Y = std::nullptr_t>
class FunctorProduct : public Functor {
  public:
    FunctorProduct(T f1, U f2, V f3 = nullptr, W f4 = nullptr, X f5 = nullptr, Y f6 = nullptr) :
        functor1(f1), functor2(f2), functor3(f3), functor4(f4), functor5(f5), functor6(f6) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == true) {
            int numConstants   = pc.getNumConstants();
            int numComponens   = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == true) {
            int numConstants   = pc.getNumConstants();
            int numComponents  = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            // Functor2
            { ret *= functor2(evt, pc) * pc.getNormalization(0); }

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == true) {
            int numConstants   = pc.getNumConstants();
            int numComponens   = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            // Functor2
            { ret *= functor2(evt, pc) * pc.getNormalization(0); }

            // Functor3
            { ret *= functor3(evt, pc) * pc.getNormalization(0); }

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == true) {
            int numConstants   = pc.getNumConstants();
            int numComponens   = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            // Functor2
            { ret *= functor2(evt, pc) * pc.getNormalization(0); }

            // Functor3
            { ret *= functor3(evt, pc) * pc.getNormalization(0); }

            // Functor4
            { ret *= functor4(evt, pc) * pc.getNormalization(0); }

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == true) {
            int numConstants   = pc.getNumConstants();
            int numComponens   = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            // Functor2
            { ret *= functor2(evt, pc) * pc.getNormalization(0); }

            // Functor3
            { ret *= functor3(evt, pc) * pc.getNormalization(0); }

            // Functor4
            { ret *= functor4(evt, pc) * pc.getNormalization(0); }

            // Functor5
            { ret *= functor5(evt, pc) * pc.getNormalization(0); }

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == false) {
            int numConstants   = pc.getNumConstants();
            int numComponens   = pc.getConstant(0);
            int numObservables = pc.getNumObservables();

            fptype ret = 1;

            // Increment to the next function
            pc.incrementIndex(1, 0, numConstants, numObservables, 1);

            // Functor1
            { ret *= functor1(evt, pc) * pc.getNormalization(0); }

            // Functor2
            { ret *= functor2(evt, pc) * pc.getNormalization(0); }

            // Functor3
            { ret *= functor3(evt, pc) * pc.getNormalization(0); }

            // Functor4
            { ret *= functor4(evt, pc) * pc.getNormalization(0); }

            // Functor5
            { ret *= functor5(evt, pc) * pc.getNormalization(0); }

            // Functor6
            { ret *= functor6(evt, pc) * pc.getNormalization(0); }

            return ret;
        }
    }

  protected:
    T functor1;
    U functor2;
    V functor3;
    W functor4;
    X functor5;
    Y functor6;
};
