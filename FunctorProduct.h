#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

template <typename T> class FunctorProduct1 : public Functor {
  public:
    FunctorProduct1(T f1) : functor1(f1) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int numConstants   = pc.getNumConstants();
        int numComponents  = pc.getConstant(0);
        int numObservables = pc.getNumObservables();

        fptype ret = 1;

        // Increment to the next function
        pc.incrementIndex(1, 0, numConstants, numObservables, 1);

        // Functor1
        ret *= functor1(evt, pc) * pc.getNormalization(0);

        return ret;
    }

  protected:
    T functor1;
};

template <typename T, typename U> class FunctorProduct2 : public Functor {
  public:
    FunctorProduct2(T f1, U f2) : functor1(f1), functor2(f2) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int numConstants   = pc.getNumConstants();
        int numComponents  = pc.getConstant(0);
        int numObservables = pc.getNumObservables();

        fptype ret = 1;

        // Increment to the next function
        pc.incrementIndex(1, 0, numConstants, numObservables, 1);

        // Functor1
        ret *= functor1(evt, pc) * pc.getNormalization(0);

        // Functor2
        ret *= functor2(evt, pc) * pc.getNormalization(0);

        return ret;
    }

  protected:
    T functor1;
    U functor2;
};

template <typename T, typename U, typename V> class FunctorProduct3 : public Functor {
  public:
    FunctorProduct3(T f1, U f2, V f3) : functor1(f1), functor2(f2), functor3(f3) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int numConstants   = pc.getNumConstants();
        int numComponents  = pc.getConstant(0);
        int numObservables = pc.getNumObservables();

        fptype ret = 1;

        // Increment to the next function
        pc.incrementIndex(1, 0, numConstants, numObservables, 1);

        // Functor1
        ret *= functor1(evt, pc) * pc.getNormalization(0);

        // Functor2
        ret *= functor2(evt, pc) * pc.getNormalization(0);

        // Functor3
        ret *= functor3(evt, pc) * pc.getNormalization(0);

        return ret;
    }

  protected:
    T functor1;
    U functor2;
    V functor3;
};

template <typename T, typename U, typename V, typename W> class FunctorProduct4 : public Functor {
  public:
    FunctorProduct4(T f1, U f2, V f3, W f4) : functor1(f1), functor2(f2), functor3(f3), functor4(f4) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int numConstants   = pc.getNumConstants();
        int numComponents  = pc.getConstant(0);
        int numObservables = pc.getNumObservables();

        fptype ret = 1;

        // Increment to the next function
        pc.incrementIndex(1, 0, numConstants, numObservables, 1);

        // Functor1
        ret *= functor1(evt, pc) * pc.getNormalization(0);

        // Functor2
        ret *= functor2(evt, pc) * pc.getNormalization(0);

        // Functor3
        ret *= functor3(evt, pc) * pc.getNormalization(0);

        // Functor4
        ret *= functor4(evt, pc) * pc.getNormalization(0);

        return ret;
    }

  protected:
    T functor1;
    U functor2;
    V functor3;
    W functor4;
};

template <typename T> FunctorProduct1<T> FunctorProduct(T t) { return FunctorProduct1<T>(t); }
template <typename T, typename U> FunctorProduct2<T, U> FunctorProduct(T t, U u) { return FunctorProduct2<T, U>(t, u); }
template <typename T, typename U, typename V> FunctorProduct3<T, U, V> FunctorProduct(T t, U u, V v) {
    return FunctorProduct3<T, U, V>(t, u, v);
}
template <typename T, typename U, typename V, typename W>
FunctorProduct4<T, U, V, W> FunctorProduct(T t, U u, V v, W w) {
    return FunctorProduct4<T, U, V, W>(t, u, v, w);
}

/*
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
*/
