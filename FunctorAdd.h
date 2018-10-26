#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

template <typename T, typename U> struct FunctorAdd2 : public Functor {
  public:
    FunctorAdd2(T f1, U f2) : functor1(f1), functor2(f2) {}
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

template <typename T, typename U, typename V> struct FunctorAdd3 : public Functor {
  public:
    FunctorAdd3(T f1, U f2, V f3) : functor1(f1), functor2(f2), functor3(f3) {}

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

        // Functor2
        {
            fptype weight = pc.getParameter(1);
            totalWeight += weight;

            fptype norm = pci.getNormalization(0);

            fptype curr = functor2(evt, pci);

            ret += weight * curr * norm;
        }

        // resource our container structure
        pc = pci;

        fptype normFactor = pc.getNormalization(0);

        fptype last = functor3(evt, pc);
        ret += (1.0 - totalWeight) * last * normFactor;

        return ret;
    }

  protected:
    T functor1;
    U functor2;
    V functor3;
};

template <typename T, typename U, typename V, typename W> struct FunctorAdd4 : public Functor {
    FunctorAdd4(T f1, U f2, V f3, W f4) : functor1(f1), functor2(f2), functor3(f3), functor4(f4) {}

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

        // Functor2
        {
            fptype weight = pc.getParameter(1);
            totalWeight += weight;

            fptype norm = pci.getNormalization(0);

            fptype curr = functor2(evt, pci);

            ret += weight * curr * norm;
        }

        // Functor3
        {
            fptype weight = pc.getParameter(2);
            totalWeight += weight;

            fptype norm = pci.getNormalization(0);

            fptype curr = functor3(evt, pci);

            ret += weight * curr * norm;
        }

        // resource our container structure
        pc = pci;

        fptype normFactor = pc.getNormalization(0);

        fptype last = functor4(evt, pc);
        ret += (1.0 - totalWeight) * last * normFactor;

        return ret;
    }

  protected:
    T functor1;
    U functor2;
    V functor3;
    W functor4;
};

// These are overloading FunctorAdd so that construction can happen with the same name
template <typename T, typename U> FunctorAdd2<T, U> FunctorAdd(T t, U u) { return FunctorAdd2(t, u); }
template <typename T, typename U, typename V> FunctorAdd3<T, U, V> FunctorAdd(T t, U u, V v) {
    return FunctorAdd3(t, u, v);
}
template <typename T, typename U, typename V, typename W> FunctorAdd4<T, U, V, W> FunctorAdd(T t, U u, V v, W w) {
    return FunctorAdd4(t, u, v, w);
}

// Below is an implementation using if constexpr, but not using variadic templates. I chose to unroll by hand the loop
// to handle the last case

/*
template <typename T = std::nullptr_t,
          typename U = std::nullptr_t,
          typename V = std::nullptr_t,
          typename W = std::nullptr_t,
          typename X = std::nullptr_t,
          typename Y = std::nullptr_t>
class FunctorAdd : public Functor {
  public:
    FunctorAdd(T f1, U f2, V f3 = nullptr, W f4 = nullptr, X f5 = nullptr, Y f6 = nullptr) :
        functor1(f1), functor2(f2), functor3(f3), functor4(f4), functor5(f5), functor6(f6) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == true) {
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

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == true) {
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

            // Functor2
            {
                fptype weight = pc.getParameter(1);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor2(evt, pci);

                ret += weight * curr * norm;
            }

            // resource our container structure
            pc = pci;

            fptype normFactor = pc.getNormalization(0);

            fptype last = functor3(evt, pc);
            ret += (1.0 - totalWeight) * last * normFactor;

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == true) {
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

            // Functor2
            {
                fptype weight = pc.getParameter(1);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor2(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor3
            {
                fptype weight = pc.getParameter(2);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor3(evt, pci);

                ret += weight * curr * norm;
            }

            // resource our container structure
            pc = pci;

            fptype normFactor = pc.getNormalization(0);

            fptype last = functor4(evt, pc);
            ret += (1.0 - totalWeight) * last * normFactor;

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == true) {
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

            // Functor2
            {
                fptype weight = pc.getParameter(1);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor2(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor3
            {
                fptype weight = pc.getParameter(2);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor3(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor4
            {
                fptype weight = pc.getParameter(3);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor4(evt, pci);

                ret += weight * curr * norm;
            }

            // resource our container structure
            pc = pci;

            fptype normFactor = pc.getNormalization(0);

            fptype last = functor5(evt, pc);
            ret += (1.0 - totalWeight) * last * normFactor;

            return ret;
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == true) {
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

            // Functor2
            {
                fptype weight = pc.getParameter(1);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor2(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor3
            {
                fptype weight = pc.getParameter(2);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor3(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor4
            {
                fptype weight = pc.getParameter(3);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor4(evt, pci);

                ret += weight * curr * norm;
            }

            // Functor5
            {
                fptype weight = pc.getParameter(4);
                totalWeight += weight;

                fptype norm = pci.getNormalization(0);

                fptype curr = functor5(evt, pci);

                ret += weight * curr * norm;
            }

            // resource our container structure
            pc = pci;

            fptype normFactor = pc.getNormalization(0);

            fptype last = functor6(evt, pc);
            ret += (1.0 - totalWeight) * last * normFactor;

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
