#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

template <typename T, typename U, typename V> class FunctorMapped3 : public Functor {
  public:
    FunctorMapped3(T f, U f2, V f3) : functor(f), functor2(f2), functor3(f3) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        unsigned int numTargets = pc.getConstant(0);

        pc.incrementIndex();

        fptype ret = 0.0;

        // Based on the targetFunction, will map to different PDFs
        int targetFunction = static_cast<int>(floor(0.5 + functor(evt, pc)));

        if(targetFunction == 0) {
            ret = functor2(evt, pc) * pc.getNormalization(0);
            pc.incrementIndex();
        } else if(targetFunction == 1) {
            pc.incrementIndex();
            ret = functor3(evt, pc) * pc.getNormalization(0);
        }

        return ret;
    }

  protected:
    T functor;
    U functor2;
    V functor3;
};

template <typename T, typename U, typename V, typename W> class FunctorMapped4 : public Functor {
  public:
    FunctorMapped4(T f, U f2, V f3, W f4) : functor(f), functor2(f2), functor3(f3), functor4(f4) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        unsigned int numTargets = pc.getConstant(0);

        pc.incrementIndex();

        fptype ret = 0.0;

        // Based on the targetFunction, will map to different PDFs
        int targetFunction = static_cast<int>(floor(0.5 + functor(evt, pc)));

        if(targetFunction == 0) {
            ret = functor2(evt, pc) * pc.getNormalization(0);
            pc.incrementIndex();
        } else if(targetFunction == 1) {
            pc.incrementIndex();
            ret = functor3(evt, pc) * pc.getNormalization(0);
        }

        return ret;
    }

  protected:
    T functor;
    U functor2;
    V functor3;
    W functor4;
};

template <typename T, typename U, typename V> FunctorMapped3<T, U, V> FunctorMapped(T t, U u, V v) {
    return FunctorMapped3(t, u, v);
}

template <typename T, typename U, typename V, typename W> FunctorMapped4<T, U, V, W> FunctorMapped(T t, U u, V v, W w) {
    return FunctorMapped4(t, u, v, w);
}

// Below is an implementation using if constexpr, but not using variadic. This was

/*
template <typename T,
          typename U = std::nullptr_t,
          typename V = std::nullptr_t,
          typename W = std::nullptr_t,
          typename X = std::nullptr_t,
          typename Y = std::nullptr_t>
class FunctorMapped : public Functor {
  public:
    FunctorMapped(T f1, U f2, V f3 = nullptr, W f4 = nullptr, X f5 = nullptr, Y f6 = nullptr) :
        functor(f1), functor2(f2), functor3(f3), functor4(f4), functor5(f5), functor6(f6) {}

    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        unsigned int numTargets = pc.getConstant(0);

        pc.incrementIndex();

        fptype ret = 0.0;

        // Based on the targetFunction, will map to different PDFs
        int targetFunction = static_cast<int>(floor(0.5 + functor(evt, pc)));

        // Does this make sense to have a mapped function only going to one pdf?
        // if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
        //             && std::is_null_pointer_v<V> == true) {
        if(functor != nullptr && functor2 != nullptr && functor3 == false) {
            ret = functor2(evt, pc) * pc.getNormalization(0);
        }

        // if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
        //             && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == true) {
        if(functor != nullptr && functor2 != nullptr && functor3 != nullptr && functor4 == nullptr) {
            if(targetFunction == 0) {
                ret = functor2(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
            } else if(targetFunction == 1) {
                pc.incrementIndex();
                ret = functor3(evt, pc) * pc.getNormalization(0);
            }
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == true) {
            if(targetFunction == 0) {
                ret = functor2(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 1) {
                pc.incrementIndex();
                ret = functor3(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
            } else if(targetFunction == 2) {
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor4(evt, pc) * pc.getNormalization(0);
            }
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == true) {
            if(targetFunction == 0) {
                ret = functor2(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 1) {
                pc.incrementIndex();
                ret = functor3(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 2) {
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor4(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
            } else if(targetFunction == 3) {
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor5(evt, pc) * pc.getNormalization(0);
            }
        }

        if constexpr(std::is_null_pointer_v<T> == false && std::is_null_pointer_v<U> == false
                     && std::is_null_pointer_v<V> == false && std::is_null_pointer_v<W> == false
                     && std::is_null_pointer_v<X> == false && std::is_null_pointer_v<Y> == false) {
            if(targetFunction == 0) {
                ret = functor2(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 1) {
                pc.incrementIndex();
                ret = functor3(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 2) {
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor4(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
                pc.incrementIndex();
            } else if(targetFunction == 3) {
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor5(evt, pc) * pc.getNormalization(0);
                pc.incrementIndex();
            } else if(targetFunction == 4) {
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
                pc.incrementIndex();
                ret = functor6(evt, pc) * pc.getNormalization(0);
            }
        }

        return ret;
    }

  protected:
    T functor;
    U functor2;
    V functor3;
    W functor4;
    X functor5;
    Y functor6;
};
*/
