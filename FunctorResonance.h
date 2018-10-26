#pragma once

#include "Functor.h"
#include "ParameterContainer.h"

#include <type_traits>

struct FunctorResonanceFlatte : public Functor {
  public:
    __device__ FunctorResonanceFlatte(fptype g1, fptype rg2og1, fptype cyc) : m_g1(g1), m_rg2og1(rg2og1), m_cyc(cyc) {}

    // This is argus_upper
    __device__ fpcomplex operator()(fptype m12, fptype m13, fptype m23, ParameterContainer &pc) const {
        fptype resmass            = pc.getParameter(0);
        fptype g1                 = pc.getParameter(1);
        fptype g2                 = pc.getParameter(2) * g1;
        unsigned int cyclic_index = pc.getConstant(0);
        unsigned int doSwap       = pc.getConstant(1);

        fptype pipmass = 0.13957018;
        fptype pi0mass = 0.1349766;
        fptype kpmass  = 0.493677;
        fptype k0mass  = 0.497614;

        fptype twopimasssq  = 4 * pipmass * pipmass;
        fptype twopi0masssq = 4 * pi0mass * pi0mass;
        fptype twokmasssq   = 4 * kpmass * kpmass;
        fptype twok0masssq  = 4 * k0mass * k0mass;

        fpcomplex ret(0.0, 1.0);

        for(int i = 0; i < 1 + doSwap; i++) {
            fptype rhopipi_real = 0, rhopipi_imag = 0;
            fptype rhokk_real = 0, rhokk_imag = 0;

            fptype s = (0 == cyclic_index ? m12 : (1 == cyclic_index ? m13 : m23));

            if(s >= twopimasssq)
                rhopipi_real += (2. / 3) * sqrt(1 - twopimasssq / s); // Above pi+pi- threshold
            else
                rhopipi_imag += (2. / 3) * sqrt(-1 + twopimasssq / s);
            if(s >= twopi0masssq)
                rhopipi_real += (1. / 3) * sqrt(1 - twopi0masssq / s); // Above pi0pi0 threshold
            else
                rhopipi_imag += (1. / 3) * sqrt(-1 + twopi0masssq / s);
            if(s >= twokmasssq)
                rhokk_real += 0.5 * sqrt(1 - twokmasssq / s); // Above K+K- threshold
            else
                rhokk_imag += 0.5 * sqrt(-1 + twokmasssq / s);
            if(s >= twok0masssq)
                rhokk_real += 0.5 * sqrt(1 - twok0masssq / s); // Above K0K0 threshold
            else
                rhokk_imag += 0.5 * sqrt(-1 + twok0masssq / s);
            fptype A = (resmass * resmass - s) + resmass * (rhopipi_imag * g1 + rhokk_imag * g2);
            fptype B = resmass * (rhopipi_real * g1 + rhokk_real * g2);
            fptype C = 1.0 / (A * A + B * B);
            fpcomplex retur(A * C, B * C);
            ret += retur;
            if(doSwap) {
                fptype swpmass = m12;
                m12            = m13;
                m13            = swpmass;
            }
        }

        pc.incrementIndex(1, 3, 2, 0, 1);

        return ret;
    }

  protected:
    fptype m_g1;
    fptype m_rg2og1;
    fptype m_cyc;
};

struct FunctorResonanceGS : public Functor {
  public:
    __device__ FunctorResonanceGS(fptype sp, fptype cyc) : m_sp(sp), m_cyc(cyc) {}

    // This is argus_upper
    __device__ fpcomplex operator()(fptype m12, fptype m13, fptype m23, ParameterContainer &pc) const {
        int spin         = pc.getConstant(0);
        int cyclic_index = pc.getConstant(1);

        fptype resmass  = pc.getParameter(0);
        fptype reswidth = pc.getParameter(1);

        fptype rMassSq   = (0 == cyclic_index ? m12 : (1 == cyclic_index ? m13 : m23));
        fptype frFractor = 1;

        resmass *= resmass;

        return fpcomplex(0.0, 1.0);
    }

  protected:
    fptype m_sp;
    fptype m_cyc;
};
