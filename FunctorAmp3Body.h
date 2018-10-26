#pragma once

#include <cmath>

#include "Functor.h"
#include "ParameterContainer.h"

__device__ fpcomplex *cResonances[16];

__host__ __device__ bool inDalitz(const fptype &m12,
                                  const fptype &m13,
                                  const fptype &bigM,
                                  const fptype &dm1,
                                  const fptype &dm2,
                                  const fptype &dm3) {
    fptype dm1pdm2  = dm1 + dm2;
    fptype bigMmdm3 = bigM - dm3;

    bool m12less = (m12 < dm1pdm2 * dm1pdm2) ? false : true;
    // if (m12 < dm1pdm2*dm1pdm2) return false; // This m12 cannot exist, it's less than the square of the (1,2)
    // particle mass.
    bool m12grea = (m12 > bigMmdm3 * bigMmdm3) ? false : true;
    // if (m12 > bigMmdm3*bigMmdm3) return false;   // This doesn't work either, there's no room for an at-rest 3
    // daughter.

    fptype sqrtM12 = sqrt(m12);
    fptype dm11    = dm1 * dm1;
    fptype dm22    = dm2 * dm2;
    fptype dm33    = dm3 * dm3;

    // Calculate energies of 1 and 3 particles in m12 rest frame.
    // fptype e1star = 0.5 * (m12 - dm2*dm2 + dm1*dm1) / sqrt(m12);
    fptype e1star = 0.5 * (m12 - dm22 + dm11) / sqrtM12;
    // fptype e3star = 0.5 * (bigM*bigM - m12 - dm3*dm3) / sqrt(m12);
    fptype e3star = 0.5 * (bigM * bigM - m12 - dm33) / sqrtM12;

    fptype rte1mdm11 = sqrt(e1star * e1star - dm11);
    fptype rte3mdm33 = sqrt(e3star * e3star - dm33);

    // Bounds for m13 at this value of m12.
    // fptype minimum = (e1star + e3star)*(e1star + e3star) - pow(sqrt(e1star1 - dm11) + sqrt(e3star*e3star - dm33), 2);
    fptype minimum = (e1star + e3star) * (e1star + e3star) - (rte1mdm11 + rte3mdm33) * (rte1mdm11 + rte3mdm33);

    bool m13less = (m13 < minimum) ? false : true;
    // if (m13 < minimum) return false;

    // fptype maximum = pow(e1star + e3star, 2) - pow(sqrt(e1star*e1star - dm1*dm1) - sqrt(e3star*e3star - dm3*dm3), 2);
    fptype maximum = (e1star + e3star) * (e1star + e3star) - (rte1mdm11 - rte3mdm33) * (rte1mdm11 - rte3mdm33);
    bool m13grea   = (m13 > maximum) ? false : true;
    // if (m13 > maximum) return false;

    return m12less && m12grea && m13less && m13grea;
}

__constant__ fptype c_motherMass;
__constant__ fptype c_daug1Mass;
__constant__ fptype c_daug2Mass;
__constant__ fptype c_daug3Mass;

template <typename T> struct FunctorAmp3Body : public Functor {
  public:
    FunctorAmp3Body(T f, fptype m0, fptype slope, fptype power) :
        m_functor(f), m_m0(m0), m_slope(slope), m_power(power) {}

    // This is argus_upper
    __device__ fptype operator()(fptype *evt, ParameterContainer &pc) const {
        int num_obs = pc.getNumObservables();
        int id_m12  = pc.getObservable(0);
        int id_m13  = pc.getObservable(1);
        int id_num  = pc.getObservable(2);

        fptype m12 = evt[id_m12];
        fptype m13 = evt[id_m13];

        unsigned int numResonances = pc.getConstant(0);

        if(!inDalitz(m12, m13, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass)) {
            pc.incrementIndex(1, numResonances * 2, 2, num_obs, 1);

            // loop over resonances and efficiency functions
            for(int i = 0; i < numResonances; i++)
                pc.incrementIndex();

            // increment the efficiency function
            pc.incrementIndex();
            return 0;
        }

        fptype evtIndex = evt[id_num];

        int evtNum = floor(0.5 + evtIndex);

        fpcomplex totalAmp(0.0, 0.0);
        for(int i = 0; i < numResonances; i++) {
            totalAmp += fpcomplex(pc.getParameter(i * 2), pc.getParameter(i * 2 + 1)) * cResonances[i][evtNum];
        }

        fptype ret = thrust::norm(totalAmp);

        pc.incrementIndex(1, numResonances * 2, 2, num_obs, 1);

        fptype eff = functor(evt, pc);

        return ret * eff;
    }

  protected:
    T m_functor;
    fptype m_m0;
    fptype m_slope;
    fptype m_power;
};
