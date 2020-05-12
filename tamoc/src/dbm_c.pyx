# Using C++ for the complex type
# it's suppsed to be available in C, but I couldn't get it to work.

"""
dmb_c.pyx : Cython version of the fortran dbm code

All in one file: it's easier that way with Cython

! Provides root-finding capability in Fortran for cubic equations by
! an analytical solution, based on equations at:
!     http://en.wikipedia.org/wiki/Cubic_function
!
! S. Socolofsky
! June 2013
! Texas A&M University

"""
import sys

cimport cython

import numpy as np
cimport numpy as np

from cpython cimport array
import array

ctypedef double FLOAT_TYPE
ctypedef double complex COMPLEX_TYPE

FLOAT_ARR_TEMPLATE = array.array('d', [])
#COMPLEX_ARR_TEMPLATE = array.array('d', [])

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp, isnan, log

cdef extern from "<complex.h>" nogil:
    COMPLEX_TYPE csqrt(double complex z)
    COMPLEX_TYPE I

## Various constants
# There should be a way to get this from a C header, but this works
cdef FLOAT_TYPE EPS = sys.float_info.epsilon

## Some constants

# "standard" gravity is:  9.80665 m/s^2 using double, may be a few more digits here?
cdef FLOAT_TYPE G = 9.81
cdef FLOAT_TYPE PI = 3.141592653589793

# ! Define constants for use by the Peng-Robinson equation of state
# ! Ru in J / (mol K)
cdef FLOAT_TYPE Ru = 8.314510

## utilities that aren't in C

cpdef inline sum(FLOAT_TYPE[:] arr):
    cdef unsigned int i
    cdef FLOAT_TYPE result = 0.0
    for i in range(arr.size):
        result += arr[i]
    return result

cpdef inline sum_mult(FLOAT_TYPE[:] arr1,
                      FLOAT_TYPE[:] arr2,
                      ):
    """
    compute the sum of the items multiplied together:

    sum(arr1[:] * arr2[:])
    """
    cdef unsigned int i
    cdef FLOAT_TYPE result = 0.0
    for i in range(arr1.size):
        result += arr1[i] * arr2[i]
    return result


## Assorted non-dimensional numbers

cpdef inline FLOAT_TYPE eotvos(FLOAT_TYPE de,
                               FLOAT_TYPE rho_p,
                               FLOAT_TYPE rho,
                               FLOAT_TYPE sigma
                               ):
    """
    ! Calculate the Eotvos number per Clift et al. page 26
    !
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    !     sigma = interfacial tension (N/m)
    !
    ! Returns the non-dimensional Eo number
    !
    ! S. Socolofsky
    ! June 2013
    """
    return G * (rho - rho_p) * de**2 / sigma

cpdef inline FLOAT_TYPE morton(FLOAT_TYPE rho_p,
                               FLOAT_TYPE rho,
                               FLOAT_TYPE mu,
                               FLOAT_TYPE sigma):
    """
    !
    ! Calculate the Morton number per Clift et al. page 26
    !
    ! Input variables:
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     sigma = interfacial tension (N/m)
    !
    ! Returns the non-dimensional M number
    !
    ! S. Socolofsky
    ! June 2013
    """
    return G * mu**4 * (rho - rho_p) / (rho**2 * sigma**3)

cdef inline FLOAT_TYPE reynolds(FLOAT_TYPE de,
                                FLOAT_TYPE us,
                                FLOAT_TYPE rho,
                                FLOAT_TYPE mu):
    """
    ! Calculate the Reynolds number per Clift et al. page 26
    !
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !
    ! Returns the non-dimensional Re number
    !
    ! S. Socolofsky
    ! June 2013
    """
    return rho * de * us / mu


cdef inline FLOAT_TYPE h_parameter(FLOAT_TYPE Eo,
                                   FLOAT_TYPE M,
                                   FLOAT_TYPE mu,
                                   ):
    """
    ! Calculate H in equation (7-7) of Clift et al. page 176
    !
    ! Input variables:
    !     Eo = non-dimensional Eotvos number
    !     M = non-dimensional Morton number
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !
    ! Returns the non-dimensional parameter H in equation (7-7) of Clift et
    ! al. 1978 page 176
    !
    ! S. Socolofsky
    ! June 2013
    """

    return 4.0 / 3.0 * Eo * M**(-0.149) * (mu / 0.0009)**(-0.14)


# ! ----------------------------------------------------------------------------
# ! Slip velocity and shape functions
# ! ----------------------------------------------------------------------------


cpdef int particle_shape(FLOAT_TYPE de,
                         FLOAT_TYPE rho_p,
                         FLOAT_TYPE rho,
                         FLOAT_TYPE mu,
                         FLOAT_TYPE sigma):
    """
    ! Calculate the shape of a fluid particle
    !
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     sigma = interfacial tension (N/m)
    !
    ! Returns an integer flag:
    !     1 : sphere
    !     2 : ellipsoid
    !     3 : spherical cap
    !
    ! S. Socolofsky
    ! June 2013
    """

    cdef int shape_p

    # Declare the variables internal to the subroutine
    cdef FLOAT_TYPE Eo, M, H

    Eo = eotvos(de, rho_p, rho, sigma)
    M = morton(rho_p, rho, mu, sigma)
    H = h_parameter(Eo, M, mu)

    # Select the appropriate shape classification
    if (H < 2.0):
        shape_p = 1
    elif ((Eo < 40.0) and (M < 0.001) and (H < 1000.0)):
        shape_p = 2
    else:
        shape_p = 3
    return shape_p


cpdef FLOAT_TYPE theta_w_sc(FLOAT_TYPE de,
                            FLOAT_TYPE us,
                            FLOAT_TYPE rho,
                            FLOAT_TYPE mu):
    """
    ! Compute the wake angle for a spherical cap bubble
    !
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !
    ! Returns the wake angle (rad) for a spherical cap bubble using equation
    ! (8-1) in Clift et al. (1978) p. 204.
    !
    ! S. Socolofsky
    ! June 2013
    """

    cdef FLOAT_TYPE Re

    # Get the Reynolds number
    Re = reynolds(de, us, rho, mu)

    # Return the wake angle
    return PI * (50.0 + 190.0 * exp(-0.62 * Re**(0.4))) / 180.0

# ! ----------------------------------------------------------------------------
# ! Peng-Robinson Equations of State for Density and Fugacity
# ! ----------------------------------------------------------------------------

from cython.view cimport array as cvarray



cpdef FLOAT_TYPE trial_nparray(int n,
                               FLOAT_TYPE[:] arr1,
                               FLOAT_TYPE[:,:] arr2,
                               ):
    cdef FLOAT_TYPE result
    cdef unsigned int i

    cdef FLOAT_TYPE[:] carr = array.clone(FLOAT_ARR_TEMPLATE, n, zero=False)

    cdef FLOAT_TYPE arr[4]
    arr[:] = [1, 2, 3, 4]

    for i in range(n):
        carr[i] = arr1[i]

#    cdef FLOAT_TYPE[:] arr3 = np.zeros((n,), dtype=np.float64)
    cdef FLOAT_TYPE[:] arr3 = cvarray(shape=(n,), itemsize=sizeof(FLOAT_TYPE), format="d")
    for i in range(n):
        arr3[i] = carr[i]

    result = 0.0
    for i in range(n):
        result += arr3[i]

    return result

def mole_fraction(FLOAT_TYPE[:] mass,
                  FLOAT_TYPE[:] Mol_wt,
                  ):
    """
    Python wrapper around the mole_fraction computation

    Compute the mole fraction of a mixture from the mass

    Converts the masses of each component in a mixture to the mole fraction
    of each component in the mixture
    """
    cdef unsigned int nc = mass.shape[0]
    if Mol_wt.shape[0] != nc:
        raise ValueError("mass and Mol_wt arrays must be the same size")
    yk = np.empty((nc,), dtype=np.float64)

    mole_fraction_c(nc, mass, Mol_wt, yk)

    return yk


# # fixme: uncomment these once all tests pass!
# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef mole_fraction_c(unsigned int nc,
                     FLOAT_TYPE[:] mass,
                     FLOAT_TYPE[:] Mol_wt,
                     FLOAT_TYPE[:] yk):
    """
    ! Compute the mole fraction of a mixture from the mass
    !
    ! Converts the masses of each component in a mixture to the mole fraction
    ! of each component in the mixture.
    !
    ! Input variables:
    !     nc = number of components in the mixture
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !
    ! Returns the mole fraction (--) of the mixture.
    !
    ! S. Socolofsky
    ! July 2013
    """

    # ! Declare the input and output variable types
    # integer, intent(in) :: nc
    # real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt

    # real(kind = DP), intent(out), dimension(nc) :: yk

    # ! Declare the variables internal to the function
    # real(kind = DP), dimension(nc) :: n_moles

    # various options for a local array -- not sure which is best ...
    # NOTE: make sure to call free if you use this!
    # cdef FLOAT_TYPE *n_moles = <FLOAT_TYPE *> malloc(nc * sizeof(FLOAT_TYPE))

    cdef FLOAT_TYPE[:] n_moles = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)

    # NOPE -- doesn't work cdef FLOAT_TYPE[:] n_moles = <FLOAT_TYPE> bytearray(nc * sizeof(FLOAT_TYPE))
    # memoryview on Cython array
    # cyarr = cvarray(shape=(nc,), itemsize=sizeof(FLOAT_TYPE), format="d")
    # cdef FLOAT_TYPE[:] n_moles = cyarr
    # # plain cython array
    # cdef cvarray n_moles = cvarray(shape=(nc,), itemsize=sizeof(FLOAT_TYPE), format="d")

    cdef FLOAT_TYPE sum_moles = 0.0
    cdef unsigned int i

    # ! Compute the total number of moles
    for i in range(nc):
        n_moles[i] = mass[i] / Mol_wt[i]
        sum_moles += n_moles[i]

    # ! Compute the mole fraction
    for i in range(nc):
        yk[i] = n_moles[i] / sum_moles


cdef coefs(unsigned int nc,
               FLOAT_TYPE T,
               FLOAT_TYPE P,
               FLOAT_TYPE[:] mass,
               FLOAT_TYPE[:] Mol_wt,
               FLOAT_TYPE[:] Pc,
               FLOAT_TYPE[:] Tc,
               FLOAT_TYPE[:] omega,
               FLOAT_TYPE[:,:] delta_in,  # (nc, nc)
               FLOAT_TYPE[:,:] Aij,  # (15, 15)
               FLOAT_TYPE[:,:] Bij,  # (15, 15)
               FLOAT_TYPE[:,:] delta_groups,  # (nc,15)
               int calc_delta,
               FLOAT_TYPE *A,
               FLOAT_TYPE *B,
               FLOAT_TYPE[:] Ap,
               FLOAT_TYPE[:] Bp,
               FLOAT_TYPE[:] yk
               ):

    """
    ! Computes the mixture coefficients for the P-R EOS
    !
    ! Computes the mixing rules for the coefficients of the Peng-Robinson
    ! equation of state as described in McCain (1990), Properties of Petroleum
    ! Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    !
    ! Input variables are:
    !     nc = number of components in the mixture
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !     Pc = array of critical point pressures for each component (Pa)
    !     Tc = array of critical point temperatures for each component (K)
    !     omega = array of Pitzer acentric factors for each component (--)
    !     delta = matrix of binary interaction coefficients (--)
    !     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
    !     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
    !     delta_groups = group contribution numbers (normalized) for each
    !         component in the mixture (--)
    !     calc_groups = flag indicating whether or not delta_groups has
    !         been provided (1 = yes, -1 = no)
    !
    ! Output variables are:
    !     A = aT coefficient in P-R EOS
    !     B = b coefficient in P-R EOS
    !     Ap = non-dimensional array of mixture aT-coefficients
    !     Bp = non-dimensional array of mixture b-coefficients
    !
    ! S. Socolofsky
    ! June 2013
    """

    # ! Declare the input and output variable types
    # integer, intent(in) :: nc, calc_delta
    # real(kind = DP), intent(in) :: T, P
    # real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
    #                                             & omega
    # real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    # real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    # real(kind = DP), intent(in), dimension(nc, nc) :: delta_in
    # real(kind = DP), intent(out) :: A, B
    # real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk

    # ! Declare the variables internal to the function
    # integer :: i, j, k, l
    cdef unsigned int i, j, k, l
    # real(kind = DP) :: bd, aT, sum_term, sum1
    cdef FLOAT_TYPE bd, aT, sum_term, sum1
    # real(kind = DP), dimension(nc) :: mu, alpha, aTk, bk
    cdef np.ndarray[FLOAT_TYPE, ndim=1] mu = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] alpha = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] aTk = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] bk = np.zeros((nc,), dtype=np.float64)

    # real(kind = DP), dimension(nc, nc) :: delta
    cdef np.ndarray[FLOAT_TYPE, ndim=2] delta = np.zeros((nc, nc), dtype=np.float64)

#     ! Convert the masses to mole fraction
#     call mole_fraction(nc, mass, Mol_wt, yk)
    mole_fraction_c(nc, mass, Mol_wt, yk)

#     ! Compute the coefficient values for each gas in the mixture.  Use the
#     ! modified Peng-Robinson (1978) equations for mu
    for i in range(nc):
        if omega[i] > 0.49:
            mu[i] = (0.379642 + 1.48503 * omega[i] - 0.164423 *
                     omega[i]**2 + 0.016666 * omega[i]**3)
        else:
            mu[i] = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i]**2
    for i in range(nc):
        alpha[i] = (1.0 + mu[i] * (1.0 - (T / Tc[i])**(1.0/2.0)))**2
        aTk[i] = 0.45724 * Ru**2 * Tc[i]**2 / Pc[i] * alpha[i]
        bk[i] = 0.07780 * Ru * Tc[i] / Pc[i]

#     ! Initialize the output vector for delta to the input values
#     delta(:,:) = delta_in(:,:)
    # could I use a memcpy here?
    for i in range(nc):
        for j in range(nc):
            delta[i, j] = delta_in[i, j]

#     ! Get the temperature-dependent binary interaction coefficients (if
#     ! the user provided the group contributions)
    if (calc_delta > 0):
        # do j = 2, nc
        for j in range(1, nc):
            # do i = 1, j-1
            for i in range(j-1):
                sum1 = 0.0
                # do l = 1, 15
                for l in range(0, 15):
                    # do k = 1, 15
                    for k in range(0, 15):
                        sum_term = ((delta_groups[i, k] -
                                     delta_groups[j, k]) * (delta_groups[i, l] -
                                     delta_groups[j, l]) * Aij[k, l] *
                                     (298.15 / T) ** (Bij[k, l] / Aij[k, l] -
                                     1.0))
                        if (not isnan(sum_term)):
                            sum1 = sum1 + sum_term

                delta[i, j] = (- (0.5 * sum1 + (sqrt(aTk[i]) / bk[i] -
                              sqrt(aTk[j]) / bk[j]) ** 2) /
                             (2.0 * sqrt(aTk[i] * aTk[j]) /
                             (bk[i] * bk[j])))
                delta[j, i] = delta[i, j]

#     ! Use the mixing rules in McCain (1990)
    bd = 0.0
    for i in range(nc):
        bd += yk[i] * bk[i]
    aT = 0.0
    for i in range(nc):
        for j in range(nc):
            aT = aT + yk[i] * yk[j] * (aTk[i] * aTk[j])**(1.0/2.0) * (1.0 - delta[i,j])

#     ! Compute the coefficients of the polynomials for z-factor and fugacity
    A[0] = aT * P / (Ru**2 * T**2)
    B[0] = bd * P / (Ru * T)
    Bp = bk / bd
    for i in range(nc):
        temp = 0.0
        for j in range(nc):
            temp += yk[j] * aTk[j] ** (1.0 / 2.0) * (1.0 - delta[j, i])
        Ap[i] = (1.0 / aT * (2.0 * aTk[i]**(1.0 / 2.0) * temp))
              # sum(yk(:) * aTk(:)**(1.0/2.0) * (1.0 - delta(:,i))))

cdef int volume_trans(unsigned int nc,
                       FLOAT_TYPE T,
                       FLOAT_TYPE P,
                       FLOAT_TYPE[:] mass,
                       FLOAT_TYPE[:] Mol_wt,
                       FLOAT_TYPE[:] Pc,
                       FLOAT_TYPE[:] Tc,
                       FLOAT_TYPE[:] Vc,
                       FLOAT_TYPE[:] vt,
                       ) except -1:

    """
    ! Computes the volume translation parameter to correct the density
    !
    ! Computes the volume translation parameter to correct the density from
    ! the Peng-Robinson Equation of State based on Lin and Duan (2005),
    ! "Empirical correction to the Peng-Robinson equation of state for the
    ! saturated region," Fluid Phase Equilibria, 233: 194-203.  The volume
    ! translation parameter has a value for each component in the mixture.
    !
    ! Input Variables are:
    !     nc = number of components in the mixture
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     Pc = array of critical point pressures for each component (Pa)
    !     Tc = array of critical point temperatures for each component (K)
    !     Vc = array of critical point molar volumes for each component
    !          (m^3/mol)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !
    ! Output variable is:
    !     vt = volume translation parameter (m^3/mol)
    !
    ! S. Socolofsky
    ! December 2014
    """

    # ! Declare the variables internal to the function
    # real(kind = DP), dimension(nc) :: Zc, beta, gamma, f_Tr, cc
    cdef FLOAT_TYPE[:] Zc = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)
    cdef FLOAT_TYPE[:] beta = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)
    cdef FLOAT_TYPE[:] gamma = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)
    cdef FLOAT_TYPE[:] f_Tr = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)
    cdef FLOAT_TYPE[:] cc = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)

    # ! Compute the compressibility factor (--) for each component of the
    # ! mixture
    # Zc = Pc(:) * Vc(:) / (Ru * Tc(:))
    for i in range(nc):
        Zc[i] = Pc[i] * Vc[i] / (Ru * Tc[i])

    # ! Calculate the parameters in the Lin and Duan (2005) paper:  beta is
    # ! from equation (12)
    # beta = -2.8431D0 * exp(-64.2184D0 * (0.3074D0 - Zc(:))) + 0.1735D0
    for i in range(nc):
        beta[i] = -2.8431 * exp(-64.2184 * (0.3074 - Zc[i])) + 0.1735

    # ! and gamma is from Equation (13)
    # gamma = -99.2558D0 + 301.6201D0 * Zc(:)
    for i in range(nc):
        gamma[i] = -99.2558 + 301.6201 * Zc[i]

    # ! Account for the temperature dependence (equation 10)
    # f_Tr = beta(:) + (1.0D0 - beta(:)) * exp(gamma(:) * abs(1.0D0-T / Tc(:)))
    for i in range(nc):
        f_Tr[i] = beta[i] + (1.0 - beta[i]) * exp(gamma[i] * abs(1.0 - T / Tc[i]))

    # ! Compute the volume translation for the critical point (equation 9)
    # cc = (0.3074D0 - Zc(:)) * Ru * Tc(:) / Pc(:)
    for i in range(nc):
        cc[i] = (0.3074 - Zc[i]) * Ru * Tc[i] / Pc[i]

    # ! Finally, the volume translation at the given state is (equation 8)
    # vt = f_Tr * cc
    for i in range(nc):
        vt[i] = f_Tr[i] * cc[i]

    return 0

cdef int z_pr(unsigned int nc,  # int return for error code
               FLOAT_TYPE T,
               FLOAT_TYPE P,
               FLOAT_TYPE[:] mass,
               FLOAT_TYPE[:] Mol_wt,
               FLOAT_TYPE[:] Pc,
               FLOAT_TYPE[:] Tc,
               FLOAT_TYPE[:] omega,
               FLOAT_TYPE[:, :] delta,  # (nc, nc)
               FLOAT_TYPE[:, :] Aij,  # (15, 15)
               FLOAT_TYPE[:, :] Bij,  # (15, 15)
               FLOAT_TYPE[:, :] delta_groups,  # (nc,15)
               int calc_delta,
               # rest used as output
               FLOAT_TYPE[:, :] z,
               FLOAT_TYPE * A,
               FLOAT_TYPE * B,
               FLOAT_TYPE[:] Ap,
               FLOAT_TYPE[:] Bp,
               FLOAT_TYPE[:] yk
               ) except -1:

    """
    ! Computes the z-factor for gas and liquid of a mixture using the P-R EOS
    !
    ! Computes the z-factor of a mixture using the Peng-Robinson equation of
    ! state as described in McCain (1990), Properties of Petroleum Fluids, 2nd
    ! Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    !
    ! The approach results in a cubic equation for the z-factor in which the
    ! largest root is for the liquid phase and the smallest root is for the
    ! gas phase; the middle root is discarded.  If the temperature is above
    ! the critical temperature, only one real root is obtained for the
    ! critical state.
    !
    ! Input variables are:
    !     nc = number of components in the mixture
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !     Pc = array of critical point pressures for each component (Pa)
    !     Tc = array of critical point temperatures for each component (K)
    !     omega = array of Pitzer acentric factors for each component (--)
    !     delta = matrix of binary interaction coefficients (--)
    !     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
    !     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
    !     delta_groups = group contribution numbers (normalized) for each
    !         component in the mixture (--)
    !     calc_groups = flag indicating whether or not delta_groups has
    !         been provided (1 = yes, -1 = no)
    !
    ! Output variables are:
    !     z = array of the z-factor (gas, liquid) for the mixture (--)
    !     A, B, Ap, Bp = coefficients for the P-R EOS defined in coefs
    !
    ! S. Socolofsky
    ! June 2013
    """
#     use EOS_Constants
#     implicit none

#     ! Declare the input and output variable types
#     integer, intent(in) :: nc, calc_delta
#     real(kind = DP), intent(in) :: T, P
#     real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
#                                                 & omega
#     real(kind = DP), intent(in), dimension(nc, nc) :: delta
#     real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
#     real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij

#     real(kind = DP), intent(out) :: A, B
#     real(kind = DP), intent(out), dimension(2, 1) :: z
#     real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk

#     ! Declare the variables internal to the function
#     integer :: i
    cdef int i
#     real(kind = DP) :: z_max, z_min
    cdef FLOAT_TYPE z_max, z_min
#     real(kind = DP), dimension(4) :: p_coefs
    # cdef FLOAT_TYPE[:] p_coefs = array.clone(FLOAT_ARR_TEMPLATE, 4, zero=False)
    cdef FLOAT_TYPE p_coefs_carr[4]
    cdef FLOAT_TYPE[:] p_coefs = p_coefs_carr
#     complex(kind = DP), dimension(3) :: z_roots
    cdef COMPLEX_TYPE z_roots_carr[3]
    cdef COMPLEX_TYPE[:] z_roots = z_roots_carr

#     ! Compute the coefficients of the polynomial for z-factor
#     call coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
#         &      delta_groups, calc_delta, &
#         &      A, B, Ap, Bp, yk)

    coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij,
          delta_groups, calc_delta, A, B, Ap, Bp, yk)

    p_coefs[0] = 1.0
    p_coefs[1] = B[0] - 1.0
    p_coefs[2] = A[0] - 2.0 * B[0] - 3.0 * B[0]**2
    p_coefs[3] = B[0]**3 + B[0]**2 - A[0] * B[0]

# #     ! Find the roots of the cubic equation of state
#     # call cubic_roots(p_coefs, z_roots)
    cubic_roots_c(p_coefs, z_roots)

#     ! Extract the correct z-factors
    z_max = 0.0
    for i in range(3):
        if z_roots[i].imag == 0.0:
            if z_roots[i].real > z_max:
                z_max = z_roots[i].real
    z_min = z_max
    for i in range(3):
        if z_roots[i].imag == 0.0:
            if ((z_roots[i].real < z_min) and (z_roots[i].real > 0.0)):
                z_min = z_roots[i].real

#     ! Return the z-factors in z
    z[0, 0] = z_max
    z[1, 0] = z_min

    return 0


def viscosity(FLOAT_TYPE T,
              FLOAT_TYPE P,
              FLOAT_TYPE[:] mass,
              FLOAT_TYPE[:] Mol_wt,
              FLOAT_TYPE[:] Pc,
              FLOAT_TYPE[:] Tc,
              FLOAT_TYPE[:] Vc,
              FLOAT_TYPE[:] omega,
              FLOAT_TYPE[:,:] delta,  # (nc, nc)
              FLOAT_TYPE[:,:] Aij,  # (15, 15)
              FLOAT_TYPE[:,:] Bij,  # (15, 15)
              FLOAT_TYPE[:,:] delta_groups, # (nc,15)
              int calc_delta,
              ):
    """
    wrapper function around the c viscosity function:

    f2py automatically converts output parameters, this is doing
    the same thing by hand.
    """
    nc = mass.shape[0]
    cdef np.ndarray[FLOAT_TYPE, ndim=2] mu = np.zeros((2,1), dtype=np.float64) # (2, 1)

    viscosity_c(
        nc,
        T,
        P,
        mass,
        Mol_wt,
        Pc,
        Tc,
        Vc,
        omega,
        delta,  # (nc, nc)
        Aij,  # (15, 15)
        Bij,  # (15, 15)
        delta_groups,  # (nc,15)
        calc_delta,
        mu,  # (2, 1)
    )

    return mu


cdef int viscosity_c(nc,
                     FLOAT_TYPE T,
                     FLOAT_TYPE P,
                     FLOAT_TYPE[:] mass,
                     FLOAT_TYPE[:] Mol_wt,
                     FLOAT_TYPE[:] Pc,
                     FLOAT_TYPE[:] Tc,
                     FLOAT_TYPE[:] Vc,
                     FLOAT_TYPE[:] omega,
                     FLOAT_TYPE[:, :] delta,  # (nc, nc)
                     FLOAT_TYPE[:, :] Aij,  # (15, 15)
                     FLOAT_TYPE[:, :] Bij,  # (15, 15)
                     FLOAT_TYPE[:, :] delta_groups,  # (nc,15)
                     int calc_delta,
                     FLOAT_TYPE[:, :] mu,  # (2, 1)
                     ) except -1:

    """
    ! Computes the viscosity of a petroleum fluid
    !
    ! Computes the viscosity of the given fluid mixture for the gas and
    ! liquid phases following the method in Pedersen et al. "Phase Behavior
    ! of Petroleum Reservoir Fluids", 2nd edition, Chapter 10.
    !
    ! This method correlates the viscosity of the mixture to the viscosity
    ! of methane taken at a specialized corresponding state.  The function
    ! has the properties of methane hard-wired so that any mixture can be
    ! evaluated.
    !
    ! Input variables:
    !     nc = number of components in the mixture
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !     Pc = array of critical point pressures for each component (Pa)
    !     Tc = array of critical point temperatures for each component (K)
    !     Vc = array of critical point molar volumes for each component
    !          (m^3/mol)
    !     omega = array of Pitzer acentric factors for each component (--)
    !     delta = matrix of binary interaction coefficients (--)
    !     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
    !     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
    !     delta_groups = group contribution numbers (normalized) for each
    !         component in the mixture (--)
    !     calc_groups = flag indicating whether or not delta_groups has
    !         been provided (1 = yes, -1 = no)
    !
    ! Output variable is:
    !     mu = numpy array of the viscosity [gas, liquid] of the mixture
    !         (Pa s)
    !
    ! S. Socolofsky
    ! June 2015
    !
    """

#     ! Declare the variables internal to the function
#     integer :: i, j
    cdef unsigned int i, j
#     real(kind = DP) :: A, B, C, F, rho_c0, eta_0, eta_1, delta_T, htan, &
#                      & numerator, denominator, Tc_mix, Pc_mix, M_bar_n, &
#                      & M_bar_w, M_mix
    cdef FLOAT_TYPE A, B, C, F, rho_c0, eta_0, eta_1, delta_T, htan, \
               numerator, denominator, Tc_mix, Pc_mix, M_bar_n, \
               M_bar_w, M_mix

#     real(kind = DP), dimension(1) :: M0, Tc0, Pc0, omega0, Vc0
    cdef FLOAT_TYPE M0[1], Tc0[1], Pc0[1], omega0[1], Vc0[1], M0_temp[1]

    cdef FLOAT_TYPE ONE[1]
    ONE[0] = 1.0
#     real(kind = DP), dimension(2) :: T0, P0
    cdef FLOAT_TYPE T0[2], P0[2]
#     real(kind = DP), dimension(7) :: jc, kc
    cdef FLOAT_TYPE jc[7], kc[7]
#     real(kind = DP), dimension(9) :: GV
    cdef FLOAT_TYPE GV[9]
#     real(kind = DP), dimension(nc) :: z, M
    cdef FLOAT_TYPE[:] z = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)
    cdef FLOAT_TYPE[:] M = array.clone(FLOAT_ARR_TEMPLATE, nc, zero=False)

#     real(kind = DP), dimension(1,1) :: delta0
    cdef FLOAT_TYPE delta0[1][1]
#     real(kind = DP), dimension(1,15) :: delta_groups0
    cdef FLOAT_TYPE delta_groups0[1][15]

#     real(kind = DP), dimension(2, 1) :: theta, delta_eta_p, delta_eta_pp, &
#                                       & rho0, eta_ch4, rho_r, alpha_mix, &
#                                       & alpha0
    cdef FLOAT_TYPE theta[2][1], delta_eta_p[2][1], delta_eta_pp[2][1], \
                    rho0[2][1], eta_ch4[2][1], rho_r[2][1], alpha_mix[2][1], \
                    alpha0[2][1]
#     ! Enter the parameter values from Table 10.1
#     GV = [-2.090975D5, 2.647269D5, -1.472818D5, 4.716740D4, -9.491872D3, &
#         & 1.219979D3, -9.627993D1, 4.274152D0, -8.141531D-2]
    GV[:] = [-2.090975E5, 2.647269E5, -1.472818E5, 4.716740E4, -9.491872E3, \
             1.219979E3, -9.627993E1, 4.274152E0, -8.141531E-2]
    A = 1.696985927
    B = -0.133372346
    C = 1.4
    F = 168.0
#     jc = [-10.3506D0, 17.5716D0, -3019.39D0, 188.730D0, 0.0429036D0, &
#         & 145.290D0, 6127.68D0]
    jc[:] = [-10.3506, 17.5716, -3019.39, 188.730, 0.0429036,
             145.290, 6127.68]
#     kc = [-9.74602D0, 18.0834D0, -4126.66D0, 44.6055D0, 0.976544D0, &
#         & 81.8134D0, 15649.9D0]
    kc[:] = [-9.74602, 18.0834, -4126.66, 44.6055, 0.976544,
             81.8134, 15649.9]
    print("GV:", GV)
    print("jc:", jc)
    print("kc:", kc)
#     ! Enter the properties for the reference fluid (methane)
#     M0(1) = 16.043D-3
#     Tc0(1) = 190.56D0
#     Pc0(1) = 4599000.0D0
#     omega0(1) = 0.011D0
#     Vc0(1) = 9.86D-5
#     delta0(1,1) = 0.0D0
    M0[0] = 16.043E-3
    Tc0[0] = 190.56
    Pc0[0] = 4599000.0
    omega0[0] = 0.011
    Vc0[0] = 9.86E-5
    delta0[0][0] = 0.0

#     delta_groups0(1,:) = [0.0D0, 0.0D0, 0.0D0, 0.0D0, 1.0D0, 0.0D0, 0.0D0, &
#         &                 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, &
#         &                 0.0D0]
    delta_groups0[0][:] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0]
#     rho_c0 = 162.84D0
    rho_c0 = 162.84

#     ! 1.  Prepare the variables to determine the corresponding states between
#     !     the given mixture and the reference fluid (methane) ----------------

#     ! Get the mole fraction of the components of the mixture
#     call mole_fraction(nc, mass, Mol_wt, z)
    mole_fraction_c(nc, mass, Mol_wt, z)

#     ! Compute equation (10.19)
#     numerator = 0.0D0
#     denominator = 0.0D0
#     do i = 1, nc
#         do j = 1, nc
#             numerator = numerator + z(i) * z(j) * ((Tc(i) / Pc(i)) &
#                 &       **(1.0D0/3.0D0) + (Tc(j) / Pc(j))**(1.0D0/3.0D0)) &
#                 &       **3 * sqrt(Tc(i) * Tc(j))
#             denominator = denominator + z(i) * z(j) * ((Tc(i) / Pc(i)) &
#                 &       **(1.0D0/3.0D0) + (Tc(j) / Pc(j))**(1.0D0/3.0D0)) &
#                 &       **3
#         end do
#     end do
#.    Tc_mix = numerator / denominator
    # Compute equation (10.19)
    numerator = 0.0
    denominator = 0.0
    for i in range(nc):
        for j in range(nc):
            # fixme: compute denominator, then take sqrt for numerator?
            numerator = (numerator + z[i] * z[j] * ((Tc[i] / Pc[i])
                         **(1.0/3.0) + (Tc[j] / Pc[j])**(1.0/3.0))**3 *
                         sqrt(Tc[i] * Tc[j])
                         )
            denominator = (denominator + z[i] * z[j] * ((Tc[i] / Pc[i])
                           **(1.0/3.0) + (Tc[j] / Pc[j])**(1.0/3.0))**3
                           )
    Tc_mix = numerator / denominator

    print("Tc_mix:", Tc_mix)
#     ! Compute equation (10.22)
#     Pc_mix = 8.0D0 * numerator / denominator**2
    # Compute equation (10.22)
    Pc_mix = 8.0 * numerator / denominator**2

    print("Pc_mix", Pc_mix)
#     ! Get the density of methane at TTc0/Tc_mix and PPc0/Pc_mix
#     call density(1, T * Tc0(1) / Tc_mix, P * Pc0(1) / Pc_mix, [1.0D0], M0, &
#         &        Pc0, Tc0, Vc0, omega0, delta0, Aij, Bij, delta_groups0, &
#         &        -1, rho0)

    # Get the density of methane at TTc0/Tc_mix and PPc0/Pc_mix
    # not sure this matters, but...
    cdef FLOAT_TYPE T_temp = Tc0[0] / Tc_mix
    cdef FLOAT_TYPE P_temp = P * Pc0[0] / Pc_mix
    density_c(1,
              T_temp,
              P_temp,
              ONE,
              M0,
              Pc0,
              Tc0,
              Vc0,
              omega0,
              delta0,
              Aij, Bij,
              delta_groups0,
              -1,
              rho0,
              )

#     ! Compute equation (10.27)
#     rho_r(:,1) = rho0(:,1) / rho_c0
    rho_r[0][0] = rho0[0][0] / rho_c0
    rho_r[1][0] = rho0[1][0] / rho_c0

#     ! Compute equation (10.23), where M is in g/mol
    # M = Mol_wt(:) * 1.0D3
    # M_bar_n = sum(z(:) * M(:))
    # M_bar_w = sum(z(:) * M(:)**2) / M_bar_n
    # M_mix = 1.304D-4 * (M_bar_w**2.303D0 - M_bar_n**2.303D0) + M_bar_n
    for i in range(nc):
        M[i] = Mol_wt[i] * 1.0  # fixme: why * 1.0?
    M_bar_n = sum_mult(z, M)
    for i in range(nc):
        M[i] = M[i]**2
    M_bar_w = sum_mult(z, M) / M_bar_n
    M_mix = 1.304E-4 * (M_bar_w**2.303 - M_bar_n**2.303) + M_bar_n

#     ! Compute equation (10.26), where M is in g/mol
#     M0 = M0(:) * 1.0D3
#     alpha_mix(:,1) = 1.0D0 + 7.378D-3 * rho_r(:,1)**1.847D0 * M_mix**0.5173D0
#     alpha0(:,1) = 1.0D0 + 7.378D-3 * rho_r(:,1)**1.847D0 * M0(1)**0.5173D0
    for i in range(nc):
        M0[i] = M0[i] * 1.0E3
    alpha_mix[0][0] = 1.0 + 7.378E-3 * rho_r[0][0]**1.847 * M_mix**0.5173
    alpha_mix[1][0] = 1.0 + 7.378E-3 * rho_r[1][0]**1.847 * M_mix**0.5173
    alpha0[0][0] = 1.0 + 7.378E-3 * rho_r[0][0]**1.847 * M0[0]**0.5173
    alpha0[1][0] = 1.0 + 7.378E-3 * rho_r[1][0]**1.847 * M0[0]**0.5173

#     ! 2.  Compute the viscosity of methane at the corresponding state --------
#     T0 = T * Tc0(1) / Tc_mix * alpha0(:,1) / alpha_mix(:,1)
#     P0 = P * Pc0(1) / Pc_mix * alpha0(:,1) / alpha_mix(:,1)
    T0[0] = T * Tc0[0] / Tc_mix * alpha0[0][0] / alpha_mix[0][0]
    T0[1] = T * Tc0[0] / Tc_mix * alpha0[1][0] / alpha_mix[1][0]

    print("T0:", T0)

    P0[0] = P * Pc0[0] / Pc_mix * alpha0[0][0] / alpha_mix[0][0]
    P0[1] = P * Pc0[0] / Pc_mix * alpha0[1][0] / alpha_mix[1][0]
    print("P0:", P0)

#     ! Compute each state separately
#     do i = 1,2
    # fixme: maybe making a function and calling it twice would be better?
    for i in range(2):
        # ! Get the density of methane at T0 and P0.  Be sure to use molecular
        # ! weight in kg/mol
        # call density(1, T0(i), P0(i), [1.0D0], M0*1.0D-3, Pc0, Tc0, Vc0, &
        #     &        omega0, delta0, Aij, Bij, delta_groups0, -1, rho0)
        M0_temp[0] = M0[0] * 1.0E-3
        T_temp = T0[i]  # probably not neccesary, but removes a warning
        P_temp = P0[i]
        density_c(1, T_temp, P_temp, ONE, M0_temp, Pc0, Tc0, Vc0,
                  omega0, delta0, Aij, Bij, delta_groups0, -1, rho0)

        print("rho0 after density call:", rho0)

#         ! Compute equation (10.10)
#         theta(:,1) = (rho0(:,1) - rho_c0) / rho_c0
        theta[0][0] = (rho0[0][0] - rho_c0) / rho_c0
        theta[1][0] = (rho0[1][0] - rho_c0) / rho_c0

        print("theta:", theta)
#         ! Equation (10.9) with T in K and rho in g/cm^3
#         rho0(:,1) = rho0(:,1) * 1.0D-3
        print("rho0 before rescaling:", rho0)
        rho0[0][0] = rho0[0][0] * 1.0E-3
        rho0[1][0] = rho0[1][0] * 1.0E-3

        print("rho0 after scaling:", rho0)
#         delta_eta_p(:,1) = exp(jc(1) + jc(4) / T0(i)) * (exp(rho0(:,1) &
#             &              **0.1D0 * (jc(2) + jc(3) / T0(i)**1.5D0) + &
#             &              theta(:,1) * rho0(:,1)**0.5D0 * (jc(5) + jc(6) &
#             &              / T0(i) + jc(7) / T0(i)**2)) - 1.0D0)

        print("Components of delta_eta_p:")

        print(exp(jc[0] + jc[3] / T0[i]))

        print("a", (rho0[0][0]**0.1 * (jc[1] + jc[2] / T0[i]**1.5))
              )

        ## this part of the equation is huge -- leading to an inf in the exp() call
        print("b", theta[0][0] * rho0[0][0]**0.5
              )

        print("c", (jc[4] + jc[5] / T0[i] + jc[6] / T0[i]**2)
              )

        print("d", (theta[0][0] * rho0[0][0]**0.5 *
                                (jc[4] + jc[5] / T0[i] + jc[6] / T0[i]**2)
                                  )
              )

        print("e", (exp(rho0[0][0]**0.1 * (jc[1] + jc[2] / T0[i]**1.5) +
                                theta[0][0] * rho0[0][0]**0.5 *
                                (jc[4] + jc[5] / T0[i] + jc[6] / T0[i]**2)
                                  ) - 1.0
                                 )
              )

        delta_eta_p[0][0] = (exp(jc[0] + jc[3] / T0[i]) *
                             (exp(rho0[0][0]**0.1 * (jc[1] + jc[2] / T0[i]**1.5) +
                                theta[0][0] * rho0[0][0]**0.5 *
                                (jc[4] + jc[5] / T0[i] + jc[6] / T0[i]**2)
                                  ) - 1.0
                                 )
                             )

        delta_eta_p[1][0] = (exp(jc[0] + jc[3] / T0[i]) * (exp(rho0[1][0] \
                             **0.1 * (jc[1] + jc[2] / T0[i]**1.5) + \
                             theta[1][0] * rho0[1][0]**0.5 * (jc[4] + jc[4] \
                             / T0[i] + jc[5] / T0[i]**2)) - 1.0))
        print("delta_eta_p:", delta_eta_p)

#         ! Equation (10.28)
#         delta_eta_pp(:,1) = exp(kc(1) + kc(4) / T0(i)) * (exp(rho0(:,1) &
#             &               **0.1D0 * (kc(2) + kc(3) / T0(i)**1.5D0) + &
#             &               theta(:,1) * rho0(:,1)**0.5D0 * (kc(5) + kc(6) &
#             &               / T0(i) + kc(7) / T0(i)**2)) - 1.0D0)

        delta_eta_pp[0][0] = exp(kc[0] + kc[3] / T0[i]) * (exp(rho0[0][0] \
                             **0.1 * (kc[1] + kc[2] / T0[i]**1.5) + \
                             theta[0][0] * rho0[0][0]**0.5 * (kc[4] + kc[5] \
                             / T0[i] + kc[6] / T0[i]**2)) - 1.0)
        delta_eta_pp[1][0] = exp(kc[0] + kc[3] / T0[i]) * (exp(rho0[1][0] \
                             **0.1 * (kc[1] + kc[2] / T0[i]**1.5) + \
                             theta[1][0] * rho0[1][0]**0.5 * (kc[4] + kc[5] \
                             / T0[i] + kc[6] / T0[i]**2)) - 1.0)
        print("delta_eta_pp:", delta_eta_pp)

#         ! Equation (10.7)
#         eta_0 = GV(1) / T0(i) + GV(2) / T0(i)**(2.0D0/3.0D0) + GV(3) / &
#             &   T0(i)**(1.0D0/3.0D0) + GV(4) + GV(5) * T0(i)**(1.0D0/3.0D0) &
#             &   + GV(6) * T0(i)**(2.0D0/3.0D0) + GV(7) * T0(i) + GV(8) * &
#             &   T0(i)**(4.0D0/3.0D0) + GV(9) * T0(i)**(5.0D0/3.0D0)

        eta_0 = GV[0] / T0[i] + GV[1] / T0[i]**(2.0/3.0) + GV[2] / \
                T0[i]**(1.0/3.0) + GV[3] + GV[4] * T0[i]**(1.0/3.0) \
                + GV[5] * T0[i]**(2.0/3.0) + GV[6] * T0[i] + GV[7] * \
                T0[i]**(4.0/3.0) + GV[8] * T0[i]**(5.0/3.0)
        print("eta_0:", eta_0)

#         ! Equation (10.8)
#         eta_1 = A + B * (C - log(T0(i) / F))**2
        eta_1 = A + B * (C - log(T0[i] / F))**2
        print("eta_1:", eta_1)
#         ! Equation (10.32)
#         delta_T = T0(i) - 91.0D0
        delta_T = T0[i] - 91.0
        print("delta_T", delta_T)
#         ! Equation (10.31)
#         htan = (exp(delta_T) - exp(-delta_T)) / (exp(delta_T) + exp(-delta_T))
        ## fixme: this is the hyperbolic tangent, yes? better to use math.tanh()
        #         and that's GOT to be there in Fortran!
        #.        Also: withthe test code, I'm getting 1.0 anyway :-)
        htan = (exp(delta_T) - exp(-delta_T)) / (exp(delta_T) + exp(-delta_T))
        print("htan:", htan)
#         ! Viscosity of methane (Equation 10.29) -- reported in (Pa s)
        # eta_ch4(i,1) = (eta_0 + eta_1 + (htan + 1.0D0) / 2.0D0 * &
        #     &          delta_eta_p(i,1) + (1.0D0 - htan) / 2.0D0 * &
        #     &          delta_eta_pp(i,1)) * 1.0e-7
        eta_ch4[i][0] = (eta_0 + eta_1 + (htan + 1.0) / 2.0 * \
                         delta_eta_p[i][0] + (1.0 - htan) / 2.0 * \
                         delta_eta_pp[i][0]) * 1.0E-7
        print("i, eta_ch4:", i, eta_ch4)

#     end do

#     ! Compute the viscosity of the mixture at the given T and P
#     mu(:,1) = (Tc_mix / Tc0(1))**(-1.0D0/6.0D0) * (Pc_mix / Pc0(1))** &
#         &     (2.0D0/3.0D0) * (M_mix / M0(1))**(0.5D0) * alpha_mix(:,1) / &
#         &     alpha0(:,1) * eta_ch4(:,1)
    print("mu: before setting", mu[0, 0], mu[1, 0])
    mu[0][0] = ((Tc_mix / Tc0[0])**(-1.0 / 6.0) * (Pc_mix / Pc0[0])**
                (2.0 / 3.0) * (M_mix / M0[0])**(0.5) * alpha_mix[0][0] /
                alpha0[0][0] * eta_ch4[0][0])
    mu[1][0] = ((Tc_mix / Tc0[0])**(-1.0 / 6.0) * (Pc_mix / Pc0[0])**
                (2.0 / 3.0) * (M_mix / M0[0])**(0.5) * alpha_mix[1][0] /
                alpha0[1][0] * eta_ch4[1][0])
    print("mu: after setting", mu[0, 0], mu[1, 0])

# end subroutine viscosity


def density(FLOAT_TYPE T,
            FLOAT_TYPE P,
            FLOAT_TYPE[:] mass,
            FLOAT_TYPE[:] Mol_wt,
            FLOAT_TYPE[:] Pc,
            FLOAT_TYPE[:] Tc,
            FLOAT_TYPE[:] Vc,
            FLOAT_TYPE[:] omega,
            FLOAT_TYPE[:,:] delta,  # (nc, nc)
            FLOAT_TYPE[:,:] Aij,  # (15, 15)
            FLOAT_TYPE[:,:] Bij,  # (15, 15)
            FLOAT_TYPE[:,:] delta_groups, # (nc,15)
            int calc_delta):
    """
    wrapper to call the C density function
    """
    cdef unsigned int nc = mass.shape[0]

    # # Memoryview on a NumPy array
    narr = np.arange(27, dtype=np.dtype("i")).reshape((3, 3, 3))
    cdef int [:, :, :] narr_view = narr

    # cdef np.ndarray[FLOAT_TYPE, ndim=2] rho_arr = np.zeros((2, 1), dtype=np.float64)
    rho_arr = np.zeros((2, 1), dtype=np.float64)
    # rho_arr = np.arange(2, dtype=np.dtype("d")).reshape((2, 1))
    cdef FLOAT_TYPE [:, :] rho = rho_arr

    density_c(nc,
              T,
              P,
              mass,
              Mol_wt,
              Pc,
              Tc,
              Vc,
              omega,
              delta,  # (nc, nc)
              Aij,  # (15, 15)
              Bij,  # (15, 15)
              delta_groups,  # (nc,15)
              calc_delta,
              rho,  # should be (2,1) in size
              )

    return rho_arr

cdef int density_c(unsigned int nc,
                   FLOAT_TYPE T,
                   FLOAT_TYPE P,
                   FLOAT_TYPE[:] mass,
                   FLOAT_TYPE[:] Mol_wt,
                   FLOAT_TYPE[:] Pc,
                   FLOAT_TYPE[:] Tc,
                   FLOAT_TYPE[:] Vc,
                   FLOAT_TYPE[:] omega,
                   FLOAT_TYPE[:, :] delta,  # (nc, nc)
                   FLOAT_TYPE[:, :] Aij,  # (15, 15)
                   FLOAT_TYPE[:, :] Bij,  # (15, 15)
                   FLOAT_TYPE[:, :] delta_groups, # (nc,15)
                   int calc_delta,
                   FLOAT_TYPE[:, :] rho , # should be (2,1) in size
                   ) except -1:
    # cdef FLOAT_TYPE rho_arr[2][1]
    # cdef FLOAT_TYPE[:, :] rho = rho_arr
    # rho[0, 0] = 0.0
    # rho[1, 0] = 0.0
    """
    ! Computes the liquid and gas density of a mixture from the P-R EOS
    !
    ! Computes the density of a mixture using the Peng-Robinson equation
    ! of state as described in McCain (1990), Properties of Petroleum
    ! Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    !
    ! Input Variables are:
    !     nc = number of components in the mixture
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)
    !     Pc = array of critical point pressures for each component (Pa)
    !     Tc = array of critical point temperatures for each component (K)
    !     Vc = array of critical point molar volumes for each component
    !          (m^3/mol)
    !     omega = array of Pitzer acentric factors for each component (--)
    !     delta = matrix of binary interaction coefficients (--)
    !     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
    !     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
    !     delta_groups = group contribution numbers (normalized) for each
    !         component in the mixture (--)
    !     calc_groups = flag indicating whether or not delta_groups has
    !         been provided (1 = yes, -1 = no)
    !
    ! Output variable is:
    !     rho = numpy array of the density [gas, liquid] of the mixture
    !         (kg/m^3)
    !
    ! S. Socolofsky
    ! June 2013
    """
    # ! Declare the input and output variable types
    # real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, Vc, &
    #                                             & omega
    # real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    # real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    # real(kind = DP), intent(in), dimension(nc, nc) :: delta
    # real(kind = DP), intent(out), dimension(2, 1) :: rho

    # cdef unsigned int nc  # the number of components -- should match array sizes
    # nc = mass.shape[0]
    # ## fixme: check other arrays for the right size?
    # cdef np.ndarray[FLOAT_TYPE, ndim=2] rho = np.zeros((2, 1), dtype=np.float64)  # should be (2,1) in size

#     ! Declare the variables internal to the function
#     real(kind = DP) :: A, B, R
    cdef FLOAT_TYPE A = 0.0
    cdef FLOAT_TYPE B = 0.0
    cdef FLOAT_TYPE R = 0.0  # FIXME: is R used?
    cdef FLOAT_TYPE temp = 0.0  #

#     real(kind = DP), dimension(2, 1) :: z, nu
    cdef np.ndarray[FLOAT_TYPE, ndim=2]  z = np.zeros((2, 1), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=2] nu = np.zeros((2, 1), dtype=np.float64)
#     real(kind = DP), dimension(nc) :: Ap, Bp, yk, vt
    cdef np.ndarray[FLOAT_TYPE, ndim=1] Ap = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] Bp = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] yk = np.zeros((nc,), dtype=np.float64)
    cdef np.ndarray[FLOAT_TYPE, ndim=1] vt = np.zeros((nc,), dtype=np.float64)

#     ! Get the z-factor using the Peng-Robinson equation of state
    z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij,
         delta_groups, calc_delta,
         z, &A, &B, Ap, Bp, yk
         )

#     ! Convert the masses to mole fraction
    mole_fraction_c(nc, mass, Mol_wt, yk)

#     ! Compute the volume translation coefficient
    volume_trans(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, vt)

#     ! Compute the molar volume
#     nu = z * Ru * T / P - sum(yk(:) * vt(:))
    for i in range(2):
        temp = 0.0
        for j in range(nc):  # fixme: use sum_mult() here
            temp += yk[j] * vt[j]
        nu[i, 0] = z[i, 0] * Ru * T / P - temp

#     !0Compute and return the density
#     r1o = 1.0D0 / nu * sum(yk(:) * Mol_wt(:))
    rho[0, 0] = 1.0 / nu[0, 0] * sum_mult(yk, Mol_wt)
    rho[1, 0] = 1.0 / nu[1, 0] * sum_mult(yk, Mol_wt)

    return 0


# def identity(np.ndarray [np.complex128_t, ndim=1] weights):
#     return weights


def cubic_roots(np.ndarray[FLOAT_TYPE, ndim = 1] p):
    """
    a python wrapper that returns the result, rather than having
    to pass in an array to fill in
    """
    roots = np.empty((3,), dtype=np.complex128)
    cubic_roots_c(p, roots)
    return roots

# fixme: uncomment these once all tests pass!
# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)
cdef int cubic_roots_c(FLOAT_TYPE[:] p,
                       COMPLEX_TYPE[:] x0  # output
                       ) except -1:
# cdef int cubic_roots_c(np.ndarray[FLOAT_TYPE, ndim=1] p,
#                        np.ndarray[COMPLEX_TYPE, ndim=1] x0  # output
#                        ) except -1:
    """
    ! Computes the roots of a cubic polynomial with coefficients p().
    !
    ! Computes the roots of an 3rd-order polynomial with real-valued
    ! coefficients specified in p().  The order of the coefficients in
    ! p() are given by
    !
    !     p(1) * x**3 + p(2) * x**2 + p(3) * x + p(4) = 0
    !
    ! Input variables are:
    !     p = array (order 4) of polynomial coefficients (must be real)
    !
    ! Output variable is:
    !     x0 = array (order 3) of roots (real or complex)
    !
    ! S. Socolofsky
    ! June 2013
    Ported to Cython: C. Barker May 2020
    !
    """
    # maybe use.a C macro for this?
    # cdef int N = 3

    # declared in function prototype
    # real(kind = DP), intent(in), dimension(N + 1) :: p
    # complex(kind = DP), intent(out), dimension(N) :: x0
    # for the results
    # cdef np.ndarray[COMPLEX_TYPE, ndim=1] x0 = np.zeros((3,), dtype=np.complex128)

#     ! Declare the variables internal to the function
#     integer :: k
    cdef unsigned int k
#     real(kind = DP) :: Delta, a, b, c, d, eps
    cdef FLOAT_TYPE Delta, a, b, c, d, s3
#     real(kind = DP), dimension(N) :: Dk
    cdef FLOAT_TYPE Dk[3]  # is there a way to make this dynamic sized?
#     complex(kind = DP), parameter :: I = cmplx(0.0, 1.0)
    cdef COMPLEX_TYPE I = 0.0 + 1.0j
#     complex(kind = DP) :: C0
    cdef COMPLEX_TYPE C0 = 0.0 + 1.0j
#     complex(kind = DP), dimension(N) :: u
    # cdef COMPLEX_TYPE u[3]
    cdef np.ndarray[COMPLEX_TYPE, ndim=1] u = np.zeros((3,), dtype=np.complex128)
#     ! Extract the coefficient values for easier reference and understanding
#     ! in the code
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]

#     ! Compute the ingredients to the solution
#     Delta = 18.0 * a * b * c * d - 4.0 * b**3 * d + b**2 * c**2 - &
#         &   4.0 * a * c**3 - 27.0*a**2*d**2
    Delta = (18.0 * a * b * c * d - 4.0 * b**3 * d + b**2 * c**2 -
             4.0 * a * c**3 - 27.0 * a**2 * d**2)

#     Dk = [b**2 - 3.0 * a * c, &
#         & 2.0 * b**3 - 9.0 * a * b * c + 27.0 * a**2 * d, &
#         & -27.0 * a**2 * Delta]
    Dk[0] = b**2 - 3.0 * a * c
    Dk[1] = 2.0 * b**3 - 9.0 * a * b * c + 27.0 * a**2 * d
    Dk[2] = -27.0 * a**2 * Delta


#     u = [1.0 + 0.0*I, (-1.0 + I * sqrt(3.0)) / 2.0, &
#         & (-1.0 - I * sqrt(3.0)) / 2.0]
    s3 = 1.7320508075688772  # s3 = sqrt(3.0)
    u[0] = 1.0 + 0.0 * I
    u[1] = (-1.0 + I * s3) / 2.0
    u[2] = (-1.0 - I * s3) / 2.0


#     C0 = ((Dk(2) + sqrt(Dk(3)+0.0*I)) / 2.0)**(1.0/3.0)
    C0 = ((Dk[1] + csqrt(Dk[2] + 0.0*I)) / 2.0)**(1.0 / 3.0)

#     ! Compute the solution
#     x0(:) = -1.0 / (3.0 * a) * (b + u(:) * C0 + Dk(1) / (u(:) * C0))
    for i in range(3):
        x0[i] = -1.0 / (3.0 * a) * (b + u[i] * C0 + Dk[0] / (u[i] * C0))

#     ! Get the machine precision
#     eps = epsilon(0.0)

#     ! Convert appropriately small numbers to zero
#   Fixme: EPS is only appropriate if we are expecting O(1.0) numbers -- are we?
    for k in range(3):
        if (abs(x0[k].real) < EPS):
            x0[k] = 0.0 + x0[k].imag
#         ! And imaginary part
        if (abs(x0[k].imag) < EPS):
            x0[k] = x0[k].real + 0.0*I
    return 0  # no Exception

