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

ctypedef double FLOAT_TYPE
ctypedef double complex COMPLEX_TYPE

from libc.math cimport sqrt, exp, isnan

cdef extern from "<complex.h>" nogil:
    COMPLEX_TYPE csqrt(double complex z)
    COMPLEX_TYPE I

## Varios constants
# for epsilson
# There should be a way to get this from a C header, but this works
cdef FLOAT_TYPE eps = sys.float_info.epsilon

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

#    cdef FLOAT_TYPE[:] arr3 = np.zeros((n,), dtype=np.float64)
    cdef FLOAT_TYPE[:] arr3 = cvarray(shape=(n,), itemsize=sizeof(FLOAT_TYPE), format="d")
    for i in range(n):
        arr3[i] = arr1[i]

    result = 0.0
    for i in range(n):
        result += arr3[i]

    return result

cdef mole_fraction(nc,
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

    cdef FLOAT_TYPE[:] n_moles = np.zeros((nc,), dtype=np.float64)

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
    mole_fraction(nc, mass, Mol_wt, yk)

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


cdef z_pr(unsigned int nc,
               FLOAT_TYPE T,
               FLOAT_TYPE P,
               np.ndarray[FLOAT_TYPE, ndim=1] mass,
               np.ndarray[FLOAT_TYPE, ndim=1] Mol_wt,
               np.ndarray[FLOAT_TYPE, ndim=1] Pc,
               np.ndarray[FLOAT_TYPE, ndim=1] Tc,
               np.ndarray[FLOAT_TYPE, ndim=1] omega,
               np.ndarray[FLOAT_TYPE, ndim=2] delta,  # (nc, nc)
               np.ndarray[FLOAT_TYPE, ndim=2] Aij,  # (15, 15)
               np.ndarray[FLOAT_TYPE, ndim=2] Bij,  # (15, 15)
               np.ndarray[FLOAT_TYPE, ndim=2] delta_groups,  # (nc,15)
               int calc_delta,
               np.ndarray[FLOAT_TYPE, ndim=2] z,  # rest used as output
               FLOAT_TYPE *A,
               FLOAT_TYPE *B,
               np.ndarray[FLOAT_TYPE, ndim=1] Ap,
               np.ndarray[FLOAT_TYPE, ndim=1] Bp,
               np.ndarray[FLOAT_TYPE, ndim=1] yk
               ):

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
    cdef FLOAT_TYPE[4] p_coefs
#     complex(kind = DP), dimension(3) :: z_roots
    cdef COMPLEX_TYPE[3] z_roots

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

#     ! Find the roots of the cubic equation of state
    # call cubic_roots(p_coefs, z_roots)
    z_roots = cubic_roots(p_coefs)

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
    z[1, 1] = z_max
    z[2, 1] = z_min


#  output is a (2, 1) array -- how is this different than a 2, array?
cpdef np.ndarray[FLOAT_TYPE, ndim=2] density(FLOAT_TYPE T,
                                             FLOAT_TYPE P,
                                             np.ndarray[FLOAT_TYPE, ndim=1] mass,
                                             np.ndarray[FLOAT_TYPE, ndim=1] Mol_wt,
                                             np.ndarray[FLOAT_TYPE, ndim=1] Pc,
                                             np.ndarray[FLOAT_TYPE, ndim=1] Tc,
                                             np.ndarray[FLOAT_TYPE, ndim=1] Vc,
                                             np.ndarray[FLOAT_TYPE, ndim=1] omega,
                                             np.ndarray[FLOAT_TYPE, ndim=2] delta,  # (nc, nc)
                                             np.ndarray[FLOAT_TYPE, ndim=2] Aij,  # (15, 15)
                                             np.ndarray[FLOAT_TYPE, ndim=2] Bij,  # (15, 15)
                                             np.ndarray[FLOAT_TYPE, ndim=2] delta_groups, # (nc,15)
                                             int calc_delta):
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

    cdef unsigned int nc  # the number of components -- should match array sizes
    nc = mass.shape[0]
    ## fixme: check other arrays for the right size?
    cdef np.ndarray[FLOAT_TYPE, ndim=2] rho = np.zeros((2, 1), dtype=np.float64)  # should be (2,1) in size

#     ! Declare the variables internal to the function
#     real(kind = DP) :: A, B, R
    cdef FLOAT_TYPE A, B, R
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
         z, &A, &B, Ap, Bp, yk)

#     ! Convert the masses to mole fraction
    mole_fraction(nc, mass, Mol_wt, yk)

#     ! Compute the volume translation coefficient
#     call volume_trans(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, vt)

#     ! Compute the molar volume
#     nu = z * Ru * T / P - sum(yk(:) * vt(:))

#     ! Compute and return the density
#     rho = 1.0D0 / nu * sum(yk(:) * Mol_wt(:))

    return rho
# end subroutine density



# def identity(np.ndarray [np.complex128_t, ndim=1] weights):
#     return weights


#cpdef cubic_roots(np.ndarray[FLOAT_TYPE, ndim=1] p,
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef np.ndarray[COMPLEX_TYPE, ndim=1] cubic_roots(np.ndarray p,
                  # x0 is output
                  # np.ndarray[FLOAT_TYPE, ndim=1] x0
                  ):

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
    cdef np.ndarray[COMPLEX_TYPE, ndim=1] x0 = np.zeros((3,), dtype=np.complex128)

#     ! Declare the variables internal to the function
#     integer :: k
    cdef unsigned int k
#     real(kind = DP) :: Delta, a, b, c, d, eps
    cdef FLOAT_TYPE Delta, a, b, c, d, s3
#     real(kind = DP), dimension(N) :: Dk
    cdef FLOAT_TYPE[3] Dk  # is there a way to make this dynamic sized?
#     complex(kind = DP), parameter :: I = cmplx(0.0, 1.0)
    cdef COMPLEX_TYPE I = 0.0+1.0j
#     complex(kind = DP) :: C0
    cdef COMPLEX_TYPE C0 = 0.0+1.0j
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
    x0[:] = -1.0 / (3.0 * a) * (b + u[:] * C0 + Dk[0] / (u[:] * C0))

#     ! Get the machine precision
#     eps = epsilon(0.0)

#     ! Convert appropriately small numbers to zero
    for k in range(3):
        if (abs(x0[k].real) < eps):
            x0[k] = 0.0 + x0[k].imag
#         ! And imaginary part
        if (abs(x0[k].imag) < eps):
            x0[k] = x0[k].real + 0.0*I
    return x0



