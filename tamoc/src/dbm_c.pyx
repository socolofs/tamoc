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

from libc.math cimport sqrt

cdef extern from "<complex.h>" nogil:
    COMPLEX_TYPE csqrt(double complex z)
    COMPLEX_TYPE I

# for epsilson
# There should be a way to get this from a C header, but this works
cdef FLOAT_TYPE eps = sys.float_info.epsilon



# subroutine cubic_roots(p, x0)

#     !
#     ! Computes the roots of a cubic polynomial with coefficients p().
#     !
#     ! Computes the roots of an 3rd-order polynomial with real-valued
#     ! coefficients specified in p().  The order of the coefficients in
#     ! p() are given by
#     !
#     !     p(1) * x**3 + p(2) * x**2 + p(3) * x + p(4) = 0
#     !
#     ! Input variables are:
#     !     p = array (order 4) of polynomial coefficients (must be real)
#     !
#     ! Output variable is:
#     !     x0 = array (order 3) of roots (real or complex)
#     !
#     ! S. Socolofsky
#     ! June 2013
#     !

#     use Math_Constants
#     implicit none

#     ! Declare the input and output variable types
#     integer, parameter :: N = 3
#     real(kind = DP), intent(in), dimension(N + 1) :: p
#     complex(kind = DP), intent(out), dimension(N) :: x0

#     ! Declare the variables internal to the function
#     integer :: k
#     real(kind = DP) :: Delta, a, b, c, d, eps
#     real(kind = DP), dimension(N) :: Dk
#     complex(kind = DP), parameter :: I = cmplx(0.0, 1.0)
#     complex(kind = DP) :: C0
#     complex(kind = DP), dimension(N) :: u

#     ! Extract the coefficient values for easier reference and understanding
#     ! in the code
#     a = p(1)
#     b = p(2)
#     c = p(3)
#     d = p(4)

#     ! Compute the ingredients to the solution
#     Delta = 18.0D0 * a * b * c * d - 4.0D0 * b**3 * d + b**2 * c**2 - &
#         &   4.0D0 * a * c**3 - 27.0D0*a**2*d**2
#     Dk = [b**2 - 3.0D0 * a * c, &
#         & 2.0D0 * b**3 - 9.0D0 * a * b * c + 27.0D0 * a**2 * d, &
#         & -27.0D0 * a**2 * Delta]
#     u = [1.0D0 + 0.0D0*I, (-1.0D0 + I * sqrt(3.0D0)) / 2.0D0, &
#         & (-1.0D0 - I * sqrt(3.0D0)) / 2.0D0]
#     C0 = ((Dk(2) + sqrt(Dk(3)+0.0D0*I)) / 2.0D0)**(1.0D0/3.0D0)

#     ! Compute the solution
#     x0(:) = -1.0D0 / (3.0D0 * a) * (b + u(:) * C0 + Dk(1) / (u(:) * C0))

#     ! Get the machine precision
#     eps = epsilon(0.0)

#     ! Convert appropriately small numbers to zero
#     do k = 1, N
#         ! Real part first
#         if (abs(real(x0(k))) < eps) then
#             x0(k) = cmplx(0.0, aimag(x0(k)))
#         end if
#         ! And imaginary part
#         if (abs(aimag(x0(k))) < eps) then
#             x0(k) = cmplx(real(x0(k)), 0.0)
#         end if
#     end do

# end subroutine cubic_roots

def identity(np.ndarray [np.complex128_t, ndim=1] weights):
    return weights


#cpdef cubic_roots(np.ndarray[FLOAT_TYPE, ndim=1] p,
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef cubic_roots(np.ndarray p,
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
#     Delta = 18.0D0 * a * b * c * d - 4.0D0 * b**3 * d + b**2 * c**2 - &
#         &   4.0D0 * a * c**3 - 27.0D0*a**2*d**2
    Delta = (18.0 * a * b * c * d - 4.0 * b**3 * d + b**2 * c**2 -
             4.0 * a * c**3 - 27.0 * a**2 * d**2)

#     Dk = [b**2 - 3.0D0 * a * c, &
#         & 2.0D0 * b**3 - 9.0D0 * a * b * c + 27.0D0 * a**2 * d, &
#         & -27.0D0 * a**2 * Delta]
    Dk[0] = b**2 - 3.0 * a * c
    Dk[1] = 2.0 * b**3 - 9.0 * a * b * c + 27.0 * a**2 * d
    Dk[2] = -27.0 * a**2 * Delta


#     u = [1.0D0 + 0.0D0*I, (-1.0D0 + I * sqrt(3.0D0)) / 2.0D0, &
#         & (-1.0D0 - I * sqrt(3.0D0)) / 2.0D0]
    s3 = 1.7320508075688772  # s3 = sqrt(3.0)
    u[0] = 1.0 + 0.0 * I
    u[1] = (-1.0 + I * s3) / 2.0
    u[2] = (-1.0 - I * s3) / 2.0


#     C0 = ((Dk(2) + sqrt(Dk(3)+0.0D0*I)) / 2.0D0)**(1.0D0/3.0D0)
    C0 = ((Dk[1] + csqrt(Dk[2] + 0.0*I)) / 2.0)**(1.0 / 3.0)

#     ! Compute the solution
#     x0(:) = -1.0D0 / (3.0D0 * a) * (b + u(:) * C0 + Dk(1) / (u(:) * C0))
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



