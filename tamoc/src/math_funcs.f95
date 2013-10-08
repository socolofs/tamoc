! File: MATH_FUNCS.f95
! 
! Provides root-finding capability in Fortran for cubic equations by 
! an analytical solution, based on equations at:
!     http://en.wikipedia.org/wiki/Cubic_function
! 
! S. Socolofsky
! June 2013
! Texas A&M University
! 

module Math_Constants
    
    ! Define constants for use by the contained math routines
    implicit none
    integer, parameter :: DP = 8
    
end module Math_Constants
    
subroutine cubic_roots(p, x0)
    
    ! 
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
    !
    
    use Math_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, parameter :: N = 3
    real(kind = DP), intent(in), dimension(N + 1) :: p
    complex(kind = DP), intent(out), dimension(N) :: x0
    
    ! Declare the variables internal to the function
    integer :: k
    real(kind = DP) :: Delta, a, b, c, d, eps
    real(kind = DP), dimension(N) :: Dk
    complex(kind = DP), parameter :: I = cmplx(0.0, 1.0)
    complex(kind = DP) :: C0
    complex(kind = DP), dimension(N) :: u
    
    ! Extract the coefficient values for easier reference and understanding
    ! in the code
    a = p(1)
    b = p(2)
    c = p(3)
    d = p(4)
    
    ! Compute the ingredients to the solution
    Delta = 18.0D0 * a * b * c * d - 4.0D0 * b**3 * d + b**2 * c**2 - & 
        &   4.0D0 * a * c**3 - 27.0D0*a**2*d**2
    Dk = [b**2 - 3.0D0 * a * c, &
        & 2.0D0 * b**3 - 9.0D0 * a * b * c + 27.0D0 * a**2 * d, &
        & -27.0D0 * a**2 * Delta]
    u = [1.0D0 + 0.0D0*I, (-1.0D0 + I * sqrt(3.0D0)) / 2.0D0, &
        & (-1.0D0 - I * sqrt(3.0D0)) / 2.0D0]
    C0 = ((Dk(2) + sqrt(Dk(3)+0.0D0*I)) / 2.0D0)**(1.0D0/3.0D0)
    
    ! Compute the solution
    x0(:) = -1.0D0 / (3.0D0 * a) * (b + u(:) * C0 + Dk(1) / (u(:) * C0))
    
    ! Get the machine precision
    eps = epsilon(0.0)
    
    ! Convert appropriately small numbers to zero
    do k = 1, N
        ! Real part first
        if (abs(real(x0(k))) < eps) then
            x0(k) = cmplx(0.0, aimag(x0(k)))
        end if
        ! And imaginary part
        if (abs(aimag(x0(k))) < eps) then
            x0(k) = cmplx(real(x0(k)), 0.0)
        end if
    end do
    
end subroutine cubic_roots

! End File: MATH_FUNCS.f95