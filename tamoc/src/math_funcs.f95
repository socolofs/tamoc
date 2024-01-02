! File: MATH_FUNCS.f95
! 
! Provides root-finding capability in Fortran for cubic equations by analytical
! solutions.  The original root-finder in TAMOC was based on equations at:
!		 http://en.wikipedia.org/wiki/Cubic_function
! 
! This new math_funcs file uses subroutines from the Public Domain Aeronautical
! Software (PDAS) site.	The Quartic polynomial program may be downloaded from:
!		 https://www.pdas.com/quarticdownload.html
!
! This program makes use of the CubicRoots function from quartic.f90. These
! tools were developed by Alfred H. Morris, William L. Davis, Alan Miller, and
! Ralph L. Carmichael.	The Fortran source code is public domain and 
! distributed without a license.
! 
! The original CubicRoots subroutine has been modified as little as possible to
! cast the code in the format and style of the other Fortran routines in TAMOC
! and to compile without any further modification of the TAMOC setup.py build
! process.  For example, f2py supports subroutines, but not functions.
!
! This new file was a collaborative effort of Scott A. Socolofsky and J. 
! Samuel Arey.  It is subject to the same license as all other TAMOC files.
!
! S. Socolofsky, Texas A&M University
! J. Samuel Arey, Oleolytics, LLC
! June 2013 and modified October 2023
! 

module Math_Constants
    
    ! Define constants for use by the contained math routines
    implicit none
    integer, parameter :: DP = 8
	complex(kind = DP), parameter :: CZERO = (0.D0, 0.D0)
	real(kind = DP), parameter :: ZERO = 0.D0, FOURTH = 0.25D0
	real(kind = DP), parameter :: HALF = 0.5D0
	real(kind = DP), parameter :: ONE = 1.D0, TWO = 2.D0
	real(kind = DP), parameter :: THREE = 3.D0, FOUR = 4.D0
    real(kind = DP), parameter :: EPS = epsilon(ONE)
	
end module Math_Constants


subroutine swap(a, b)
	
	! 
	! Swap the contents of a and b
	! 
	! Swap the value stored in a with the value stored in b and vice versa
	!  
	! Input variables are:
	!     a = float value
	!     b = float value
	!
	! Output variables are:
	!     a = float value containing the original value stored in b
	!     b = float value containing the original value stored in a
	! 
	! This function had a generic interface to a single- and double-precision
	! version of the swap subroutine in the original files from PDAS.  Here, 
	! we only use the double-precision version
	!
	! S. Socolofsky
	! October 2023
	! 
	
	use Math_Constants
	implicit none
	
	! Declare the input and output variable types
	real(kind = DP), intent(in out) :: a, b
	
	! Declare the variables internal to the function
	real(kind = DP) :: t
	
	! Swap the values
	t = b
	b = a
	a = t
	return

end subroutine swap

subroutine cube_root(x, f)
	
	! 
	! Computes the cube root of a real number. 
	! 
	! Computes the cube root of a real number and preserves the sign so that
	! if the argument is negative, the cube root is also negative.
	! 
	! Input variable is:
	!     x = real number we want the cube root of
	! 
	! Output variable is:
	!     f = cube root of the input value x
	!
	! Adapted from the PDAS quartic.f90 module
	! 
	! S. Socolofsky
	! October 2023
	!
	
	use Math_Constants
	implicit none
	
	! Declare the input variable type
	real(kind = DP), intent(in) :: x
	real(kind = DP), intent(out) :: f
	
    IF (x < ZERO) THEN
      	f = -EXP(LOG(-x) / THREE)
    ELSE IF (x > ZERO) THEN
      	f = EXP(LOG(x) / THREE)
    ELSE
      	f = ZERO
    END IF
	
    RETURN

END subroutine cube_root
	

subroutine quadratic_roots(a, z)
	
	! 
	! Computes the roots of a quadratic polynomial with coefficients a().
	! 
	! Computes the roots of the quadratic equation:
	! 	  a(1) + a(2) * z + a(3) * z**3 = 0
	! 
	! This subroutine is used by cubic_roots() when one of the roots is 
	! zero (as when the constant coefficient a(1) of the cubic equation 
	! is zero).  Like the cubic_roots() subroutine, this subroutine is 
	! from he quartic.f90 module provided by the Public Domain
	! Aeronautical Software (PDAS) project, downloaded from:
	! 	  https://www.pdas.com/quarticdownload.html
	!
	! Minimal changes have been made to this subroutine to match the formatting
	! of other Fortran codes in TAMOC and to allow the subroutine to be used
	! in this stand-alone version.
    ! 
    ! S. Socolofsky
    ! October 2023
    !
	
	use Math_Constants
	implicit none
	
	! Declare the input and output variable types
	integer, parameter :: N = 2
	real(kind = DP), intent(in), dimension(N + 1) :: a
	complex(kind = DP), intent(out), dimension(N) :: z

	! Declare the variables internal to the function
	real(kind =DP) :: d, r, w, x, y
	
	!--------------------------------------------------------------------------
    IF (a(1) == 0.0) THEN
		! one root is obviously zero
      	z(1) = CZERO               
		! remainder is a linear eq.
      	z(2) = CMPLX(-a(2)/a(3), ZERO, kind = DP)    
      	RETURN
    END IF

	! the discriminant
    d = a(2) * a(2) - FOUR * a(1) * a(3)             
    IF (ABS(d) <= TWO * EPS * a(2) * a(2)) THEN
		! discriminant is tiny
      	z(1) = CMPLX(-HALF * a(2) / a(3), ZERO, kind = DP) 
      	z(2) = z(1)
      	RETURN
    END IF

    r = SQRT(ABS(d))
    IF (d < ZERO) THEN
		! negative discriminant => roots are complex   
      	x = -HALF * a(2) / a(3)        
      	y = ABS(HALF * r / a(3))
      	z(1) = CMPLX(x, y, kind = DP)
		! its conjugate
      	z(2) = CMPLX(x, -y, kind = DP)   
    END IF

    IF (a(2) /= ZERO) THEN
		! see Numerical Recipes, sec. 5.5
      	w = -(a(2) + SIGN(r,a(2)))
      	z(1) = CMPLX(TWO * a(1) / w,  ZERO, kind = DP)
      	z(2) = CMPLX(HALF * w / a(3), ZERO, kind = DP)
      	RETURN
    END IF

	! a(2) = 0 if you get here
    x = ABS(HALF * r / a(3))   
    z(1) = CMPLX( x, ZERO, kind = DP)
    z(2) = CMPLX(-x, ZERO, kind = DP)

    RETURN

end subroutine quadratic_roots   ! -------------------------------------------
	
subroutine cubic_roots(a_t, z)
	
    ! 
    ! Computes the roots of a cubic polynomial with coefficients a().
    ! 
    ! Computes the roots of a 3rd-order polynomial with real-valued 
    ! coefficients specified in p().  The order of the coefficients in 
    ! a() are given by
    ! 
    !     a_t(1) * x**3 + a_t(2) * x**2 + a_t(3) * x + a_t(4) = 0
    ! 
    ! Input variable is:
    !     a_t = array (order 4) of polynomial coefficients (must be real)
    ! 
    ! Output variable is:
    !     z = array (order 3) of roots (real or complex)
	!
	! The original TAMOC cubic_roots() subroutine was replaced with this one
	! in October 2023.  The original solver had difficulty computing roots for
	! some single-phase compositions where there is one real root.  This
	! function is from the quartic.f90 module provided by the Public Domain
	! Aeronautical Software (PDAS) project, downloaded from:
	! 	  https://www.pdas.com/quarticdownload.html
	! 
	! Minimal changes have been made to this subroutine to match the formatting
	! of other Fortran codes in TAMOC and to allow the subroutine to be used
	! in this stand-alone version.
    ! 
    ! S. Socolofsky
    ! June 2013 -- Updated October 2023
    !
	
	use Math_Constants
	implicit none
	
	! Declare the input and output variable types
	integer, parameter :: N = 3
	real(kind = DP), intent(in), dimension(N + 1) :: a_t
	complex(kind = DP), intent(out), dimension(N) :: z
	
	! Declare the variables internal to the function
	real(kind = DP), dimension(N + 1) :: a
	real(kind = DP), parameter :: RT3 = 1.7320508075689D0   ! (Sqrt(3)
    real(kind = DP) :: aq(3), arg, c, cf, d, p, p1, q, q1
    real(kind = DP) :: r, ra, rb, rq, rt
    real(kind = DP) :: r1, s, sf, sq, sum, t, tol, t1, w
    real(kind = DP) :: w1, w2, x, x1, x2, x3, y, y1, y2, y3
	
	! NOTE -   It is assumed that a(4) is non-zero. No test is made here.
	!--------------------------------------------------------------------------
 
 	! TAMOC sends the coefficients in the reverse order to those used in 
	! this subroutine
	a(4) = a_t(1)
	a(3) = a_t(2)
	a(2) = a_t(3)
	a(1) = a_t(4)
	
	! check constant coefficient
    IF (a(1)==0.0) THEN
		! one root is obviously zero
		z(1) = CZERO  
		! remaining 2 roots here
        CALL quadratic_roots(a(2:4), z(2:3))   
        RETURN
    END IF

	! Set up some constant parameters
    p = a(3) / (THREE*a(4))
    q = a(2) / a(4)
    r = a(1) / a(4)
    tol = FOUR * EPS

    c = ZERO
    t = a(2) - p * a(3)
    IF (ABS(t) > tol * ABS(a(2))) c = t/a(4)

    t = TWO * p * p - q
    IF (ABS(t) <= tol * ABS(q)) t = ZERO
    d = r + p * t
	
	! Source of jump to line 110...
    IF (ABS(d) <= tol*ABS(r)) THEN
		GO TO 110
	END IF

    ! SET  SQ = (A(4)/S)**2 * (C**3/27 + D**2/4)
    s = MAX(ABS(a(1)), ABS(a(2)), ABS(a(3)))
    p1 = a(3) / (THREE * s)
    q1 = a(2) / s
    r1 = a(1) / s

    t1 = q - 2.25D0 * p * p
    IF (ABS(t1) <= tol * ABS(q)) t1 = ZERO
    w = FOURTH * r1 * r1
    w1 = HALF * p1 * r1 * t
    w2 = q1 * q1 * t1 / 27.0D0

    IF (w1 >= ZERO) THEN
    	w = w + w1
     	sq = w + w2
    ELSE IF (w2 < ZERO) THEN
      	sq = w + (w1 + w2)
    ELSE
      	w = w + w2
      	sq = w + w1
    END IF

    IF (ABS(sq) <= tol * w) sq = ZERO
    rq = ABS(s / a(4)) * SQRT(ABS(sq))
    
	! Source of jump to line 40...
	IF (sq >= ZERO) GO TO 40

  	! If code reaches this point, ALL ROOTS ARE REAL
    arg = ATAN2(rq, -HALF * d)
    cf = COS(arg / THREE)
    sf = SIN(arg / THREE)
    rt = SQRT(-c / THREE)
    y1 = TWO * rt * cf
    y2 = -rt * (cf + rt3 * sf)
    y3 = -(d / y1) / y2

    x1 = y1 - p
    x2 = y2 - p
    x3 = y3 - p

    IF (ABS(x1) > ABS(x2)) CALL swap(x1,x2)
    IF (ABS(x2) > ABS(x3)) CALL swap(x2,x3)
    IF (ABS(x1) > ABS(x2)) CALL swap(x1,x2)

    w = x3

	! Source of jump to line 70
    IF (ABS(x2) < 0.1D0 * ABS(x3)) GO TO 70
    IF (ABS(x1) < 0.1D0 * ABS(x2)) x1 = - (r / x3) / x2
    z(1) = CMPLX(x1, ZERO, kind = DP)
    z(2) = CMPLX(x2, ZERO, kind = DP)
    z(3) = CMPLX(x3, ZERO, kind = DP)
    RETURN

  	! REAL AND COMPLEX ROOTS
 40 call cube_root(-HALF * d - SIGN(rq,d), ra)
    rb = -c / (THREE * ra)
    t = ra + rb
    w = -p
    x = -p
	
	! Source of jump to line 41
    IF (ABS(t) <= tol * ABS(ra)) GO TO 41
    w = t - p
    x = -HALF * t - p
    IF (ABS(x) <= tol*ABS(p)) x = ZERO
	
 41 t = ABS(ra - rb)
    y = HALF * rt3 * t
  
  	! Source of jump to line 60
    IF (t <= tol * ABS(ra)) GO TO 60
	
	! Source of jump to line 50
    IF (ABS(x) < ABS(y)) GO TO 50

	s = ABS(x)
    t = y / x
	! Source of jump to line 51
    GO TO 51

 50 s = ABS(y)
    t = x/y
	
	! Source of jump to line 70
 51 IF (s < 0.1D0 * ABS(w)) GO TO 70
    w1 = w / s
    sum = ONE + t * t
    IF (w1 * w1 < 0.01D0 * sum) w = - ((r / sum) / s) / s
    z(1) = CMPLX(w, ZERO, kind = DP)
    z(2) = CMPLX(x, y, kind = DP)
    z(3) = CMPLX(x, -y, kind = DP)
    RETURN

  	! AT LEAST TWO ROOTS ARE EQUAL
	
	! Source of jump to line 61
 60 IF (ABS(x) < ABS(w)) GO TO 61
    IF (ABS(w) < 0.1D0 * ABS(x)) w = - ( r / x) / x
    z(1) = CMPLX(w, ZERO, kind = DP)
    z(2) = CMPLX(x, ZERO, kind = DP)
    z(3) = z(2)
    RETURN
	
	! Source of jump to line 70
 61 IF (ABS(x) < 0.1D0 * ABS(w)) GO TO 70
    z(1) = CMPLX(x, ZERO, kind = DP)
    z(2) = z(1)
    z(3) = CMPLX(w, ZERO, kind = DP)
    RETURN

  	! HERE W IS MUCH LARGER IN MAGNITUDE THAN THE OTHER ROOTS.
  	! AS A RESULT, THE OTHER ROOTS MAY BE EXCEEDINGLY INACCURATE
  	! BECAUSE OF ROUNDOFF ERROR.  TO DEAL WITH THIS, A QUADRATIC
  	! IS FORMED WHOSE ROOTS ARE THE SAME AS THE SMALLER ROOTS OF
  	! THE CUBIC.  THIS QUADRATIC IS THEN SOLVED.

  	! THIS CODE WAS WRITTEN BY WILLIAM L. DAVIS (NSWC).

 70 aq(1) = a(1)
    aq(2) = a(2) + a(1) / w
    aq(3) = -a(4) * w
    CALL quadratic_roots(aq, z)
    z(3) = CMPLX(w, ZERO, kind = DP)
  
    IF (AIMAG(z(1)) == ZERO) RETURN
    z(3) = z(2)
    z(2) = z(1)
    z(1) = CMPLX(w, ZERO, kind = DP)
    RETURN
  
  	! CASE WHEN D = 0

 110 z(1) = CMPLX(-p, ZERO, kind = DP)
    w = SQRT(ABS(c))
	
	! Source of jump to 120
    IF (c < ZERO) GO TO 120
	
    z(2) = CMPLX(-p, w, kind = DP)
    z(3) = CMPLX(-p,-w, kind = DP)
    RETURN

	! Source of jump to 130
 120 IF (p /= ZERO) GO TO 130
    
	z(2) = CMPLX(w, ZERO, kind = DP)
    z(3) = CMPLX(-w, ZERO, kind = DP)
    RETURN

 130 x = -(p + SIGN(w, p))
 	z(3) = CMPLX(x, ZERO, kind = DP)
    t = THREE * a(1) / (a(3) * x)
	
	! Source of jump to 131
    IF (ABS(p) > ABS(t)) GO TO 131
	
    z(2) = CMPLX(t, ZERO, kind = DP)
    RETURN
	
 131 z(2) = z(1)
    z(1) = CMPLX(t, ZERO, kind = DP)
    RETURN

end subroutine cubic_roots

! End File: MATH_FUNCS.f95