! File: DBM_PHYS.f95
! 
! Discrete Bubble Model Physics Equations from Clift et al. (1978)
! 
! These subroutines compute the governing non-dimensional parameters, slip 
! velocity equations, and equations for mass transfer coefficient based on 
! models described in Clift et al. (1978).
! 
! REQUIRES:
!     None.
! 
! Complies in Mac OSX using:
!     f2py -c -m dbm dbm.f95
!
! S. Socolofsky
! June 2013
! Texas A&M University

module Phys_Constants
    
    ! Define constants for use by the Discrete Bubble Model 
    implicit none
    integer, parameter :: DP = 8
    real(kind = DP), parameter :: G = 9.81D0, PI = 3.141592653589793D0
    
end module Phys_Constants


! ----------------------------------------------------------------------------
! Governing Non-dimensional Variables
! ----------------------------------------------------------------------------

subroutine eotvos(de, rho_p, rho, sigma, &
    &             Eo)
    
    ! 
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, rho_p, rho, sigma
    real(kind = DP), intent(out) :: Eo
    
    Eo = G * (rho - rho_p) * de**2 / sigma
    
end subroutine eotvos


subroutine morton(rho_p, rho, mu, sigma, &
    &             M)
    
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: rho_p, rho, mu, sigma
    real(kind = DP), intent(out) :: M
    
    M = G * mu**4 * (rho - rho_p) / (rho**2 * sigma**3)
    
end subroutine morton


subroutine reynolds(de, us, rho, mu, &
    &               Re)
    
    ! 
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, us, rho, mu
    real(kind = DP), intent(out) :: Re
    
    Re = rho * de * us / mu
    
end subroutine reynolds

subroutine h_parameter(Eo, M, mu, &
    &                  H)
    
    ! 
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: Eo, M, mu
    real(kind = DP), intent(out) :: H
    
    H = 4.0D0 / 3.0D0 * Eo * M**(-0.149D0) * (mu / 0.0009D0)**(-0.14D0)
    
end subroutine h_parameter


! ----------------------------------------------------------------------------
! Slip velocity and shape functions
! ----------------------------------------------------------------------------

subroutine particle_shape(de, rho_p, rho, mu, sigma, &
    &                     shape_p)
    
    ! 
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, rho_p, rho, mu, sigma
    integer, intent(out) :: shape_p
        
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: Eo, M, H
    
    ! Calculate the non-dimensional variables
    call eotvos(de, rho_p, rho, sigma, Eo)
    call morton(rho_p, rho, mu, sigma, M)
    call h_parameter(Eo, M, mu, H)
    
    ! Select the appropriate shape classification
    if (H < 2.0D0) then
        shape_p = 1
    else if (Eo < 40.0D0 .and. M < 0.001D0 .and. H < 1000.0D0) then
        shape_p = 2
    else
        shape_p = 3
    end if
    
end subroutine particle_shape


subroutine theta_w_sc(de, us, rho, mu, &
    &                 theta_w)
    
    ! 
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
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, us, rho, mu
    real(kind = DP), intent(out) :: theta_w
        
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: Re
    
    ! Get the Reynolds number
    call reynolds(de, us, rho, mu, Re)
    
    ! Return the wake angle
    theta_w = PI * (50.0D0 + 190.0D0 * exp(-0.62D0 * Re**(0.4D0))) / 180.0D0
    
end subroutine theta_w_sc


subroutine surface_area_sc(de, theta_w, &
    &                      area)
    
    ! 
    ! Compute the surface area for a spherical cap fluid particle
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     theta_w = wake angle for the partial sphere model (rad)
    ! 
    ! Returns the surface area (m^2) of a spherical cap fluid particle per the 
    ! model sketched in figure 8.1 of Clift et al. (1978) p. 204.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, theta_w
    real(kind = DP), intent(out) :: area
    
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: V, r_sc, Af, Ar
    
    ! Match the volume
    V = 4.0D0 / 3.0D0 * PI * (de / 2.0D0)**3
    
    ! Find the radius at the bottom of the partial sphere
    r_sc = (V / PI / (2.0D0 / 3.0D0 - cos(theta_W) + cos(theta_W)**3 / &
         & 3.0D0))**(1.0D0 / 3.0D0)
    
    ! Surface area of the frontal sphere
    Af = 2.0D0 * PI * r_sc**2 * (1.0D0 - cos(theta_W))
    
    ! Surface area of the real bottom of the partial sphere
    Ar = PI * (r_sc * sin(theta_W))**2
    
    ! Return the surface area
    area = Af + Ar
    
end subroutine surface_area_sc


subroutine surface_area_sphere(de, &
    &                          area)
    
    ! 
    ! Compute the surface area for a spherical cap fluid particle
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     theta_w = wake angle for the partial sphere model (rad)
    ! 
    ! Returns the surface area (m^2) of a spherical cap fluid particle per the 
    ! model sketched in figure 8.1 of Clift et al. (1978) p. 204.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de
    real(kind = DP), intent(out) :: area
    
    ! Return the surface area
    area = PI * de**2
    
end subroutine surface_area_sphere


subroutine us_sphere(de, rho_p, rho, mu, &
    &                u_slip)
    
    ! 
    ! Compute the slip velocity of a rigid sphere
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    ! 
    ! Returns the slip velocity (m/s) of a rigid spherical particle per 
    ! equation (5-15) and following in Clift et al. (1978) p. 133ff.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, rho_p, rho, mu
    real(kind = DP), intent(out) :: u_slip
        
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: Nd, W, Re
    
    ! Compute the non-dimensional independent parameters
    Nd = 4.0D0 * rho * abs(rho - rho_p) * G * de**3 / (3.0D0 * mu**2)
    W = log10(Nd)
    
    ! Compute the Reynolds number from the correlations
    if (Nd <= 73.0D0) then
        Re = (Nd / 24.0D0 - 1.7569D-4 * Nd**2 + 6.9252D-7 * Nd**3 &
           &  - 2.3027D-10 * Nd**4)
    else if (Nd <= 580.0D0) then
        Re = 10.0D0**(-1.7095D0 + 1.33438D0 * W - 0.11591D0 * W**2)
    else if (Nd <= 1.55D7) then
        Re = 10.0D0**(-1.81391D0 + 1.34671D0 * W - 0.12427D0 * W**2 + &
           & 0.006344D0 * W**3)
    else if (Nd <= 5.0D10) then
        Re = 10.0D0**(5.33283D0 - 1.21728D0 * W + 0.19007D0 * W**2 - &
           & 0.007005D0 * W**3)
    else
        print *, "US_SPHERE: Outside range of Nd -- RE not assigned"
    end if 
    
    ! Return the slip velocity
    u_slip = mu / (rho * de) * Re
    
end subroutine us_sphere


subroutine us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status, &
    &                   u_slip)
    
    ! 
    ! Compute the slip velocity of an elliptical-wobbling fluid particle
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    !     mu_p = dynamic viscosity of the dispersed phase (Pa s)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     sigma = interfacial tension (N/m)
    !     status = flag indicating whether the interface is clean (status = 1)
    !              or dirty (status = -1)
    ! 
    ! Returns the slip velocity (m/s) of an elliptical-wobbling fluid 
    ! particle per equation (7-4) and following in Clift et al. (1978) 
    ! p. 175ff.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: status
    real(kind = DP), intent(in) :: de, rho_p, rho, mu_p, mu, sigma
    real(kind = DP), intent(out) :: u_slip
    
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: Eo, M, H, J, Re, us_dirty, kappa, xi, gamma
    
    ! Calculate the non-dimensional variables
    call eotvos(de, rho_p, rho, sigma, Eo)
    call morton(rho_p, rho, mu, sigma, M)
    call h_parameter(Eo, M, mu, H)
    
    ! Compute the correlation equations
    if (H > 59.3D0) then
        J = 3.42D0 * H**(0.441D0)
    else
        J = 0.94D0 * H**(0.757D0)
    end if
    
    ! Calculate the Reynolds number
    Re = M**(-0.149D0) * (J - 0.857D0)
    
    ! Compute the dirty-bubble the slip velocity
    us_dirty =  mu / (rho * de) * Re
    
    ! Return the correct slip velocity
    if (status > 0) then
        ! Compute the clean-bubble correction from Figure 7.7 and Eqn. 7-10 in 
        ! Clift et al. (1978)
        kappa = mu_p / mu
        xi = Eo * (1.0D0 + 0.15D0 * kappa) / (1.0D0 + kappa)
        gamma = 2.0D0 * exp( -(log10(Xi) + 0.6383D0)**2 / &
              & (0.2598D0 + 0.2D0*(log10(Xi) + 1.0D0))**2 )
        u_slip = us_dirty * (1.0D0 + gamma/(1.0D0 + kappa))
    else
        u_slip = us_dirty
    end if
    
end subroutine us_ellipsoid


subroutine us_spherical_cap(de, rho_p, rho, &
    &                       u_slip)
    
    ! 
    ! Compute the slip velocity of a spherical cap fluid particle
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     rho_p = dispersed phase density (kg/m^3)
    !     rho = continuous phase density (kg/m^3)
    ! 
    ! Returns the slip velocity (m/s) of a spherical cap fluid particle using
    ! equation (8-11) in Clift et al. (1978) p. 206.  This is the equation 
    ! also suggested by Zheng and Yapa (2000).  This is strictly valid for
    ! Re > 150 and Eo > 40, though these limits are not tested here.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    real(kind = DP), intent(in) :: de, rho_p, rho
    real(kind = DP), intent(out) :: u_slip
    
    u_slip = 0.711D0 * sqrt(G * de * (rho - rho_p) / rho)
    
end subroutine  us_spherical_cap


! ----------------------------------------------------------------------------
! Mass Transfer Coefficients
! ----------------------------------------------------------------------------

subroutine xfer_johnson(de, us, D, nc, &
    &                   beta)
    
    ! Compute the mass transfer coefficient for clean particles
    ! 
    ! Computes the mass transfer coefficient for clean particles given by 
    ! equation (42) in Johnson et al. (1969), Canadian Journal of Chemical
    ! Engineering, vol. 47, pp. 559-564.
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     D = diffusion coefficients of the dispersed phase components in the
    !         continuous phase fluid (m^2/s)
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: de, us
    real(kind = DP), intent(in), dimension(nc) :: D
    real(kind = DP), intent(out), dimension(nc) :: beta
    
    ! Compute equation (42) in Johnson et al. (1969)
    beta(:) = 1.13D0 * sqrt(D(:) * us * 100.0D0**3 / (0.45D0 + &
            & 0.2D0 * de * 100.0D0)) / 100.0D0
    
end subroutine xfer_johnson


subroutine xfer_sphere(de, us, rho, mu, D, nc, &
    &                  beta)
    
    ! 
    ! Compute the mass transfer coefficients for a rigid sphere
    ! 
    ! Computes the mass transfer coefficients for a rigid sphere in water from
    ! equations in Clift et al. (1978).  For Re < 100, equation (5-25) on 
    ! page 121 is used; for Re < 1e5, equations in Table 5.4 on page 123 are 
    ! used.  All of these equations assume high Schmidt number.
    ! 
    ! These equations may be used for fluid particles in contaminated systems
    ! when slight impurities result in the bubble or droplet having no 
    ! internal circulations.  Currently, this function uses these equations 
    ! for all spherical particles.
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     D = diffusion coefficients of the dispersed phase components in the
    !         continuous phase fluid (m^2/s)
    !     nc = number of tracked dispersed phase chemical components
    ! 
    ! Returns the mass transfer coefficients (m/s) for each component in the
    ! dispersed phase.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: de, us, rho, mu
    real(kind = DP), intent(in), dimension(nc) :: D
    real(kind = DP), intent(out), dimension(nc) :: beta
    
    ! Declare the variables internal to the subroutine
    integer :: i
    real(kind = DP) :: Re
    real(kind = DP), dimension(nc) :: Sc, Pe, Sh
    
    ! Compute the non-dimensional governing parameters
    Sc = mu / (rho * D)
    Pe = us * de / D
    call reynolds(de, us, rho, mu, Re)
    
    ! Compute the Sherwood Number
    do i = 1, nc
        if (D(i) > 0.0D0) then
            if (Re < 100) then
                Sh(i) = 1.0D0 + (1.0D0 + 1.0D0 / Pe(i))**(1.0D0/3.0D0) * &
                    & Re**0.41D0 * Sc(i)**(1.0D0/3.0D0)
            else if (Re < 2000) then
                Sh(i) = 1.0D0 + 0.724D0 * Re**(0.48D0) * Sc(i)**(1.0D0/3.0D0)
            else
                Sh(i) = 1.0D0 + 0.425D0 * Re**0.55D0 * Sc(i)**(1.0D0/3.0D0)
            end if
        else
            Sh(i) = 0.0D0
        end if
    end do
    
    ! Return the mass transfer coefficient
    beta(:) = Sh(:) * D(:) / de
    
end subroutine xfer_sphere


subroutine xfer_ellipsoid(de, us, rho, mu, D, status, nc, &
    &                     beta)
    
    ! 
    ! Compute the mass transfer coefficients for an ellipsoidal fluid particle
    ! 
    ! Clift is not very clear on what equations should be used for ellipsoidal
    ! fluid particles (drops and bubbles), but indicates that in contaminanted
    ! liquids, the mass transfer is close to that of rigid particles (i.e.,
    ! there is no internal circulation due to the contamination).  Thus, this
    ! subroutine currently returns the result for rigid spheres if the 
    ! particles are dirty.  For clean fluid particles, this function returns
    ! the result from equation (42) in Johnson et al. (1969).
    ! 
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     rho = continuous phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     D = diffusion coefficients of the dispersed phase components in the
    !         continuous phase fluid (m^2/s)
    !     status = flag indicating whether the interface is clean (status = 1)
    !              or dirty (status = -1)
    !     nc = number of tracked dispersed phase chemical components
    ! 
    ! Returns the mass transfer coefficients (m/s) for each component in the
    ! dispersed phase (assuming rigid spheres).
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: status, nc
    real(kind = DP), intent(in) :: de, us, rho, mu
    real(kind = DP), intent(in), dimension(nc) :: D
    real(kind = DP), intent(out), dimension(nc) :: beta
    
    ! Compute the correct mass transfer coefficients
    if (status > 0) then
        call xfer_johnson(de, us, D, nc, beta)
    else
        call xfer_sphere(de, us, rho, mu, D, nc, beta)
    end if
    
end subroutine xfer_ellipsoid


subroutine xfer_spherical_cap(de, us, rho, rho_p, mu, D, status, nc, &
    &                         beta)
    
    ! 
    ! Compute the mass transfer coefficients for spherical cap fluid particles
    ! 
    ! Computes the mass transfer coefficient for spherical-cap bubbles or
    ! droplets.  If the particles are clean, it uses equation (42) in 
    ! Johnson et al. (1969).  If the particles are dirty, it uses equation 
    ! (8-28) in Clift et al. (1978), p. 214.  
    !
    ! Input variables:
    !     de = equivalent spherical diameter (m)
    !     us = slip velocity of the dispersed phase (m/s)
    !     rho = continuous phase density (kg/m^3)
    !     rho_p = dispersed phase density (kg/m^3)
    !     mu = dynamic viscosity of the continuous phase (Pa s)
    !     D = diffusion coefficients of the dispersed phase components in the
    !         continuous phase fluid (m^2/s)
    !     status = flag indicating whether the interface is clean (status = 1)
    !              or dirty (status = -1)    
    !     nc = number of tracked dispersed phase chemical components
    ! 
    ! Returns the mass transfer coefficients (m/s) for each component in the
    ! dispersed phase.
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use Phys_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: status, nc
    real(kind = DP), intent(in) :: de, us, rho, rho_p, mu
    real(kind = DP), intent(in), dimension(nc) :: D
    real(kind = DP), intent(out), dimension(nc) :: beta
    
    ! Declare the variables internal to the subroutine
    real(kind = DP) :: theta_w, A, Ae
    
    ! Compute the correct mass transfer coefficients
    if (status > 0) then
        ! Use the Johnson et al. (1969) equation for clean bubbles
        call xfer_johnson(de, us, D, nc, beta)
        
    else
        ! Use the Clift et al. (1978) equation for spherical cap bubbles
        ! Compute the wake angle for the partial sphere model (equation 8-1)
        call theta_w_sc(de, us, rho, mu, theta_w)
        
        ! Compute the surface area of the spherical cap and equivalent sphere
        call surface_area_sc(de, theta_w, A)
        Ae = 4.0D0 * PI * (de / 2.0D0)**2
        
        ! Compute the mass transfer (equation 8-28)
        beta = (1.25D0 * (G * (rho - rho_p) / rho)**(0.25D0) * sqrt(D(:)) / &
             & de**(0.25D0)) * Ae / A
    
    end if
    
end subroutine xfer_spherical_cap


! End File: DBM_PHYS.f95