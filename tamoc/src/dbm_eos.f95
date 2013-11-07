! File: DBM_EOS_F.f95
! 
! Fortran Implementation of the Diserpsed Phase Equations of State
! 
! This module provides fluid particle properties to the discrete bubble model
! (DBM) on various equations of state.  These include the Peng-Robinson 
! equation of state for density and fugacity and the modified Henry's law
! for solubility.
! 
! REQUIRES:
!     math_funcs.f95 : Fortran library of math functions, including the 
!         cubic_roots solver for solutions to cubic equations.
! 
! Compiles in Mac OSX using:
!     f2py -c -m dbm_eos dbm_eos_f.f95 math_funcs.f95
! 
! S. Socolofsky 
! June 2013
! Texas A&M University


module EOS_Constants
    
    ! Define constants for use by the Peng-Robinson equation of state
    implicit none
    integer, parameter :: DP = 8
    real(kind = DP), parameter :: RU = 8.314510D0
    
end module EOS_Constants


! ----------------------------------------------------------------------------
! Peng-Robinson Equations of State for Density and Fugacity
! ----------------------------------------------------------------------------

subroutine density(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
    &              rho)
    
    ! 
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
    !     omega = array of Pitzer acentric factors for each component (--)
    !     delta = matrix of binary interaction coefficients (--)
    ! 
    ! Output variable is:
    !     rho = numpy array of the density [gas, liquid] of the mixture (kg/m^3)
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out), dimension(2, 1) :: rho
    
    ! Declare the variables internal to the function
    real(kind = DP) :: A, B, R
    real(kind = DP), dimension(2, 1) :: z
    real(kind = DP), dimension(nc) :: Ap, Bp, yk
    
    ! Get the z-factor using the Peng-Robinson equation of state
    call z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
        &     z, A, B, Ap, Bp, yk)
    
    ! Compute and return the density
    R = Ru * sum(mass(:) / Mol_wt(:)) / sum(mass(:))
    rho = P / (z * R * T)

end subroutine density


subroutine fugacity(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
    &               fug)
    
    ! 
    ! Computes the liquid and gas fugacity of a mixture from the P-R EOS
    ! 
    ! Computes the gas and liquid fugacity of a mixture using the Peng-
    ! Robinson equation of state as described in McCain (1990), Properties of 
    ! Petroleum Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, 
    ! Oklahoma.
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
    ! 
    ! Output variable is:
    !     fug = array of the fugacities [gas, liquid] of the mixture (Pa)
    !         The first row of f are the gas component fugacities and the 
    !         second row of f are the liquid component fugacities
    ! 
    ! S. Socolofsky
    ! June 2013
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out), dimension(2, nc) :: fug
    
    ! Declare the variables internal to the function
    integer :: i
    real(kind = DP) :: A, B
    real(kind = DP), dimension(2, 1) :: z
    real(kind = DP), dimension(nc) :: Ap, Bp, yk
    
    ! Get the z-factor using the Peng-Robinson equation of state
    call z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
        &     z, A, B, Ap, Bp, yk)
    
    do i = 1, 2
        fug(i,:) = exp((z(i,1) - 1.0D0) * Bp(:) - log(z(i,1) - B) - A / &
            &      (2.0D0**(1.5D0) * B) * (Ap(:) - Bp(:)) * log((z(i,1) + &
            &      (sqrt(2.0D0) + 1.0D0) * B) / (z(i,1) - (sqrt(2.0D0) - &
            &      1.0D0) * B))) * yk(:) * P
    end do 
    
end subroutine fugacity


subroutine z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
    &           z, A, B, Ap, Bp, yk)
    
    ! 
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
    ! 
    ! Output variables are:
    !     z = array of the z-factor (gas, liquid) for the mixture (--)
    !     A, B, Ap, Bp = coefficients for the P-R EOS defined in coefs
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out) :: A, B
    real(kind = DP), intent(out), dimension(2, 1) :: z
    real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk
    
    ! Declare the variables internal to the function
    integer :: i
    real(kind = DP) :: z_max, z_min
    real(kind = DP), dimension(4) :: p_coefs
    complex(kind = DP), dimension(3) :: z_roots
    
    ! Compute the coefficients of the polynomial for z-factor
    call coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
        &      A, B, Ap, Bp, yk)
    p_coefs(1) = 1.0D0
    p_coefs(2) = B - 1.0D0
    p_coefs(3) = A - 2.0D0 * B - 3.0D0 * B**2
    p_coefs(4) = B**3 + B**2 - A * B
    
    ! Find the roots of the cubic equation of state
    call cubic_roots(p_coefs, z_roots)
    
    ! Extract the correct z-factors
    z_max = 0.0D0
    do i = 1, 3
        if (aimag(z_roots(i)) == 0.0) then
            if (real(z_roots(i)) > z_max) then
                z_max = real(z_roots(i))
            end if
        end if
    end do
    z_min = z_max
    do i = 1, 3
        if (aimag(z_roots(i)) == 0.0) then
            if ((real(z_roots(i)) < z_min) .and. &
              & (real(z_roots(i)) > 0.0D0)) then
                z_min = real(z_roots(i))
            end if
        end if
    end do
    
    ! Return the z-factors in z
    z(1,1) = z_max
    z(2,1) = z_min
    
end subroutine z_pr


subroutine coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, &
    &            A, B, Ap, Bp, yk)
    
    ! 
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
    ! 
    ! Output variables are:
    !     A = aT coefficient in P-R EOS
    !     B = b coefficient in P-R EOS
    !     Ap = non-dimensional array of mixture aT-coefficients
    !     Bp = non-dimensional array of mixture b-coefficients
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out) :: A, B
    real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk
    
    ! Declare the variables internal to the function
    integer :: i, j
    real(kind = DP) :: bd, aT
    real(kind = DP), dimension(nc) :: mu, alpha, aTk, bk
    
    ! Convert the masses to mole fraction
    call mole_fraction(nc, mass, Mol_wt, yk)
    
    ! Compute the coefficient values for each gas in the mixture
    mu = 0.37464D0 + 1.54226D0 * omega(:) - 0.26992D0 * omega(:)**2
    alpha = (1.0D0 + mu(:) * (1.0D0 - (T / Tc(:))**(1.0D0/2.0D0)))**2
    aTk = 0.45724D0 * Ru**2 * Tc(:)**2 / Pc(:) * alpha(:)
    bk = 0.07780D0 * Ru * Tc(:) / Pc(:)
    
    ! Use the mixing rules in McCain (1990)
    bd = sum(yk(:) * bk(:))
    aT = 0.0D0
    do i = 1, nc
        do j = 1, nc
            aT = aT + yk(i) * yk(j) * (aTk(i) * aTk(j))**(1.0D0/2.0D0) * &
               & (1.0D0 - delta(i,j))
        end do
    end do
    
    ! Compute the coefficients of the polynomials for z-factor and fugacity
    A = aT * P / (Ru**2 * T**2)
    B = bd * P / (Ru * T)
    Bp = bk / bd
    do i = 1, nc
        Ap(i) = 1.0D0 / aT * (2.0D0 * aTk(i)**(1.0D0/2.0D0) * &
              & sum(yk(:) * aTk(:)**(1.0D0/2.0D0) * (1.0D0 - delta(:,i))))
    end do

end subroutine coefs


subroutine mole_fraction(nc, mass, Mol_wt, &
    &                    yk)
    
    ! 
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
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt
    real(kind = DP), intent(out), dimension(nc) :: yk
    
    ! Declare the variables internal to the function
    real(kind = DP), dimension(nc) :: n_moles
    
    ! Compute the total number of moles
    n_moles(:) = mass(:) / Mol_wt(:)
    
    ! Compute the mole fraction
    yk(:) = n_moles(:) / sum(n_moles(:))
    
end subroutine mole_fraction


! ----------------------------------------------------------------------------
! Modified Henry's Law for Solubility Calculations
! ----------------------------------------------------------------------------

subroutine kh_insitu(T, P, S, kh_0, dH_solR, nu_bar, nc, &
    &                kh)
    
    ! 
    ! Compute the in-situ Henry's law constant
    ! 
    ! Compute the in-situ Henry's law constant per the algorithm in McGinnis
    ! et al. (2006).  This involves adjustment from Henry's coefficients at 
    ! STP to the appropriate values at ambient temperature, pressure, and 
    ! continuous phase salinity.  The conditions at STP are specified in the
    ! source documentation for the input Henry's coefficients.  Adjustments
    ! for temperature and pressure are per appropriate thermodynamic equations
    ! of state.  The adjust for salinity is taken from detailed calculations 
    ! for dissolution of CO2 in seawater.  The form of the equation is 
    ! likely correct for a wide range of chemicals; however, the fit 
    ! coefficients used here were derived for CO2 and should be adjusted 
    ! when applied to other components.  No available method for adjustment
    ! is provided in this function.
    ! 
    ! Input variables are:
    !     T = temperature (K)
    !     P = pressure (Pa)
    !     S = salinity (psu)
    !     mass = array of masses for each component in the mixture (kg)
    !     Mol_wt = array of molecular weights for each component (kg/mol)    
    !     kh_0 = Henry's Law constant at 298.15 K (kg/(m^3 Pa))
    !     dH_solR = enthalpy of solution / R (K)
    !     nu_bar = partial molar volume at infinite dilution (m^3/mol)
    ! 
    ! Returns an array of Henry's law coefficients (kg/(m^3 Pa))
    ! 
    ! S. Socolofsky
    ! July 2013
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P, S
    real(kind = DP), intent(in), dimension(nc) :: kh_0, dH_solR, nu_bar
    real(kind = DP), intent(out), dimension(nc) :: kh
    
    ! Declare the variables internal to the function 
    real(kind = DP), parameter :: P_ATM = 101325.0D0
    
    ! Adjust from STP to ambient temperature
    kh(:) = kh_0(:) * exp(dH_solR(:) * (1.0D0 / T - 1.0D0 / 298.15D0))
    
    ! Adjust to the ambient pressure
    kh(:) =  kh(:) * exp((P_ATM - P) * nu_bar(:) / (RU * T))
    
    ! Adjust for the salting out effect of salinity.  Note that this specific
    ! equation was derived for CO2 and that we use it here assuming the 
    ! effect is similar for all gases.  This is consistent with McGinnis 
    ! et al. (2006).
    kh(:) = kh(:) * exp(S * (0.027766D0 - 0.025888D0 * T / 100.0D0 + &
          & 0.0050578D0 * (T / 100.0D0)**2))
    
end subroutine kh_insitu


subroutine sw_solubility(f, kh, nc, &
    &                    Cs)
    
    ! 
    ! Compute the solubility of each component in a dispersed-phase mixture
    ! 
    ! Computes the solubility of each component in a dispersed-phase mixture
    ! into the surrounding continuous phase.  The calculation follows 
    ! McGinnis et al. (2006) using the modified Henry's law coefficients and
    ! the mixture fugacities.
    ! 
    ! Input variables are:
    !     f = fugacity of each component in the dispersed-phase (Pa)
    !     kh = modified Henry's law coefficients (kg/(m^3 Pa))
    ! 
    ! Returns an array of solubilities (kg/m^3) of dispersed-phase components
    ! into the continuous phase.
    ! 
    ! S. Socolofsky
    ! July 2013
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in), dimension(nc) :: f, kh
    real(kind = DP), intent(out), dimension(nc) :: Cs
    
    Cs(:) = f(:) * kh(:)
    
end subroutine sw_solubility


subroutine diffusivity(T, B, dE, nc, &
    &                  D)
    
    ! 
    ! Compute the diffusivity of each component in a mixture into seawater
    ! 
    ! Computes the diffusivity of each component in a fluid mixture into 
    ! seawater at the given temperature.  The calculation is from Wise and 
    ! Houghton (1966) for slightly soluble gases in water at 10 and 60 deg C.
    ! Data in the paper are presented for H2, O2, N2, air, He, Ar, CH4, C2H6, 
    ! C3H8, and n-C4H10.  
    ! 
    ! Input variables are:
    !     T = temperature of the ambient seawater (K)
    !     B, dE = fit parameters from Wise and Houghton
    ! 
    ! Returns an array of diffusivities (m^2/s) for each component into water.
    ! 
    ! S. Socolofsky
    ! July 2013
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T
    real(kind = DP), intent(in), dimension(nc) :: B, dE
    real(kind = DP), intent(out), dimension(nc) :: D
    
    D(:) = B(:) * exp(-dE(:) / (RU * T))
    
end subroutine diffusivity


! ----------------------------------------------------------------------------
! Hydrate formation predictions from Sloan and Koh (2008)
! ----------------------------------------------------------------------------

subroutine Kvsi_hydrate(T_in, P_in, mass, &
    &                   K_vsi, yk)
    
    !
    ! Determine whether or not hydrate is stable for the given conditions
    !
    ! Solve the K_vsi method for hydrate partition coefficient to determine
    ! whether the given gas composition and thermodynamic state yields a 
    ! stable hydrate.  The masses of each component of the gas must be 
    ! organized in the following order:
    !
    !     CH4
    !     C2H6
    !     C3H8
    !     i-C4H10
    !     n-C4H10
    !     N2
    !     CO2
    !     H2S
    !
    ! Input variables are:
    !     T_in = temperature of the gas mixture (K)
    !     P_in = pressure (Pa)
    !     mass = array of masses for each component in the mixture (kg)   
    !
    ! Returns
    !
    ! S. Socolofsky
    ! October 2013
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, parameter :: NC = 8
    real(kind = DP), intent(in) :: T_in, P_in
    real(kind = DP), intent(in), dimension(nc) :: mass
    real(kind = DP), intent(out), dimension(nc) :: K_vsi
    real(kind = DP), intent(out), dimension(nc) :: yk
    
    ! Declare the variables internal to the function
    integer :: i
    real(kind = DP) :: T, P
    real(kind = DP), dimension(nc) :: Mol_wt
    real(kind = DP), dimension(8,18) :: coef
    
    ! Define the molecular weight of each gas component
    Mol_wt = [16.0426D0, 30.0694D0, 44.0962D0, 58.123D0, 58.123D0, 28.0134D0, &
        &     44.0098D0, 34.0818D0]
    
    ! Fill the parameter matrix coef with data from Table 4.4a, p. 223
    coef = reshape( &
        & [1.63636D0, 6.41934D0, -7.8499D0, -2.17137D0, -37.211D0, 1.78857D0, &
        &  9.0242D0, -4.7071D0, &
        &  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.86564D0, 0.0D0, &
        &  0.0D0, 0.06192D0, &
        &  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, -0.001356D0, &
        &  0.0D0, 0.0D0, &
        &  31.6621D0, -290.283D0, 47.056D0, 0.0D0, 732.20D0, -6.187D0, &
        &  -207.033D0, 82.627D0, &
        &  -49.3534D0, 2629.10D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, &
        &  0.0D0, 0.0D0, &
        &  -5.31D-6, 0.0D0, -1.17D-6, 0.0D0, 0.0D0, 0.0D0, &
        &  4.66D-5, -7.39D-6, &
        &  0.0D0, 0.0D0, 7.145D-4, 1.251D-3, 0.0D0, 0.0D0, &
        &  -6.992D-3, 0.0D0, &
        &  0.0D0, -9.0D-8, 0.0D0, 1.0D-8, 9.37D-6, 2.5D-7, &
        &  -2.89D-6, 0.0D0, &
        &  0.128525D0, 0.129759D0, 0.0D0, 0.166097D0, -1.07657D0, 0.0D0, &
        &  -6.223D-3, 0.240869D0, &
        &  -0.78338D0, -1.19703D0, 0.12348D0, -2.75945D0, 0.0D0, 0.0D0, &
        &  0.0D0, -0.64405D0, &
        &  0.0D0, -8.46D4, 1.669D4, 0.0D0, 0.0D0, 0.0D0, &
        &  0.0D0, 0.0D0, &
        &  0.0D0, -71.0352D0, 0.0D0, 0.0D0, -66.221D0, 0.0D0, &
        &  0.0D0, 0.0D0, &
        &  0.0D0, 0.596404D0, 0.23319D0, 0.0D0, 0.0D0, 0.0D0, &
        &  0.27098D0, 0.0D0, &
        &  -5.3569D0, -4.7437D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, &
        &  0.0D0, -12.704D0, &
        &  0.0D0, 7.82D4, -4.48D4, -8.84D2, 9.17D5, 5.87D5, &
        &  0.0D0, 0.0D0, &
        &  -2.3D-7, 0.0D0, 5.5D-6, 0.0D0, 0.0D0, 0.0D0, &
        &  8.82D-5, -1.3D-6, &
        &  -2.0D-8, 0.0D0, 0.0D0, -5.4D-7, 4.98D-6, 1.0D-8, &
        &  2.55D-6, 0.0D0, &
        &  0.0D0, 0.0D0, 0.0D0, -1.0D-8, -1.26D-6, 1.1D-7, &
        &  0.0D0, 0.0D0], shape(coef))
    
    ! Convert the masses to mole fraction
    call mole_fraction(nc, mass, Mol_wt, yk)
    
    ! Convert pressure to psia and temperature to deg F
    P = 0.0001450377377D0 * P_in
    T = 9.0D0 / 5.0D0 * (T_in- 273.15D0) + 32.0D0
    
    ! Compute equation 4.2 in Sloan and Koh
    do i = 1, 8
        K_vsi(i) = exp(coef(i,1) + coef(i,2) * T + coef(i,3) * P & 
            &      + coef(i,4) / T + coef(i,5) / P + coef(i,6) * P * T &
            &      + coef(i,7) * T**2 + coef(i,8) * P**2 &
            &      + coef(i,9) * P / T + coef(i,10) * log(P / T) &
            &      + coef(i,11) / P**2 + coef(i,12) * T / P &
            &      + coef(i,13) * T**2 / P + coef(i,14) * P / T**2 &
            &      + coef(i,15) * T / P**3 + coef(i,16) * T**3 &
            &      + coef(i,17) * P**3 / T**2 + coef(i,18) * T**4)
    end do

end subroutine Kvsi_hydrate

! End File: DBM_EOS.f95