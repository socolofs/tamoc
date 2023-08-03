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
    ! Ru in J / (mol K)
    implicit none
    integer, parameter :: DP = 8
    real(kind = DP), parameter :: RU = 8.314510D0
    
end module EOS_Constants


! ----------------------------------------------------------------------------
! Peng-Robinson Equations of State for Density and Fugacity
! ----------------------------------------------------------------------------

subroutine density(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta, Aij, & 
    &              Bij, delta_groups, calc_delta, C_pen, C_pen_T, &
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
	!     C_pen = Peneloux volume translation coefficient (m^3/mol)
	!     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
    ! 
    ! Output variable is:
    !     rho = numpy array of the density [gas, liquid] of the mixture 
    !         (kg/m^3)
    ! 
    ! S. Socolofsky
    ! June 2013
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc, calc_delta
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, Vc, &
                                                & omega, C_pen, C_pen_T
    real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out), dimension(2, 1) :: rho
    
    ! Declare the variables internal to the function
    real(kind = DP) :: A, B
    real(kind = DP), dimension(2, 1) :: z, nu
    real(kind = DP), dimension(nc) :: Ap, Bp, yk, vt
    
    ! Get the z-factor using the Peng-Robinson equation of state
    call z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
        &     delta_groups, calc_delta, & 
        &     z, A, B, Ap, Bp, yk)
    
    ! Convert the masses to mole fraction
    call mole_fraction(nc, mass, Mol_wt, yk)
    
    ! Compute the volume translation coefficient
    call volume_trans(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, C_pen, C_pen_T, vt)
    
    ! Compute the molar volume
    nu = z * Ru * T / P - sum(yk(:) * vt(:))
    
    ! Compute and return the density
    rho = 1.0D0 / nu * sum(yk(:) * Mol_wt(:))

end subroutine density


subroutine fugacity(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
    &               delta_groups, calc_delta, &
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
    !     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
    !     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
    !     delta_groups = group contribution numbers (normalized) for each 
    !         component in the mixture (--)
    !     calc_groups = flag indicating whether or not delta_groups has 
    !         been provided (1 = yes, -1 = no)
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
    integer, intent(in) :: nc, calc_delta
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out), dimension(2, nc) :: fug
    
    ! Declare the variables internal to the function
    integer :: i
    real(kind = DP) :: A, B
    real(kind = DP), dimension(2, 1) :: z
    real(kind = DP), dimension(nc) :: Ap, Bp, yk
    
    ! Get the z-factor using the Peng-Robinson equation of state
    call z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
        &     delta_groups, calc_delta, &
        &     z, A, B, Ap, Bp, yk)
    
    do i = 1, 2
        fug(i,:) = exp((z(i,1) - 1.0D0) * Bp(:) - log(z(i,1) - B) - A / &
            &      (2.0D0**(1.5D0) * B) * (Ap(:) - Bp(:)) * log((z(i,1) + &
            &      (sqrt(2.0D0) + 1.0D0) * B) / (z(i,1) - (sqrt(2.0D0) - &
            &      1.0D0) * B))) * yk(:) * P
    end do 
    
end subroutine fugacity


subroutine volume_trans(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, C_pen, C_pen_T, &
    &                   vt)
    
    ! 
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
	!     C_pen = Peneloux volume translation coefficient (m^3/mol)
	!     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
    ! 
    ! Output variable is:
    !     vt = volume translation parameter (m^3/mol)
    ! 
    ! S. Socolofsky
    ! December 2014
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, Vc, &
	                                            & C_pen, C_pen_T
    real(kind = DP), intent(out), dimension(nc) :: vt
    
    ! Declare the variables internal to the function
    real(kind = DP), dimension(nc) :: Zc, beta, gamma, f_Tr, cc
    
	! Decide how to get the Peneloux shift parameters
	if (C_pen(1) == 0.) then
		! Compute the compressibility factor (--) for each component of the 
	    ! mixture
	    Zc = Pc(:) * Vc(:) / (Ru * Tc(:))
   
	    ! Calculate the parameters in the Lin and Duan (2005) paper:  beta is 
	    ! from equation (12)
	    beta = -2.8431D0 * exp(-64.2184D0 * (0.3074D0 - Zc(:))) + 0.1735D0
   
	    ! and gamma is from Equation (13)
	    gamma = -99.2558D0 + 301.6201D0 * Zc(:)
   
	    ! Account for the temperature dependence (equation 10)
	    f_Tr = beta(:) + (1.0D0 - beta(:)) * exp(gamma(:) * &
	&		   abs(1.0D0-T / Tc(:)))
   
	    ! Compute the volume translation for the critical point (equation 9)
	    cc = (0.3074D0 - Zc(:)) * Ru * Tc(:) / Pc(:)
   
	    ! Finally, the volume translation at the given state is (equation 8)
	    vt = f_Tr * cc
		
	else
		! Use the user-defined Peneloux parameters following equation 5.9 in
		! Pedersen et al. (2015) Phase Behavior of Petroleum Reservoir Fluids
		vt = C_pen(:) + C_pen_T(:) * (T - 288.15)
		
	end if
	
end subroutine volume_trans


subroutine z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
    &           delta_groups, calc_delta, &
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
    ! 
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc, calc_delta
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    real(kind = DP), intent(out) :: A, B
    real(kind = DP), intent(out), dimension(2, 1) :: z
    real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk
    
    ! Declare the variables internal to the function
    integer :: i
    real(kind = DP) :: z_max, z_min
    real(kind = DP), dimension(4) :: p_coefs
    complex(kind = DP), dimension(3) :: z_roots
    
    ! Compute the coefficients of the polynomial for z-factor
    call coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij, &
        &      delta_groups, calc_delta, &
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


subroutine coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta_in, Aij, Bij, &
    &            delta_groups, calc_delta, &
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
    ! 
    
    use EOS_Constants
    use ieee_arithmetic
    
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc, calc_delta
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, &
                                                & omega
    real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    real(kind = DP), intent(in), dimension(nc, nc) :: delta_in
    real(kind = DP), intent(out) :: A, B
    real(kind = DP), intent(out), dimension(nc) :: Ap, Bp, yk
    
    ! Declare the variables internal to the function
    integer :: i, j, k, l
    real(kind = DP) :: bd, aT, sum_term, sum1
    real(kind = DP), dimension(nc) :: mu, alpha, aTk, bk
    real(kind = DP), dimension(nc, nc) :: delta
    
    ! Convert the masses to mole fraction
    call mole_fraction(nc, mass, Mol_wt, yk)
    
    ! Compute the coefficient values for each gas in the mixture.  Use the 
    ! modified Peng-Robinson (1978) equations for mu
    do i = 1, nc
        if (omega(i) > 0.49D0) then
            mu(i) = 0.379642D0 + 1.48503D0 * omega(i) - 0.164423D0 * &
                  & omega(i)**2 + 0.016666D0 * omega(i)**3
        else
            mu(i) = 0.37464D0 + 1.54226D0 * omega(i) - 0.26992D0 * omega(i)**2
        end if
    end do
    alpha = (1.0D0 + mu(:) * (1.0D0 - (T / Tc(:))**(1.0D0/2.0D0)))**2
    aTk = 0.45724D0 * Ru**2 * Tc(:)**2 / Pc(:) * alpha(:)
    bk = 0.07780D0 * Ru * Tc(:) / Pc(:)
    
    ! Initialize the output vector for delta to the input values
    delta(:,:) = delta_in(:,:)
    
    ! Get the temperature-dependent binary interaction coefficients (if 
    ! the user provided the group contributions)
    if (calc_delta > 0) then
        do j = 2, nc
            do i = 1, j-1
                sum1 = 0.0D0
                do l = 1, 15
                    do k = 1, 15
                        sum_term =  (delta_groups(i,k) - & 
                                 & delta_groups(j,k)) * (delta_groups(i,l) - &
                                 & delta_groups(j,l)) * Aij(k, l) * &
                                 & (298.15D0 / T) ** (Bij(k,l) / Aij(k,l) - &
                                 & 1.0D0)
                        if (.not. ieee_is_nan(sum_term)) then
                            sum1 = sum1 + sum_term
                        end if
                    end do
                end do
                
                delta(i, j) = - (0.5D0 * sum1 + (sqrt(aTk(i)) / bk(i) - &
                            & sqrt(aTk(j)) / bk(j)) ** 2) / & 
                            & (2.0D0 * sqrt(aTk(i) * aTk(j)) / &
                            & (bk(i) * bk(j)))
                delta(j, i) = delta(i,j)
            end do
        end do
    end if
    
    ! Use the mixing rules in McCain (1990)
    bd = sum(yk(:) * bk(:))
    aT = 0.0D0
    do j = 1, nc
        do i = 1, nc
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
! Other Fluid Properties (viscosity, surface tension, etc...)
! ----------------------------------------------------------------------------

subroutine viscosity(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta, Aij, & 
    &              Bij, delta_groups, calc_delta, C_pen, C_pen_T, &
    &              mu)
    
    !
    ! Computes the viscosity of a petroleum fluid
    !
    ! Computes the viscosity of the given fluid mixture for the gas and 
    ! liquid phases following the method in Pedersen et al. "Phase Behavior
    ! of Petroleum Reservoir Fluids", 2nd edition, Chapeter 10.
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
	!     C_pen = Peneloux volume translation coefficient (m^3/mol)
	!     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
    ! 
    ! Output variable is:
    !     mu = numpy array of the viscosity [gas, liquid] of the mixture
    !         (Pa s)    
    !
    ! S. Socolofsky 
    ! June 2015
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc, calc_delta
    real(kind = DP), intent(in) :: T, P
    real(kind = DP), intent(in), dimension(nc) :: mass, Mol_wt, Pc, Tc, Vc, &
                                                & omega, C_pen, C_pen_T
    real(kind = DP), intent(in), dimension(nc, 15) :: delta_groups
    real(kind = DP), intent(in), dimension(15, 15) :: Aij, Bij
    real(kind = DP), intent(in), dimension(nc, nc) :: delta
    real(kind = DP), intent(out), dimension(2, 1) :: mu
    
    ! Declare the variables internal to the function
    integer :: i, j
    real(kind = DP) :: A, B, C, F, rho_c0, eta_0, eta_1, delta_T, htan, &
                     & numerator, denominator, Tc_mix, Pc_mix, M_bar_n, &
                     & M_bar_w, M_mix
    real(kind = DP), dimension(1) :: M0, Tc0, Pc0, omega0, Vc0
    real(kind = DP), dimension(2) :: T0, P0
    real(kind = DP), dimension(7) :: jc, kc
    real(kind = DP), dimension(9) :: GV
    real(kind = DP), dimension(nc) :: z, M
    real(kind = DP), dimension(1,1) :: delta0
    real(kind = DP), dimension(1,15) :: delta_groups0
    real(kind = DP), dimension(2, 1) :: theta, delta_eta_p, delta_eta_pp, &
                                      & rho0, eta_ch4, rho_r, alpha_mix, &
                                      & alpha0
    
    ! Enter the parameter values from Table 10.1
    GV = [-2.090975D5, 2.647269D5, -1.472818D5, 4.716740D4, -9.491872D3, &
        & 1.219979D3, -9.627993D1, 4.274152D0, -8.141531D-2]
    A = 1.696985927D0
    B = -0.133372346D0
    C = 1.4D0
    F = 168.0D0
    jc = [-10.3506D0, 17.5716D0, -3019.39D0, 188.730D0, 0.0429036D0, &
        & 145.290D0, 6127.68D0]
    kc = [-9.74602D0, 18.0834D0, -4126.66D0, 44.6055D0, 0.976544D0, &
        & 81.8134D0, 15649.9D0]
    
    ! Enter the properties for the reference fluid (methane)
    M0(1) = 16.043D-3
    Tc0(1) = 190.56D0
    Pc0(1) = 4599000.0D0
    omega0(1) = 0.011D0
    Vc0(1) = 9.86D-5
    delta0(1,1) = 0.0D0
    delta_groups0(1,:) = [0.0D0, 0.0D0, 0.0D0, 0.0D0, 1.0D0, 0.0D0, 0.0D0, &
        &                 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, &
        &                 0.0D0]
    rho_c0 = 162.84D0
    
    ! 1.  Prepare the variables to determine the corresponding states between
    !     the given mixture and the reference fluid (methane) ----------------
    
    ! Get the mole fraction of the components of the mixture
    call mole_fraction(nc, mass, Mol_wt, z)
    
    ! Compute equation (10.19)
    numerator = 0.0D0
    denominator = 0.0D0
    do i = 1, nc
        do j = 1, nc
            numerator = numerator + z(i) * z(j) * ((Tc(i) / Pc(i)) & 
                &       **(1.0D0/3.0D0) + (Tc(j) / Pc(j))**(1.0D0/3.0D0)) & 
                &       **3 * sqrt(Tc(i) * Tc(j))
            denominator = denominator + z(i) * z(j) * ((Tc(i) / Pc(i)) &
                &       **(1.0D0/3.0D0) + (Tc(j) / Pc(j))**(1.0D0/3.0D0)) &
                &       **3
        end do
    end do
    Tc_mix = numerator / denominator
    
    ! Compute equation (10.22)
    Pc_mix = 8.0D0 * numerator / denominator**2
    
    ! Get the density of methane at TTc0/Tc_mix and PPc0/Pc_mix
    call density(1, T * Tc0(1) / Tc_mix, P * Pc0(1) / Pc_mix, [1.0D0], M0, &
        &        Pc0, Tc0, Vc0, omega0, delta0, Aij, Bij, delta_groups0, &
        &        -1, C_pen, C_pen_T, rho0)
    
    ! Compute equation (10.27)
    rho_r(:,1) = rho0(:,1) / rho_c0
    
    ! Compute equation (10.23), where M is in g/mol
    M = Mol_wt(:) * 1.0D3
    M_bar_n = sum(z(:) * M(:))
    M_bar_w = sum(z(:) * M(:)**2) / M_bar_n
    M_mix = 1.304D-4 * (M_bar_w**2.303D0 - M_bar_n**2.303D0) + M_bar_n
    
    ! Compute equation (10.26), where M is in g/mol
    M0 = M0(:) * 1.0D3
    alpha_mix(:,1) = 1.0D0 + 7.378D-3 * rho_r(:,1)**1.847D0 * M_mix**0.5173D0
    alpha0(:,1) = 1.0D0 + 7.378D-3 * rho_r(:,1)**1.847D0 * M0(1)**0.5173D0
    
    ! 2.  Compute the viscosity of methane at the corresponding state --------
    
    ! Corresponding state
    T0 = T * Tc0(1) / Tc_mix * alpha0(:,1) / alpha_mix(:,1)
    P0 = P * Pc0(1) / Pc_mix * alpha0(:,1) / alpha_mix(:,1)
    
    ! Compute each state separately
    do i = 1,2
        
        ! Get the density of methane at T0 and P0.  Be sure to use molecular
        ! weight in kg/mol
        call density(1, T0(i), P0(i), [1.0D0], M0*1.0D-3, Pc0, Tc0, Vc0, &
            &        omega0, delta0, Aij, Bij, delta_groups0, -1, &
			&        C_pen, C_pen_T, rho0)
        
        ! Compute equation (10.10)
        theta(:,1) = (rho0(:,1) - rho_c0) / rho_c0
        
        ! Equation (10.9) with T in K and rho in g/cm^3
        rho0(:,1) = rho0(:,1) * 1.0D-3
        
        delta_eta_p(:,1) = exp(jc(1) + jc(4) / T0(i)) * (exp(rho0(:,1) & 
            &              **0.1D0 * (jc(2) + jc(3) / T0(i)**1.5D0) + &
            &              theta(:,1) * rho0(:,1)**0.5D0 * (jc(5) + jc(6) &
            &              / T0(i) + jc(7) / T0(i)**2)) - 1.0D0)
        
        ! Equation (10.28)
        delta_eta_pp(:,1) = exp(kc(1) + kc(4) / T0(i)) * (exp(rho0(:,1) &
            &               **0.1D0 * (kc(2) + kc(3) / T0(i)**1.5D0) + &
            &               theta(:,1) * rho0(:,1)**0.5D0 * (kc(5) + kc(6) &
            &               / T0(i) + kc(7) / T0(i)**2)) - 1.0D0)
        
        ! Equation (10.7)
        eta_0 = GV(1) / T0(i) + GV(2) / T0(i)**(2.0D0/3.0D0) + GV(3) / &
            &   T0(i)**(1.0D0/3.0D0) + GV(4) + GV(5) * T0(i)**(1.0D0/3.0D0) &
            &   + GV(6) * T0(i)**(2.0D0/3.0D0) + GV(7) * T0(i) + GV(8) * &
            &   T0(i)**(4.0D0/3.0D0) + GV(9) * T0(i)**(5.0D0/3.0D0)
        
        ! Equation (10.8)
        eta_1 = A + B * (C - log(T0(i) / F))**2
        
        ! Equation (10.32)
        delta_T = T0(i) - 91.0D0
        
        ! Equation (10.31)
        htan = (exp(delta_T) - exp(-delta_T)) / (exp(delta_T) + exp(-delta_T))
        
        ! Viscosity of methane (Equation 10.29) -- reported in (Pa s)
        eta_ch4(i,1) = (eta_0 + eta_1 + (htan + 1.0D0) / 2.0D0 * &
            &          delta_eta_p(i,1) + (1.0D0 - htan) / 2.0D0 * &
            &          delta_eta_pp(i,1)) * 1.0e-7
        
    end do
    
    ! Compute the viscosity of the mixture at the given T and P
    mu(:,1) = (Tc_mix / Tc0(1))**(-1.0D0/6.0D0) * (Pc_mix / Pc0(1))** &
        &     (2.0D0/3.0D0) * (M_mix / M0(1))**(0.5D0) * alpha_mix(:,1) / &
        &     alpha0(:,1) * eta_ch4(:,1)

end subroutine viscosity


! ----------------------------------------------------------------------------
! Modified Henry's Law for Solubility Calculations
! ----------------------------------------------------------------------------

subroutine kh_insitu(T, P, S, kh_0, dH_solR, nu_bar, Mol_wt, K_salt, nc, &
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
    !     kh_0 = Henry's Law constant at 298.15 K (kg/(m^3 Pa))
    !     dH_solR = enthalpy of solution / R (K)
    !     nu_bar = partial molar volume at infinite dilution (m^3/mol)
    !     Mol_wt = array of molecular weights for each component (kg/mol) 
    !     K_salt = Setschenow constant (m^3/mol)
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
    real(kind = DP), intent(in), dimension(nc) :: kh_0, dH_solR, nu_bar, &
                                                & Mol_wt, K_salt
    real(kind = DP), intent(out), dimension(nc) :: kh
    
    ! Declare the variables internal to the function 
    integer :: i
    real(kind = DP), parameter :: P_ATM = 101325.0D0
    real(kind = DP), parameter :: M_SEA = 0.06835D0
    
    do i = 1, nc
        
        if (kh_0(i) < 0.0D0) then
            ! These are low solubility compounds for which we do not know 
            ! the solubility...set kh to zero.
            kh(i) = 0.0D0
        
        else
            ! Adjust from STP to ambient temperature
            kh(i) = kh_0(i) * exp(dH_solR(i) * (1.0D0 / T - 1.0D0 / 298.15D0))
            
            ! Adjust to the ambient pressure
            kh(i) =  kh(i) * exp((P_ATM - P) * nu_bar(i) / (RU * T))
            
            ! Adjust for the salting out effect of salinity.   
            kh(i) = kh(i) * 10.0D0 ** (-S / M_SEA * K_salt(i))
            
        end if
        
    end do
    
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


subroutine diffusivity(mu, Vb, nc, &
    &                  D)
    
    ! 
    ! Compute the diffusivity of each component in a mixture into seawater
    ! 
    ! Computes the diffusivity of each component in a fluid mixture into 
    ! seawater at the given temperature.  The calculation is from Hayduk and
    ! Laudie (1974), AIChE J., vol. 20, pp. 611-615.
    ! 
    ! Input variables are:
    !     mu = viscosity of seawater at the ambient conditions (Pa s)
    !     Vb = molar volume of each compound at its boiling point (m^3/mol)
    ! 
    ! Returns an array of diffusivities (m^2/s) for each component into water.
    ! 
    ! S. Socolofsky
    ! February 2015
    !
    
    use EOS_Constants
    implicit none
    
    ! Declare the input and output variable types
    integer, intent(in) :: nc
    real(kind = DP), intent(in) :: mu
    real(kind = DP), intent(in), dimension(nc) :: Vb
    real(kind = DP), intent(out), dimension(nc) :: D
    
    ! Declare the internal variables
    integer :: i
    
    do i = 1, nc
        
        if (Vb(i) < 0.0D0) then
            ! For some insoluble compounds, we do not know the inputs...
            ! set diffusivity to zero
            D(i) = 0.0D0
            
        else
            ! Use the Hayduk and Laudie formula
            D(i) = 13.26D-9 / ((mu * 1.0D3)**1.14D0 * (Vb(i) * 1.0D6) &
                &  **0.589D0)
        
        end if
    
    end do
    
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