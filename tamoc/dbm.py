"""
DBM Module
==========

Define objects that interface with the DBM functions in ``dbm_f``.

This module defines high-level Python class objects that wrap the individual 
functions that comprise the Discrete Bubble Model (DBM) in ``dbm_f``.  

These are particularly useful as an interface to the chemical property data
contained in ``./data/ChemData.csv`` and for pre- and post-processing of data
needed by the ``TAMOC`` simulation modules. These classes provide efficient 
data management of the chemical properties needed by the ``dbm_f`` functions 
and ensure proper behavior for the equations of state.

Notes
-----
The functions defining most equations of state and fluid particle physics are
contained in the ``dbm_f`` library.  ``dbm_f`` contains object code compiled
from the Fortran sources::

    ./src/dbm_eos.f95
    ./src/dbm_phys.f95
    ./src/math_funcs.f95

There are two additional functions defined in this module to complete the 
equations of state calculations.  These are::

    dbm.equilibrium
    dbm.gas_liq_eq

which compute the partitioning between gas and liquid of each component in 
a mixture.  For this calculation, iteration is required until the fugacities 
of each component in the mixture are equal in both gas and liquid.  Because 
for ``TAMOC`` this would generally only be done once at the start of a 
simulation to establish initial conditions and because `scipy.optimize` 
provides a nice Python interface to a fast zero-solver, these two elements of 
the discrete bubble model have not been ported to Fortran and reside in the 
`dbm` module instead of the ``dbm_f`` library.  

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

# Use these imports for deployment
from tamoc import chemical_properties as chem
from tamoc import dbm_f
from tamoc import seawater

import os
import numpy as np
from scipy.optimize import fsolve

class FluidMixture(object):
    """
    Class object for a fluid mixture
    
    This object defines the behavior of a fluid mixture defined as a standard
    thermodynamic system.  The mixture may contain just liquid phase, a 
    gas and liquid phase together, or a pure gas phase.  The Peng-Robinson
    equation of state returns the properties of each phase in the mixture.
    If the mixture is pure liquid or pure gas, the properties of each phase
    will be the same; otherwise, the gas properties will be in the first 
    row of all two-dimensional return variables and the liquid properties in 
    the second row.
    
    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    user_data : dict
        A dictionary of chemical property data.  If not specified, the data
        loaded from `/tamoc/data/ChemData.csv` by ``chemical_properties`` will
        be used.  To load a different properties database, use the 
        ``chemical_properties.load_data`` function to load in a custom 
        database, and pass that data to this object as `user_data`.
    delta_groups : ndarray (nc, 15)
        Provides the group contribution numbers (normalized) for each 
        component in the mixture for the 15 groups used by the Privat and
        Jaubert (2012) group contribution method for binary interaction 
        coefficients.  Default is None, in which case the values in `delta`
        will be used.
    air : bool
        Flag indicating whether or not fluid is air.  The methods for 
        viscosity and interfacial tension below use correlations developed
        for hydocarbons.  If `air` is False (default value), these built
        in methods are used.  If `air` is True, then these methods are 
        replaced with correlations between air and seawater.
    
    Attributes
    ----------
    nc : integer
        Number of chemical components in the mixture
    issoluble : logical, True
        Indicates the object contents are soluble
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    kh_0 : ndarray, size (nc)
        Henry's law constants at 298.15 K and 101325 Pa (kg/(m^3 Pa))
    neg_dH_solR : ndarray, size (nc)
        The negative of the enthalpies of solution / Ru (K).
    nu_bar : ndarray, size (nc)
        Partial molar volumes at infinite dilution (m^3/mol)
    B : ndarray, size (nc)
        White and Houghton (1966) pre-exponential factor (m^2/s)
    dE : ndarray, size (nc)
        Activation energy (J/mol)
    K_salt : Setschenow constants (m^3/mol)
    
    See Also
    --------
    chemical_properties, FluidParticle, InsolubleParticle
    
    Examples
    --------
    >>> air = FluidMixture(['nitrogen', 'oxygen'])
    >>> yk = np.array([0.79, 0.21])
    >>> m = air.masses(yk)
    >>> air.density(m, 273.15+10., 101325.)
    array([[ 1.24260911],
           [ 1.24260911]])
    
    """
    def __init__(self, composition, delta=None, user_data={}, 
                 delta_groups=None, air=False):
        super(FluidMixture, self).__init__()
        
        # Check the data type of the inputs and fix if necessary
        if not isinstance(composition, list):
            composition = [composition]
        
        if isinstance(delta, float) or isinstance(delta, list):
            delta = np.atleast_2d(delta)
        
        # Store the input variables and some of their derived properties
        self.composition = composition
        self.nc = len(composition)
        if delta is None:
            self.delta = np.zeros((self.nc, self.nc))
        else:
            if delta.shape[0] is self.nc:
                if delta.shape[1] is self.nc:
                    self.delta = delta
                else:
                    print '\nError: Delta wrong shape, should be (%d, %d)' % \
                          (self.nc, self.nc)
                    print 'Set to np.zeros((%d, %d))\n' % (self.nc, self.nc)
                    self.delta = np.zeros((self.nc, self.nc))
        
        # Store all of the chemical data
        self.chem_db = chem.data
        
        # Initialize the chemical composition variables used in TAMOC
        self.M = np.zeros(self.nc)
        self.Pc = np.zeros(self.nc)
        self.Tc = np.zeros(self.nc)
        self.Vc = np.zeros(self.nc)
        self.Tb = np.zeros(self.nc)
        self.Vb = np.zeros(self.nc)
        self.omega = np.zeros(self.nc)
        self.kh_0 = np.zeros(self.nc)
        self.neg_dH_solR = np.zeros(self.nc)
        self.nu_bar = np.zeros(self.nc)
        self.B = np.zeros(self.nc)
        self.dE = np.zeros(self.nc)
        self.K_salt = np.zeros(self.nc)
        
        # Fill the chemical composition variables from the chem database
        for i in range(self.nc):
            if composition[i] in user_data:
                # Get the properties from the user-specified dataset
                properties = user_data[composition[i]]
            else:
                # Get the properties from the default dataset supplied with 
                # TAMOC
                if composition[i] in chem.data:
                    properties = chem.data[composition[i]]
                else:
                    print '\nERROR:  %s is not in the ' % composition[i] + \
                          'Chemical Properties database\n' 
            
            # Store the properties in the object attributes
            self.M[i] = properties['M']
            self.Pc[i] = properties['Pc']
            self.Tc[i] = properties['Tc']
            self.Vc[i] = properties['Vc']
            self.Tb[i] = properties['Tb']
            if properties['Vb'] < 0.:
                # Use Tyn & Calus estimate in Poling et al. (2001)
                self.Vb[i] = (0.285 * (self.Vc[i]*1.e6)**1.048)*1.e-6
            else:
                self.Vb[i] = properties['Vb']
            self.omega[i] = properties['omega']
            self.kh_0[i] = properties['kh_0']
            self.neg_dH_solR[i] = properties['-dH_solR']
            if properties['nu_bar'] < 0.:
                # Use empirical equation from Jonas Gros
                self.nu_bar[i] = (1.148236984 * self.M[i] + 6.789136822) \
                                 / 100.**3
            else:
                self.nu_bar[i] = properties['nu_bar']
            
            if properties['B'] < 0.:
                self.B[i] = 5.0 * 1.e-2 / 100.**2.
            else:
                self.B[i] = properties['B']
            
            if properties['dE'] < 0.:
                self.dE[i] = 4000. / 0.238846
            else:
                self.dE[i] = properties['dE']
            
            if properties['K_salt'] < 0.:
                self.K_salt[i] = (-1.345 * self.M[i] + 2799.4 * 
                                 self.nu_bar[i] +  0.083556) / 1000.
            else:
                self.K_salt[i] = properties['K_salt']
        
        # If we are using group contribution method (Privat and Jaubert 2012) 
        # for the binary interaction matrix, then we must import Aij and Bij
        if delta_groups is not None:
            self.calc_delta = 1
            self.delta_groups = delta_groups
            aij_file = os.path.join(os.path.realpath(os.path.join(os.getcwd(), 
                       os.path.dirname(__file__), 'data')),'Aij.csv')
            bij_file = os.path.join(os.path.realpath(os.path.join(os.getcwd(), 
                       os.path.dirname(__file__), 'data')),'Bij.csv')
            self.Aij = np.loadtxt(aij_file, delimiter=',') * 1.e6 # Pa
            self.Bij = np.loadtxt(bij_file, delimiter=',') * 1.e6 # Pa
        
        else:
            self.calc_delta = -1
            self.delta_groups = np.zeros((self.nc, 15))
            self.Aij = np.zeros((15, 15))
            self.Bij = np.zeros((15, 15))
        
        # Specify that adequate information is contained in the object to 
        # run the solubility methods
        self.issoluble = True
        
        # Ideal gas constant
        self.Ru = 8.314510  # (J/(kg K))
        
        # Store whether or not the fluid is air
        self.air = air
    
    def masses(self, n):
        """
        Convert the moles of each component in a mixture to their masses (kg).
        
        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        
        """
        return n * self.M
    
    def mass_frac(self, n):
        """
        Calculate the mass fraction (--) from the number of moles of each 
        component in a mixture.
        
        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        Returns
        -------
        mf : ndarray, size (nc)
            mass fractions of each component in a mixture (--)
        
        """
        m = self.masses(n)
        return m / np.sum(m)
    
    def moles(self, m):
        """
        Convert the masses of each component in a mixture to their moles 
        (mol).
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        
        Returns
        -------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        """
        return m / self.M
    
    def mol_frac(self, m):
        """
        Calcualte the mole fraction (--) from the masses of each component in 
        a mixture.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)        
        
        Returns
        -------
        yk : ndarray, size (nc)
            mole fraction of each component in a mixture (--)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.mole_fraction(m, self.M)
    
    def partial_pressures(self, m, P):
        """
        Compute the partial pressure (Pa) of each component in a mixture.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        Pk : ndarray, size (nc)
            partial pressures of each component in a mixture (Pa)
        
        """
        yk = self.mol_frac(m)
        return P * yk
    
    def density(self, m, T, P):
        """
        Compute the gas and liquid density (kg/m^3) of a fluid mixture at the 
        given state.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        rho_p : ndarray, size (2,1)
            density of the gas phase (row 1) and liquid phase (row 2) of a 
            fluid mixture (kg/m^3)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.density(T, P, m, self.M, self.Pc, self.Tc, self.Vc, 
                             self.omega, self.delta, self.Aij, self.Bij, 
                             self.delta_groups, self.calc_delta)
    
    def fugacity(self, m, T, P):
        """
        Compute the gas and liquid fugacity (Pa) of a fluid mixture at the 
        given state.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        fk : ndarray, size (2, nc)
            fugacity coefficients of the gas phase (row 1) and liquid phase
            (row 2) for each component in a mixture (Pa)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.fugacity(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                              self.delta, self.Aij, self.Bij, 
                              self.delta_groups, self.calc_delta)
    
    def viscosity(self, m, T, P):
        """
        Computes the dynamic viscosity of the gas/liquid mixture.
        
        Computes the dynamic viscosity of gas and liquid using correlation
        equations in Pedersen et al. (2014) "Phase Behavior of Petroleum
        Reservoir Fluids", 2nd edition, chapter 10.  This function has been
        tested for non-hydrocarbon mixtures (oxygen, carbon dioxide) and 
        shown to give reasonable results; hence, the same equations are used
        for all mixtures.
        
        Parameters size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        mu_p : ndarray, size (2)
            dynamic viscosity for gas (row 1) and liquid (row 2) (Pa s)
        
        """
        return dbm_f.viscosity(T, P, m, self.M, self.Pc, self.Tc, self.Vc, 
                               self.omega, self.delta, self.Aij, self.Bij, 
                               self.delta_groups, self.calc_delta)
    
    def interface_tension(self, m, T, S, P):
        """
        Computes the interfacial tension between gas/liquid and water
        
        If `air` is False (thus, assume hydrocarbon), this method computes
        the interfacial tension between the gas and liquid phases of the
        mixture and water using equations in Danesh (1998) "PVT and Phase
        Behaviour Of Petroleum Reservoir Fluids," Chapter 8.  Otherwise, we
        treat the fluid like air and use the surface tension of seawater from
        Sharqawy et al. (2010), Table 6.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        sigma_p : ndarray, size (2)
            interfacial tension for gas (row 1) and liquid (row 2) (N/m)
        
        """
        if self.air:
            # Use the surface tension of seawater with air
            sigma_0 = seawater.sigma(T, S)
            sigma = np.array([[sigma_0], [sigma_0]])
            
        else:
            # Compute the local density of water
            rho_w = seawater.density(T, S, P)
            
            # Compute the density of the mixture phases
            rho_p = FluidMixture.density(self, m, T, P)
            
            # Get the density difference in g/cm^3
            delta_rho = (rho_w - rho_p) / 1000.
            
            # Compute the pseudo critical temperature using mole fractions as 
            # weights
            xi = self.mol_frac(m)
            Tc = np.sum(self.Tc * xi)
            
            # Get the interfacial tension
            sigma = 0.111 * delta_rho**1.024 * (T / Tc)**(-1.25) 
        
        # Return the Interfacial tension
        return sigma
    
    def equilibrium(self, m, T, P, K=None):
        """
        Computes the equilibrium composition of a gas/liquid mixture.
        
        Computes the equilibrium composition of a gas/liquid mixture using the
        procedure in Michelson and Mollerup (2007) and McCain (1990).  
        
        Parameters
        ----------
        dbm_obj : dbm.FluidMixture or dbm.FluidParticle
            DBM FluidMixture or FluidParticle object
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        K : ndarray, size (nc)
            array of partition coefficients to use as an initial guess for
            K-factor.  Default is `None`, in which case the standard initial
            guess will be used.
        
        Returns
        -------
        A tuple containing:
        
            m : ndarray, size(2, nc)
                masses of each component in a mixture at equilibrium between 
                the gas and liquid phases (kg)
            xi : ndarray, size(2, nc)
                mole fractions of each component in the mixture at equilibrium 
                between the gas and liquid phases (--)
            K : ndarray, size(nc)
                partition coefficients expressed as K-factor (--)
        
        Notes
        -----
        Uses the function equil_MM which uses the Michelsen and Mollerup (2007)
        procedure to find a stable solution
        
        """
        # Get the mole fractions and K-factors at equilibrium
        xi, beta, K = equil_MM(m, T, P, self.M, self.Pc, self.Tc,
                               self.omega, self.delta, self.Aij, 
                               self.Bij, self.delta_groups,
                               self.calc_delta, K)
        
        # Get the total moles of each molecule (both phases together)
        n_tot = self.moles(m)
        
        # Get the total number of moles in gas phase using the first 
        # component in the mixture (note that this is independent of 
        # which component you pick):
        ng = np.abs((n_tot[0] - (xi[1,0] * np.sum(n_tot)))/(xi[0,0]-xi[1,0]))
        
        # Get the moles of each component in gas (line 1) and liquid (line 2) 
        # phase
        n = np.zeros((2, self.nc))
        n[0,:] = xi[0,:] * ng
        n[1,:] = xi[1,:] * (np.sum(n_tot) - ng)
        
        # Finally converts to mass
        m = np.zeros((2, self.nc))
        for i in range(2):
            m[i,:] = self.masses(n[i,:])
        
        return (m, xi, K)
    
    def solubility(self, m, T, P, Sa):
        """
        Compute the solubility (kg/m^3) of each component of a mixture in both
        gas and liquid dissolving into seawater.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        Sa : float
            salinity of the ambient seawter (psu)
        
        Returns
        -------
        Cs : ndarray, size (2, nc)
            solubility of the gas phase (row 1) and liquid phase (row 2) for 
            each component in a mixture (kg/m^3)
        
        Notes
        -----
        It is assumed that the mixture is at the same pressure as the ambient 
        seawater and that the temperature at the interface is that of the 
        mixture.
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Compute the Henry's law coefficients using the temperature of the
        # seawater
        kh = dbm_f.kh_insitu(T, P, Sa, self.kh_0, self.neg_dH_solR, 
                             self.nu_bar, self.M, self.K_salt)
        
        # Compute the mixture fugacity using the temperature of the mixture
        f = FluidMixture.fugacity(self, m, T, P)
        
        # Get the solubility of each phase separately
        Cs = np.zeros((2,self.nc))
        Cs[0,:] = dbm_f.sw_solubility(f[0,:], kh)
        Cs[1,:] = dbm_f.sw_solubility(f[1,:], kh)
        return Cs
    
    def diffusivity(self, Ta, Sa, P):
        """
        Compute the diffusivity (m^2/s) of each component of a mixture into 
        seawater at the given temperature.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        Ta : float
            temperature of ambient seawater (K)
        Sa : float
            salinity of ambient seawater (psu)
        P : float
            pressure of ambient seawater (Pa)
        
        Returns
        -------
        D : ndarray, size (nc)
            diffusion coefficients for each component of a mixture into 
            seawater (m^2/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Compute the viscosity of seawater
        mu = seawater.mu(Ta, Sa, P)
        
        # Return the diffusivities
        return dbm_f.diffusivity(mu, self.Vb)
    
    def hydrate_stability(self, m, P):
        """
        Compute the hydrate formation temperature at the given pressure
        
        Use the K_vsi method from Sloan and Koh (2008) to compute the hydrate
        formation/dissociation temperature at the given pressure.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        P : float
            ambient pressure (Pa)
        
        Returns
        -------
        T_hyd : float
            critical hydrate stability temperature (K)
        
        Notes
        -----
        This method relys on the data fitted in Equation 4-1 in Sloan and Koh
        (2008), which is over a restricted range of temperature and pressure.
        In particular, when the pressure is outside (usually lower than) the 
        range of data, the model can predict spurious results.
        
        TODO (S. Socolofsky, October 1, 2013):  Get the original papers and 
        understand the limits of the range of applicability of this model.  
        Use this understanding to put bounds on the computation and ensure
        that accurate results are always returned.
        
        """
        # Get the mole fraction of the gas components in the K_vsi model in 
        # the order assumed in the model.
        gases = ['methane', 'ethane', 'propane', 'i-butane', 'n-butane', 
                 'nitrogen', 'carbon_dioxide', 'hydrogen_sulfide']
        m_gases = np.zeros(len(gases))
        for i in range(len(gases)):
            if gases[i] in self.composition:
                m_gases[i] = m[self.composition.index(gases[i])]
        
        def residual(T_hyd):
            """
            Compute the residual of sum (yi / K_vsi) - 1.
            
            Computes the residual of Equation 4-3 in Sloan and Koh for use
            in obtaining the critical formation temperature using a root-
            finding algorithm.
            
            Parameters
            ----------
            T_hyd : float
                current guess for the hydrate formation temperature (K)
            
            Returns
            -------
            res : float
                difference between sum (yi / K_vsi) - 1.
            
            """
            # Get the current value of the partition coefficients
            (K_vsi, yk) = dbm_f.kvsi_hydrate(T_hyd, P, m_gases)
            
            return np.sum(yk / K_vsi) - 1
        
        # Use root-finding to get the critical formation pressure
        T_hyd = fsolve(residual, 283.15)
        
        # Return the formation temperature
        return T_hyd


class FluidParticle(FluidMixture):
    """
    Class object for a soluble fluid particle
    
    This object defines the behavior of a soluble fluid particle.  The object
    inherits the internal variables and methods from the `FluidMixture` 
    object, but limits the output to a single phase, defined by the internal 
    variable `fp_type`.  It further extends the `FluidMixture` class to 
    include the properties inherent to particles (e.g., shape, diameter, slip 
    velocity, etc.).  
    
    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    fp_type : integer
        Defines the fluid type (0 = gas, 1 = liquid) that is expected to be 
        contained in the bubble.  This is needed because the Peng-Robinson
        equation of state returns values for both phases of a mixture.  This
        variable allows the class to automatically return the values for the
        desired phase.
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    user_data : dict
        A dictionary of chemical property data.  If not specified, the data
        loaded from `/tamoc/data/ChemData.csv` by ``chemical_properties`` will
        be used.  To load a different properties database, use the 
        ``chemical_properties.load_data`` function to load in a custom 
        database, and pass that data to this object as `user_data`.
    delta_groups : ndarray (nc, 15)
        Provides the group contribution numbers (normalized) for each 
        component in the mixture for the 15 groups used by the Privat and
        Jaubert (2012) group contribution method for binary interaction 
        coefficients.  Default is None, in which case the values in `delta`
        will be used.
    air : bool
        Flag indicating whether or not fluid is air.  The methods for 
        viscosity and interfacial tension below use correlations developed
        for hydocarbons.  If `air` is False (default value), these built
        in methods are used.  If `air` is True, then these methods are 
        replaced with correlations between air and seawater.
    
    Notes
    -----
    The attributes are identical to those defined for a `FluidMixture`
    
    See Also
    --------
    FluidMixture, chemical_properties, InsolubleParticle
    
    Examples
    --------
    >>> bub = FluidParticle(['nitrogen', 'oxygen'], fp_type=0)
    >>> yk = np.array([0.79, 0.21])
    >>> T = 273.15 + 30.
    >>> P = 10.e5
    >>> Sa = 35.
    >>> Ta = 273.15 + 20.
    >>> m = bub.masses_by_diameter(0.01, T, P, yk)
    array([  4.61873994e-06,   1.40243772e-06])
    >>> bub.density(m, T, P)
    11.499602249012074
    >>> bub.slip_velocity(m, T, P, Sa, Ta)
    0.22197023589052
    
    """
    def __init__(self, composition, fp_type=0., delta=None, user_data={},
                 delta_groups=None, air=False):
        super(FluidParticle, self).__init__(composition, delta, user_data,
                                            delta_groups, air)
        
        # Store the input variables
        self.fp_type = fp_type
    
    def density(self, m, T, P):
        """
        Compute the particle density (kg/m^3) of the fluid in the phase given 
        by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in the particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        rho_p : float
            density of the fluid particle (kg/m^3)
        
        Notes
        -----
        Uses the density method in the `FluidMixture` object, but only returns
        the value for the phase given by `fp_type`.
        
        """
        return FluidMixture.density(self, m, T, P)[self.fp_type, 0]
    
    def fugacity(self, m, T, P):
        """
        Compute the particle fugacities (Pa) of the fluid in the phase given 
        by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        fk : ndarray, size (nc)
            fugacities of each component of the fluid particle (Pa)
        
        Notes
        -----
        Uses the fugacity method in the `FluidMixture` object, but only 
        returns the values for the phase given by `fp_type`.
        
        """
        return FluidMixture.fugacity(self, m, T, P)[self.fp_type, :]
    
    def viscosity(self, m, T, P):
        """
        Computes the dynamic viscosity of the fluid in the phase given by 
        `fp_type`
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        mu_p : float
            dynamic viscosity (Pa s)
        
        Notes
        -----
        Uses the density method in the `FluidMixture` object, but only returns
        the value for the phase given by `fp_type`.
        
        """
        return FluidMixture.viscosity(self, m, T, P)[self.fp_type, 0]
    
    def interface_tension(self, m, T, S, P):
        """
        Computes the interfacial tension between the particle and water
        
        Computes the interfacial tension between the particle and water.  This
        method uses equations in Danesh (1998).
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in the particle (kg)
        T : float
            particle temperature (K)
        S : float
            salinity of the ambient seawter (psu)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        sigma_p : float
            interfacial tension (N/m)
        
        Notes
        -----
        Uses the density method in the `FluidMixture` object, but only returns
        the value for the phase given by `fp_type`.
        
        """
        return FluidMixture.interface_tension(self, m, T, S, P)[self.fp_type, 0]
    
    def solubility(self, m, T, P, Sa):
        """
        Compute the solubility (kg/m^3) of each component of a particle into 
        seawater for the phase given by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        S : float
            salinity of the ambient seawter (psu)
        
        Returns
        -------
        Cs : ndarray, size (nc)
            solubilities of each component of the fluid particle into 
            seawater (kg/m^3)
        
        Notes
        -----
        It is assumed that the mixture is at the same pressure as the ambient 
        seawater and that the temperature at the interface is that of the 
        particle.
        
        Uses the solubility method in the `FluidMixture` object, but only 
        returns the values for the phase given by `fp_type`.
        
        """
        return FluidMixture.solubility(self, m, T, P, Sa)[self.fp_type, :]
    
    def masses_by_diameter(self, de, T, P, yk):
        """
        Find the masses (kg) of each component in a particle with equivalent 
        spherical diameter `de` and mole fractions `yk`.
        
        Parameters
        ----------
        de : float
            equivalent spherical diameter (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        yk : ndarray, size (nc)
            mole fractions of each component in the particle (--)
        
        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a fluid particle (kg)
        
        """
        if isinstance(yk, list):
            yk = np.array(yk)
        
        # Get the masses for one mole of fluid
        m = self.masses(yk)
        
        # Compute the actual total mass of the fluid particle using the 
        # density computed from one mole of fluid
        m_tot = 1.0 / 6.0 * np.pi * de**3 * self.density(m, T, P)
        
        # Determine the number of moles in the fluid particle
        n = yk * m_tot / np.sum(m)
        
        return self.masses(n)
    
    def diameter(self, m, T, P):
        """
        Compute the equivalent spherical diameter (m) of a single fluid 
        particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        de : float
            equivalent spherical diameter of a fluid particle (m)
        
        """
        return (6.0 * np.sum(m) / (np.pi * self.density(m, T, P)))**(1.0/3.0)
    
    def particle_shape(self, m, T, P, Sa, Ta):
        """
        Determine the shape of a fluid particle from the properties of the 
        particle and surrounding fluid.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        A tuple containing:
            
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            rho : float
                ambient seawater density (kg/m^3)
            mu : float
                ambient seawater dynamic viscosity (Pa s)
            mu_p : float
                dispersed phase dynamic viscosity (Pa s)
            sigma : float
                interfacial tension (N/m)
        
        Notes
        -----
        As for the solubility calculation, we use the particle temperature to 
        calculate properties at the interface (e.g., to calculate the 
        interfacial tension) and the ambient temperature for properties of 
        the bulk continuous phase (e.g., density and viscosity).
        
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Compute the fluid particle and ambient properties
        de = self.diameter(m, T, P)
        rho_p = self.density(m, T, P)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        mu_p = self.viscosity(m, T, P)
        sigma = self.interface_tension(m, T, Sa, P)
        
        shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        
        return (shape, de, rho_p, rho, mu_p, mu, sigma)
    
    def slip_velocity(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the slip velocity (m/s) of a fluid particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        us : float
            slip velocity of the fluid particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 1:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
        
        return us
    
    def surface_area(self, m, T, P, Sa, Ta):
        """
        Compute the surface area (m^2) of a fluid particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        A : float
            surface area of the fluid particle (m^2)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 3:
            # Compute the surface area of a spherical cap bubble
            us = self.slip_velocity(m, T, P, Sa, Ta)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
        else:
            # Compute the area of the equivalent sphere:
            A = np.pi * de**2
        
        return A
    
    def mass_transfer(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the mass transfer coefficients (m/s) for each component in a 
        fluid particle
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        beta : ndarray, size (nc)
            mass transfer coefficient for each component in a fluid particle
            (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.  This method
        checks for hydrate stability and returns a reduced mass transfer 
        coefficient when hydrate shells are predicted to be present.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta, status)
        
        # Get the diffusivities
        D = self.diffusivity(Ta, Sa, P)
        
        # Compute the appropriate mass transfer coefficients
        if shape == 1:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 
                                     self.fp_type, status)
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 
                                        self.fp_type, status)
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D, status)
        return beta
    
    def heat_transfer(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the heat transfer coefficient (m/s) for a fluid particle
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        beta_T : float
            heat transfer coefficient for a fluid particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Get the thermal conductivity of seawater
        k = seawater.k(Ta, Sa, P) / (seawater.density(Ta, Sa, P) * 
            seawater.cp())
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta, status)
        
        # Compute the appropriate heat transfer coefficients.  Assume the 
        # heat transfer has the same form as the mass transfer with the 
        # diffusivity replaced by the thermal conductivity
        if shape == 1:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, 
                                     self.fp_type, status)
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, 
                                        self.fp_type, status)
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k, status)
        
        return beta
    
    def return_all(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute all of the dynamic properties of the bubble in an efficient
        manner (e.g., minimizing replicate calls to functions).
        
        This method repeats the calculations in the individual property 
        methods, and does not call the methods already defined.  This is done
        so that multiple calls to functions (e.g., slip velocity) do not 
        occur.  
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        A tuple containing:
            
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            us : float
                slip velocity (m/s)
            A : float 
                surface area (m^2)
            Cs : ndarray, size (nc)
                solubility (kg/m^3)
            beta : ndarray, size (nc)
                mass transfer coefficient (m/s)
            beta_T : float
                heat transfer coefficient (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.  This method
        checks for hydrate stability and returns a reduced mass transfer 
        coefficient when hydrate shells are predicted to be present.
        
        """
        # Ambient properties of seawater
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        sigma = self.interface_tension(m, T, Sa, P)
        D = dbm_f.diffusivity(mu, self.Vb)
        k = seawater.k(Ta, Sa, P) / (rho * seawater.cp())
        
        # Particle density, equivalent diameter and shape
        rho_p = dbm_f.density(T, P, m, self.M, self.Pc, self.Tc, self.Vc, 
                              self.omega, self.delta, self.Aij, self.Bij, 
                              self.delta_groups, 
                              self.calc_delta)[self.fp_type, 0]
        de = (6.0 * np.sum(m) / (np.pi * rho_p))**(1.0/3.0)
        shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        
        # Other particle properties
        mu_p = self.viscosity(m, T, P)
                
        # Solubility
        f = dbm_f.fugacity(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                           self.delta, self.Aij, self.Bij, 
                           self.delta_groups, self.calc_delta)
        kh = dbm_f.kh_insitu(T, P, Sa, self.kh_0, self.neg_dH_solR, 
                             self.nu_bar, self.M, self.K_salt)
        Cs = dbm_f.sw_solubility(f[self.fp_type,:], kh)
        
        # Check hydrate stability
        K_hyd = 1.
        T_hyd = self.hydrate_stability(m, P)
        if T < T_hyd:
            K_hyd = 1.0
        
        # Shape-specific properties
        if shape == 1:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
            A = np.pi * de**2
            beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 
                                     self.fp_type, status)
            beta_T = dbm_f.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, 
                                       self.fp_type, status)[0]
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
            A = np.pi * de**2
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 
                                        self.fp_type, status)
            beta_T = dbm_f.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, 
                                          self.fp_type, status)[0]
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D, status)
            beta_T = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k, 
                                              status)[0]
        
        return (shape, de, rho_p, us, A, Cs, K_hyd * beta, beta_T)
    

class InsolubleParticle(object):
    """
    Class object for an insoluble (inert) fluid particle
    
    This object defines the behavior of an inert fluid particle.  The purpose
    of this class is to simulate particle that cannot be described by the 
    Peng-Robinson equation of state, such as sand, or that do not require
    such computational expense, such as dead oil.
    
    Parameters
    ----------
    isfluid : logical
        `True` or `False`; states whether or not the inert particle could have 
        a mobile interface.  For example, choose `True` for oil and `False` 
        for sand.
    iscompressible : logical
        `True` or `False`; selects the equation of state for density.  `True` 
        uses the API gravity, isothermal compression and isobaric thermal 
        expansion; whereas, `False` returns the constant density specified at 
        instantiation.
    rho_p : float
        particle density (default value is 930 kg/m^3)
    gamma : float
        API gravity (default value is 30 deg API)
    beta : float
        thermal expansion coefficient (default value is 0.0007 K^(-1))
    co : float
        isothermal compressibility coefficient (default value is 
        2.90075e-9 Pa^(-1))
    fp_type : integer
        Defines the fluid type (0 = gas, 1 = liquid) that is expected to be 
        contained in the particle.  This is needed because the heat transfer
        equations are different for gas and liquid.  The default value is 1.
    air : bool
        Flag indicating whether or not fluid is air.  The methods for 
        viscosity and interfacial tension below use correlations developed
        for hydocarbons.  If `air` is False (default value), these built
        in methods are used.  If `air` is True, then these methods are 
        replaced with correlations between air and seawater.
    
    Attributes
    ----------
    composition : string list
        Set equal to ['inert']
    nc : integer
        Number of components, set equal to 1
    issoluble : logical, False
        Indicates the particle is not soluble
    
    See Also
    --------
    FluidMixture, chemical_properties, FluidParticle
    
    Examples
    --------
    >>> oil = InsolubleParticle(True, True)
    >>> T = 273.15 + 30.
    >>> P = 10.e5
    >>> Sa = 35.
    >>> Ta = 273.15 + 20.
    >>> m = oil.mass_by_diameter(0.01, T, P, Sa, Ta)
    0.00045487710681354078
    >>> oil.density(T, P, Sa, Ta)
    868.75128058458085
    >>> oil.slip_velocity(m, T, P, Sa, Ta)
    0.14332887200025926
    
    """
    def __init__(self, isfluid, iscompressible, rho_p=930., gamma=30., 
                 beta=0.0007, co=2.90075e-9, fp_type=1, air=False):
        super(InsolubleParticle, self).__init__()
        
        # Store the input variables
        self.isfluid = isfluid
        self.iscompressible = iscompressible
        self.rho_p = rho_p
        self.gamma = gamma
        self.beta = beta
        self.co = co
        self.fp_type = fp_type
        
        # Specify that the particle is not soluble and is therefore treated
        # like a single substance and store whether or not the fluid is 
        # like air
        self.issoluble = False
        self.nc = 1
        self.composition = ['inert']
        self.air = air
    
    def density(self, T, P, Sa, Ta):
        """
        Compute the density (kg/m^3) of an inert fluid particle.
        
        If the particle is compressible, this method computes the particle 
        density following McCain (1990), using the API gravity (ee pages 224 
        and following in McCain).  This would be typical of an oil.
        
        Otherwise, the method returns the constant density stored in the 
        internal variable `rho_p`.
        
        Parameters
        ----------
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        rho_p : float
            density of the inert particle (kg/m^3)
        
        """
        if self.iscompressible:
            
            # Set the standard conditions for the API gravity
            P_stp = 101325.0
            T_stp = 273.15 + (60.0 - 32.0) * 5.0 / 9.0
            
            # Get the density of water at standard conditions
            rho_stp = seawater.density(T_stp, 0., P_stp)
            
            # Use the API gravity equation (8-2) in McCain (1990) to get the
            # fluid particle density at standard conditions
            gamma_0 = 141.5 / (self.gamma + 131.5)
            rho_p = gamma_0 * rho_stp
            
            # Isothermal compression to in-situ pressure using equation (8-19) 
            # in McCain (1990).
            rho_p = rho_p * np.exp(self.co * (P - P_stp))
            
            # Isobaric compression to oil temperature using equation (8-28) in
            # McCain (1990).
            rho_p = rho_p * (1 - self.beta * (T - T_stp))
        
        else:
            rho_p = self.rho_p
        
        return rho_p
    
    def viscosity(self, T):
        """
        Computes the dynamic viscosity of the liquid if applicable.
        
        Computes the dynamic viscosity of gas and liquid using correlation 
        equations in McCain (1990).  This is the only method we can use
        since an `InsolubleParticle` has not `composition`.  If solid, the
        viscosity is returned as infinite.
        
        Parameters
        ----------
        T : float
            mixture temperature (K)
        
        Returns
        -------
        mu_p : ndarray, size (2)
            dynamic viscosity (Pa s)
        
        """
        if self.isfluid:
            # Use equation B-53 for dead oil in McCain (1990)
            TF = (T - 273.15) * 9.0 / 5.0 + 32.0
            mu = (10. ** (10. ** (1.8653 - 0.025086 * self.gamma - 
                 0.5644 * np.log10(TF))) - 1.0) / 1000.
        else:
            # Particle is solid; thus, viscosity is infinite
            mu = np.inf
        
        return mu
    
    def interface_tension(self, T):
        """
        Computes the interfacial tension between the particle and water
        
        Computes the interfacial tension between a fluid particle and water.
        Since for `InsolubleParticle` there is no `composition` we have very
        little to go on.  This function currently returns the surface 
        tension of seawater.  If solid, the surface tension is returned as
        infinite.
        
        Parameters
        ----------
        T : float
            particle temperature (K)
        
        Returns
        -------
        sigma_p : float
            interfacial tension (N/m)
        
        Notes
        -----
        Returns the air-water interfacial tension for lack of any better
        knowledge about this compound
        
        """
        if self.isfluid:
            S = 34.5
            sigma = seawater.sigma(T, S)
            
        else:
            sigma = np.inf
        
        return sigma
    
    def mass_by_diameter(self, de, T, P, Sa, Ta):
        """
        Compute the mass (kg) of an inert fluid particle with equivalent 
        spherical diameter `de`.
        
        Parameters
        ----------
        de : float
            equivalent spherical diameter (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        m : float
            mass of the fluid particle (kg)
        
        """
        return 1.0 / 6.0 * np.pi * de**3 * self.density(T, P, Sa, Ta)
    
    def diameter(self, m, T, P, Sa, Ta):
        """
        Compute the diameter (m) of an inert fluid particle of mass `m`.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        de : diameter of the fluid particle (m)
        
        """
        return (6.0 * m / (np.pi * self.density(T, P, Sa, Ta)))**(1.0/3.0)
    
    def particle_shape(self, m, T, P, Sa, Ta):
        """
        Determine the shape of an inert fluid particle from the properties of 
        the particle and surrounding fluid.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        A tuple containing:
        
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap, 4 - rigid
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            rho : float
                ambient seawater density (kg/m^3)
            mu : float
                ambient seawater dynamic viscosity (Pa s)
            mu_p : float
                dispersed phase dynamic viscosity (Pa s)
            sigma : float
                interfacial tension (N/m)
        
        Notes
        -----
        We use the particle temperature to calculate properties at the 
        interface (e.g., to calculate the interfacial tension) and the 
        ambient temperature for properties of the bulk continuous phase 
        (e.g., density and viscosity).
        
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Compute the fluid particle properties
        de = self.diameter(m, T, P, Sa, Ta)
        rho_p = self.density(T, P, Sa, Ta)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        mu_p = self.viscosity(T)
        sigma = self.interface_tension(T)
        
        # Compute the particle shape
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        return (shape, de, rho_p, rho, mu_p, mu, sigma)
    
    def slip_velocity(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the slip velocity (m/s) of an inert fluid particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        us : float
            slip velocity of the inert particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 1 or shape == 4:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
        
        return us
    
    def surface_area(self, m, T, P, Sa, Ta):
        """
        Compute the surface area (m^2) of an inert fluid particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        A : float
            surface area of the inert particle (m^2)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 3:
            # Compute the surface area of a spherical cap bubble
            us = self.slip_velocity(m, T, P, Sa, Ta)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
        else:
            # Compute the area of the equivalent sphere:
            A = np.pi * de**2
        
        return A
    
    def heat_transfer(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the heat transfer coefficients (m/s) for an inert fluid 
        particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        beta_T : float
            heat transfer coefficient for the inert particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
       """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Get the thermal conductivity of seawater
        k = seawater.k(Ta, Sa, P) / (seawater.density(Ta, Sa, P) * 
            seawater.cp())
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta, status)
        
        # Compute the appropriate heat transfer coefficients.  Assume the 
        # heat transfer has the same form as the mass transfer with the 
        # diffusivity replaced by the thermal conductivity
        if shape == 1 or shape == 4:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, 
                                     self.fp_type, status)
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, 
                                        self.fp_type, status)
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k, status)
        
        return beta
    
    def return_all(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute all of the dynamic properties of an inert fluid particle in 
        an efficient manner (e.g., minimizing replicate calls to functions).
        
        This method repeats the calculations in the individual property 
        methods, and does not call the methods already defined.  This is done
        so that multiple calls to functions (e.g., slip velocity) do not 
        occur.  
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
        Returns
        -------
        A tuple containing:
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap, 4 - rigid
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            us : float
                slip velocity (m/s)
            A : float
                surface area (m^2)
            beta_T : float
                heat transfer coefficient (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Ambient properties of seawater
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        sigma = self.interface_tension(T)
        k = seawater.k(Ta, Sa, P) / (rho * seawater.cp())
        
        # Particle density, equivalent diameter and shape
        rho_p = self.density(T, P, Sa, Ta)
        de = (6.0 * m / (np.pi * rho_p))**(1.0/3.0)
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        # Other particle properties
        mu_p = self.viscosity(T)
        
        # Shape-specific properties
        if shape == 1 or shape == 4:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
            A = np.pi * de**2
            beta_T = dbm_f.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, 
                                       self.fp_type, status)[0]
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
            A = np.pi * de**2
            beta_T = dbm_f.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, 
                                          self.fp_type, status)[0]
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
            beta_T = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, 
                                              k, status)[0]
        
        return (shape, de, rho_p, us, A, beta_T)
    

# ----------------------------------------------------------------------------
# Functions used by classes to compute gas/liquid equilibrium of a mixture
# ----------------------------------------------------------------------------

def equil_MM(m, T, P, M, Pc, Tc, omega, delta, Aij, Bij, delta_groups, 
             calc_delta, K_0):
    """
    Compute the equilibrium composition of a mixture using the P-R EOS
    
    Computes the mole fraction composition for the gas and liquid phases of a
    mixture using the Peng-Robinson equation of state and the methodology
    described Michelsen and Mollerup (2007).  For multiphase equilibria, 
    the successive substition method is used.  If several iterations suggest
    a single-phase equilibrium, stability analysis is used to verify the
    prediction.  If a two-phase result is predicted by stability analysis, 
    successive substitution continues with an improved estimate for the 
    composition; otherwise, the single phase result is returned.
    
    Parameters
    ----------
    m : ndarray, size (nc)
        masses of each component present in the whole mixture (gas plus 
        liquid, kg)
    T : float
        temperature (K)
    P : float
        pressure (Pa)
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  
    Aij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    Bij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    delta_groups : ndarray, (nc, 15)
        Specification of the fractional groups for each component of the 
        mixture for the group contribution method of Privat and Jaubert (2012)
        for delta_ij
    calc_delta : int
        Flag specifying whether or not to compute delta_ij (1: True, -1: 
        False) using the group contribution method
    K_0 : ndarray, size (nc)
        Initial guess for the partition coefficients.  If K = None, this 
        function will use initial estimates from Wilson (see Michelsen and
        Mollerup, 2007, page 259, equation 26)
    
    Returns
    -------
    xi : ndarray, size(2, nc)
        Mole fraction of each component in the mixture.  Row 1 gives the
        values for the gas phase and Row 2 gives the values for the liquid 
        phase (--)
    
    Notes
    -----
    The method estimates the K-factors giving the mole fractions in gas and
    liquid and optimizes the K-factor estimates until they converge on the
    ratio of the actual fugacity coefficients (phi_liq / phi_gas).
    Convergence uses a squared relative error as in McCain (1990).
    
    """
    # Compute the some constant properties of the mixture
    moles = m / M
    zi = moles / np.sum(moles)
    f_zi = dbm_f.fugacity(T, P, zi*M, M, Pc, Tc, omega, delta, 
                          Aij, Bij, delta_groups, calc_delta)[0,:]
    phi_zi = f_zi / (zi * P)
    di = np.log(zi) + np.log(phi_zi)
    
    # Compute the total Gibbs energy
    def gibbs_energy(K):
        """
        Compute the Gibbs energy difference between the feed and the current
        composition given by K using equation (41) on page 266
        
        """
        # Use the current K to compute the equilibrium
        xi, beta = gas_liq_eq(m, M, K)
        
        # Compute the fugacities of the new composition
        f_gas = dbm_f.fugacity(T, P, xi[0,:]*M, M, Pc, Tc, omega, delta, 
                               Aij, Bij, delta_groups, calc_delta)[0,:]
        f_liq = dbm_f.fugacity(T, P, xi[1,:]*M, M, Pc, Tc, omega, delta, 
                               Aij, Bij, delta_groups, calc_delta)[1,:]
        
        # Get the fugacity coefficients
        phi_gas = f_gas / (xi[0,:] * P)
        phi_liq = f_liq / (xi[1,:] * P)
        
        # Compute the reduced tangent plane distances
        tpdx = np.nansum(xi[1,:] * (np.log(xi[1,:]) + np.log(phi_liq) - di))
        tpdy = np.nansum(xi[0,:] * (np.log(xi[0,:]) + np.log(phi_gas) - di))
        
        # Compute the change in the total Gibbs energy between the feed 
        # and this present composition
        DG_RT = (1. - beta) * tpdx + beta * tpdy
        
        # Return the results
        return (DG_RT, tpdx, tpdy, phi_liq, phi_gas)
    
    # Get an initial estimate for the K-factors 
    if K_0 is None:
        # Use equation (26) on page 259 of Michelson and Mollerup (2007)
        K = np.exp(5.37 * (1. + omega) * (1 - Tc / T)) / (P / Pc)
    else:
        K = K_0
    
    # Follow the procedure on page 266ff of Michelsen and Mollerup (2007).    
    # Start with three iterations of successive substitution
    K, beta, xi, exit_flag = successive_substitution(
                                 m, T, P, 3, M, Pc, Tc, omega, delta, Aij, 
                                 Bij, delta_groups, calc_delta, K)
    
    # Test the outcome of the iterations to determine how to proceed.
    if exit_flag > 0:
        # The solution already converged.
        pass
        
    else:
        # The solution has not converged, test the total Gibbs energy to 
        # decide how to proceed.
        Delta_G_RT, tpdx, tpdy, phi_liq, phi_gas = gibbs_energy(K)
        
        if Delta_G_RT < 0.:
            # The current composition is converging on a lower total Gibbs
            # energy than the feed: continue successive substitution
            K, beta, xi, exit_flag = successive_substitution(
                                         m, T, P, np.inf, M, Pc, Tc, omega, 
                                         delta, Aij, Bij,  delta_groups, 
                                         calc_delta, K)
        
        elif tpdy < 0.:
            # The feed is unstable, but we need a better estimate of K
            K = phi_zi / phi_gas
            
            # Continue with successive substitution
            K, beta, xi, exit_flag = successive_substitution(
                                         m, T, P, np.inf, M, Pc, Tc, omega, 
                                         delta, Aij, Bij,  delta_groups, 
                                         calc_delta, K)
        
        elif tpdx < 0.:
            # The feed is unstable, but we need a better estimate of K
            K = phi_liq / phi_zi
            
            # Continue with successive substitution
            K, beta, xi, exit_flag = successive_substitution(
                                         m, T, P, np.inf, M, Pc, Tc, omega, 
                                         delta, Aij, Bij,  delta_groups, 
                                         calc_delta, K)
        
        else:
            # We are not sure of the stability of the feed:  do stability 
            # analysis.
            K, phases = stability_analysis(m, T, P, M, Pc, Tc, omega, delta, 
                                           Aij, Bij, delta_groups, 
                                           calc_delta, K, zi, di)
            if phases > 1:
                # The mixture is unstable, continue with successive 
                # substitution
                K, beta, xi, exit_flag = successive_substitution(
                                             m, T, P, np.inf, M, Pc, Tc, omega, 
                                             delta, Aij, Bij,  delta_groups, 
                                             calc_delta, K)
            else:
                # The mixture is single-phase
                xi = np.zeros((2,len(zi)))
                if beta > 0.5:
                    beta = 1.
                    xi[0,:] = zi
                else:
                    beta = 0.
                    xi[1,:] = zi
        
    # Return the optimized mixture composition
    return (xi, beta, K)


def stability_analysis(m, T, P, M, Pc, Tc, omega, delta, Aij, Bij, 
                       delta_groups, calc_delta, K, zi, di):
    """
    Perform stability analysis to determine the stability of a mixture
    
    Perform the stabilty analysis steps in Michelsen and Mollerup (2007) to
    determine the stability of a mixture
    
    Parameters
    ----------
    m : ndarray, size (nc)
        masses of each component present in the whole mixture (gas plus 
        liquid, kg)
    T : float
        temperature (K)
    P : float
        pressure (Pa)
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  
    Aij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    Bij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    delta_groups : ndarray, (nc, 15)
        Specification of the fractional groups for each component of the 
        mixture for the group contribution method of Privat and Jaubert (2012)
        for delta_ij
    calc_delta : int
        Flag specifying whether or not to compute delta_ij (1: True, -1: 
        False) using the group contribution method
    K : ndarray, size (nc)
        Initial guess for the partition coefficients.  If K = None, this 
        function will use initial estimates from Wilson (see Michelsen and
        Mollerup, 2007, page 259, equation 26)
    di : ndarray, size (nc)
        Mixture property ln(zi) + ln(phi(zi)); see Michelsen and Mollerup
        (2007) page 267
    
    Returns
    -------
    K : ndarray, size (nc)
        Updated estimate for the K factors after stability analysis
    phases : int
        Number of phases in the mixture (2 or 1)
    
    """
    # Compute the mole fraction of the total mixture (called the feed in 
    # Michelsen and Mollerup, 2007)
    moles = m / M
    zi = moles / np.sum(moles)
    
    # Generate the update equation for finding W that minizes tm
    def update_W(W, phase):
        """
        Update the estimate for W to minimize the modified tangent plane
        distance using equation (51) on page 269 in Michelsen and Mollerup
        (2007).
        
        Parameters
        ----------
        W : ndarray
            Current estimate for the composition (moles)
        phase : int
            Assumed phase of the current composition (0: gas, 1: liquid)
        
        Returns
        -------
        W : ndarray
            New estimate of W (moles)
        
        """
        # Compute the fugacity at the composition W
        f_W = dbm_f.fugacity(T, P, W*M, M, Pc, Tc, omega, delta, 
                             Aij, Bij, delta_groups, calc_delta)[phase,:]
        
        # Get the fugacity coefficients
        phi_W = f_W / (W / np.sum(W) * P)
        
        # Return a new estimate of W
        return np.exp(di - np.log(phi_W))
    
    # Compute the modified tangent plane distance
    def compute_tm(W, phase):
        """
        Compute the modified tangent plane distance according to equation (44)
        in Michelsen and Mollerup (2007) on page 267.
        
        Parameters
        ----------
        W : ndarray
            Current estimate for the composition (moles)
        phase : int
            Assumed phase of the current composition (0: gas, 1: liquid)
        
        Returns
        -------
        tm : float
            Value of the modified tangent plane distance for the given 
            composition
        
        """
        # Compute the fugacity at the composition W
        f_W = dbm_f.fugacity(T, P, W*M, M, Pc, Tc, omega, delta, 
                             Aij, Bij, delta_groups, calc_delta)[phase,:]
        
        # Get the fugacity coefficients
        phi_W = f_W / (W / np.sum(W) * P)
        
        # Return the modified tangent plane distance, equation (44) on page
        # 267
        return 1. + np.sum(W * (np.log(W) + np.log(phi_W) - di - 1.))
    
    # Solve for W that minimizes tm
    def find_W(W, phase):
        """
        Use successive subsitution to find a value of W that minimizes the 
        modified tangent plane distance and then interpret the stability 
        of the mixture based on the results
        
        Parameters
        ----------
        W : ndarray
            Current estimate for the composition (moles)
        phase : int
            Assumed phase of the current composition (0: gas, 1: liquid)
        
        Returns
        -------
        W : ndarray
            Final value of the composition (moles)
        tm : float
            Value of the modified tangent plane distance for the final
            composition
        phases : int
            Evaluation of the number of phases present (1 or 2)
        
        """
        # Set up the iteration parameters
        tol = 1.49012e-8  # Use same value as for K-factor iteration
        err = 1.
        
        # Iterate to find the final value of W
        while err > tol:
            # Save the current value of W
            W_old = W
            
            # Update the estimate of W using the update equation
            W = update_W(W, phase)
            
            # Compute the current error based on the squared relative error 
            # suggested by McCain (1990)
            err = np.nansum((W - W_old)**2 / (W * W_old))
        
        # Compute the modified tangent plane distance
        tm = compute_tm(W, phase)
        
        # Determine if we found a trivial solution
        trivial = True
        for i in range(len(W)):
            if np.abs(W[i] - zi[i]) > 1.e-5:
                trivial = False
        
        # Evaluate the stability of the outcome
        if tm < 0. and not trivial:
            phases = 2
        else:
            # This is a single-phase gas
            phases = 1
        
        # Return the results
        return (W, tm, phases)
    
    # First, do a test vapor-like composition
    W = K * zi
    W_gas, tm_gas, phases_gas = find_W(W, 0)
    K_gas = W_gas / (zi * np.sum(W_gas))
    
    # Second, to be conservative, do a test liquid-like composition
    W = zi / K
    W_liq, tm_liq, phases_liq = find_W(W, 1)
    K_liq = zi * np.sum(W_liq)/ W_liq
    
    if phases_gas > 1 and phases_liq > 1:
        if tm_gas < tm_liq:
            # This is probably a gas-like mixture
            K = K_gas
            phases = 2
        else:
            # This is probably a liquid-like mixture
            K = K_liq
            phases = 2
    elif phases_gas > 1:
        # This is proably a gas-like mixture
        K = K_gas
        phases = 2
    elif phases_liq > 1:
        # This is probably a liquid-like mixture
        K = K_liq
        phases = 2
    else:
        # This is a single-phase mixture
        K = np.ones(K.shape)
        phases = 1
    
    # Return the results
    return (K, phases)


def successive_substitution(m, T, P, max_iter, M, Pc, Tc, omega, delta, Aij, 
                            Bij, delta_groups, calc_delta, K):
    """
    Find K-factors by successive substitution
    
    Iterate to find a converged set of K-factors defining the gas/liquid
    partitioning of a mixture using successive substitution.  We follow the
    algorithms in McCain (1990) and Michelsen and Mollerup (2007).
    
    Parameters
    ----------
    m : ndarray, size (nc)
        masses of each component present in the whole mixture (gas plus 
        liquid, kg)
    T : float
        temperature (K)
    P : float
        pressure (Pa)
    max_iter : int
        maximum number of iterations to perform.  Set max_iter to np.inf if
        you want the algorithm to guarantee to iterate to convergenece, but 
        beware that you may create an infinite loop.
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  
    Aij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    Bij : ndarray, (15, 15)
        Coefficients in matrix A_ij for the group contribution method for 
        delta_ij following Privat and Jaubert (2012)
    delta_groups : ndarray, (nc, 15)
        Specification of the fractional groups for each component of the 
        mixture for the group contribution method of Privat and Jaubert (2012)
        for delta_ij
    calc_delta : int
        Flag specifying whether or not to compute delta_ij (1: True, -1: 
        False) using the group contribution method
    K : ndarray, size (nc)
        Initial guess for the partition coefficients.  If K = None, this 
        function will use initial estimates from Wilson (see Michelsen and
        Mollerup, 2007, page 259, equation 26)
    
    Returns
    -------
    K : ndarray, size (nc)
        Final value of the K-factors
    
    Notes
    -----
    The max_iter parameter controls how many steps of successive iteration 
    are performed.  If set to None, the iteration will continue until the 
    tolerance criteria are reached.
    
    """
    # Update the value of K using successive substitution
    def update_K(K):
        """
        Evaluate the update function for finding K-factor
        
        Evaluates the new guess for K-factor following McCain (1990) p. 426, 
        equation (15-23) as explained on p. 430 in connection with equation
        (15-31).  This is the update equation for the successive substitution 
        method.
        
        Parameters
        ----------
        T, P, m_0, M, Pc, Tc, omega, delta = constant and inherited
            from above
        K : ndarray
            The current guess for the K-factor (--)
        
        Returns
        -------
        K_new : ndarray
            New guess for K-factor
        
        """
        # Get the mixture composition for the current K-factor
        xi, beta = gas_liq_eq(m, M, K)
        
        # Get tha gas and liquid fugacities for the current composition
        f_gas = dbm_f.fugacity(T, P, xi[0,:]*M, M, Pc, Tc, omega, delta, 
                               Aij, Bij, delta_groups, calc_delta)[0,:]
        f_liq = dbm_f.fugacity(T, P, xi[1,:]*M, M, Pc, Tc, omega, delta, 
                               Aij, Bij, delta_groups, calc_delta)[1,:]
        
        # Update K using K = (phi_liq / phi_gas)
        K_new = (f_liq / (xi[1,:] * P)) / (f_gas / (xi[0,:] * P))
        
        # If the mass of any component in the mixture is zero, make sure the
        # K-factor is also zero.
        K_new[np.isnan(K_new)] = 0.
        
        # Return an updated value for the K factors
        return K_new
    
    # Set up the iteration parameters
    tol = 1.49012e-8  # Suggested by McCain (1990)
    err = 1.
    steps = 0
    
    # Iterate to find the final value of K factor using successive 
    # substitution
    while err > tol and steps < max_iter:
        # Save the current value of K factor
        K_old = K
        
        # Update the estimate of K factor using the present fugacities
        K = update_K(K)
        
        # Compute the current error basedo on the squared relative error 
        # suggested by McCain (1990) and update the iteration counter
        err = np.nansum((K - K_old)**2 / (K * K_old))
        steps += 1
    
    # Determine the exit condition
    if steps < max_iter:
        # This solution is converged
        flag = 1
    else:
        flag = 0
    
    # Update the equilibrium and return the last value of K-factor
    xi, beta = gas_liq_eq(m, M, K)
    return (K, beta, xi, flag)


def gas_liq_eq(m, M, K):
    """
    docstring for gas_liq_eq(m, M, K)
    
    This function follows the procedure in Michelsen and Mollerup (2007).  
    All page and equation numbers below are from this book unless otherwise
    noted.
    
    """
    # Compute the mole fraction of the total mixture (called the feed in 
    # Michelsen and Mollerup, 2007)
    moles = m / M
    zi = moles / np.sum(moles)
    
    # Define the Rachford-Rice equation for beta as gas fraction.  
    def g_gas(beta):
        """
        Computes the Rachford-Rice equation, which defines a root-finding 
        problem for the solution of beta, the gas mole fraction in a mixture.
        
        Parameters
        ----------
        zi, K = global variables defined in the main function containing this
            subfunction
        beta : float
            Fraction of moles of mixture in the gas phase, [0, 1]
        
        Returns
        -------
        g : float
            Value of the Rachford-Rice equation, the roots of which are the
            solution for beta.
        
        """
        # Equation (2) on page 252
        return np.sum(zi * (K - 1.) / (1. + beta * (K - 1.)))
    
    def g_gas_p(beta):
        """
        Computes the gradient of the Rachford-Rice equation, which defines a 
        root-finding problem for the solution of beta, the gas mole fraction 
        in a mixture.  This is used in Newton's method to solve for beta.
        
        Parameters
        ----------
        zi, K = global variables defined in the main function containing this
            subfunction
        beta : float
            Fraction of moles of mixture in the gas phase, [0, 1]
        
        Returns
        -------
        gp : float
            Value of the beta-derivative of the Rachford-Rice equation
        
        """
        # Equation (3) on page 252
        return -np.sum(zi * (K - 1.)**2 / (1. + beta * (K - 1.))**2)
    
    # Define the Rachford-Rice equation for beta_l as liquid fraction
    def g_liq(beta_l):
        """
        Computes the modified Rachford-Rice equation, which defines a root-
        finding problem for the solution of beta_l, the liquid mole fraction 
        in a mixture.
        
        Parameters
        ----------
        zi, K = global variables defined in the main function containing this
            subfunction
        beta_l : float
            Fraction of moles of mixture in the liquid phase, [0, 1]
        
        Returns
        -------
        g : float
            Value of the modified Rachford-Rice equation, the roots of which 
            are the solution for beta_l.
        
        """
        # Unlabeled equation at bottom of page 253
        return np.sum(zi * (K - 1.) / (K - beta_l * (K - 1.)))
    
    def g_liq_p(beta_l):
        """
        Computes the gradient of the modified Rachford-Rice equation, which 
        defines a root-finding problem for the solution of beta_l, the liquid
        mole fraction in a mixture.  This is used in Newton's method to solve 
        for beta_l.
        
        Parameters
        ----------
        zi, K = global variables defined in the main function containing this
            subfunction
        beta_l : float
            Fraction of moles of mixture in the liquid phase, [0, 1]
        
        Returns
        -------
        gp : float
            Value of the beta_l-derivative of the modified Rachford-Rice 
            equation
        
        """
        # beta_l-derivative of g_liq equation above
        return np.sum(zi * (K - 1.)**2 / (K - beta_l * (K - 1.))**2)    
    
    # Step i on page 253:  Check conditions of equations (4) and (5) on page
    # 252 for existence of a two-phase solution for beta.
    if np.sum(zi * K) - 1. <= 0.:
        # This is subcooled liquid, beta = 0.
        beta = 0.
        
    elif 1. - np.sum(zi / K) > 0.:
        # This is superheated gas, beta = 1.
        beta = 1.
        
    else:
        # This is a two-phase mixture, so search for a solution for beta.
        # Step ii on page 253:  Check equations (7) and (8) on page 253 for 
        # tighter bounds on the possible range of beta
        beta_min = 0.
        beta_max = 1.
        for i in range(len(K)):
            if K[i] >= 1.:
                # Apply equation (7) on page 253
                beta_min = np.max([beta_min, (K[i] * zi[i] - 1.) / 
                           (K[i] - 1.)])
            else:
                # Apply equation (8) on page 253
                beta_max = np.min([beta_max, (1. - zi[i]) / (1. - K[i])])
        
        # Step iii on page 254:  Select initial guess for beta and choose
        # which objective function to use.
        beta = 0.5 * (beta_min + beta_max)
        
        if g_gas(beta) > 0.:
            # Solution will have excess gas
            eqn = 1.
            beta_var = beta
            
        else:
            # Solution will have excess liquid
            eqn = 0.
            beta_var = 1. - beta
            beta_min_hold = beta_min
            beta_min = 1. - beta_max
            beta_max = 1. - beta_min_hold
        
        # Set up an iterative solution to find beta_var using the optimal
        # root-finding equation
        tol = 1.e-6
        err = 1.
        
        while err > tol:
            # Store the current value of beta
            beta_old = beta_var
            
            # Step iv on page 254:  Perform one iteration of Newton's method
            # and narrow the possible range of the solution for beta
            if eqn > 0.:
                # Use the equations for excess gas
                g = g_gas(beta_var)
                gp = g_gas_p(beta_var)
                beta_new = beta_var - g / gp
                
                # Update bounds on beta per criteria in step iv on page 254
                if g > 0:
                    beta_min = beta_var
                else:
                    beta_max = beta_var
            
            else:
                # Use the equations for excess liqiud
                g = g_liq(beta_var)
                gp = g_liq_p(beta_var)
                beta_new = beta_var - g / gp
                
                # Update bounds on beta per criteria in step iv on page 254
                if g > 0:
                    beta_max = beta_var
                else:
                    beta_min = beta_var
            
            # Step v on page 254:  Select best update for beta
            if beta_new <= beta_max and beta_new >= beta_min:
                # Newton's method is converging within allowable range for 
                # the independent variable:  use the Newton's method solution
                beta_var = beta_new
                
            else:
                # Newton's method suggests a solution outside the allowable
                # range for the independent variable:  use the bisection
                # method
                beta_var = 0.5 * (beta_min + beta_max)
            
            # Step vi on page 254:  Check for convergence.  Note:  do not
            # use relative error since beta_var ~= 0 is an acceptable answer
            err = np.abs(beta_var - beta_old)
        
        # Get the final value of beta, the gas mole fraction
        if eqn > 0.:
            # We found beta
            beta = beta_var
        else:
            # We found beta_l
            beta = 1. - beta_var
    
    # Return the solution for gas and liquid mole fractions based on the
    # converged value of beta
    return (np.array([zi * K / (1. + beta * (K - 1.)), 
                     zi / (1. + beta * (K - 1.))]), beta)

