"""
Particle Size Models
====================

Compute particle size distributions for jets of oil and gas

This module provides an interface to available empirical models for bubble
and droplet size distributions for jet releases.  The empirical model
functions are in `psf.py`; this module provides an object-oriented interface
to these functions.  Gas bubble size distributions can be created from:

* Li et al. (2017)
* Wang et al. (2018)

Oil droplet size distributions can be created from:

* Johansen et al. (2013)
* Li et al. (2017)

All models support log-normal and Rosin-Rammler distributions.

Notes
-----

The particle size computational algorithms are contained in the module

    `psf.py` - Particle Size Fuctions

This module is a function library used by the objects in this module.  In
general, the `psf` module can be replace by any module having the same
application programming interface, whether programmed in Python, or another
language and wrapped in Python.

See Also
--------

To date, only simple, analytical equations have been implemented in the
partice size functions. More complex, physics-based models using a population
dynamic approach are also often used. These may be added in the future. See,
for example:

    Zhao, L., Boufadel, M. C., Socolofsky, S. A., Adams, E., King, T., and
    Lee, K. (2014). "Evolution of droplets in subsea oil and gas blowouts:
    Development and validation of the numerical model VDROP-J." Mar Pollut
    Bull, 83(1), 58-69.

"""
# S. Socolofsky, March 2020, Texas A&M University, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

from tamoc import psf, seawater, dbm

import numpy as np
# import matplotlib.pyplot as plt

class ModelBase(object):
    """
    Master class object for interfacing with functions in the `psf` module

    This base model class contains the attributes necessary to directly call
    the functions in the `psf` module.  This class is also initialized with
    these fluid properties; hence, this class can be used independently of
    ``TAMOC`` or the discrete particle module (`dbm`) in TAMOC.

    Parameters
    ----------
    rho_gas : float
        Density of the gas phase released from the jet (kg/m^3)
    mu_gas : float
        Dynamic viscosity of the gas phase released from the jet (Pa s)
    sigma_gas : float
        Interfacial tension between the gas phase released from the jet and
        the continuous-phase receiving fluid (N/m)
    rho_oil : float
        Density of the liquid phase released from the jet (kg/m^3)
    mu_oil : float
        Dynamic viscosity of the liquid phase released from the jet (Pa s)
    sigma_oil : float
        Interfacial tension between the liquid phase released from the jet
        and the continuous-phase receiving fluid (N/m)
    rho : float
        Density of the continuous-phase receiving fluid (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous-phase receiving fluid (Pa s)

    Attributes
    ----------
    rho_gas : float
        Density of the gas phase released from the jet (kg/m^3)
    mu_gas : float
        Dynamic viscosity of the gas phase released from the jet (Pa s)
    sigma_gas : float
        Interfacial tension between the gas phase released from the jet and
        the continuous-phase receiving fluid (N/m)
    rho_oil : float
        Density of the liquid phase released from the jet (kg/m^3)
    mu_oil : float
        Dynamic viscosity of the liquid phase released from the jet (Pa s)
    sigma_oil : float
        Interfacial tension between the liquid phase released from the jet
        and the continuous-phase receiving fluid (N/m)
    rho : float
        Density of the continuous-phase receiving fluid (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous-phase receiving fluid (Pa s)
    sim_stored : bool
        Flag indicating whether or not the particle size distribution
        algorithm has been calculated since the last property update.
    distribution_stored : bool
        Flag indicating whether or not the particle size distribution has
        been computed and stored since the last property update.
    d0 : float
        Equivalent circular diameter of the release orifice (m)
    m_gas : float
        Mass flow rate of gas released from the jet (kg/s)
    m_oil : float
        Mass flow rate of liquid released from the jet (kg/s)
    model_gas : str
        Name of the model used for computing the gas bubble size
        distribution.  Choices are 'wang_etal' or 'li_etal':.
    model_oil : str
        Name of the model used for computing the oil droplet size
        distribution.  Choices are 'sintef' or 'li_etal'.
    pdf_gas : str
        Probability density function to use for the gas bubble size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    pdf_oil : str
        Probability density function to use for the oil droplet size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    d50_gas : float
        Median equivalent spherical diameter of gas bubbles (m)
    de_max_gas : float
        Maximum stable bubble size (equivalent spherical diameter) of gas
        bubbles (m).
    de_gas : ndarray
        Array of equivalent spherical diameters of bubbles in the bubble
        size distribution (log-distributed, m)
    vf_gas : ndarray
        Array of volume fractions of gas corresponding to each bubble size
        in the  `de_gas` bubble size distribution (--)
    d50_oil : float
        Median equivalent spherical diameter of oil droplets (m)
    de_max_oil : float
        Maximum stable droplet size (equivalent spherical diameter) of oil
        droplets (m).
    de_oil : ndarray
        Array of equivalent spherical diameters of droplets in the droplet
        size distribution (log-distributed, m)
    vf_oil : ndarray
        Array of volume fractions of oil corresponding to each droplet size
        in the  `de_oil` droplet size distribution (--)

    See Also
    --------
    PureJet, Model

    Notes
    -----
    This class should only be used is the chemical properties are not going
    to be computed using the ``TAMOC`` `dbm`.  If the `dbm` is going to be
    used, then the regular `Model` class should be used instead.

    """
    def __init__(self, rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                 sigma_oil, rho, mu):
        super(ModelBase, self).__init__()

        # Record the release properties
        self.update_properties(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                               sigma_oil, rho, mu)

    def update_properties(self, rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                          sigma_oil, rho, mu):
        """
        Set the thermodynamic properties of the released and receiving fluids

        Store the density, viscosity, and interfacial tension of the fluids
        involved in a jet breakup scenario.

        Parameters
        ----------
        rho_gas : float
            Density of the gas phase released from the jet (kg/m^3)
        mu_gas : float
            Dynamic viscosity of the gas phase released from the jet (Pa s)
        sigma_gas : float
            Interfacial tension between the gas phase released from the jet
            and the continuous-phase receiving fluid (N/m)
        rho_oil : float
            Density of the liquid phase released from the jet (kg/m^3)
        mu_oil : float
            Dynamic viscosity of the liquid phase released from the jet (Pa s)
        sigma_oil : float
            Interfacial tension between the liquid phase released from the
            jet and the continuous-phase receiving fluid (N/m)
        rho : float
            Density of the continuous-phase receiving fluid (kg/m^3)
        mu : float
            Dynamic viscosity of the continuous-phase receiving fluid (Pa s)

        """
        # Set the flags initially to False
        self.sim_stored = False
        self.distribution_stored = False

        # Record the properties
        self.rho_gas = rho_gas
        self.mu_gas = mu_gas
        self.sigma_gas = sigma_gas
        self.rho_oil = rho_oil
        self.mu_oil = mu_oil
        self.sigma_oil = sigma_oil
        self.rho = rho
        self.mu = mu

    def simulate(self, d0, m_gas, m_oil, model_gas='wang_etal',
                 model_oil='sintef', pdf_gas='lognormal',
                 pdf_oil='rosin-rammler', Pj=4.e6, Tj=288.15):
        """
        Compute the parameters of the particle size distribution

        Computes the median bubble and droplet sizes and the spread
        of the selected size distributions.  Models for gas bubble median
        size are `wang_etal` or `li_etal`; models for oil droplet median
        size are `sintef` or `li_etal`.  Size distributions are either
        `lognormal` or `rosin-rammler`.  No matter what model is selected,
        the `d_95`-rule is used when the predicted size distribution would
        exceed the maximum stable bubble or droplet size.  Under that rule,
        the 95-percentile of the volume size distribution is set to the
        maximum stable size, and the median size is adjusted downward.

        Parameters
        ----------
        d0 : float
            Equivalent circular diameter of the release orifice (m)
        m_gas : float
            Mass flow rate of gas released from the jet (kg/s)
        m_oil : float
            Mass flow rate of liquid released from the jet (kg/s)
        model_gas : str, default='wang_etal'
            Name of the model used for computing the gas bubble size
            distribution.  Choices are 'wang_etal' or 'li_etal':.
        model_oil : str, default='sintef'
            Name of the model used for computing the oil droplet size
            distribution.  Choices are 'sintef' or 'li_etal'.
        pdf_gas : str, default='lognormal'
            Probability density function to use for the gas bubble size
            distribution.  Choices are 'lognormal' or 'rosin-rammler'.
        pdf_oil : str, default='rosin-rammler'
            Probability density function to use for the oil droplet size
            distribution.  Choices are 'lognormal' or 'rosin-rammler'.
        Pj : float, default=4.e6
            Pressure at the release point.  Used to compute the speed of
            sound in gas.
        Tj : float, default=288.15
            Temperature of the released fluids.  Used to compute the
            speed of sound of gas.

        Notes
        -----
        This method does not return any values.  Instead, the computed values
        are stored as attributes of the class object.  To report the computed
        values, use the `get`-methods.

        See Also
        --------
        get_d50, get_de_max, get_distributions

        """
        # Record the state of the release
        self.d0 = d0
        self.m_gas = m_gas
        self.m_oil = m_oil
        self.model_gas = model_gas
        self.model_oil = model_oil
        self.pdf_gas = pdf_gas
        self.pdf_oil = pdf_oil
        
        # Get the gas bubble size distribution
        if model_gas == 'wang_etal':
            # Get the parameters of the distribution
            self.d50_gas, m_gas, m_oil, self.de_max_gas, self.sigma_ln_gas = \
                psf.wang_etal(
                    self.d0, self.m_gas, self.rho_gas, self.mu_gas,
                    self.sigma_gas, self.rho, self.mu, m_l=self.m_oil,
                    rho_l=self.rho_oil, P=Pj, T=Tj
                )
            if pdf_gas == 'rosin-rammler':
                # Convert lognormal parameters to Rosin-Rammler
                self.d50_gas, self.k_gas, self.sigma_gas = psf.ln2rr(
                        self.d50_gas, self.sigma_ln_gas
                    )
        elif model_gas == 'li_etal':
            # Get the parameters of the distribution
            self.d50_gas, self.de_max_gas, self.k_gas, self.alpha_gas = \
                psf.li_etal(
                   self.d0, self.m_gas, self.rho_gas, self.m_oil,
                   self.rho_oil, self.mu_gas, self.sigma_gas, self.rho,
                   self.mu, fp_type=0
                )
            if pdf_gas == 'lognormal':
                # Convert Rosin-Rammler parameters to lognormal
                self.d50_gas, self.sigma_ln_gas = psf.rr2ln(
                        self.d50_gas, self.k_gas, self.alpha_gas
                    )

        # Get the oil droplet size distribution
        if model_oil == 'sintef':
            # Get the parameters of the distribution
            self.d50_oil, self.de_max_oil, self.k_oil, self.alpha_oil = \
                psf.sintef(
                   self.d0, self.m_gas, self.rho_gas, self.m_oil,
                   self.rho_oil, self.mu_oil, self.sigma_oil, self.rho,
                   self.mu, fp_type=1
                )
        elif model_oil == 'li_etal':
            # Get the parameters of the distribution
            self.d50_oil, self.de_max_oil, self.k_oil, self.alpha_oil = \
                psf.li_etal(
                   self.d0, self.m_gas, self.rho_gas, self.m_oil,
                   self.rho_oil, self.mu_oil, self.sigma_oil, self.rho,
                   self.mu, fp_type=1
                )
        if pdf_oil == 'lognormal':
            # Convert Rosin-Rammler parameters to lognormal
            self.d50_oil, self.sigma_ln_oil = psf.rr2ln(
                    self.d50_oil, self.k_oil, self.alpha_oil
                )

        # Set the simulation flag to True
        self.sim_stored = True

    def get_de_max(self, fp_type):
        """
        Report the maximum stable particle size of a fluid at the release

        Parameters
        ----------
        fp_type : int
            Fluid for which the maximum stable particle size is desired:
            0 = gas, 1 = liquid.

        Returns
        -------
        de_max : float
            Equivalent spherical diameter of the maximum stable particle
            size (m)

        """
        if self.sim_stored:
            # Use the simulated values stored in the present object
            if fp_type == 0:
                de_max = self.de_max_gas
            else:
                de_max = self.de_max_oil
        else:
            # Compute new values since these are independent of orifice
            # conditions
            if fp_type == 0:
                # Get maximum stable particle size for gas
                de_max = psf.grace(self.rho, self.rho_gas, self.mu,
                                   self.mu_gas, self.sigma_gas, fp_type)
            elif fp_type == 1:
                # Get the maximum stable particle size for oil
                de_max = psf.de_max_oil(self.rho_oil, self.sigma_oil,
                                        self.rho)

        return de_max

    def get_d50(self, fp_type):
        """
        Report the median particle size of a fluid at the release

        Parameters
        ----------
        fp_type : int
            Fluid for which the maximum stable particle size is desired:
            0 = gas, 1 = liquid.

        Returns
        -------
        d50 : float
            Equivalent spherical diameter of the median particle size of
            the volume size distribution (m)

        Notes
        -----
        This method uses the parameters of the particle size distributions
        determined by the `simulate()` method of the object.  You must
        run this method before calling this method to create the particle
        size distributions.

        """
        if self.sim_stored:
            if fp_type == 0:
                d50 = self.d50_gas
            elif fp_type == 1:
                d50 = self.d50_oil
            return d50
        else:
            print("You should run the .simulate() method first")
            return 0

    def get_distributions(self, nbins_gas, nbins_oil):
        """
        Report the bubble and droplet size distributions

        Parameters
        ----------
        nbins_gas : int
            Number of bin sizes to use in the gas bubble volume size
            distribution
        nbins_oil : int
            Number of bin sizes to use in the oil droplet volume size
            distribution

        Returns
        -------
        de_gas : ndarray
            Array of equivalent spherical diameters of bubbles in the bubble
            size distribution (log-distributed, m)
        vf_gas : ndarray
            Array of volume fractions of gas corresponding to each bubble size
            in the  `de_gas` bubble size distribution (--)
        de_oil : ndarray
            Array of equivalent spherical diameters of droplets in the droplet
            size distribution (log-distributed, m)
        vf_oil : ndarray
            Array of volume fractions of oil corresponding to each droplet
            size in the `de_oil` droplet size distribution (--)

        Notes
        -----
        This method uses the parameters of the particle size distributions
        determined by the `simulate()` method of the object.  You must
        run this method before calling this method to create the particle
        size distributions.

        """
        if self.sim_stored:
        
            # Record the input parameters
            self.nbins_gas = nbins_gas
            self.nbins_oil = nbins_oil

            if self.d50_gas == 0.:
                # There is no gas in this mixture
                self.de_gas = np.array([])
                self.vf_gas = np.array([])
            else:
                if self.pdf_gas == 'rosin-rammler':
                    # Use Rosin-Rammler directly
                    self.de_gas, self.vf_gas = psf.rosin_rammler(self.nbins_gas,
                        self.d50_gas, self.k_gas, self.alpha_gas
                        )
                elif self.pdf_gas == 'lognormal':
                    # Use the fitted lognormal distribution
                    self.de_gas, self.vf_gas = psf.log_normal(self.nbins_gas,
                        self.d50_gas, self.sigma_ln_gas
                        )
            if self.d50_oil == 0.:
                # There is no liquid in this mixture
                self.de_oil = np.array([])
                self.vf_oil = np.array([])
            else:                
                if self.pdf_oil == 'rosin-rammler':
                    # Use Rosin-Rammler directly
                    self.de_oil, self.vf_oil = psf.rosin_rammler(self.nbins_oil,
                        self.d50_oil, self.k_oil, self.alpha_oil
                        )
                elif self.pdf_oil == 'lognormal':
                    # Use the fitted lognormal distribution
                    self.de_oil, self.vf_oil = psf.log_normal(self.nbins_oil,
                        self.d50_oil, self.sigma_ln_oil
                        )

            # Set the distribution flag to true
            self.distribution_stored = True

        else:
            print("You should run the .simulate() method first")
            return 0

        return (self.de_gas, self.vf_gas, self.de_oil, self.vf_oil)


    def plot_psd(self, fig=1, fp_type=None):
        """
        Create plots of the bubble and droplet size distribution

        Plots a standard presentation of the present gas bubble and oil
        droplet size distributions

        Parameters
        ----------
        fig : int, default=1
            Figure number to plot
        fp_type : int, default=None
            Fluid to plot.  If `fp_type` = None, then both the gas bubbles
            and liquid droplets are plotted.  Otherwise, only the
            distribution defined by this parameter is plotted:  0 = gas,
            1 = liquid.

        Notes
        -----
        This method relies on the distribution already being computed,
        which requires first calling the methods `simulate()` and
        `get_distributions()`.  If these have not been computed, an
        error message will display and not plots will be created.

        """
        import matplotlib.pyplot as plt

        if self.sim_stored:
            if not self.distribution_stored:
                # Prepare to plot d50 only
                self.get_distributions(1, 1)

            # Create the standard plots for gas and oil size distributions
            plt.figure(fig)
            plt.clf()

            title_font = {'fontsize': 10,
                          'fontweight' : 1,
                          'verticalalignment': 'baseline',
                          'horizontalalignment': 'left'}

            # Gas bubble distribution
            if fp_type == 0 or fp_type == None:
                plt.subplot(211)
                plot_phase(self.nbins_gas, self.de_gas, self.vf_gas, 'b')
                plt.xlabel('Gas bubble diameter (mm)')
                plt.ylabel('Gas mass flux (kg/s)')
                fig_title = ' Gas model = ' + self.model_gas + \
                            '; PDF = ' + self.pdf_gas + \
                            '; d_50 = ' + \
                            '%2.2f mm' % (self.get_d50(0) * 1000.)
                plt.title(fig_title, loc='left', fontdict=title_font, pad=-15)

            # Oil droplet distribution
            if fp_type == 1 or fp_type == None:
                if fp_type == 1:
                    plt.subplot(211)
                elif fp_type == None:
                    plt.subplot(212)
                plot_phase(self.nbins_oil, self.de_oil, self.vf_oil, 'r')
                plt.xlabel('Oil droplet diameter (mm)')
                plt.ylabel('Oil mass flux (kg/s)')
                fig_title = ' Oil model = ' + self.model_oil + \
                            '; PDF = ' + self.pdf_oil + \
                            '; d_50 = ' + \
                            '%3.3f mm' % (self.get_d50(1) * 1000.)
                plt.title(fig_title, loc='left', fontdict=title_font,
                          pad=-15)

            plt.show()
        else:
            print("You should run the .simulate() and .get_distribution()")
            print("methods first")


class PureJet(ModelBase):
    """
    Class object for pure gas or pure oil plumes

    This class uses the `ModelBase` class, but provides an interface that
    only expects one released fluid (based on `fp_type`; 0 = gas,
    1 = liquid).

    Parameters
    ----------
    rho_p : float
        Density of the gas phase released from the jet (kg/m^3)
    mu_p : float
        Dynamic viscosity of the gas phase released from the jet (Pa s)
    sigma_p : float
        Interfacial tension between the gas phase released from the jet and
        the continuous-phase receiving fluid (N/m)
    rho : float
        Density of the continuous-phase receiving fluid (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous-phase receiving fluid (Pa s)
    fp_type : int, default=1
        Phase of the released fluid; 0 = gas, 1 = liquid.

    Attributes
    ----------
    rho_gas : float
        Density of the gas phase released from the jet (kg/m^3)
    mu_gas : float
        Dynamic viscosity of the gas phase released from the jet (Pa s)
    sigma_gas : float
        Interfacial tension between the gas phase released from the jet and
        the continuous-phase receiving fluid (N/m)
    rho_oil : float
        Density of the liquid phase released from the jet (kg/m^3)
    mu_oil : float
        Dynamic viscosity of the liquid phase released from the jet (Pa s)
    sigma_oil : float
        Interfacial tension between the liquid phase released from the jet
        and the continuous-phase receiving fluid (N/m)
    rho : float
        Density of the continuous-phase receiving fluid (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous-phase receiving fluid (Pa s)
    sim_stored : bool
        Flag indicating whether or not the particle size distribution
        algorithm has been calculated since the last property update.
    distribution_stored : bool
        Flag indicating whether or not the particle size distribution has
        been computed and stored since the last property update.
    d0 : float
        Equivalent circular diameter of the release orifice (m)
    m_gas : float
        Mass flow rate of gas released from the jet (kg/s)
    m_oil : float
        Mass flow rate of liquid released from the jet (kg/s)
    model_gas : str
        Name of the model used for computing the gas bubble size
        distribution.  Choices are 'wang_etal' or 'li_etal':.
    model_oil : str
        Name of the model used for computing the oil droplet size
        distribution.  Choices are 'sintef' or 'li_etal'.
    pdf_gas : str
        Probability density function to use for the gas bubble size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    pdf_oil : str
        Probability density function to use for the oil droplet size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    d50_gas : float
        Median equivalent spherical diameter of gas bubbles (m)
    de_max_gas : float
        Maximum stable bubble size (equivalent spherical diameter) of gas
        bubbles (m).
    de_gas : ndarray
        Array of equivalent spherical diameters of bubbles in the bubble
        size distribution (log-distributed, m)
    vf_gas : ndarray
        Array of volume fractions of gas corresponding to each bubble size
        in the  `de_gas` bubble size distribution (--)
    d50_oil : float
        Median equivalent spherical diameter of oil droplets (m)
    de_max_oil : float
        Maximum stable droplet size (equivalent spherical diameter) of oil
        droplets (m).
    de_oil : ndarray
        Array of equivalent spherical diameters of droplets in the droplet
        size distribution (log-distributed, m)
    vf_oil : ndarray
        Array of volume fractions of oil corresponding to each droplet size
        in the  `de_oil` droplet size distribution (--)

    See Also
    --------
    ModelBase, Model

    Notes
    -----
    In the attributes above, the attributes attached to the fluid phase of
    interest will have values; whereas, the other phase will contain `None`.
    For example, if this is a pure oil release, `rho_oil` will have a value,
    but `rho_gas` will store `None`.

    """
    def __init__(self, rho_p, mu_p, sigma_p, rho, mu, fp_type=1):

        # Send the particle properties to the correct phase
        self.update_properties(rho_p, mu_p, sigma_p, rho, mu, fp_type)

    def update_properties(self, rho_p, mu_p, sigma_p, rho, mu, fp_type):
        """
        Set the thermodynamic properties of the release and receiving fluids

        Parameters
        ----------
        rho_p : float
            Density of the gas phase released from the jet (kg/m^3)
        mu_p : float
            Dynamic viscosity of the gas phase released from the jet (Pa s)
        sigma_p : float
            Interfacial tension between the gas phase released from the jet
            and the continuous-phase receiving fluid (N/m)
        rho : float
            Density of the continuous-phase receiving fluid (kg/m^3)
        mu : float
            Dynamic viscosity of the continuous-phase receiving fluid (Pa s)
        fp_type : int, default=1
            Phase of the released fluid; 0 = gas, 1 = liquid.

        """
        # Set the flags initially to False
        self.sim_stored = False
        self.distribution_stored = False

        # Record the fluid type
        self.fp_type = fp_type

        # Send the particle properties to the correct phase
        if fp_type == 0:
            ModelBase.update_properties(self, rho_p, mu_p, sigma_p, None,
                None, None, rho, mu)

        elif fp_type == 1:
            ModelBase.update_properties(self, None, None, None, rho_p, mu_p,
                sigma_p, rho, mu)

    def simulate(self, d0, m, model='sintef', pdf='rosin-rammler'):
        """
        Compute the parameters of the particle size distribution

        Computes the median bubble or droplet sizes and the spread
        of the selected size distributions.  Models for gas bubble median
        size are `wang_etal` or `li_etal`; models for oil droplet median
        size are `sintef` or `li_etal`.  Size distributions are either
        `lognormal` or `rosin-rammler`.  No matter what model is selected,
        the `d_95`-rule is used when the predicted size distribution would
        exceed the maximum stable bubble or droplet size.  Under that rule,
        the 95-percentile of the volume size distribution is set to the
        maximum stable size, and the median size is adjusted downward.

        This method only computes size distribution values for the fluid
        type selected by the class attribute fp_type (0 = gas, 1 = liquid).

        Parameters
        ----------
        d0 : float
            Equivalent circular diameter of the release orifice (m)
        m : float
            Mass flow rate of fluid released from the jet (kg/s)
        model : str, default='wang_etal'
            Name of the model used for computing the particle size
            distribution.  Choices are 'wang_etal', 'li_etal', or 'sintef'.
            Note that the model choice must agree with the fluid of interest;
            see paragraph above for details.
        pdf : str, default='rosin-rammler'
            Probability density function to use for the particle size
            distribution.  Choices are 'lognormal' or 'rosin-rammler'.

        Notes
        -----
        This method does not return any values.  Instead, the computed values
        are stored as attributes of the class object.  To report the computed
        values, use the `get`-methods.

        See Also
        --------
        get_d50, get_de_max, get_distributions

        """
        if self.fp_type == 0:
            ModelBase.simulate(self, d0, m, 0. * m, model_gas=model,
                               pdf_gas=pdf)
        elif self.fp_type == 1:
            ModelBase.simulate(self, d0, 0. * m, m, model_oil=model,
                               pdf_oil=pdf)

    def get_de_max(self, fp_type=None):
        """
        Report the maximum stable particle size of the fluid at the release

        Returns
        -------
        de_max : float
            Equivalent spherical diameter of the maximum stable particle
            size (m)
        """
        return ModelBase.get_de_max(self, self.fp_type)

    def get_d50(self, fp_type=None):
        """
        Report the median particle size of a fluid at the release

        Returns
        -------
        d50 : float
            Equivalent spherical diameter of the median particle size of
            the volume size distribution (m)

        Notes
        -----
        This method uses the parameters of the particle size distributions
        determined by the `simulate()` method of the object.  You must
        run this method before calling this method to create the particle
        size distributions.

        """
        return ModelBase.get_d50(self, self.fp_type)

    def get_distributions(self, nbins):
        """
        Report the particle size distributions

        Parameters
        ----------
        nbins : int
            Number of bin sizes to use in the volume size distribution

        Returns
        -------
        de : ndarray
            Array of equivalent spherical diameters of particles in the
            volume size distribution (log-distributed, m)
        vf : ndarray
            Array of volume fractions of particles corresponding to each
            particle size in the  `de` volume size distribution (--)

        Notes
        -----
        This method uses the parameters of the particle size distributions
        determined by the `simulate()` method of the object.  You must
        run this method before calling this method to create the particle
        size distributions.

        """
        # Compute the appropriate distribution
        if self.fp_type == 0:
            de, vf, de_none, vf_none = ModelBase.get_distributions(
                    self, nbins, 0
                )
        elif self.fp_type == 1:
            de_none, vf_none, de, vf = ModelBase.get_distributions(
                    self, 0, nbins
                )

        return (de, vf)

    def plot_psd(self, fig_num):
        """
        Create a plot of the particle size distribution

        Plots a standard presentation of the present particle size
        distribution

        Parameters
        ----------
        fig_num : int, default=1
            Figure number to plot

        Notes
        -----
        This method relies on the distribution already being computed,
        which requires first calling the methods `simulate()` and
        `get_distributions()`.  If these have not been computed, an
        error message will display and not plots will be created.

        """
        ModelBase.plot_psd(self, fig_num, self.fp_type)


class Model(ModelBase):
    """
    Master lass object for computing bubble and droplet size distributions

    This model class contains handles the interface to the `ModelBase`,
    allowing particle size distributions to be easily computed given a
    `dbm.FluidMixture` description of the released fluids.  This is the
    main class object that should be used to compute bubble and droplet
    size distributions for the plume models in ``TAMOC``.

    Parameters
    ----------
    profile : `ambient.Profile` object
        Profile containing ambient CTD data
    oil_mixture : `dbm.FluidMixture` object
        A `dbm.FluidMixture` object that contains the chemical description
        of an oil mixture.
    m_mixture : ndarray
        An array of mass fluxes (kg/s) of each pseudo-component in the live-
        oil mixture.
    z0 : float
        Release point of the jet orifice (m)
    Tj : float
        Temperature of the released fluids (K)

    Attributes
    ----------
    profile : `ambient.Profile` object
        Profile containing ambient CTD data
    oil_mixture : `dbm.FluidMixture` object
        A `dbm.FluidMixture` object that contains the chemical description
        of an oil mixture.
    m_mixture : ndarray
        An array of mass fluxes (kg/s) of each pseudo-component in the live-
        oil mixture.
    z0 : float
        Release point of the jet orifice (m)
    Tj : float
        Temperature of the released fluids (K)
    T : float
        Temperature of the receiving fluid at the release (K)
    S : float
        Salinity of the receiving fluid at the release (psu)
    P : float
        Pressure of the receiving fluid at the release (Pa)
    gas : `dbm.FluidParticle` object
        A `dbm.FluidParticle` object for the gas phase fluid at the
        release
    m_gas : ndarray
        An array of mass fluxes (kg/s) of each pseudo-component in the gas
        phase of the released fluids.
    m_oil : ndarray
        An array of mass fluxes (kg/s) of each pseudo-component in the oil
        phase of the released fluids.
    rho_gas : float
        Density of the gas phase released from the jet (kg/m^3)
    mu_gas : float
        Dynamic viscosity of the gas phase released from the jet (Pa s)
    sigma_gas : float
        Interfacial tension between the gas phase released from the jet and
        the continuous-phase receiving fluid (N/m)
    oil : `dbm.FluidParticle` object
        A `dbm.FluidParticle` object for the liquid phase fluid at the
        release
    rho_oil : float
        Density of the liquid phase released from the jet (kg/m^3)
    mu_oil : float
        Dynamic viscosity of the liquid phase released from the jet (Pa s)
    sigma_oil : float
        Interfacial tension between the liquid phase released from the jet
        and the continuous-phase receiving fluid (N/m)
    rho : float
        Density of the continuous-phase receiving fluid (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous-phase receiving fluid (Pa s)
    sim_stored : bool
        Flag indicating whether or not the particle size distribution
        algorithm has been calculated since the last property update.
    distribution_stored : bool
        Flag indicating whether or not the particle size distribution has
        been computed and stored since the last property update.
    d0 : float
        Equivalent circular diameter of the release orifice (m)
    model_gas : str
        Name of the model used for computing the gas bubble size
        distribution.  Choices are 'wang_etal' or 'li_etal':.
    model_oil : str
        Name of the model used for computing the oil droplet size
        distribution.  Choices are 'sintef' or 'li_etal'.
    pdf_gas : str
        Probability density function to use for the gas bubble size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    pdf_oil : str
        Probability density function to use for the oil droplet size
        distribution.  Choices are 'lognormal' or 'rosin-rammler'.
    d50_gas : float
        Median equivalent spherical diameter of gas bubbles (m)
    de_max_gas : float
        Maximum stable bubble size (equivalent spherical diameter) of gas
        bubbles (m).
    de_gas : ndarray
        Array of equivalent spherical diameters of bubbles in the bubble
        size distribution (log-distributed, m)
    vf_gas : ndarray
        Array of volume fractions of gas corresponding to each bubble size
        in the  `de_gas` bubble size distribution (--)
    d50_oil : float
        Median equivalent spherical diameter of oil droplets (m)
    de_max_oil : float
        Maximum stable droplet size (equivalent spherical diameter) of oil
        droplets (m).
    de_oil : ndarray
        Array of equivalent spherical diameters of droplets in the droplet
        size distribution (log-distributed, m)
    vf_oil : ndarray
        Array of volume fractions of oil corresponding to each droplet size
        in the  `de_oil` droplet size distribution (--)

    See Also
    --------
    ModelBase

    Notes
    -----
    This is the main class object that should be used for particle size
    distributions using the ``TAMOC`` plume models.

    """
    def __init__(self, profile, oil, m, z0, Tj_user=None, Pj_user=None):

        # Record the initial user input for Tj and Pj
        self.Tj_user = Tj_user
        self.Pj_user = Pj_user
        
        # Compute and store the oil properties
        self.update_properties(profile, oil, m, z0, Tj_user, Pj_user)

    def update_properties(self, profile, oil_mixture, m_mixture, z0, Tj_user,
        Pj_user):
        """
        Set the thermodynamic properties of the released and receiving fluids

        Store the density, viscosity, and interfacial tension of the fluids
        involved in a jet breakup scenario.

        Parameters
        ----------
        profile : `ambient.Profile` object
            Profile containing ambient CTD data
        oil_mixture : `dbm.FluidMixture` object
            A `dbm.FluidMixture` object that contains the chemical description
            of an oil mixture.
        m_mixture : ndarray
            An array of mass fluxes (kg/s) of each pseudo-component in the
            live-oil mixture.
        z0 : float
            Release point of the jet orifice (m)
        Tj_user : float, default=None
            Temperature of the released fluids (K). The default value of `None`
            means that the ambient value should be used.
        Pj_user : float, default=None
            Pressure of the released fluids (Pa). If the fluids are undergoing
            a phase change, they may not be able to adjust immediately to the
            ambient pressure. This allows the user to control the pressure used
            to compute properties of the jet fluid. The default value of `None`
            means that the ambient value should be used.

        Notes
        -----
        This method allows the complete release to be redefined.  If you
        only want to update the release depth or release temperature, use
        `.update_z0()` or `update_Tj()`, instead.

        """
        # Set the flags initially to False
        self.sim_stored = False
        self.distribution_stored = False

        # Record the input parameters
        self.profile = profile
        self.oil_mixture = oil_mixture
        self.m_mixture = m_mixture
        self.z0 = z0

        # Compute the properties of seawater
        self.T, self.S, self.P = self.profile.get_values(self.z0,
               ['temperature', 'salinity', 'pressure']
            )
        self.rho = seawater.density(self.T, self.S, self.P)
        self.mu = seawater.mu(self.T, self.S, self.P)

        # Set jet temperature and pressure either to ambient or input value
        if Tj_user == None:
            # Use ambient temperature
            self.Tj = self.T
        else:
            # Use input temperature
            self.Tj = Tj_user
        if Pj_user == None:
            # Use ambient pressure
            self.Pj = self.P
        else:
            self.Pj = Pj_user
        
        # Perform the equilibrium calculation
        m_eq, xi, K = self.oil_mixture.equilibrium(self.m_mixture, self.Tj,
            self.Pj)
                  
        # Compute the gas phase properties
        if np.sum(m_eq[0,:]) == 0:
            self.gas = None
            self.m_gas = m_eq[0,:]
            self.rho_gas = None
            self.mu_gas = None
            self.sigma_gas = None
        else:
            self.gas = dbm.FluidParticle(self.oil_mixture.composition,
                                         fp_type=0,
                                         delta=oil_mixture.delta,
                                         user_data=oil_mixture.user_data)
            self.m_gas = m_eq[0,:]
            self.rho_gas = self.gas.density(self.m_gas, self.Tj, self.Pj)
            self.mu_gas = self.gas.viscosity(self.m_gas, self.Tj, self.Pj)
            self.sigma_gas = self.gas.interface_tension(self.m_gas, self.Tj,
                                                        self.S, self.Pj)

        # Compute the liquid phase properties
        if np.sum(m_eq[1,:]) == 0:
            self.oil = None
            self.m_oil = m_eq[1,:]
            self.rho_oil = None
            self.mu_oil = None
            self.sigma_oil = None
        else:
            self.oil = dbm.FluidParticle(self.oil_mixture.composition,
                                         fp_type=1,
                                         delta=oil_mixture.delta,
                                         user_data=oil_mixture.user_data)

            self.m_oil = m_eq[1,:]
            self.rho_oil = self.oil.density(self.m_oil, self.Tj, self.Pj)
            self.mu_oil = self.oil.viscosity(self.m_oil, self.Tj, self.Pj)
            self.sigma_oil = self.oil.interface_tension(self.m_oil, self.Tj,
                                                    self.S, self.Pj)

    def update_z0(self, z0):
        """
        Update the release depth of the jet

        Parameters
        ----------
        z0 : float
            Release point of the jet orifice (m)

        """
        self.z0 = z0
        self.update_properties(self.profile, self.oil_mixture,
                               self.m_mixture, self.z0, self.Tj_user, 
                               self.Pj_user)

    def update_Tj(self, Tj):
        """
        Update the temperature of the released fluids in the jet

        Parameters
        ----------
        Tj : float
            Temperature of the released fluids (K)

        """
        self.Tj_user = Tj
        self.update_properties(self.profile, self.oil_mixture,
                               self.m_mixture, self.z0, self.Tj_user, 
                               self.Pj_user)

    def update_Pj(self, Pj):
        """
        Update the pressure of the released fluids in the jet

        Parameters
        ----------
        Pj : float
            Pressure of the released fluids (Pa)

        """
        self.Pj_user = Pj
        self.update_properties(self.profile, self.oil_mixture,
                               self.m_mixture, self.z0, self.Tj_user, 
                               self.Pj_user)
    
    def update_m_mixture(self, m_mixture):
        """
        Update the total mass flux of the released fluids in the jet

        Parameters
        ----------
        m_mixture : float
            An array of mass fluxes (kg/s) of each pseudo-component in the
            live-oil mixture.

        """
        self.m_mixture = m_mixture
        self.update_properties(self.profile, self.oil_mixture,
                               self.m_mixture, self.z0, self.Tj_user, 
                               self.Pj_user)

    def simulate(self, d0, model_gas='wang_etal', pdf_gas='lognormal',
                 model_oil='sintef', pdf_oil='rosin-rammler'):
        """
        Compute the parameters of the particle size distribution

        Computes the median bubble and droplet sizes and the spread
        of the selected size distributions.  Models for gas bubble median
        size are `wang_etal` or `li_etal`; models for oil droplet median
        size are `sintef` or `li_etal`.  Size distributions are either
        `lognormal` or `rosin-rammler`.  No matter what model is selected,
        the `d_95`-rule is used when the predicted size distribution would
        exceed the maximum stable bubble or droplet size.  Under that rule,
        the 95-percentile of the volume size distribution is set to the
        maximum stable size, and the median size is adjusted downward.

        Parameters
        ----------
        d0 : float
            Equivalent circular diameter of the release orifice (m)
        model_gas : str, default='wang_etal'
            Name of the model used for computing the gas bubble size
            distribution.  Choices are 'wang_etal' or 'li_etal':.
        model_oil : str, default='sintef'
            Name of the model used for computing the oil droplet size
            distribution.  Choices are 'sintef' or 'li_etal'.
        pdf_gas : str, default='lognormal'
            Probability density function to use for the gas bubble size
            distribution.  Choices are 'lognormal' or 'rosin-rammler'.
        pdf_oil : str, default='rosin-rammler'
            Probability density function to use for the oil droplet size
            distribution.  Choices are 'lognormal' or 'rosin-rammler'.

        Notes
        -----
        This method does not return any values.  Instead, the computed values
        are stored as attributes of the class object.  To report the computed
        values, use the `get`-methods.

        See Also
        --------
        get_d50, get_de_max, get_distributions

        """
        ModelBase.simulate(self, d0, self.m_gas, self.m_oil,
                           model_gas=model_gas, model_oil=model_oil,
                           pdf_gas=pdf_gas, pdf_oil=pdf_oil, Pj=self.Pj,
                           Tj=self.Tj)


def plot_phase(nbins, de, vf, color):
    """
    docstring for plot_phase

    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    # Prepare data for plotting
    if np.sum(vf) > 0:
        index = np.arange(nbins)
        bar_width = 0.75
        opacity = 0.4
        plt.bar(index, vf, bar_width, alpha=opacity, color=color)
        ntics = 10
        ticlocs = np.linspace(0, nbins-1, ntics)
        ticnums = []
        de_vals = interp1d(index, de)
        for i in range(ntics):
            ticnums.append('%2.2f' % (de_vals(ticlocs[i]) * 1000))
        plt.xticks(ticlocs, ticnums)

