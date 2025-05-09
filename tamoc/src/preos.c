/* preos.c
 *  -------
 *
 * This file is a library of C functions to implement the Peng-Robinson
 * equation of state (EOS) with volume translation.  This file is translated
 * directly from the Fortran functions in dbm_eos.f95 and math_funcs.f95
 * distributed in the TAMOC Python Package (the Texas A&M Oil spill / outfall
 * calculator).  The C-syntax follows styles from the Scipy package.  Some
 * of the code choices are taken from the f2py-generated dbm_fmodule.c
 * file created from the original Fortran files mentioned above.
 *
 * This library contains the following functions:
 *
 *   . density:  gas and liquid densities at given conditions
 *   . fugacity:  gas and liquid fugacities at given conditions
 *   . volume_trans:  computes the volume-translation parameter to improve
 *        density estimates.
 *   . z_pr:  z-factor for gas and liquid
 *   . coefs:  mixture coefficients for the Peng-Robinson EOS
 *   . cubic_roots:  general solution for the roots of a cubic polynomial
 *   . quadratic_roots:  general solution for the roos of a quadratic
 *        polynomial
 *   . swap:  helper function to swap the values stored in two variables
 *
 * Because these functions emulate Fortran subroutines, all of the input
 * variables will normally be passed by reference, and the outputs from
 * these functions will be returned by changing the values of the input
 * variables.  The intent (input or output) of each variable will be noted
 * in the function comments.
 *
 * Each of these functions is thoroughly documented in the code below.  The
 * intention of this file is to code all parts in pure C.  To use this
 * library in Python, it should be compiled and linked with the wrapper
 * functions in preos_cmodule.c.
 *
 * June 2013 / October 2023 -> Fortran 95; May 2025 -> C.
 */

// S. Socolofsky, Texas A&M University, May 2025, <socolofs@tamu.edu>

/* Use flat arrays for all 2D arrays:
 *
 * Index using:  arr[i * cols + j]
 *
 * Allocate using
 * double *A = alloc_array(rows * cols, sizeof(double), "A");
 *
 * Where,
 *
 * static void *alloc_2d_array_flat(size_t rows, size_t cols,
 *     size_t elem_size, const char *var_name)
 * {
 *     size_t len = rows * cols;
 *     return alloc_array(len, elem_size, var_name);
 * }

 * Fill using:
 *
 * for (size_t i = 0; i < rows; ++i) {
 *     for (size_t j = 0; j < cols; ++j) {
 *         A[i * cols + j] = some_function(i, j);
 *     }
 * }
 */

#ifndef PY_SSIZE_T_CLEAN // Py_SSIZE_T_CLEAN statement from f2py
#define PY_SSIZE_T_CLEAN
#endif

#include "numpy/arrayobject.h"
#include <Python.h>

#include <float.h>
#include <math.h>

// Define some global variables
#define F_INT int

// Borrow a complex number type from f2py
typedef struct {
    double r, i;
} complex_double;

/* Prototype some functions coming from preos_cmodule.c and that get used
 * before the full declaration in the code to follow */
void *alloc_1d_array(size_t len, size_t elem_size, const char *var_name);
void *alloc_2d_array_flat(size_t rows, size_t cols, size_t elem_size,
    const char *var_name);
void mole_fraction(int nc, double *mass, double *Mol_wt, double *yk);
void volume_trans(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *C_pen, double *C_pen_T,
    double *vt);
void z_pr(int nc, double T, double P, double *mass, double *Mol_wt, double *Pc,
    double *Tc, double *omega, double *delta, double *yk, double *Aij,
    double *Bij, double *delta_groups, int calc_delta, double *z, double *A,
    double *B, double *Ap, double *Bp);
void coefs(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *omega, double *delta_in, double *yk,
    double *Aij, double *Bij, double *delta_groups, int calc_delta, double *A,
    double *B, double *Ap, double *Bp);
void _cubic_roots(double a_t[4], complex_double z[3]);

// Define some global variables
static complex_double czero = { 0., 0. };
static double zero = 0., fourth = 0.25, half = 0.5, one = 1., two = 2.;
static double three = 3., four = 4.;
static double eps = DBL_EPSILON;
static double RU = 8.314510;

double dot(const double *a, const double *b, int n)
{
    /*
     * Compute the inner product of the length 'n' arrays 'a' and 'b'
     */
    int i;
    double sum_ab = 0.;
    for (i = 0; i < n; i++) {
        sum_ab += a[i] * b[i];
    }
    return sum_ab;
}

/* --------------------------------------------------------------------------
 * Functions from dbm_eos.f95 for the Peng Robinson EOS
 * ------------------------------------------------------------------------*/

void mole_fraction(int nc, double *mass, double *Mol_wt, double *yk)
{
    /*
     * Compute the mole fraction of a mixture from the mass
     *
     * Converts the masses of each component in a mixture to the mole fraction
     * of each component in the mixture.
     *
     * Input variables:
     *     nc = number of components in the mixture
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *
     * Returns the mole fraction (--) of the mixture, yk.
     *
     * S. Socolofsky
     * July 2013 -> Fortran 95; May 2025 -> C
     */

    int i;
    double sum_n_moles;
    double *n_moles = NULL;

    // Allocate an array to hold the moles of each component
    n_moles = alloc_1d_array(nc, sizeof(*n_moles), "n_moles in mole_fraction");
    if (n_moles == NULL) {
        goto fail;
    }

    // Compute the total number of moles of each component
    for (i = 0; i < nc; i++) {
        n_moles[i] = mass[i] / Mol_wt[i];
    }

    // Sum the total moles
    sum_n_moles = 0.;
    for (i = 0; i < nc; i++) {
        sum_n_moles += n_moles[i];
    }

    // Compute the mole fraction
    for (i = 0; i < nc; i++) {
        yk[i] = n_moles[i] / sum_n_moles;
    }

    // Computed mole fraction successully
    goto success;

success:
    free(n_moles);
    return;

fail:
    if (n_moles != NULL)
        free(n_moles);
    return;
}

void _density(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *omega, double *delta,
    double *Aij, double *Bij, double *delta_groups, int calc_delta,
    double *C_pen, double *C_pen_T, double *rho)
{
    /*
     * Computes the liquid and gas density of a mixture from the P-R EOS
     *
     * Computes the density of a mixture using the Peng-Robinson equation
     * of state as described in McCain (1990), Properties of Petroleum
     * Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
     *
     * Input Variables are:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     Vc = array of critical point molar volumes for each component
     *          (m^3/mol)
     *     omega = array of Pitzer acentric factors for each component (--)
     *     delta = matrix of binary interaction coefficients (--)
     *     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
     *     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
     *     delta_groups = group contribution numbers (normalized) for each
     *         component in the mixture (--)
     *     calc_groups = flag indicating whether or not delta_groups has
     *         been provided (1 = yes, -1 = no)
     *     C_pen = Peneloux volume translation coefficient (m^3/mol)
     *     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
     *
     * Output variable is:
     *     rho = array of the densities [gas, liquid] of the mixture, kg/m^3.
     *         The return array is a flat C array mimicing a 2D array in
     *         which row 0 is the gas densities and row 1 is the liquid
     *         densities; this array has only one column.
     *
     * S. Socolofsky
     * June 2013 -> Fortran 95; May 2025 -> C.
     */

    int i;
    double A, B;
    double sum_yk_vt, sum_yk_Mol_wt;
    double z[2], nu[2];
    double *Ap = NULL, *Bp = NULL, *yk = NULL, *vt = NULL;

    // Allocate the local arrays
    Ap = alloc_1d_array(nc, sizeof(*Ap), "Ap in density");
    if (Ap == NULL) {
        goto fail;
    }
    Bp = alloc_1d_array(nc, sizeof(*Bp), "Bp in density");
    if (Bp == NULL) {
        goto fail;
    }
    yk = alloc_1d_array(nc, sizeof(*yk), "yk in density");
    if (yk == NULL) {
        goto fail;
    }
    vt = alloc_1d_array(nc, sizeof(*vt), "vt in density");
    if (vt == NULL) {
        goto fail;
    }

    // Convert the masses to mole fraction
    mole_fraction(nc, mass, Mol_wt, yk);

    // Get the z-factor using the Peng-Robinson equation of state
    z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, yk, Aij, Bij,
        delta_groups, calc_delta, z, &A, &B, Ap, Bp);

    // Compute the volume translation coefficient
    volume_trans(nc, T, P, mass, Mol_wt, Pc, Tc, Vc, C_pen, C_pen_T, vt);

    // Compute the molar volume and density, which is returned by reference
    sum_yk_vt = dot(yk, vt, nc);
    sum_yk_Mol_wt = dot(yk, Mol_wt, nc);
    for (i = 0; i < 2; i++) {
        nu[i] = z[i] * RU * T / P - sum_yk_vt;
        rho[i] = 1. / nu[i] * sum_yk_Mol_wt;
    }

    // Completed successfully
    goto success;

success:
    free(Ap);
    free(Bp);
    free(yk);
    free(vt);
    return;

fail:
    if (Ap != NULL)
        free(Ap);
    if (Bp != NULL)
        free(Bp);
    if (yk != NULL)
        free(yk);
    if (vt != NULL)
        free(vt);
    return;
}

void _fugacity(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *omega, double *delta, double *Aij,
    double *Bij, double *delta_groups, int calc_delta, double *fug)
{
    /*
     * Computes the gas and liquid fugacity of a mixture from the P-R EOS
     *
     * Computes the gas and liquid fugacity of a mixture using the Peng-
     * Robinson equation of state as described in McCain (1990), Properties of
     * Petroleum Fluids, 2nd Edition, PennWell Publishing Company, Tulsa,
     * Oklahoma.
     *
     * Input variables are:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     omega = array of Pitzer acentric factors for each component (--)
     *     delta = matrix of binary interaction coefficients (--)
     *     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
     *     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
     *     delta_groups = group contribution numbers (normalized) for each
     *         component in the mixture (--)
     *     calc_groups = flag indicating whether or not delta_groups has
     *         been provided (1 = yes, -1 = no)
     *
     * Output variable is:
     *     fug = array of the fugacities [gas, liquid] of the mixture (Pa).
     *         The return array is a flat C array mimicing a 2D array in
     *         which row 0 is the gas fugacities and row 1 is the liquid
     *         fugacities; each column corresponds to each component of the
     *         mixture.
     *
     * S. Socolofsky
     * July 2013 -> Fortran 95; May 2025 -> C
     */

    int i, j;
    double log_term, exponent;
    double A, B;
    double z[2];
    double *Ap = NULL, *Bp = NULL, *yk = NULL;

    // Allocate the local arrays
    Ap = alloc_1d_array(nc, sizeof(*Ap), "Ap in fugacity");
    if (Ap == NULL) {
        goto fail;
    }
    Bp = alloc_1d_array(nc, sizeof(*Bp), "Bp in fugacity");
    if (Bp == NULL) {
        goto fail;
    }
    yk = alloc_1d_array(nc, sizeof(*yk), "yk in fugacity");
    if (yk == NULL) {
        goto fail;
    }

    // Convert the masses to mole fraction
    mole_fraction(nc, mass, Mol_wt, yk);

    // Get the z-factor using the Peng-Robinson equation of state
    z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, yk, Aij, Bij,
        delta_groups, calc_delta, z, &A, &B, Ap, Bp);

    // Fill the fugacity array
    for (i = 0; i < 2; i++) {
        for (j = 0; j < nc; j++) {
            fug[i * nc + j] = exp((z[i] - 1.0) * Bp[j] - log(z[i] - B) -
                                  A / (pow(2.0, 1.5) * B) * (Ap[j] - Bp[j]) *
                                      log((z[i] + (sqrt(2.0) + 1.0) * B) /
                                          (z[i] - (sqrt(2.0) - 1.0) * B))) *
                              yk[j] * P;
        }
    }

    // Completed successfully
    goto success;

success:
    free(Ap);
    free(Bp);
    free(yk);
    return;

fail:
    if (Ap != NULL)
        free(Ap);
    if (Bp != NULL)
        free(Bp);
    if (yk != NULL)
        free(yk);
    return;
}

void _viscosity(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *omega, double *delta,
    double *Aij, double *Bij, double *delta_groups, int calc_delta,
    double *C_pen, double *C_pen_T, double *mu)
{
    /*
     * Computes the viscosity of a petroleum fluid
     *
     * Computes the viscosity of the given fluid mixture for the gas and
     * liquid phases following the method in Pedersen et al. "Phase Behavior
     * of Petroleum Reservoir Fluids", 2nd edition, Chapeter 10.
     *
     * This method correlates the viscosity of the mixture to the viscosity
     * of methane taken at a specialized corresponding state.  The function
     * has the properties of methane hard-wired so that any mixture can be
     * evaluated.
     *
     * Input variables:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     Vc = array of critical point molar volumes for each component
     *          (m^3/mol)
     *     omega = array of Pitzer acentric factors for each component (--)
     *     delta = matrix of binary interaction coefficients (--)
     *     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
     *     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
     *     delta_groups = group contribution numbers (normalized) for each
     *         component in the mixture (--)
     *     calc_groups = flag indicating whether or not delta_groups has
     *         been provided (1 = yes, -1 = no)
     *     C_pen = Peneloux volume translation coefficient (m^3/mol)
     *     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
     *
     * Output variable is:
     *     mu = array of the viscosity [gas, liquid] of the mixture, Pa s.
     *         The return array is a flat C array mimicing a 2D array in
     *         which row 0 is the gas densities and row 1 is the liquid
     *         densities; this array has only one column.
     *
     * S. Socolofsky
     * July 2013 -> Fortran 95; May 2025 -> C
     */

    int i, j;
    double rho_c0, eta_0, eta_1, delta_T, htan, numerator, denominator, Tc_mix,
        Pc_mix, M_bar_n, M_bar_w_sum, M_bar_w, M_mix;
    double M0[1], Tc0[1], Pc0[1], omega0[1], Vc0[1], C_pen0[1], C_pen_T0[1];
    double T0[2], P0[2];
    double *z, *M;
    double delta0[1];
    double theta[2], delta_eta_p[2], delta_eta_pp[2], rho0[2], eta_ch4[2],
        rho_r[2], alpha_mix[2], alpha0[2];

    // Enter the parameter values from Table 10.1
    double GV[] = { -2.090975e5, 2.647269e5, -1.472818e5, 4.716740e4,
        -9.491872e3, 1.219979e3, -9.627993e1, 4.274152e0, -8.141531e-2 };
    double A = 1.696985927;
    double B = -0.133372346;
    double C = 1.4;
    double F = 168.0;
    double jc[] = { -10.3506, 17.5716, -3019.39, 188.730, 0.0429036, 145.290,
        6127.68 };
    double kc[] = { -9.74602, 18.0834, -4126.66, 44.6055, 0.976544, 81.8134,
        15649.9 };

    // Set the mass and delta_groups for a mixture of pure methane
    double mass0[] = { 1. };
    double delta_groups0[] = { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Allocate the local arrays
    z = alloc_1d_array(nc, sizeof(*z), "z in viscosity");
    if (z == NULL) {
        goto fail;
    }
    M = alloc_1d_array(nc, sizeof(*M), "M in viscosity");
    if (M == NULL) {
        goto fail;
    }

    // Enter the properties for the reference fluid (methane)
    M0[0] = 16.043e-3;
    Tc0[0] = 190.56;
    Pc0[0] = 4599000.0;
    omega0[0] = 0.011;
    Vc0[0] = 9.86e-5;
    delta0[0] = 0.0;
    C_pen0[0] = 0.;
    C_pen_T0[0] = 0.;
    rho_c0 = 162.84;

    /* 1. Prepare the variables to determine the corresponding states between
     *    the given mixture and the reference fluid (methane) */

    // Get the mole fraction of the components of the mixture
    mole_fraction(nc, mass, Mol_wt, z);

    // Compute equation (10.19)
    numerator = 0.0;
    denominator = 0.0;
    for (i = 0; i < nc; i++) {
        for (j = 0; j < nc; j++) {
            numerator += z[i] * z[j] *
                         pow(pow(Tc[i] / Pc[i], 1.0 / 3.0) +
                                 pow(Tc[j] / Pc[j], 1.0 / 3.0),
                             3) *
                         sqrt(Tc[i] * Tc[j]);
            denominator += z[i] * z[j] *
                           pow(pow(Tc[i] / Pc[i], 1.0 / 3.0) +
                                   pow(Tc[j] / Pc[j], 1.0 / 3.0),
                               3);
        }
    }
    Tc_mix = numerator / denominator;

    // Compute equation (10.22)
    Pc_mix = 8.0 * numerator / pow(denominator, 2);

    // Get the density of methane at TTc0/Tc_mix and PPc0/Pc_mix
    _density(1,              // nc
        T * Tc0[0] / Tc_mix, // T
        P * Pc0[0] / Pc_mix, // P
        mass0,               // mass
        M0,                  // Mol_wt
        Pc0,                 // Pc
        Tc0,                 // Tc
        Vc0,                 // Vc
        omega0,              // omega
        delta0,              // delta
        Aij,                 // Aij
        Bij,                 // Bij
        delta_groups0,       // delta_groups
        -1,                  // calc_delta
        C_pen,               // C_pen
        C_pen_T,             // C_pen_T
        rho0                 // rho
    );

    // Compute equation (10.27)
    for (i = 0; i < 2; i++) {
        rho_r[i] = rho0[i] / rho_c0;
    }

    // Compute equation (10.23), where M is in g/mol
    M_bar_n = 0.;
    for (i = 0; i < nc; i++) {
        M[i] = Mol_wt[i] * 1.0e3;
        M_bar_n += z[i] * M[i];
    }
    M_bar_w_sum = 0.;
    for (i = 0; i < nc; i++) {
        M_bar_w_sum += z[i] * pow(M[i], 2);
    }
    M_bar_w = M_bar_w_sum / M_bar_n;
    M_mix = 1.304e-4 * (pow(M_bar_w, 2.303) - pow(M_bar_n, 2.303)) + M_bar_n;

    // Compute equation (10.26), where M is in g/mol
    for (i = 0; i < 2; i++) {
        alpha_mix[i] =
            1.0 + 7.378e-3 * pow(rho_r[i], 1.847) * pow(M_mix, 0.5173);
        alpha0[i] =
            1.0 + 7.378e-3 * pow(rho_r[i], 1.847) * pow(M0[0] * 1.0e3, 0.5173);
    }

    /* 2.  Compute the viscosity of methane at the corresponding state */

    // Corresponding state
    for (i = 0; i < 2; i++) {
        T0[i] = T * Tc0[0] / Tc_mix * alpha0[i] / alpha_mix[i];
        P0[i] = P * Pc0[0] / Pc_mix * alpha0[i] / alpha_mix[i];
    }

    // Compute each corresponding state separately
    for (i = 0; i < 2; i++) {

        // Get the density of methane at T0 and P0.  Be sure to use molecular
        // weight in kg/mol
        _density(1, T0[i], P0[i], mass0, M0, Pc0, Tc0, Vc0, omega0, delta0,
            Aij, Bij, delta_groups0, -1, C_pen, C_pen_T, rho0);

        // Compute each phase within a state separately
        for (j = 0; j < 2; j++) {

            // Compute equation (10.10)
            theta[j] = (rho0[j] - rho_c0) / rho_c0;

            // Equation (10.9) with T in K and rho in g/cm^3
            rho0[j] = rho0[j] * 1.0e-3;

            delta_eta_p[j] =
                exp(jc[0] + jc[3] / T0[i]) *
                (exp(pow(rho0[j], 0.1) * (jc[1] + jc[2] / pow(T0[i], 1.5)) +
                     theta[j] * sqrt(rho0[j]) *
                         (jc[4] + jc[5] / T0[i] + jc[6] / pow(T0[i], 2))) -
                    1.0);

            // Equation (10.28)
            delta_eta_pp[j] =
                exp(kc[0] + kc[3] / T0[i]) *
                (exp(pow(rho0[j], 0.1) * (kc[1] + kc[2] / pow(T0[i], 1.5)) +
                     theta[j] * sqrt(rho0[j]) *
                         (kc[4] + kc[5] / T0[i] + kc[6] / pow(T0[i], 2))) -
                    1.0);
        }

        // Equation (10.7)
        eta_0 = GV[0] / T0[i] + GV[1] / pow(T0[i], 2.0 / 3.0) +
                GV[2] / pow(T0[i], 1.0 / 3.0) + GV[3] +
                GV[4] * pow(T0[i], 1.0 / 3.0) + GV[5] * pow(T0[i], 2.0 / 3.0) +
                GV[6] * T0[i] + GV[7] * pow(T0[i], 4.0 / 3.0) +
                GV[8] * pow(T0[i], 5.0 / 3.0);

        // Equation (10.8)
        eta_1 = A + B * pow(C - log(T0[i] / F), 2);

        // Equation (10.32)
        delta_T = T0[i] - 91.0;

        // Equation (10.31)
        htan = (exp(delta_T) - exp(-delta_T)) / (exp(delta_T) + exp(-delta_T));

        // Viscosity of methane (Equation 10.29) -- reported in (Pa s)
        eta_ch4[i] = (eta_0 + eta_1 + (htan + 1.0) / 2.0 * delta_eta_p[i] +
                         (1.0 - htan) / 2.0 * delta_eta_pp[i]) *
                     1.0e-7;
    }

    // Compute the viscosity of the mixture at the given T and P
    for (i = 0; i < 2; i++) {
        mu[i] = pow(Tc_mix / Tc0[0], -1.0 / 6.0) *
                pow(Pc_mix / Pc0[0], 2.0 / 3.0) *
                sqrt(M_mix / (M0[0] * 1.e3)) * alpha_mix[i] / alpha0[i] *
                eta_ch4[i];
    }

    // Completed successfully
    goto success;

success:
    free(z);
    free(M);
    return;

fail:
    if (z != NULL)
        free(z);
    if (M != NULL)
        free(M);
    return;
}

void volume_trans(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *C_pen, double *C_pen_T,
    double *vt)
{
    /*
     * Computes the volume translation parameter to correct the density
     *
     * Computes the volume translation parameter to correct the density from
     * the Peng-Robinson Equation of State.
     *
     * TAMOC provides too methods.  First, temperature-dependent volume
     * translation coefficients can be estimated for atmospheric gases and
     * hydrocarbons using the approach in Lin and Duan (2005), "Empirical
     * correction to the Peng-Robinson equation of state for the saturated
     * region," Fluid Phase Equilibria, 233: 194-203.  The implementation
     * of this code was in collaboration with Jonas Gros.
     *
     * Second, a simpler equation from Pedersen et al.(2015) Phase Behavior of
     * Petroleum Reservoir Fluids can be used, which has the form:
     *
     *    vt = C_pen * C_pen_T * (T - 288.15)
     *
     * This may be convenient, for example, if the user has fit density
     * estimates to measurements using this equation in the PVT Sim software
     * package.
     *
     * This function uses the Lin and Duan (2005) approach as long as the
     * first element of the C_pen array is zero.  Otherwise, it uses the
     * simpler empirical equation.
     *
     * The volumetranslation parameter has a value for each component in the
     * mixture.
     *
     * Input Variables are:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     Vc = array of critical point molar volumes for each component
     *         (m^3/mol)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     C_pen = Peneloux volume translation coefficient (m^3/mol)
     *     C_pen_T = Peneloux parameter temperature correction (m^3/(mol K))
     *
     * Output variable is:
     *     vt = an array of volume translation parameters, m^3/mol.
     *         The return array is a flat C array corresponding of length
     *         nc.
     *
     * S. Socolofsky
     * December 2014 -> Fortran 95; May 2025 -> C
     */

    int i;
    double *Zc = NULL, *beta = NULL, *gamma = NULL, *f_Tr = NULL, *cc = NULL;

    // Allocate the local arrays
    Zc = alloc_1d_array(nc, sizeof(*Zc), "Zc in volume_trans");
    if (Zc == NULL) {
        goto fail;
    }
    beta = alloc_1d_array(nc, sizeof(*beta), "beta in volume_trans");
    if (beta == NULL) {
        goto fail;
    }
    gamma = alloc_1d_array(nc, sizeof(*gamma), "gamma in volume_trans");
    if (gamma == NULL) {
        goto fail;
    }
    f_Tr = alloc_1d_array(nc, sizeof(*f_Tr), "f_Tr in volume_trans");
    if (f_Tr == NULL) {
        goto fail;
    }
    cc = alloc_1d_array(nc, sizeof(*cc), "cc in volume_trans");
    if (cc == NULL) {
        goto fail;
    }

    // Decide how to get the Peneloux shift parameters
    if (C_pen[0] == 0.) {
        // Use the Lin and Duan (2005) method

        for (i = 0; i < nc; i++) {

            /* Compute the compressibility factor (--) for each component of
             * the mixture */
            Zc[i] = Pc[i] * Vc[i] / (RU * Tc[i]);

            /* Calculate the parameters in the Lin and Duan (2005) paper:  beta
             * is from equation (12) */
            beta[i] = -2.8431 * exp(-64.2184 * (0.3074 - Zc[i])) + 0.1735;

            // and gamma is from Equation (13) */
            gamma[i] = -99.2558 + 301.6201 * Zc[i];

            // Account for the temperature dependence (equation 10)
            f_Tr[i] = beta[i] +
                      (1.0 - beta[i]) * exp(gamma[i] * fabs(1.0 - T / Tc[i]));

            /* Compute the volume translation for the critical point
             * (equation 9) */
            cc[i] = (0.3074 - Zc[i]) * RU * Tc[i] / Pc[i];

            /* Finally, the volume translation at the given state is
             * (equation 8) */
            vt[i] = f_Tr[i] * cc[i];
        }
    } else {
        /* Use the user-defined Peneloux parameters following equation
         * 5.9 in Pedersen et al.(2015) Phase Behavior of Petroleum
         * Reservoir Fluids */
        for (i = 0; i < nc; i++) {
            vt[i] = C_pen[i] + C_pen_T[i] * (T - 288.15);
        }
    }

    // Successfully computed the volume translation coefficients
    goto success;

success:
    free(Zc);
    free(beta);
    free(gamma);
    free(f_Tr);
    free(cc);
    return;

fail:
    if (Zc != NULL)
        free(Zc);
    if (beta != NULL)
        free(beta);
    if (gamma != NULL)
        free(gamma);
    if (f_Tr != NULL)
        free(f_Tr);
    if (cc != NULL)
        free(cc);
    return;
}

void z_pr(int nc, double T, double P, double *mass, double *Mol_wt, double *Pc,
    double *Tc, double *omega, double *delta, double *yk, double *Aij,
    double *Bij, double *delta_groups, int calc_delta, double *z, double *A,
    double *B, double *Ap, double *Bp)
{
    /*
     * Computes the z-factor for gas and liquid of a mixture using the P-R EOS
     *
     * Computes the z-factor of a mixture using the Peng-Robinson equation of
     * state as described in McCain (1990), Properties of Petroleum Fluids, 2nd
     * Edition, PennWell Publishing Company, Tulsa, Oklahoma.
     *
     * The approach results in a cubic equation for the z-factor in which the
     * largest root is for the liquid phase and the smallest root is for the
     * gas phase; the middle root is discarded.  If the temperature is above
     * the critical temperature, only one real root is obtained for the
     * critical state.
     *
     * Input variables are:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     omega = array of Pitzer acentric factors for each component (--)
     *     delta = matrix of binary interaction coefficients (--)
     *     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
     *     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
     *     delta_groups = group contribution numbers (normalized) for each
     *         component in the mixture (--)
     *     calc_groups = flag indicating whether or not delta_groups has
     *         been provided (1 = yes, -1 = no)
     *     yk = mole fractions for each component in the mixture, --
     *
     * Output variables are:
     *     z = array of the z-factor [gas, liquid] for the mixture (--).
     *         The return array is a flat C array of two values.
     *     A, B, Ap, Bp = coefficients for the P-R EOS defined in coefs.
     *         These are passed by reference back to the calling function.
     *         Each of these arrays is a flat C array of length nc.
     *
     * S. Socolofsky
     * June 2013 -> Fortran 95; May 2025 -> C
     */

    int i;
    double z_max, z_min;
    double p_coefs[4];
    complex_double z_roots[3];

    // Compute the coefficients of the polynomial for z-factor
    coefs(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta, yk, Aij, Bij,
        delta_groups, calc_delta, A, B, Ap, Bp);
    p_coefs[0] = 1.0;
    p_coefs[1] = (*B) - 1.0;
    p_coefs[2] = (*A) - 2.0 * (*B) - 3.0 * pow((*B), 2);
    p_coefs[3] = pow((*B), 3) + pow((*B), 2) - (*A) * (*B);

    // Find the roots of the cubic equation of state
    _cubic_roots(p_coefs, z_roots);

    // Extract the correct z-factors
    z_max = 0.0;
    for (i = 0; i < 3; i++) {
        if (z_roots[i].i == zero) {
            if (z_roots[i].r > z_max) {
                z_max = z_roots[i].r;
            }
        }
    }
    z_min = z_max;
    for (i = 0; i < 3; i++) {
        if (z_roots[i].i == zero) {
            if ((z_roots[i].r < z_min) && (z_roots[i].r > zero)) {
                z_min = z_roots[i].r;
            }
        }
    }

    // Return the z - factors in z
    z[0] = z_max;
    z[1] = z_min;

    // Success
    return;
}

void coefs(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *omega, double *delta_in, double *yk,
    double *Aij, double *Bij, double *delta_groups, int calc_delta, double *A,
    double *B, double *Ap, double *Bp)
{
    /*
     * Computes the mixture coefficients for the P-R EOS
     *
     * Computes the mixing rules for the coefficients of the Peng-Robinson
     * equation of state as described in McCain (1990), Properties of Petroleum
     * Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
     *
     * Input variables are:
     *     nc = number of components in the mixture
     *     T = temperature (K)
     *     P = pressure (Pa)
     *     mass = array of masses for each component in the mixture (kg)
     *     Mol_wt = array of molecular weights for each component (kg/mol)
     *     Pc = array of critical point pressures for each component (Pa)
     *     Tc = array of critical point temperatures for each component (K)
     *     omega = array of Pitzer acentric factors for each component (--)
     *     delta = matrix of binary interaction coefficients (--)
     *     yk = mole fractions for each component in the mixture, --
     *     Aij = group contribution matrix A in Privat and Jaubert (2012) (Pa)
     *     Bij = group contribution matrix B in Privat and Jaubert (2012) (Pa)
     *     delta_groups = group contribution numbers (normalized) for each
     *         component in the mixture (--)
     *     calc_groups = flag indicating whether or not delta_groups has
     *         been provided (1 = yes, -1 = no)
     *
     * Output variables are:
     *     A = aT coefficient in P-R EOS.  This is a 1D C-array of length nc
     *     B = b coefficient in P-R EOS.  This is a 1D C-array of length nc
     *     Ap = non-dimensional array of mixture aT-coefficients.  This is
     *         a 1D C-array of length nc
     *     Bp = non-dimensional array of mixture b-coefficients.  This is
     *         a 1D C-array of length nc
     *
     * S. Socolofsky
     * June 2013 -> Fortran 95; May 2025 -> C
     */

    int i, j, k, l;
    double bd, aT, sum_term, sum1, sum_for_Api;
    double *mu, *alpha, *aTk, *bk;
    double *delta;

    // Allocate the local arrays
    mu = alloc_1d_array(nc, sizeof(*mu), "mu in coefs");
    if (mu == NULL) {
        goto fail;
    }
    alpha = alloc_1d_array(nc, sizeof(*alpha), "mu in coefs");
    if (mu == NULL) {
        goto fail;
    }
    aTk = alloc_1d_array(nc, sizeof(*aTk), "aTk in coefs");
    if (aTk == NULL) {
        goto fail;
    }
    bk = alloc_1d_array(nc, sizeof(*bk), "bk in coefs");
    if (bk == NULL) {
        goto fail;
    }
    delta = alloc_2d_array_flat(nc, nc, sizeof(*delta), "delta in coefs");
    if (delta == NULL) {
        goto fail;
    }

    /* Compute the coefficient values for each component in the mixture.  Use
     * the modified Peng-Robinson (1978) equations for mu */
    for (i = 0; i < nc; i++) {
        if (omega[i] > 0.49) {
            mu[i] = 0.379642 + 1.48503 * omega[i] -
                    0.164423 * pow(omega[i], 2) + 0.016666 * pow(omega[i], 3);
        } else {
            mu[i] = 0.37464 + 1.54226 * omega[i] - 0.26992 * pow(omega[i], 2);
        }
        alpha[i] = pow(1.0 + mu[i] * (1.0 - sqrt(T / Tc[i])), 2);
        aTk[i] = 0.45724 * pow(RU, 2) * pow(Tc[i], 2) / Pc[i] * alpha[i];
        bk[i] = 0.07780 * RU * Tc[i] / Pc[i];
    }

    // Initialize the vector for delta to the input values
    for (i = 0; i < nc; i++) {
        for (j = 0; j < nc; j++) {
            delta[i * nc + j] = delta_in[i * nc + j];
        }
    }

    /* Get the temperature-dependent binary interaction coefficients (if
     * the user provided the group contributions) */
    if (calc_delta > 0) {
        for (j = 1; j < nc; j++) {
            for (i = 0; i < j - 1; i++) {
                sum1 = 0.0;
                for (l = 0; l < 15; l++) {
                    for (k = 0; k < 15; k++) {
                        sum_term =
                            (delta_groups[i * 15 + k] -
                                delta_groups[j * 15 + k]) *
                            (delta_groups[i * 15 + l] -
                                delta_groups[j * 15 + l]) *
                            Aij[k * 15 + l] *
                            pow(298.15 / T,
                                Bij[k * 15 + l] / Aij[k * 15 + l] - 1.0);
                        if (!isnan(sum_term)) {
                            sum1 += sum_term;
                        }
                    }
                }
                delta[i * nc + j] =
                    -(0.5 * sum1 +
                        pow(sqrt(aTk[i]) / bk[i] - sqrt(aTk[j]) / bk[j], 2)) /
                    (2.0 * sqrt(aTk[i] * aTk[j]) / (bk[i] * bk[j]));
                delta[j * nc + i] = delta[i * nc + j];
            }
        }
    }

    // Use the mixing rules in McCain (1990)
    bd = dot(yk, bk, nc);
    aT = 0.;
    for (i = 0; i < nc; i++) {
        for (j = 0; j < nc; j++) {
            aT += yk[i] * yk[j] * sqrt(aTk[i] * aTk[j]) *
                  (1.0 - delta[i * nc + j]);
        }
    }

    // Compute the coefficients of the polynomials for z-factor and fugacity
    *A = aT * P / (pow(RU, 2) * pow(T, 2));
    *B = bd * P / (RU * T);
    for (i = 0; i < nc; i++) {
        Bp[i] = bk[i] / bd;
        sum_for_Api = 0.;
        for (j = 0; j < nc; j++) {
            sum_for_Api += yk[j] * sqrt(aTk[j]) * (1.0 - delta[j * nc, i]);
        }
        Ap[i] = 1.0 / aT * (2.0 * sqrt(aTk[i]) * sum_for_Api);
    }

success:
    free(mu);
    free(alpha);
    free(aTk);
    free(bk);
    free(delta);
    return;

fail:
    if (mu != NULL)
        free(mu);
    if (alpha != NULL)
        free(alpha);
    if (aTk != NULL)
        free(aTk);
    if (bk != NULL)
        free(bk);
    if (delta != NULL)
        free(delta);
    return;
}

/* --------------------------------------------------------------------------
 * Functions from math_funcs.f95 to solve a cubic polynomial
 * ------------------------------------------------------------------------*/

void swap(double *a, double *b)
{
    /*
     * Swap the contents of a and b
     *
     * Swap the value stored in 'a' with the value stored in 'b' and vice
     * versa
     *
     * Input variables are:
     *     a = double-precision float value (intent in/out)
     *     b = double-precision float value (intent in/out)
     *
     * Output variables are:
     *     a = float value containing the original value stored in b
     *     b = float value containing the original value stored in a
     *
     * This function is adapted from the quartic.f90 module provided by the
     * Public Domain Aeronautical Software (PDAS) project, downloaded from:
     *     https://www.pdas.com/quarticdownload.html
     *
     * S. Socolofsky and J. Samuel Arey
     * October 2023 -> Fortran 95; May 2025 -> C.
     */

    double t;

    // swap the values
    t = *b;
    *b = *a;
    *a = t;
}

double cube_root(double x)
{
    /*
     * Computes the cube root of a real number.
     *
     * Computes the cube root of a real number and preserves the sign so
     that
     * if the argument is negative, the cube root is also negative.
     *
     * Input variable is:
     *     x = real number we want the cube root of
     *
     * Output variable is:
     *     f = cube root of the input value x
     *
     * This function is adapted from the quartic.f90 module provided by the
     * Public Domain Aeronautical Software (PDAS) project, downloaded from:
     *     https://www.pdas.com/quarticdownload.html
     *
     * S. Socolofsky

     * October 2023 -> Fortran 95; May 2025 -> C.
     */

    // Declare the output variable
    double f;

    if (x < zero) {
        f = -exp(log(-x) / three);
    } else if (x > zero) {
        f = exp(log(x) / three);
    } else {
        f = zero;
    }

    return f;
}

void quadratic_roots(double a[3], complex_double z[2])
{
    /*
     * Computes the roots of a quadratic polynomial with coefficients a().
     *
     * Computes the roots of the quadratic equation:
     *     a(0) + a(1) * z + a(2) * z**2 = 0
     *
     * Input variables are:
     *    a = array of double-precision coefficients to the quadratic
     *        equation (intent in)
     *    z = array containing two complex_double roots to the quadratic
     *        equation (intent out)
     *
     * This subroutine is used by cubic_roots() when one of the roots is
     * zero (as when the constant coefficient a(0) of the cubic equation
     * is zero).
     *
     * This function is adapted from the quartic.f90 module provided by the
     * Public Domain Aeronautical Software (PDAS) project, downloaded from:
     *     https://www.pdas.com/quarticdownload.html
     *
     * S. Socolofsky and J. Samuel Arey
     * October 2023 -> Fortran 95; May 2025 -> C.
     */

    // Declare variables internal to this function
    double d, r, w, x, y;

    /* If one root is zero */
    if (a[0] == zero) {
        /* One root is obviously zero */
        z[0] = czero;

        /* The remaining root is a linear equation */
        z[1].r = (-a[1] / a[2]);
        z[1].i = zero;
        return;
    }

    // Calculate the discriminant
    d = a[1] * a[1] - four * a[2] * a[0];

    /* If the discriminant is small, it is a double root = -b/(2a) */
    if (fabs(d) <= two * eps * a[1] * a[1]) {
        /* The disriminant is tiny */
        z[0] = (complex_double){ -half * a[1] / a[2], zero };
        z[1] = z[0];
        return;
    }

    // Calculate the root of the absolute value of the discriminant
    r = sqrt(fabs(d));

    /* Case of complex roots */
    if (d < zero) {
        x = -half * a[1] / a[2];
        y = fabs(half * r / a[2]);
        z[0] = (complex_double){ x, y };
        /* and its complex conjugate */
        z[1] = (complex_double){ x, -y };
        return;
    }

    // Case of real roots
    if (a[1] != zero) {
        // See Numerical Recipes, Section 5.5
        w = -(a[1] + copysign(r, a[1]));
        z[0] = (complex_double){ two * a[0] / w, zero };
        z[1] = (complex_double){ half * w / a[2], zero };
        return;
    }

    // Case of a[1] = 0 and real roots
    x = fabs(half * r / a[2]);
    z[0] = (complex_double){ x, zero };
    z[1] = (complex_double){ -x, zero };

    return;
}

void _cubic_roots(double a_t[4], complex_double z[3])
{
    /*
     * Computes the roots of a cubic polynomial with coefficients a_t()
     *
     * Computes the roots of a 3rd-order polynomial with real-valued
     * coefficients specified in a_t().  The order of the coefficients in
     * a_t() are given by
     *
     *     a_t(0) * x**3 + a_t(1) * x**2 + a_t(2) * x + a_t(3) = 0
     *
     * Input variables are:
     *    a_t = array of double-precision coefficients to the cubic
     *        polynomial (intent in)
     *    z = array containing two complex_double roots to the cubic
     *        polynomial (intent out)
     *
     * The original TAMOC cubic_roots() subroutine was replaced with this
     *one in October 2023.  The original solver had difficulty computing
     *roots for some single-phase compositions where there is one real
     *root.  This new solver was developed as a collaborative effort of
     *Scott A. Socolofsky and J. Samuel Arey.  It is subject to the same
     *license as all other TAMOC files.
     *
     * This function is adapted from the quartic.f90 module provided by the
     * Public Domain Aeronautical Software (PDAS) project, downloaded from:
     *     https://www.pdas.com/quarticdownload.html
     *
     * Note
     * ----
     * The original Fortran code utilizes several GO TO statements that
     *cannot be readily converted to a normal select structure (e.g., if or
     *select). We utilize the C goto functionality here.  In Fortran, the
     *jump is always to a line number.  Here, we label these a flnXX, where
     *'fln' is an acronym for 'Fortran line number' and 'XX' will be the
     *same line number as in the TAMOC math_funcs.f95 code.
     *
     *
     * S. Socolofsky and J. Samuel Arey
     * October 2023 -> Fortran 95; May 2025 -> C.
     *
     * The original header for the Fortran module containing the quartic
     * solver adapted here is pasted as follows:
     *
     *!
     *------------------------------------------------------------------------
     *! PURPOSE - Solve for the roots of a polynomial equation with real
     *!   coefficients, up to quartic order. Retrun a code indicating the
     *nature !   of the roots found.
     *!
     *! AUTHORS - Alfred H. Morris, Naval Surface Weapons Center,
     *Dahlgren,VA !           William L. Davis, Naval Surface Weapons
     *Center, Dahlgren,VA !           Alan Miller,  CSIRO Mathematical &
     *Information Sciences !                         CLAYTON, VICTORIA,
     *AUSTRALIA 3169 ! http://www.mel.dms.csiro.au/~alan !           Ralph
     *L. Carmichael, Public Domain Aeronautical Software !
     *http://www.pdas.com !     REVISION HISTORY !   DATE  VERS PERSON
     *STATEMENT OF CHANGES !    ??    1.0 AHM&WLD Original coding ! 27Feb97
     *1.1   AM    Converted to be compatible with ELF90 ! 12Jul98  1.2 RLC
     *Module format; numerous style changes !  4Jan99  1.3   RLC   Made the
     *tests for zero constant term exactly zero
     */

    // Declare variables internal to this function
    double a[4];
    static double rt3 = 1.7320508075689; /* sqrt(3) */
    double aq[3], arg, c, cf, d, p, p1, q, q1, r, ra, rb, rq, rt, r1, s, sf;
    double sq, sum, t, tol, t1, w, w1, w2, x, x1, x2, x3, y, y1, y2, y3;

    // NOTE - It is assumed that a_t(3) is non-zero.  No test is made here.

    /* TAMOC sends the coefficients in the reverse order to those used in
     * this subroutine */
    a[3] = a_t[0];
    a[2] = a_t[1];
    a[1] = a_t[2];
    a[0] = a_t[3];

    // check constant coefficient
    if (a[0] == zero) {
        // One root is obviously zero
        z[0] = czero;
        // The remaining roots are for a quadratic equation
        quadratic_roots(&a[1], &z[1]);
        return;
    }

    // Set up some constant parameters
    p = a[2] / (three * a[3]);
    q = a[1] / a[3];
    r = a[0] / a[3];
    tol = four * eps;

    c = zero;
    t = a[1] - p * a[2];
    if (fabs(t) > tol * fabs(a[1]))
        c = t / a[3];

    t = two * p * p - q;
    if (fabs(t) <= tol * fabs(q))
        t = zero;
    d = r + p * t;

    if (fabs(d) <= tol * fabs(r)) {
        goto fln110;
    }

    // Set sq = (a(3) / s)**2 * (c**3 / 27 + d**2 / 4))
    s = fmax(fmax(fabs(a[0]), fabs(a[1])), fabs(a[2]));
    p1 = a[2] / (three * s);
    q1 = a[1] / s;
    r1 = a[0] / s;

    t1 = q - 2.25 * p * p;
    if (fabs(t1) <= tol * fabs(q))
        t1 = zero;
    w = fourth * r1 * r1;
    w1 = half * p1 * r1 * t;
    w2 = q1 * q1 * t1 / 27.0;

    if (w1 >= zero) {
        w = w + w1;
        sq = w + w2;
    } else if (w2 < zero) {
        sq = w + (w1 + w2);
    } else {
        w = w + w2;
        sq = w + w1;
    }

    if (fabs(sq) <= tol * w)
        sq = zero;
    rq = fabs(s / a[3]) * sqrt(fabs(sq));

    if (sq >= zero)
        goto fln40;

    // If code reaches this point, all roots are real
    arg = atan2(rq, -half * d);
    cf = cos(arg / three);
    sf = sin(arg / three);
    rt = sqrt(-c / three);
    y1 = two * rt * cf;
    y2 = -rt * (cf + rt3 * sf);
    y3 = -(d / y1) / y2;

    x1 = y1 - p;
    x2 = y2 - p;
    x3 = y3 - p;

    if (fabs(x1) > fabs(x2))
        swap(&x1, &x2);
    if (fabs(x2) > fabs(x3))
        swap(&x2, &x3);
    if (fabs(x1) > fabs(x2))
        swap(&x1, &x2);

    w = x3;

    if (fabs(x2) < 0.1 * fabs(x3))
        goto fln70;

    if (fabs(x1) < 0.1 * fabs(x2))
        x1 = -(r / x3) / x2;

    // Set the roots
    z[0] = (complex_double){ x1, zero };
    z[1] = (complex_double){ x2, zero };
    z[2] = (complex_double){ x3, zero };
    return;

fln40:
    // Real and complex roots
    ra = cube_root(-half * d - copysign(rq, d));
    rb = -c / (three * ra);
    t = ra + rb;
    w = -p;
    x = -p;

    if (fabs(t) <= tol * fabs(ra))
        goto fln41;
    w = t - p;
    x = -half * t - p;
    if (fabs(x) <= tol * fabs(p))
        x = zero;

fln41:
    // Continuing with real and complex roots
    t = fabs(ra - rb);
    y = half * rt3 * t;

    if (t <= tol * fabs(ra))
        goto fln60;

    if (fabs(x) < fabs(y))
        goto fln50;

    s = fabs(x);
    t = y / x;
    goto fln51;

fln50:
    // Continuing with real and complex roots
    s = fabs(y);
    t = x / y;

fln51:
    // Continuing with real dna complex roots...source of jump to line 70
    if (s < 0.1 * fabs(w))
        goto fln70;
    w1 = w / s;
    sum = one + t * t;
    if (w1 * w1 < 0.01 * sum)
        w = -((r / sum) / s) / s;

    // Found one real root and a complex conjugate pair of roots
    z[0] = (complex_double){ w, zero };
    z[1] = (complex_double){ x, y };
    z[2] = (complex_double){ x, -y };
    return;

    /* At least two roots are equal */

fln60:
    if (fabs(x) < fabs(w))
        goto fln61;
    if (fabs(w) < 0.1 * fabs(x))
        w = -(r / x) / x;

    // Found real roots with one double root
    z[0] = (complex_double){ w, zero };
    z[1] = (complex_double){ x, zero };
    z[2] = z[1];
    return;

fln61:
    if (fabs(x) < 0.1 * fabs(w))
        goto fln70;
    // Found real roots with on double root
    z[0] = (complex_double){ x, zero };
    z[1] = z[0];
    z[2] = (complex_double){ w, zero };
    return;

    /* Here, 'w' is much larger in magnitude than the other roots.
     * as a result, the other roots may be exceedingly inaccurate because
     * of roundoff error.  To deal with this, a quadratic is formed whose
     * roots are the same as the smaller roots of the cubic.  This
     * quadratic is then solved.
     */

fln70:
    aq[0] = a[0];
    aq[1] = a[1] + a[0] / w;
    aq[2] = -a[3] * w;
    quadratic_roots(&aq[0], &z[0]);
    z[2] = (complex_double){ w, zero };

    if (z[0].i == zero)
        return;
    z[2] = z[1];
    z[1] = z[0];
    z[0] = (complex_double){ w, zero };
    return;

    // Case when d = 0
fln110:
    z[0] = (complex_double){ -p, zero };
    w = sqrt(fabs(c));

    if (c < zero)
        goto fln120;

    z[1] = (complex_double){ -p, w };
    z[2] = (complex_double){ -p, -w };
    return;

fln120:
    if (p != zero)
        goto fln130;
    z[1] = (complex_double){ w, zero };
    z[2] = (complex_double){ -w, zero };
    return;

fln130:
    x = -(p + copysign(w, p));
    z[2] = (complex_double){ x, zero };
    t = three * a[0] / (a[2] * x);

    if (fabs(p) > fabs(t))
        goto fln131;

    z[1] = (complex_double){ t, zero };
    return;

fln131:
    z[1] = z[0];
    z[0] = (complex_double){ t, zero };
    return;
}
