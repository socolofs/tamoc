/*
 * preos_cmodule.c
 * ---------------
 *
 * This is a C implementation of the Peng-Robinson Equation of State (EOS)
 * from the TAMOC package.  The Peng-Robinson EOS equations are implemented
 * in a C-library contained in preos.c.  This file provides wrapper code
 * to expose the density(), fugacity(), viscosity(), and cubic_roots()
 * functions to Python.  All other functions (e.g. volumne_trans, z_pr,
 * coefs, etc) are private to the preos.c C-library.
 *
 * This code was created as a direct translation from the dbm_eos.f95 Fortran
 * code in TAMOC.  The file dbm_fmodule.c created by ./numpy/f2py was
 * consulted to help with some of the module interface building.  The
 * style, syntax, and some of the code ideas were also adpated from the
 * Scipy package, following ./scipy/interpolate/src/_fitpackmodule.c
 *
 * May 2025
 */

// S. Socolofsky, Texas A&M University, May 2025, <socolofs@tamu.edu>

#ifndef PY_SSIZE_T_CLEAN // Py_SSIZE_T_CLEAN statement from f2py
#define PY_SSIZE_T_CLEAN
#endif

#include "numpy/arrayobject.h"
#include <Python.h>

#include <math.h>

static char *version_num = "0.1.0"; // Version number for this module
static PyObject *preos_c_error;     // Error object for this module

/* Define some global variables */
#define F_INT int

/* Borrow a complex number type from f2py */
typedef struct {
    double r, i;
} complex_double;

// Declare the functions coming from the preos.c library
void _cubic_roots(double[4], complex_double[3]);
void _density(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *omega, double *delta,
    double *Aij, double *Bij, double *delta_groups, int calc_delta,
    double *C_pen, double *C_pen_T, double *rho);
void _fugacity(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *omega, double *delta, double *Aij,
    double *Bij, double *delta_groups, int calc_delta, double *fug);
void _viscosity(int nc, double T, double P, double *mass, double *Mol_wt,
    double *Pc, double *Tc, double *Vc, double *omega, double *delta,
    double *Aij, double *Bij, double *delta_groups, int calc_delta,
    double *C_pen, double *C_pen_T, double *mu);

/* --------------------------------------------------------------------------
 * Write some functions to convert between local C arrays and Python
 * ndarrays.  These are used to facilitate getting data from Python to
 * this C interface and back to Python.
 * ------------------------------------------------------------------------*/

void get_arr_from_pyobj(PyObject *arr_pyin, const int np_type, int min_Dims,
    int max_Dims, const char *var_name, PyArrayObject **arr_np, double **arr)
{
    /*
     * Convert a Python Object that points to a NumPy array to a C array of
     * of identical typed data.
     *
     * Input variable are:
     *     arr_pyin = pointer to the PyObject parsed from the direct inputs
     *         from Python
     *     np_type = the NumPy datatype stored in the PyObject
     *     min_Dims = minimum allowable dimensions the NumPy array can have
     *     max_Dims = maximum allowable dimensions the NumPy array can have
     *     var_name = a string describing the data that has been parsed and
     * that will be passed to an error message if there are problems accessing
     *         the data.
     *
     * Output variables are:
     *     arr_np = a pointer to the pointer that contains the NumPy object.
     *         This input is passed by reference so that the object in the
     *         calling function is modified here in place.
     *     arr = a pointer to the pointer that contains the C array data.
     *         This input is passed by reference so that the object in the
     *         calling function is modified here in place.
     */

    // Convert the input PyObject to a NumPy object
    *arr_np = (PyArrayObject *)PyArray_ContiguousFromObject(
        arr_pyin,  /* Python object pass in to C from Python */
        np_type,   /* Datatype stored in Pyton object array */
        min_Dims,  /* Minimum allowable dimensions array can have */
        max_Dims); /* Maximum allowable dimensions array can have */
    if (*arr_np == NULL) {
        PyErr_Format(PyExc_MemoryError,
            "Could not convert input %s to Numpy array\n", var_name);
        return;
    }

    /* Convert the NumPy array to a C array.  The variable type will come
     * from the PyArrayObject *arr_np */
    *arr = (void *)PyArray_DATA(*arr_np);
    if (*arr == NULL) {
        PyErr_Format(PyExc_TypeError,
            "Could not create C double array from %s\n", var_name);
        Py_DECREF(*arr_np);
        return;
    }
    return;
}

void *alloc_1d_array(size_t len, size_t elem_size, const char *var_name)
{
    /*
     * Allocate memory to hold return values from preos.c library functions
     *
     * This function creates 1d arrays:
     *
     * Index using:  arr[i]
     *
     * Input variables are:
     *    len = number of elements to place in the array
     *    elem_size = the memory size for one element.  This is normally
     *        obtained from the build-in sizeof(datatype) function
     *    var_name = a string describing the data that has been parsed and
     *        that will be passed to an error message if there are problems
     *        the array the data.
     *
     * Output variables are:
     *    arr = a pointer to the memory block for the requested array
     */

    // Allocate memory for an array
    void *arr = malloc(len * elem_size);

    // Write an error to Python if the allocation fails...no memory to free
    if (arr == NULL) {
        PyErr_Format(PyExc_MemoryError,
            "Could not allocate memory for C-library return value %s",
            var_name);
    }
    return arr;
}

void *alloc_2d_array_flat(size_t rows, size_t cols, size_t elem_size,
    const char *var_name)
{
    /*
     * Allocate memory to hold return values from preos.c library functions
     *
     * This function creates 2d flat arrays:
     *
     * Index using:  arr[i * cols + j]
     *
     * Fill using:
     *
     * for (size_t i = 0; i < rows; ++i) {
     *     for (size_t j = 0; j < cols; ++j) {
     *         A[i * cols + j] = some_function(i, j);
     *     }
     * }
     *
     * Input variables are:
     *    rows = number of rows in the 2d array
     *    cols = number of column in the 2d array
     *    elem_size = the memory size for one element.  This is normally
     *        obtained from the build-in sizeof(datatype) function
     *    var_name = a string describing the data that has been parsed and
     *        that will be passed to an error message if there are problems
     *        the array the data.
     *
     * Output variables are:
     *    arr = a pointer to the memory block for the requested array
     */

    // Allocate memory for an array
    size_t len = rows * cols;
    return alloc_1d_array(len, elem_size, var_name);
}

PyObject *get_pyobj_from_arr(const int np_type, npy_intp *arr_Dims,
    const int arr_Rank, void *arr, const char *var_name)
{
    /*
     * Convert a C array of data to a NumPy array PyObject
     *
     * Input variables are:
     *     np_type = the NumPy datatype to store in the new PyObject
     *     arr_Dims = an array holding the NumPy shape of the array to
     *         create
     *     arr_Rank = the number of dimensions that the array holds.  This
     *         should match the number of dimensions of 'arr_Dims'
     *     arr = pointer to the memory location that holds the C array
     *     var_name = a string describing the data that has been parsed and
     * that will be passed to an error message if there are problems accessing
     *         the data.
     *
     * Output variables are:
     *     arr_pyout = a Python PyObject in the structure of a NumPy ndarray
     *         that can be passed directly back to Python
     */

    PyArray_Descr *descr = NULL;
    PyArrayObject *arr_pyout = NULL;

    // Create a NumPy description object for this NPY data type
    descr = PyArray_DescrFromType(np_type);

    // If the descr creation failed, exit with error message
    if (!descr) {
        PyErr_Format(PyExc_RuntimeError,
            "Could not create datatype description for %s\n", var_name);
        goto fail;
    }

    // Allocate a new NumPy array and fill with the passed 'arr' data
    arr_pyout = (PyArrayObject *)PyArray_NewFromDescr(
        &PyArray_Type, // subtype -- PyArray_Type is built-in
        descr,         // created above from passed np_type
        arr_Rank,      // number of dimensons
        arr_Dims,      // shape of array
        NULL,          // strides (NULL -> default C-contiguous)
        arr,           // data (*arr points to memory location or arr)
        NPY_ARRAY_C_CONTIGUOUS | // flags
            NPY_ARRAY_OWNDATA,   /* tells NumPy to take over memory
                                  * management of 'arr' */
        NULL                     // Object from which data is borrowed (none)
    );

    // If ndarray creation failed, exit with error message
    if (!arr_pyout) {
        PyErr_Format(PyExc_MemoryError, "Could not allocate ndarray for %s\n",
            var_name);
        goto fail;
    }

    return (PyObject *)arr_pyout;

fail:

    /* If we created descr but did not successfully pass it to arr_pyout,
     * we need to free the descr memory */
    if (descr && !arr_pyout) {
        Py_XDECREF(descr);
    }
    return NULL;
}

/* --------------------------------------------------------------------------
 * Create wrappers for each exposed function from the preos.c libraray
 * ------------------------------------------------------------------------*/

static char doc_cubic_roots[] = "[z] = cubic_roots(a_t)";
static PyObject *preos_c_cubic_roots(PyObject *self, PyObject *args)
{
    /*
     * Wrapper from Python to the cubic_roots() function of the
     * preos_c library.
     *
     * In Python:
     *
     * z = cubic_roots(a_t)
     *
     * Parameters
     * ----------
     * a_t : ndarray
     *     Array of length 4 containing the double-precision coefficients
     *     of the polynomial a[0] x**3 + a[1] x**2 + a[2] x + a[3] = 0
     *
     * Returns
     * -------
     * z : ndarray
     *     Array of complex roots of the polynomial (length 3)
     *
     */

    PyObject *a_t_pyin = NULL;    // Input argument a_t PyObject
    PyArrayObject *a_t_np = NULL; // NumPy array version of a_t
    double *a_t = NULL;           // C array passed to _preos library

    complex_double *z = NULL;    // C array returned from _preos library
    const int z_Rank = 1;        // Number of dimensions of z
    npy_intp z_Dims[1] = { -1 }; // Array for shape of z
    PyObject *z_pyout = NULL;    // PyOjbect output argument

    int n;

    // Convert the Python inputs to working arrays --------------------------

    /* Parse the input */
    if (!PyArg_ParseTuple(args, "O", &a_t_pyin)) {
        /* Input parameters could not be cast to local variables */
        return NULL;
    }

    // Convert the input to a C-style contiguous array
    get_arr_from_pyobj(a_t_pyin, NPY_DOUBLE, 0, 1, "polynomial coefs a_t",
        &a_t_np, &a_t);

    // Do the numerical calculations of this function -----------------------

    // Make sure the user provided exactly four polynomial coefficients
    n = PyArray_DIMS(a_t_np)[0];
    if (n != 4) {
        PyErr_SetString(PyExc_ValueError,
            "You must provide four coefficients to a cubic polynomial.\n");
        goto fail;
    }

    // Get the roots of the cubic polynomial
    z = (complex_double *)alloc_1d_array(3, sizeof(complex_double),
        "roots, z");
    if (z == NULL) {
        goto fail;
    }
    _cubic_roots(a_t, z);

    // If you want to write to the screen, use PySys_WriteStdout()

    // Convert the function results to Numpy return values ------------------

    // Create a new Numpy array to hold the return value
    z_Dims[0] = 3;
    z_pyout = get_pyobj_from_arr(NPY_CDOUBLE, z_Dims, z_Rank, z,
        "cubic roots result, z");
    if (z_pyout == NULL) {
        goto fail;
    }

    // Success
    return (PyObject *)z_pyout;

fail:
    // Clear-up allocated memory and return NULL
    if (z && !z_pyout) {
        free(z);
    }
    Py_XDECREF(a_t_np);
    return NULL;
}

static char doc_density[] =
    "[rho] = density(T, P, m, M, Pc, Tc, Vc, omega, delta, Aij, Bij,"
    "                delta_groups, calc_delta, C_pen, C_pen_T";
static PyObject *preos_c_density(PyObject *self, PyObject *args)
{
    /*
     * Wrapper from Python to the density() function of the
     * preos_c library.
     *
     * In Python:
     *
     * rho = density(T, P, m, M, Pc, Tc, Vc, omega, delta, Aij, Bij,
     *     delta_groups, calc_delta, C_pen, C_pen_T)
     *
     * Parameters
     * ----------
     * T : double float
     *     Temperature, K
     * P : double float
     *     Pressure, Pa
     * m : ndarray of length nc
     *     Masses of each component of the mixture, kg
     * M : ndarray of length nc
     *     Molecular weights, kg/mol
     * Pc : ndarray of length nc
     *     Critical pressures, Pa
     * Tc : ndarray of length nc
     *     Critical temperatures, K
     * Vc : ndarray of length nc
     *     Critical molar volume, m^3/mol
     * omega : ndarray of length nc
     *     Aacentric factors, --
     * delta : ndarray of shape (nc, nc)
     *     Binary interaction coefficients, --
     * Aij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * Bij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * delta_groups : ndarray of shape (nc, 15)
     *     Group contribution factors for the Privat and Jaubert (2012)
     *     temperature-dependent binary interaction coefficients model
     * calc_delta : int
     *     Flag determining whether to compute temperature-dependent binary
     *     interaction coefficients (1: True, -1: False)
     * C_pen : ndarray of length nc
     *     Peneloux volume translation coefficients, m^3/mol
     * C_pen_T : ndarray of length nc
     *     Temperature correction parameter for the Peneloux volume
     *     translation coefficients, m^3/(mol K)
     *
     * Returns
     * -------
     * rho : ndarray of shape (2, 1)
     *     Two dimensional array containing the gas and liquid densities,
     *     kg/m^3
     *
     */

    // PyObjects created by ArgParse
    PyObject *m_pyin = NULL, *M_pyin = NULL, *Pc_pyin = NULL, *Tc_pyin = NULL,
             *Vc_pyin = NULL, *omega_pyin = NULL, *delta_pyin = NULL,
             *Aij_pyin = NULL, *Bij_pyin = NULL, *delta_groups_pyin = NULL,
             *C_pen_pyin = NULL, *C_pen_T_pyin = NULL;

    // Numpy array version of each ndarray variable
    PyArrayObject *m_np = NULL, *M_np = NULL, *Pc_np = NULL, *Tc_np = NULL,
                  *Vc_np = NULL, *omega_np = NULL, *delta_np = NULL,
                  *Aij_np = NULL, *Bij_np = NULL, *delta_groups_np = NULL,
                  *C_pen_np = NULL, *C_pen_T_np = NULL;

    // Double and int versions of input variables
    double T, P;
    int calc_delta;

    // Declare C array variables to pass to preos library
    double *masses = NULL, *M_wts = NULL, *Pc = NULL, *Tc = NULL, *Vc = NULL,
           *omega = NULL, *delta = NULL, *Aij = NULL, *Bij = NULL,
           *delta_groups = NULL, *C_pen = NULL, *C_pen_T = NULL;

    // Declare variables for creating the output array
    double *rho = NULL;
    const int rho_Rank = 2;
    npy_intp rho_Dims[2] = { 2, 1 };
    PyObject *rho_pyout = NULL;

    int nc, dc;

    // Convert the Python inputs to working arrays --------------------------

    /* Parse the input */
    if (!PyArg_ParseTuple(args, ("ddOOOOOOOOOOiOO"), &T, &P, &m_pyin, &M_pyin,
            &Pc_pyin, &Tc_pyin, &Vc_pyin, &omega_pyin, &delta_pyin, &Aij_pyin,
            &Bij_pyin, &delta_groups_pyin, &calc_delta, &C_pen_pyin,
            &C_pen_T_pyin)) {
        /* Input parameters could not be cast to local variables */
        return NULL;
    }

    // Convert the Python objects to flat C-style contiguous arrays
    get_arr_from_pyobj(m_pyin, NPY_DOUBLE, 1, 1, "component masses, m", &m_np,
        &masses);
    get_arr_from_pyobj(M_pyin, NPY_DOUBLE, 1, 1, "molecular weights, M", &M_np,
        &M_wts);
    get_arr_from_pyobj(Pc_pyin, NPY_DOUBLE, 1, 1, "critical pressures, Pc",
        &Pc_np, &Pc);
    get_arr_from_pyobj(Tc_pyin, NPY_DOUBLE, 1, 1, "critical temperatures, Tc",
        &Tc_np, &Tc);
    get_arr_from_pyobj(Vc_pyin, NPY_DOUBLE, 1, 1, "critical temperatures, Vc",
        &Vc_np, &Vc);
    get_arr_from_pyobj(omega_pyin, NPY_DOUBLE, 1, 1,
        "accentric factors, omega", &omega_np, &omega);
    get_arr_from_pyobj(delta_pyin, NPY_DOUBLE, 2, 2, "delta matrix", &delta_np,
        &delta);
    get_arr_from_pyobj(Aij_pyin, NPY_DOUBLE, 2, 2, "Aij matrix", &Aij_np,
        &Aij);
    get_arr_from_pyobj(Bij_pyin, NPY_DOUBLE, 2, 2, "Bij matrix", &Bij_np,
        &Bij);
    get_arr_from_pyobj(delta_groups_pyin, NPY_DOUBLE, 2, 2, "delta groups",
        &delta_groups_np, &delta_groups);
    get_arr_from_pyobj(C_pen_pyin, NPY_DOUBLE, 1, 1, "coefficients, C_pen",
        &C_pen_np, &C_pen);
    get_arr_from_pyobj(C_pen_T_pyin, NPY_DOUBLE, 1, 1, "coefficients, C_pen_T",
        &C_pen_T_np, &C_pen_T);
    if (m_np == NULL || M_np == NULL || Pc_np == NULL || Tc_np == NULL ||
        omega_np == NULL || delta_np == NULL || Aij_np == NULL ||
        Bij_np == NULL || delta_groups_np == NULL || C_pen_np == NULL ||
        C_pen_T == NULL) {
        goto fail;
    }

    // Do the numerical calculations of this function -----------------------

    // Make sure the user provided exactly four polynomial coefficients
    nc = PyArray_DIMS(m_np)[0];
    dc = PyArray_DIMS(Aij_np)[0];

    // Get the density
    rho =
        (double *)alloc_2d_array_flat(2, 1, sizeof(double), "densities, rho");
    if (rho == NULL) {
        goto fail;
    }
    _density(nc, T, P, masses, M_wts, Pc, Tc, Vc, omega, delta, Aij, Bij,
        delta_groups, calc_delta, C_pen, C_pen_T, rho);

    // Convert the function results to Numpy return values ------------------

    // Create a new Numpy array to hold the return value
    rho_Dims[0] = 2;
    rho_Dims[1] = 1;
    rho_pyout = get_pyobj_from_arr(NPY_DOUBLE, rho_Dims, rho_Rank, rho,
        "densities, rho");
    if (rho_pyout == NULL) {
        goto fail;
    }

    // Success
    return (PyObject *)rho_pyout;

fail:
    // Clear-up allocated memory and return NULL
    if (rho && !rho_pyout) {
        free(rho);
    }
    Py_XDECREF(m_np);
    Py_XDECREF(M_np);
    Py_XDECREF(Pc_np);
    Py_XDECREF(Tc_np);
    Py_XDECREF(Vc_np);
    Py_XDECREF(omega_np);
    Py_XDECREF(delta_np);
    Py_XDECREF(Aij_np);
    Py_XDECREF(Bij_np);
    Py_XDECREF(delta_groups_np);
    Py_XDECREF(C_pen_np);
    Py_XDECREF(C_pen_T_np);
    return NULL;
}

static char doc_fugacity[] =
    "[fk] = fugacity(T, P, m, M, Pc, Tc, omega, delta, Aij, Bij,"
    "                delta_groups, calc_delta\n";
static PyObject *preos_c_fugacity(PyObject *self, PyObject *args)
{
    /*
     * Wrapper from Python to the fugacity() function of the
     * preos_c library.
     *
     * In Python:
     *
     * fk = density(T, P, m, M, Pc, Tc, omega, delta, Aij, Bij, delta_groups,
     *     calc_delta, C_pen, C_pen_T)
     *
     * Parameters
     * ----------
     * T : double float
     *     Temperature, K
     * P : double float
     *     Pressure, Pa
     * m : ndarray of length nc
     *     Masses of each component of the mixture, kg
     * M : ndarray of length nc
     *     Molecular weights, kg/mol
     * Pc : ndarray of length nc
     *     Critical pressures, Pa
     * Tc : ndarray of length nc
     *     Critical temperatures, K
     * omega : ndarray of length nc
     *     Aacentric factors, --
     * delta : ndarray of shape (nc, nc)
     *     Binary interaction coefficients, --
     * Aij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * Bij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * delta_groups : ndarray of shape (nc, 15)
     *     Group contribution factors for the Privat and Jaubert (2012)
     *     temperature-dependent binary interaction coefficients model
     * calc_delta : int
     *     Flag determining whether to compute temperature-dependent binary
     *     interaction coefficients (1: True, -1: False)
     *
     * Returns
     * -------
     * fk : ndarray of shape (2, nc)
     *     Two dimensional array containing the gas and liquid fugacities,
     *     Pa.  The gas values are in row 0, liquid values in row 1, and
     *     each column corresponding to one component of the mixture.
     *
     */

    // PyObjects created by ArgParse
    PyObject *m_pyin = NULL, *M_pyin = NULL, *Pc_pyin = NULL, *Tc_pyin = NULL,
             *omega_pyin = NULL, *delta_pyin = NULL, *Aij_pyin = NULL,
             *Bij_pyin = NULL, *delta_groups_pyin = NULL;

    // Numpy array version of each ndarray variable
    PyArrayObject *m_np = NULL, *M_np = NULL, *Pc_np = NULL, *Tc_np = NULL,
                  *omega_np = NULL, *delta_np = NULL, *Aij_np = NULL,
                  *Bij_np = NULL, *delta_groups_np = NULL;

    // Double and int versions of input variables
    double T, P;
    int calc_delta;

    // Declare C array variables to pass to preos library
    double *masses = NULL, *M_wts = NULL, *Pc = NULL, *Tc = NULL;
    double *omega = NULL, *delta = NULL, *Aij = NULL, *Bij = NULL;
    double *delta_groups = NULL;

    // Declare variables for creating the output array
    double *fk = NULL;
    const int fk_Rank = 2;
    npy_intp fk_Dims[2] = { -1, -1 };
    PyObject *fk_pyout = NULL;

    int nc, dc;

    // Convert the Python inputs to working arrays --------------------------

    /* Parse the input */
    if (!PyArg_ParseTuple(args, ("ddOOOOOOOOOi"), &T, &P, &m_pyin, &M_pyin,
            &Pc_pyin, &Tc_pyin, &omega_pyin, &delta_pyin, &Aij_pyin, &Bij_pyin,
            &delta_groups_pyin, &calc_delta)) {
        /* Input parameters could not be cast to local variables */
        return NULL;
    }

    // Convert the Python objects to flat C-style contiguous arrays
    get_arr_from_pyobj(m_pyin, NPY_DOUBLE, 1, 1, "component masses, m", &m_np,
        &masses);
    get_arr_from_pyobj(M_pyin, NPY_DOUBLE, 1, 1, "molecular weights, M", &M_np,
        &M_wts);
    get_arr_from_pyobj(Pc_pyin, NPY_DOUBLE, 1, 1, "critical pressures, Pc",
        &Pc_np, &Pc);
    get_arr_from_pyobj(Tc_pyin, NPY_DOUBLE, 1, 1, "critical temperatures, Tc",
        &Tc_np, &Tc);
    get_arr_from_pyobj(omega_pyin, NPY_DOUBLE, 1, 1,
        "accentric factors, omega", &omega_np, &omega);
    get_arr_from_pyobj(delta_pyin, NPY_DOUBLE, 2, 2, "delta matrix", &delta_np,
        &delta);
    get_arr_from_pyobj(Aij_pyin, NPY_DOUBLE, 2, 2, "Aij matrix", &Aij_np,
        &Aij);
    get_arr_from_pyobj(Bij_pyin, NPY_DOUBLE, 2, 2, "Bij matrix", &Bij_np,
        &Bij);
    get_arr_from_pyobj(delta_groups_pyin, NPY_DOUBLE, 2, 2, "delta groups",
        &delta_groups_np, &delta_groups);

    if (m_np == NULL || M_np == NULL || Pc_np == NULL || Tc_np == NULL ||
        omega_np == NULL || delta_np == NULL || Aij_np == NULL ||
        Bij_np == NULL || delta_groups_np == NULL) {
        goto fail;
    }

    // Do the numerical calculations of this function -----------------------

    // Make sure the user provided exactly four polynomial coefficients
    nc = PyArray_DIMS(m_np)[0];
    dc = PyArray_DIMS(Aij_np)[0];

    // Get the density
    fk =
        (double *)alloc_2d_array_flat(2, nc, sizeof(double), "fugacities, fk");
    if (fk == NULL) {
        goto fail;
    }
    _fugacity(nc, T, P, masses, M_wts, Pc, Tc, omega, delta, Aij, Bij,
        delta_groups, calc_delta, fk);

    // Convert the function results to Numpy return values ------------------

    // Create a new Numpy array to hold the return value
    fk_Dims[0] = 2;
    fk_Dims[1] = nc;
    fk_pyout =
        get_pyobj_from_arr(NPY_DOUBLE, fk_Dims, fk_Rank, fk, "fugacities, fk");
    if (fk_pyout == NULL) {
        goto fail;
    }

    // Success
    return (PyObject *)fk_pyout;

fail:
    // Clear-up allocated memory and return NULL
    if (fk && !fk_pyout) {
        free(fk);
    }
    Py_XDECREF(m_np);
    Py_XDECREF(M_np);
    Py_XDECREF(Pc_np);
    Py_XDECREF(Tc_np);
    Py_XDECREF(omega_np);
    Py_XDECREF(delta_np);
    Py_XDECREF(Aij_np);
    Py_XDECREF(Bij_np);
    Py_XDECREF(delta_groups_np);
    return NULL;
}

static char doc_viscosity[] =
    "[mu_p] = viscosity(T, P, m, M, Pc, Tc, Vc, omega, delta, Aij, Bij,"
    "                delta_groups, calc_delta, C_pen, C_pen_T\n";
static PyObject *preos_c_viscosity(PyObject *self, PyObject *args)
{
    /*
     * Wrapper from Python to the viscosity() function of the
     * preos_c library.
     *
     * In Python:
     *
     * mu_p = viscosity(T, P, m, M, Pc, Tc, Vc, omega, delta, Aij, Bij,
     *     delta_groups, calc_delta, C_pen, C_pen_T)
     *
     * Parameters
     * ----------
     * T : double float
     *     Temperature, K
     * P : double float
     *     Pressure, Pa
     * m : ndarray of length nc
     *     Masses of each component of the mixture, kg
     * M : ndarray of length nc
     *     Molecular weights, kg/mol
     * Pc : ndarray of length nc
     *     Critical pressures, Pa
     * Tc : ndarray of length nc
     *     Critical temperatures, K
     * Vc : ndarray of length nc
     *     Critical molar volume, m^3/mol
     * omega : ndarray of length nc
     *     Aacentric factors, --
     * delta : ndarray of shape (nc, nc)
     *     Binary interaction coefficients, --
     * Aij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * Bij : ndarray of shape (15, 15)
     *     Matrix used in the Privat and Jaubert (2012) temperature-dependent
     *     binary interaction coefficients model
     * delta_groups : ndarray of shape (nc, 15)
     *     Group contribution factors for the Privat and Jaubert (2012)
     *     temperature-dependent binary interaction coefficients model
     * calc_delta : int
     *     Flag determining whether to compute temperature-dependent binary
     *     interaction coefficients (1: True, -1: False)
     * C_pen : ndarray of length nc
     *     Peneloux volume translation coefficients, m^3/mol
     * C_pen_T : ndarray of length nc
     *     Temperature correction parameter for the Peneloux volume
     *     translation coefficients, m^3/(mol K)
     *
     * Returns
     * -------
     * mu_p : ndarray of shape (2, 1)
     *     Two dimensional array containing the gas and liquid dynamic
     *     viscosities, Pa s
     *
     */

    // PyObjects created by ArgParse
    PyObject *m_pyin = NULL, *M_pyin = NULL, *Pc_pyin = NULL, *Tc_pyin = NULL,
             *Vc_pyin = NULL, *omega_pyin = NULL, *delta_pyin = NULL,
             *Aij_pyin = NULL, *Bij_pyin = NULL, *delta_groups_pyin = NULL,
             *C_pen_pyin = NULL, *C_pen_T_pyin = NULL;

    // Numpy array version of each ndarray variable
    PyArrayObject *m_np = NULL, *M_np = NULL, *Pc_np = NULL, *Tc_np = NULL,
                  *Vc_np = NULL, *omega_np = NULL, *delta_np = NULL,
                  *Aij_np = NULL, *Bij_np = NULL, *delta_groups_np = NULL,
                  *C_pen_np = NULL, *C_pen_T_np = NULL;

    // Double and int versions of input variables
    double T, P;
    int calc_delta;

    // Declare C array variables to pass to preos library
    double *masses = NULL, *M_wts = NULL, *Pc = NULL, *Tc = NULL, *Vc = NULL,
           *omega = NULL, *delta = NULL, *Aij = NULL, *Bij = NULL,
           *delta_groups = NULL, *C_pen = NULL, *C_pen_T = NULL;

    // Declare variables for creating the output array
    double *mu_p = NULL;
    const int mu_p_Rank = 2;
    npy_intp mu_p_Dims[2] = { 2, 1 };
    PyObject *mu_p_pyout = NULL;

    int nc, dc;

    // Convert the Python inputs to working arrays --------------------------

    /* Parse the input */
    if (!PyArg_ParseTuple(args, ("ddOOOOOOOOOOiOO"), &T, &P, &m_pyin, &M_pyin,
            &Pc_pyin, &Tc_pyin, &Vc_pyin, &omega_pyin, &delta_pyin, &Aij_pyin,
            &Bij_pyin, &delta_groups_pyin, &calc_delta, &C_pen_pyin,
            &C_pen_T_pyin)) {
        /* Input parameters could not be cast to local variables */
        return NULL;
    }

    // Convert the Python objects to flat C-style contiguous arrays
    get_arr_from_pyobj(m_pyin, NPY_DOUBLE, 1, 1, "component masses, m", &m_np,
        &masses);
    get_arr_from_pyobj(M_pyin, NPY_DOUBLE, 1, 1, "molecular weights, M", &M_np,
        &M_wts);
    get_arr_from_pyobj(Pc_pyin, NPY_DOUBLE, 1, 1, "critical pressures, Pc",
        &Pc_np, &Pc);
    get_arr_from_pyobj(Tc_pyin, NPY_DOUBLE, 1, 1, "critical temperatures, Tc",
        &Tc_np, &Tc);
    get_arr_from_pyobj(Vc_pyin, NPY_DOUBLE, 1, 1, "critical temperatures, Vc",
        &Vc_np, &Vc);
    get_arr_from_pyobj(omega_pyin, NPY_DOUBLE, 1, 1,
        "accentric factors, omega", &omega_np, &omega);
    get_arr_from_pyobj(delta_pyin, NPY_DOUBLE, 2, 2, "delta matrix", &delta_np,
        &delta);
    get_arr_from_pyobj(Aij_pyin, NPY_DOUBLE, 2, 2, "Aij matrix", &Aij_np,
        &Aij);
    get_arr_from_pyobj(Bij_pyin, NPY_DOUBLE, 2, 2, "Bij matrix", &Bij_np,
        &Bij);
    get_arr_from_pyobj(delta_groups_pyin, NPY_DOUBLE, 2, 2, "delta groups",
        &delta_groups_np, &delta_groups);
    get_arr_from_pyobj(C_pen_pyin, NPY_DOUBLE, 1, 1, "coefficients, C_pen",
        &C_pen_np, &C_pen);
    get_arr_from_pyobj(C_pen_T_pyin, NPY_DOUBLE, 1, 1, "coefficients, C_pen_T",
        &C_pen_T_np, &C_pen_T);
    if (m_np == NULL || M_np == NULL || Pc_np == NULL || Tc_np == NULL ||
        omega_np == NULL || delta_np == NULL || Aij_np == NULL ||
        Bij_np == NULL || delta_groups_np == NULL || C_pen_np == NULL ||
        C_pen_T == NULL) {
        goto fail;
    }

    // Do the numerical calculations of this function -----------------------

    // Make sure the user provided exactly four polynomial coefficients
    nc = PyArray_DIMS(m_np)[0];
    dc = PyArray_DIMS(Aij_np)[0];

    // Get the density
    mu_p =
        (double *)alloc_2d_array_flat(2, 1, sizeof(double), "densities, rho");
    if (mu_p == NULL) {
        goto fail;
    }
    _viscosity(nc, T, P, masses, M_wts, Pc, Tc, Vc, omega, delta, Aij, Bij,
        delta_groups, calc_delta, C_pen, C_pen_T, mu_p);

    // Convert the function results to Numpy return values ------------------

    // Create a new Numpy array to hold the return value
    mu_p_Dims[0] = 2;
    mu_p_Dims[1] = 1;
    mu_p_pyout = get_pyobj_from_arr(NPY_DOUBLE, mu_p_Dims, mu_p_Rank, mu_p,
        "viscosities, mu_p");
    if (mu_p_pyout == NULL) {
        goto fail;
    }

    // Success
    return (PyObject *)mu_p_pyout;

fail:
    // Clear-up allocated memory and return NULL
    if (mu_p && !mu_p_pyout) {
        free(mu_p);
    }
    Py_XDECREF(m_np);
    Py_XDECREF(M_np);
    Py_XDECREF(Pc_np);
    Py_XDECREF(Tc_np);
    Py_XDECREF(Vc_np);
    Py_XDECREF(omega_np);
    Py_XDECREF(delta_np);
    Py_XDECREF(Aij_np);
    Py_XDECREF(Bij_np);
    Py_XDECREF(delta_groups_np);
    Py_XDECREF(C_pen_np);
    Py_XDECREF(C_pen_T_np);
    return NULL;
}

/* --------------------------------------------------------------------------
 * Create the module interface to Python for this C-library
 * ------------------------------------------------------------------------*/

static struct PyMethodDef preos_c_module_methods[] = {
    /*
     * Create the structure table describing each of the functions present
     * in this module.  The layout is:
     *
     *     A string name for the function
     *     The C function name of the wrapper
     *     METH_VARARGS
     *     The variable name of the doc string for the function
     *
     * The last entry will always be NULL, NULL, 0, NULL
     *
     */
    { "cubic_roots", preos_c_cubic_roots, METH_VARARGS, doc_cubic_roots },
    { "density", preos_c_density, METH_VARARGS, doc_density },
    { "fugacity", preos_c_fugacity, METH_VARARGS, doc_fugacity },
    { "viscosity", preos_c_viscosity, METH_VARARGS, doc_viscosity },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, "preos_c", NULL,
    -1, preos_c_module_methods, NULL, NULL, NULL, NULL };

PyMODINIT_FUNC PyInit_preos_c(void)
{
    /*
     * Conduct the module initialization process that should occur when the
     * user imports this module within Python.  General steps include:
     *
     *    Create the module
     *    Import the NumPy tools used by this module
     *    Set a doc string for the whole module
     *    Define the error message that occurs on module errors
     *    Put any extra data in the module dictionary
     *    Handle errors on creation to avoid memory leaks
     *
     * This was largely adapted from the equivalent function created by f2py
     * for the dbm_f module in TAMOC.
     *
     */

    PyObject *module, *mdict, *s;

    // Try to create this module
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        // If creation failed, return NULL immediately
        return NULL;
    }

    // Import the Numpy C-API Macro to enable array functionality
    import_array();
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_ImportError,
            "Cannot initialize module preos_c: (failed to import numpy)");
        Py_DECREF(module);
        return NULL;
    }

    // Get the module dictionary to edit
    mdict = PyModule_GetDict(module);

    // Insert a version number for this module
    s = PyUnicode_FromString(version_num);
    PyDict_SetItemString(mdict, "__version__", s);
    Py_DECREF(s);

    // Set a doc-string for this module
    s = PyUnicode_FromString(
        "This module 'preos_c' contains the following functions translated\n"
        "from 'dbm_eos.f95' in TAMOC:\n"
        "    z = cubic_roots(a)\n"
        "    rho = density(T, P, m, M, Pc, Tc, Vc, omega,delta, Aij, Bij,"
        "                  delta_groups, calc_delta, C_pen, C_pen_T\n"
        "    fk = fugacity(T, P, m, M, Pc, Tc, omega, delta, Aij, Bij,"
        "                  delta_groups, calc_delta, C_pen, C_pen_T\n"
        "    mu_p = viscosity(T, P, m, M, Pc, Tc, Vc, omega, delta, Aij, Bij,"
        "                     delta_groups, calc_delta, C_pen, C_pen_T\n");
    PyDict_SetItemString(mdict, "__doc__", s);
    Py_DECREF(s);

    // Define the local error messaging function
    preos_c_error = PyErr_NewException("preos_c.error", NULL, NULL);
    if (preos_c_error == NULL) {
        printf("Could not create the Exception object for the module preos_c");
        return NULL;
    }

    // Add the error function to the object dictionary
    if (PyDict_SetItemString(mdict, "error", preos_c_error)) {
        Py_DECREF(preos_c_error);
        return NULL;
    }

// Statement adapted from ./scipy/interpolate/src/_fitpackmodule.c
#if Py_GIL_DISABLED
    // Signal whether this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif

    // Success
    return module;
}
