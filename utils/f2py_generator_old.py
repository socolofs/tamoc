#!/usr/bin/env python3
"""
f2py_generator.py
-----------------

This script is used by Meson at build time to run `f2py` and generate the
wrapper files.  It is best practice to create the `.pyf` files locally 
and distribute those with the package source code; hence, this script takes
a `.pyf` description file of an extension module as input and uses it through
`f2py` to generate the Fortran and C wrapper files that will be used to 
finally create the extension module.

This script is entirely adapted from the `./tools/generate_f2pymod.py` 
script from the GitHub repository of Scipy.

"""
# S. Socolofsky, Texas A&M University, May 2025, <socolofs@tamu.edu>

import argparse
import os
import subprocess

def main():
    """
    Main entry point to the `f2py` processor. 
    
    """
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
        help="Path to the input file")
    parser.add_argument("-o", "--outdir", type=str,
        help="Path to the output directory")
    parser.add_argument("--free-threading",
        action=argparse.BooleanOptionalAction,
        help="Whether to add --free-threading-compatible")
    args = parser.parse_args()

    # Make sure the user passed a .pyf file
    if not args.infile.endswith('.pyf'):
        # Input file not appropriate, report error and exit
        raise ValueError(
            f"F2PY Iput file has unknown extension:  {args.infile}"
        )
    else:
        # Get a variable to hold the input file name
        fname_pyf = args.infile
    
    # Write the output files where Meson will find them
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    
    # Set the free threading parameter 
    nogil_arg = []
    if args.free_threading:
        nogil_arg = ['--freethreading-compatible']
    
    # Invoke f2py (a command-line function) to generate the C API module file.
    # This will directly write the Fortran and C wrapper files to the
    # selected destination.  Thus, no return value is needed from this function
    # or script to the calling meson.build file.
    p = subprocess.Popen(
        ['f2py', fname_pyf, '--build-dir', outdir_abs] + nogil_arg,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd()
    )

    # Check if the wrapper generation was successful
    out, err = p.communicate()
    if not (p.returncode == 0):
        raise RuntimeError(f"Processing {fname_pyf} with f2py failed!\n"
                           f"{out.decode()}\n"
                           f"{err.decode()}"
        )
    else:
        print(f"Successfully built C-wrappers for {fname_pyf}")

if __name__ == "__main__":
	main()
