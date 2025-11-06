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

This script is closely adapted from the `./tools/generate_f2pymod.py` 
script from the GitHub repository of Scipy.  It was modified through the help
of ChatGPT to handle the fact that the .f95 files expected here have 
`module` blocks that can only be correctly built into the Fortran and 
C wrappers if the Fortran source files are available.  Scipy does not
need to do this.

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
    parser.add_argument("infiles", nargs="+",
        help="Input .pyf file followed by optional Fortran source files")
    parser.add_argument("-o", "--outdir", type=str,
        help="Path to the output directory")
    parser.add_argument("--free-threading",
        action=argparse.BooleanOptionalAction,
        help="Whether to add --free-threading-compatible")
    args = parser.parse_args()
    
    # First argument is always the .pyf file
    pyf_file = args.infiles[0]
    if not pyf_file.endswith('.pyf'):
        # Input file sequence not appropriate, report error and exit
        raise ValueError(
            f"First input file must be a .pyf file:  got {pyf_file}"
        )

    # The remaining inputs are Fortran source files
    fortran_sources = args.infiles[1:]
    
    # Write the output files where Meson will find them
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    
    # Set the free threading parameter 
    nogil_arg = []
    if args.free_threading:
        nogil_arg = ['--freethreading-compatible']
    
    # Construct the f2py command
    cmd = ['f2py', '-c', pyf_file] + fortran_sources + ['--build-dir', 
        outdir_abs] + nogil_arg

    # Invoke f2py (a command-line function) to generate the C API module file.
    # This will directly write the Fortran and C wrapper files to the
    # selected destination.  Thus, no return value is needed from this function
    # or script to the calling meson.build file.
    print("Running f2py:", " ".join(cmd))
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd()
    )

    # Check if the wrapper generation was successful
    out, err = p.communicate()
    if not (p.returncode == 0):
        raise RuntimeError(f"Processing {pyf_file} with f2py failed!\n"
                           f"{out.decode()}\n"
                           f"{err.decode()}"
        )
    else:
        print(f"Successfully built C-wrappers for {pyf_file}")

if __name__ == "__main__":
	main()
