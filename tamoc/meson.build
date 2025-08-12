# This part of the meson.build cycle installs all the required package
# files and extension modules

# List all Python source files to include
python_sources = [
  '__init__.py',
  'ambient.py',
  'bent_plume_model.py',
  'blowout.py',
  'chemical_properties.py',
  'dbm_p.py',
  'dbm_utilities.py',
  'dbm.py',
  'dispersed_phases.py',
  'lmp.py',
  'model_share.py',
  'params.py',
  'particle_size_models.py',
  'psf.py',
  'seawater.py',
  'single_bubble_model.py',
  'sintef.py',
  'smp.py',
  'stratified_plume_model.py'
]

test_sources = [
  'test/__init__.py',
  'test/test_ambient.py',
  'test/test_blowout.py',
  'test/test_bpm.py',
  'test/test_dbm_f.py',
  'test/test_dbm_objects.py',
  'test/test_dbm.py',
  'test/test_params.py',
  'test/test_psm.py',
  'test/test_sbm.py',
  'test/test_sintef.py',
  'test/test_spm.py'
]

data_sources = [
  'data/Aij.csv',
  'data/Bij.csv',
  'data/BioData.csv',
  'data/C.dat',
  'data/ChemData.csv',
  'data/ctd_BM54.cnv',
  'data/CTD_data.dat',
  'data/lake.dat',
  'data/PJData.csv',
  'data/T.dat',
  'data/world_ocean_ave_ctd.dat'
]

# Install the Python source files
py3.install_sources(
  python_sources,
  subdir: 'tamoc',
)

# Install the package test files
py3.install_sources(
  test_sources,
  subdir: 'tamoc/test',
  install_tag: 'tests'
) 

# Install the data files used within the package
py3.install_sources(
  data_sources,
  subdir: 'tamoc/data',
  install_tag: 'data'
)

# Create the C extension module if requested
if get_option('with-c')
  
  # Create the C-extension module for the PR EOS
  preos_c_src = files(
    'src/preos_cmodule.c',
    'src/preos.c'
  )

  py3.extension_module('preos_c',
    [preos_c_src],
    c_args: [Wno_unused_variable],
    link_args: version_link_args,
    link_language: 'c',
    dependencies: [np_dep],
    install: true,
    subdir: 'tamoc'
  )

endif

# Create the Fortran extension module if requested
if get_option('with-fortran')

  # Set up f2py to dynamically create the dbm_f extension module
  # Set a variable to store the name of a Python script that can fun f2py
  generate_f2pywrapper = find_program('../utils/f2py_generator.py')

  # List the Fortran files to include in the dbm_f module
  dbm_f_src = [
    'src/dbm_phys.f95',
    'src/dbm_eos.f95',
    'src/math_funcs.f95',
  ]
  dbm_f_pyf = 'dbm_f.pyf'

  dbm_f_build = custom_target('f2py-build',
    input: [dbm_f_pyf] + dbm_f_src,
    output: ['dbm_fmodule.c', 'dbm_f-f2pywrappers.f', 'dbm_f-f2pywrappers2.f90'],
    command: [
      generate_f2pywrapper, '@INPUT@', '-o', meson.current_build_dir(),
    ] + f2py_freethreading_arg,
    install: false,
  )

  # Create the dbm_f extension module
  py3.extension_module('dbm_f',
    [
      dbm_f_build,
      dbm_f_src,
    ],
    c_args: [Wno_unused_variable],
    fortran_args: fortran_ignore_warnings,
    override_options: ['b_lto=false'],
    gnu_symbol_visibility: 'hidden',
    link_args: version_link_args,
    dependencies: [fortranobject_dep],
    install: true,
    link_language: 'fortran',
    subdir: 'tamoc'
  )

endif

subdir('tools')