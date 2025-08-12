# This build file has been created using elements from the Scipy meson.build
# files and from the Meson build created using::
# 
#   f2py -c -m dbm_f dbm_phys.f95 dbm_eos.f95 math_funcs.f95 --build-dir build
# 
# I have tried to carry over the minimum things from the Scipy structure to
# support the elements in the f2py bulid and to allow reasonable compatibility
# with Windows and MacOS/Unix.

project(
  'tamoc',
  'c',
  version: '4.0.0',
  license: 'MIT',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c17',
    'warning_level=1',
  ],
)

# Get the Python command
py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

# Set a variable to hold the minimum Numpy version 
min_numpy_version = '1.25.2'   # keep in sync with pyproject.toml

# Emit a warning for 32-bit Python installs on Windows as this many not work
is_windows = host_machine.system() == 'windows'
if is_windows and py3.has_variable('EXT_SUFFIX')
  ext_suffix = py3.get_variable('EXT_SUFFIX')
  if ext_suffix.contains('win32')
    warning('You are building from source on a 32-bit Python install. This is untested; install 64-bit Python if you are having issues!')
  endif
endif

# Initialize a variable to hold the configuration data
conf_data = configuration_data()

# Add the machines
machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

if not get_option('python-only')

  # Set up the C-compiler.  This is required for both C and Fortran extension
  # modules as the wrappers to the Fortran libraries are in C.
  cc = meson.get_compiler('c')

  if cc.get_id() == 'gcc'
    if not cc.version().version_compare('>=8.3.0')
      error('Tamoc requires GCC >= 8.3.0')
    endif
  elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
    if not cc.version().version_compare('>=15.0')
      error('Tamoc requires clang >= 15.0')
    endif
  elif cc.get_id() == 'msvc'
    if not cc.version().version_compare('>=19.20')
      error('Tamoc requires at least vc142 (default with ' + \
            'Visual Studio 2019) when building with MSVC')
    endif
  endif

  _global_c_args = cc.get_supported_arguments(
    '-Wno-unused-but-set-variable',
    '-Wno-unused-function',
    '-Wno-conversion',
    '-Wno-misleading-indentation',
  )
  add_project_arguments(_global_c_args, language : 'c')

  # If we write C-modules directly in the future, they will almost certainly
  # include math.  Set up -lm for all C code.  For Fortran code, Meson already 
  # adds `-lm`.
  m_dep = cc.find_library('m', required : false)
  if m_dep.found()
    add_project_link_arguments('-lm', language : 'c')
  endif

  if host_machine.system() == 'os400'
    # IBM i system, needed to avoid build errors
    add_project_link_arguments('-Wl,-bnotextro', language : ['c', 'fortran'])
  endif

  # Intel compilers default to fast-math, so disable it if we detect Intel
  # compilers. A word of warning: this may not work with the conda-forge
  # compilers, because those have the annoying habit of including lots of flags
  # that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
  # detection logic below. You have to remove the wrong flags (only `-isystem`
  # is actually needed, everything else shouldn't be there).
  _intel_cflags = []
  if cc.get_id() in ['intel', 'intel-llvm']
    _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
  elif cc.get_id() in ['intel-cl', 'intel-llvm-cl']
    _intel_cflags += cc.get_supported_arguments('/fp:strict')
  endif
  add_project_arguments(_intel_cflags, language: 'c')

  if host_machine.system() == 'darwin'
    if cc.has_link_argument('-Wl,-dead_strip')
      # Allow linker to strip unused symbols
      add_project_link_arguments('-Wl,-dead_strip', language : ['c'])
    endif
  endif

  # On Windows, if using mingw, make sure everything is set up correctly
  is_mingw = is_windows and cc.get_id() == 'gcc'
  if is_mingw
    # For mingw-w64, link statically against the UCRT.
    #gcc_link_args = ['-lucrt', '-static']
    gcc_link_args = []
    add_project_link_arguments(gcc_link_args, language: ['c'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c'])
  endif 

  # Check if compiler flags are supported. This is necessary to ensure that
  # tamoc can be built with any supported compiler. This code block is 
  # copied directly from Scipy's ./scipy/meson.build file.
  #
  # The standard convention used here is:
  #   - for C, drop the leading dash and turn remaining dashes into underscores
  #
  # C warning flags
  Wno_maybe_uninitialized = cc.get_supported_arguments('-Wno-maybe-uninitialized')
  Wno_discarded_qualifiers = cc.get_supported_arguments('-Wno-discarded-qualifiers')
  Wno_empty_body = cc.get_supported_arguments('-Wno-empty-body')
  Wno_implicit_function_declaration = cc.get_supported_arguments('-Wno-implicit-function-declaration')
  Wno_parentheses = cc.get_supported_arguments('-Wno-parentheses')
  Wno_switch = cc.get_supported_arguments('-Wno-switch')
  Wno_unused_label = cc.get_supported_arguments('-Wno-unused-label')
  Wno_unused_result = cc.get_supported_arguments('-Wno-unused-result')
  Wno_unused_variable = cc.get_supported_arguments('-Wno-unused-variable')
  Wno_unused_but_set_variable = cc.get_supported_arguments('-Wno-unused-but-set-variable')

  # Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
  if is_windows
    use_math_defines = ['-D_USE_MATH_DEFINES']
  else
    use_math_defines = []
  endif

  # Add the compiler
  compilers = { 'C': cc}

  # Set up numpy, which is needed to compile the fortran sources into dbm_f,
  # but is ALSO needed for the pure C extension module because the inputs
  # to that module from Python are NumPy Array objects.
  #
  # Uses the `numpy-config` executable (or a user's numpy.pc pkg-config file).
  # Will work for numpy>=2.0, hence not required as we don't want to require
  # higher than 1.25.2.  Using this now to be able to detect the version easily 
  # for >=2.0.
  _numpy_dep = dependency('numpy', required: false)
  f2py_freethreading_arg = []
  if _numpy_dep.found()
    if _numpy_dep.version().version_compare('>=2.1.0')
      f2py_freethreading_arg = ['--free-threading']
    endif
  endif

  # NumPy include directory.  This implementation from Scipy is a bit more 
  # complicated than that built by f2py with --build-dir.  However, it should
  # work well here.
  incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
  if incdir_numpy == 'not-given'
    incdir_numpy = run_command(py3,
      [
        '-c',
        '''
import numpy as np
import os
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
    '''
      ],
      check: true
    ).stdout().strip()

    # We do need an absolute path to feed to `cc.find_library` below
    _incdir_numpy_abs = run_command(py3,
      ['-c', 
      'import numpy; print(numpy.get_include())'
      ],
      check: true
    ).stdout().strip()
  else
    _incdir_numpy_abs = incdir_numpy
  endif
  inc_np = include_directories(incdir_numpy)
  # Don't use the deprecated NumPy C API. Define this to a fixed version instead 
  # of NPY_API_VERSION in order not to break compilation for released tamoc
  # versions when NumPy introduces a new deprecation.
  numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
  np_dep = declare_dependency(
    include_directories: inc_np, 
    compile_args: numpy_nodepr_api
  )

  # Initialize an empty variable for the fortran dependencies that will
  # eventually be passed to initialize f2py:
  fortranobject_dep = []

  if get_option('with-fortran')
    # Adding Fortran at the project level causes many spurious -lgfortran flags.
    # Add the Fortran compiler now.
    add_languages('fortran', native: false)
    fc = meson.get_compiler('fortran')

    if fc.get_id() == 'gcc'
      # -std=legacy is not supported by all Fortran compilers, but very useful with
      # gfortran since it avoids a ton of warnings that we don't care about.
      # Needs fixing in Meson, see 
      # https://github.com/mesonbuild/meson/issues/11633.
      add_project_arguments('-std=legacy', language: 'fortran')
    endif

    if fc.has_argument('-Wno-conversion')
      add_project_arguments('-Wno-conversion', language: 'fortran')
    endif

    if host_machine.system() == 'darwin'
      if cc.has_link_argument('-Wl,-dead_strip')
        # Allow linker to strip unused symbols
        add_project_link_arguments('-Wl,-dead_strip', language : ['fortran'])
      endif
    endif

    # Also deal with fast-math on intel fortran compilers
    _intel_fflags = []
    if fc.get_id() in ['intel', 'intel-llvm']
      _intel_fflags = fc.get_supported_arguments('-fp-model=strict')
      minus0_arg = ['-assume', 'minus0']
      if fc.has_multi_arguments(minus0_arg)
        _intel_fflags += minus0_arg
      endif
    elif fc.get_id() in ['intel-cl', 'intel-llvm-cl']
      # Intel Fortran on Windows does things differently, so deal with that
      # (also specify dynamic linking and the right name mangling)
      _intel_fflags = fc.get_supported_arguments(
        '/fp:strict', '/MD', '/names:lowercase', '/assume:underscore',
        '/assume:minus0'
      )
    endif
    add_project_arguments(_intel_fflags, language: 'fortran')

    # Make sure mingw is set up correctly for installing Fortran files
    if is_mingw and fc.get_id() != 'gcc'
      error('If you are using GCC on Windows, you must also use GFortran! ' + 
        'Detected ' + fc.get_id())
    endif
    if is_mingw
      # For mingw-w64, link statically against the UCRT.
      add_project_link_arguments(gcc_link_args, language: ['fortran'])
      # Flag needed to work around BLAS and LAPACK Gfortran dependence 
      add_project_arguments('-fno-optimize-sibling-calls', language: ['fortran'])
    endif

    # Check if compiler flags are supported. This is necessary to ensure that
    # tamoc can be built with any supported compiler. This code block is 
    # copied directly from Scipy's ./scipy/meson.build file.
    #
    # Fortran warning flags
    _fflag_Wno_argument_mismatch = fc.get_supported_arguments('-Wno-argument-mismatch')
    _fflag_Wno_conversion = fc.get_supported_arguments('-Wno-conversion')
    _fflag_Wno_intrinsic_shadow = fc.get_supported_arguments('-Wno-intrinsic-shadow')
    _fflag_Wno_maybe_uninitialized = fc.get_supported_arguments('-Wno-maybe-uninitialized')
    _fflag_Wno_surprising = fc.get_supported_arguments('-Wno-surprising')
    _fflag_Wno_uninitialized = fc.get_supported_arguments('-Wno-uninitialized')
    _fflag_Wno_unused_dummy_argument = fc.get_supported_arguments('-Wno-unused-dummy-argument')
    _fflag_Wno_unused_label = fc.get_supported_arguments('-Wno-unused-label')
    _fflag_Wno_unused_variable = fc.get_supported_arguments('-Wno-unused-variable')
    _fflag_Wno_tabs = fc.get_supported_arguments('-Wno-tabs')
    # The default list of warnings to ignore from Fortran code. There is a lot of
    # old, vendored code that is very bad and we want to compile it silently (at
    # least with GCC and Clang)
    fortran_ignore_warnings = fc.get_supported_arguments(
     _fflag_Wno_argument_mismatch,
     _fflag_Wno_conversion,
     _fflag_Wno_maybe_uninitialized,
     _fflag_Wno_unused_dummy_argument,
     _fflag_Wno_unused_label,
     _fflag_Wno_unused_variable,
     _fflag_Wno_tabs,
    )

    # Intel Fortran (ifort) does not run the preprocessor by default, if Fortran
    # code uses preprocessor statements, add this compile flag to it.
    _fflag_fpp = []
    if fc.get_id() in ['intel-cl', 'intel-llvm-cl']
      if is_windows
        _fflag_fpp = fc.get_supported_arguments('/fpp')
      else
        _fflag_fpp = fc.get_supported_arguments('-fpp')
      endif
    endif

    # Add the compiler
    compilers += {'FORTRAN': fc}

    # Set up f2py, which is used to compile the fortran sources into dbm_f
    incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
    inc_f2py = include_directories(incdir_f2py)
    fortranobject_c = incdir_f2py / 'fortranobject.c'

    # Create compact objects that hold all these dependencies in place
    fortranobject_lib = static_library('_fortranobject',
      fortranobject_c,
      c_args: numpy_nodepr_api,
      dependencies: py3_dep,
      include_directories: [inc_np, inc_f2py],
      gnu_symbol_visibility: 'hidden',
    )
    fortranobject_dep = declare_dependency(
      link_with: fortranobject_lib,
      include_directories: [inc_np, inc_f2py],
    )

  endif  

  # Hide symbols when building on Linux with GCC. For Python extension modules,
  # we only need `PyInit_*` to be public, anything else may cause problems. 
  _linker_script = meson.project_source_root() / 'utils/link-version-pyinit.map'
  version_link_args = ['-Wl,--version-script=' + _linker_script]
  # Note that FreeBSD only accepts version scripts when -shared is passed,
  # hence we need to pass that to `cc.links` explicitly (flag is already
  # present for `extension_module` invocations).
  if not cc.links('', name: '-Wl,--version-script', 
      args: ['-shared', version_link_args])
    version_link_args = []
  endif

  # Set compiler information
  foreach name, compiler : compilers
    conf_data.set(name + '_COMP', compiler.get_id())
    conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
    conf_data.set(name + '_COMP_VERSION', compiler.version())
    conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
    conf_data.set(name + '_COMP_ARGS', ', '.join(
        get_option(name.to_lower() + '_args')
      )
    )
    conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
        get_option(name.to_lower() + '_link_args')
      )
    )
  endforeach


  # Work on compiler configuration
  cdata = configuration_data()

  # Test variable attribute to use for thread-local storage;
  # Adapted from `numpy/_core/meson.build`.
  check_tls_attrs = [
    ['thread_local', 'HAVE_THREAD_LOCAL'],    # C23
    ['_Thread_local', 'HAVE__THREAD_LOCAL'],  # C11/C17
    ['__thread', 'HAVE__THREAD'],
  ]
  if is_windows and not is_mingw
    check_tls_attrs += ['__declspec(thread)', 'HAVE___DECLSPEC_THREAD_']
  endif
  f2py_tls_define = ''
  foreach tls_attrs: check_tls_attrs
    attr = tls_attrs[0]
    code = f'int @attr@ foo;'
    code += '''
      int
      main()
      {
        return 0;
      }
    '''
    if cc.compiles(code, name: tls_attrs[0])
      cdata.set10(tls_attrs[1], true)
      f2py_tls_define = tls_attrs[0]
    endif
  endforeach
  _f2py_c_args = [f'-DF2PY_THREAD_LOCAL_DECL=@f2py_tls_define@']
  fortranobject_dep = declare_dependency(
    dependencies: fortranobject_dep,
    compile_args: _f2py_c_args,
  )

  # Check that the f2py version is compatible with numpy
  f2py = find_program('f2py')
  f2py_version = run_command([f2py, '-v'], check: true).stdout().strip()
  if f2py_version.version_compare('<'+min_numpy_version)
    error(f'Found f2py executable too old: @f2py_version@')
  endif

endif

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py3.full_path())
conf_data.set('PYTHON_VERSION', py3.language_version())

# Install the pure Python parts of this package
# Get the directory where tamoc will be installed
tamoc_dir = py3.get_install_dir() / 'tamoc'

# Write the configuration data to __config__.py
configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration : conf_data,
  install_dir: tamoc_dir,
  install_tag: 'python-runtime',
)

subdir('tamoc')
