# This part of the meson.build cycle installs all the required package
# files and extension modules

# List all Python source files to include
util_sources = [
  '__init__.py',
  'bpm_utils.py',
  'ambient_utils.py',
  'eos_utils.py'
]


# Install the Python source files
py3.install_sources(
  util_sources,
  subdir: 'tamoc/utils',
)