# This part of the meson.build cycle installs all the required package
# files and extension modules

# List all Python source files to include
tool_sources = [
  '__init__.py',
  'bpm_tools.py',
  'ambient_tools.py',
  'eos_tools.py'
]


# Install the Python source files
py3.install_sources(
  tool_sources,
  subdir: 'tamoc/tools',
)