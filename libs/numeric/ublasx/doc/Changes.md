# Release Highlights

This page summarizes the most important changes.
For a more detailed list, please see the commit log (e.g., go to the GitHub [history page](/sguazt/boost-ublasx/commits/master) or run `git log` command from the command line).


## Version 2.x

### Breaking Changes

- Dropped support for C++ standards less than C++11.
- Changed the following operations to work like their MATLAB/Octave counterparts: `isinf`, `sign`.

### New Features

- New operations: `eye`, `realmax`.

### Fixes

### Other Changes

- Added test suite for `realmin`.


## Version 1.x

### New Features

- New operations: `linspace`, `log`, `log10`, `logspace`, `mldivide`, `pow`, `sign`, `tanh`
- New expression types: `matrix_binary_functor`, `matrix_unary_functor`, `vector_binary_functor`, `vector_unary_functor`
- New variants for the following operations: `cat`, `element_pow`.

### Fixes

- Fixed `rcond` operation for banded matrices.
- Recomputed expected results on some test suites with recent versions of MATLAB and Octave.
- Fixed some compilation issues with recent compilers

### Other Changes

- Improved documentation.
- Improved test suites.


## Version 0.x

Initial release.
