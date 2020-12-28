Boost.uBLASx
============

Extensions to Boost.uBLAS library

Building
--------

### Prerequisites

* A modern C++98 compiler (e.g., GCC v4.8 or newer is fine)
    * Tested for GCC 7.4.1, 6.3.0, 4.8.4
* [Boost](http://boost.org) C++ libraries (v1.54 or newer)
    * Tested for Boost 1.62
* [Boost Numeric Bindings](https://github.com/uBLAS/numeric_bindings)
    * One may also choose using older SVN version of [Boost Numeric Bindings](https://svn.boost.org/svn/boost/sandbox/numeric_bindings)
* [LAPACK](http://www.netlib.org/lapack/) Linear Algebra PACKage (v3.5 or newer)

### Compilation 

This is a header-only library and thus it does not need to be compiled. 

To compile test files, create `user-config.mk` on the base of
`user-config.mk.template` file. Then `make` the project.
