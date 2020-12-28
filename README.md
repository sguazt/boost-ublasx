Boost.uBLASx
============

Extensions to Boost.uBLAS library


Building
--------

### Prerequisites

* A modern C++98 compiler (e.g., GCC v4.8 or newer is fine)
    * Tested for GCC 10.2.1, 7.4.1, 6.3.0, 4.8.4
* [Boost](http://boost.org) C++ libraries (v1.54 or newer)
    * Tested for Boost 1.73, 1.62, 1.54
* [Boost Numeric Bindings](https://github.com/uBLAS/numeric_bindings)
    * One may also choose using older SVN version of [Boost Numeric Bindings](https://svn.boost.org/svn/boost/sandbox/numeric_bindings)
* [LAPACK](http://www.netlib.org/lapack/) Linear Algebra PACKage (v3.5 or newer)
    * Tested for LAPACK 3.9.0, 3.5.0

### Compilation 

This is a header-only library and thus it does not need to be compiled. 

To compile test files, create `user-config.mk` on the base of
`user-config.mk.template` file. Then `make` the project.


Documentation
-------------

The source code is annotated with [Doxygen](https://www.doxygen.nl/) tags.
To generate the HTML documentation, install Doxygen, run `make apidoc`, and the open the file `libs/numeric/ublasx/doc/apidoc/html/index.html`.


Credits
-------

I really thank you the following users for their valuable contribution given to this library:

- [Alexey Nesterenko](https://github.com/comcon1)
