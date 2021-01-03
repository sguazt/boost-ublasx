Boost.uBLASx
============

Extensions to [Boost.uBLAS](https://www.boost.org/doc/libs/release/libs/numeric/ublas/doc/index.html) library.


Overview
--------

The aim of this project is to extend the [Boost.uBLAS](https://www.boost.org/doc/libs/release/libs/numeric/ublas/doc/index.html) library with useful functions and features similar to those available in numerical packages (e.g., [MATLAB](https://www.mathworks.com/products/matlab.html) and [Octave](https://www.gnu.org/software/octave/index), and libraries for scientific computing (e.g., [Armadillo](http://arma.sourceforge.net/) and [Eigen](http://eigen.tuxfamily.org/)).

In a nutshell, Boost.uBLASx provides the following features:
* A set of vector/matrix operations (e.g., the `reshape` operation, for changing the shape of an array).
The list of currently available operations, together with a comparison with similar MATLAB/Octave functions, is available [here](libs/numeric/ublasx/doc/MATLAB).
* New container classes and adaptors (e.g., the `generalized_diagonal_matrix` class`, for representing generalized diagonal matrices).
* New expression types (e.g., the `matrix_binary_function` class, for representing binary matrix functions).
* New proxy classes (e.g., the `matrix_diagonal` class, for accessing to a specific diagonal of a given matrix)
* New storage classes (e.g., the `array_reference` class, for representing references to an array).
* New type traits (e.g., the `layout_type` class, for determining the layout of a matrix expression).


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
    * LAPACK is needed only by the followin operations: `balance`, `eigen`, `expm`, `lsq`, `ql`, `qr`, `qz`, `rcond`, `svd`.
    * Tested for LAPACK 3.9.0, 3.5.0

### Compilation 

This is a header-only library and thus it does not need to be compiled. 

To compile test and example files, create `user-config.mk` on the base of `user-config.mk.template` file.
Then `make` the project.

You can also build test files and example files separately:
- Build test files: `make clean test`
- Build example files: `make clean examples`


Getting Started
---------------

Here below is a simple C++ program showing some feature of Boost.uBLASx.
You can find more examples in the [examples](libs/numeric/ublasx/examples) and [test](libs/numeric/ublasx/test) directories.

```c++
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operations.hpp>

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;

int main()
{
    ublas::vector<double> a = ublasx::linspace(0.0, 2.0, 2); // a = [0 2]
    ublas::vector<double> b = ublasx::linspace(1.0, 2.0, 2); // b = [1 2]
    ublas::vector<double> c = 2*a + 3*b; // c = [3 10]
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "c = 2*a + 3*b = " << c << "\n";

    ublas::matrix<double> A = ublasx::rot90(2*ublas::identity_matrix<double>(2)); // A = [0 2; 2 0]
    ublas::matrix<double> B = ublasx::inv(A); // B = [0 0.5; 0.5 0]
    std::cout << "A = " << A << "\n";
    std::cout << "rank of A = " << ublasx::rank(A) << "\n";
    std::cout << "inverse of A = " << B << "\n";

    ublas::matrix<double> C = ublasx::reshape(ublasx::linspace(1.0, 9.0, 9), 3, 3); // C = [1 2 3; 4 5 6; 7 8 9]
    ublas::matrix<double> D = ublasx::pow2(C); // [2 4 8; 16 32 64; 128 256 512]
    ublas::matrix<double> E = ublasx::cat<2>(C, D); // [1 2 3 2 4 8; 4 5 6 16 32 64; 7 8 9 128 256 512]
    std::cout << "C = " << C << "\n";
    std::cout << "D = 2.^C = " << D << "\n";
    std::cout << "E = [C D] = " << E << "\n";
}
```


Documentation
-------------

The source code is annotated with [Doxygen](https://www.doxygen.nl/) tags.
To generate the documentation in HTML, install Doxygen, run `make apidoc`, and then open the file `libs/numeric/ublasx/doc/api/html/index.html`.


Authors
-------

- [Marco Guazzone](http://people.unipmn.it/sguazt)


Credits
-------

I really thank you the following users for their valuable contribution given to this library:

- [Alexey Nesterenko](https://github.com/comcon1)
