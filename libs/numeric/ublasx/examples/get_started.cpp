/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libw/numeric/ublasx/operation/balance.cpp
 *
 * \brief Test case for matrix balance operation.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail&gt;
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operations.hpp>


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


int main()
{
    // MATLAB/Octave:
    //   a = linspace(0, 2, 2)
    //   b = linspace(1, 2, 2)
    //   c = 2*a + 3*b
    ublas::vector<double> a = ublasx::linspace(0.0, 2.0, 2); // a = [0 2]
    ublas::vector<double> b = ublasx::linspace(1.0, 2.0, 2); // b = [1 2]
    ublas::vector<double> c = 2*a + 3*b; // c = [3 10]
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "c = 2*a + 3*b = " << c << "\n";

    // MATLAB/Octave:
    //   A = rot90(2*eye(2))
    //   rank(A)
    //   B = inv(A)
    ublas::matrix<double> A = ublasx::rot90(2*ublas::identity_matrix<double>(2)); // A = [0 2; 2 0]
    ublas::matrix<double> B = ublasx::inv(A); // B = [0 0.5; 0.5 0]
    std::cout << "A = " << A << "\n";
    std::cout << "rank of A = " << ublasx::rank(A) << "\n";
    std::cout << "inverse of A = " << B << "\n";

    // MATLAB/Octave:
    //   C = reshape(linspace(1, 9, 9), 3, 3)'
    //   D = pow2(C)
    //   E = cat(2, C, D)
    ublas::matrix<double> C = ublasx::reshape(ublasx::linspace(1.0, 9.0, 9), 3, 3); // C = [1 2 3; 4 5 6; 7 8 9]
    ublas::matrix<double> D = ublasx::pow2(C); // [2 4 8; 16 32 64; 128 256 512]
    ublas::matrix<double> E = ublasx::cat<2>(C, D); // [1 2 3 2 4 8; 4 5 6 16 32 64; 7 8 9 128 256 512]
    std::cout << "C = " << C << "\n";
    std::cout << "D = 2.^C = " << D << "\n";
    std::cout << "E = [C D] = " << E << "\n";
}
