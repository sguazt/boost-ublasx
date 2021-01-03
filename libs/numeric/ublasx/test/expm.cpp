/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/cond.cpp
 *
 * \brief Test suite for the \c cond operation.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */


#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/expm.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( complex_dense_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Dense Matrix");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(3);
    const std::size_t nc(3);

    value_type img = value_type(0,1);

    matrix_type gen(nr,nc);  // Generator of rotaion around z aix in group theory

    gen(0,0) = 0  ; gen(0,1) = -img; gen(0,2) = 0;
    gen(1,0) = img; gen(1,1) = 0   ; gen(1,2) = 0;
    gen(2,0) = 0  ; gen(2,1) = 0   ; gen(2,2) = 0;

    value_type theta(1.5);

    matrix_type mat(nr,nc);

    mat = img * theta * gen;

    matrix_type res;
    matrix_type expect_res(nr,nc);

    res = ublasx::expm_pad(mat);
    //cout<< "Rotation Matrix : "<< expm_pad(gen) <<"\n\n";

    // Results obtained with:
    // - MATLAB 2017a
    // - Octave 5.2.0
    // on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
    // ```octave
    // A=[0 0+1i 0; 0+1i 0 0; 0 0 0]
    // B=(0+1i)*1.5*A
    // expm(B)
    // ```

    expect_res(0,0) = value_type( 2.352409615243247,0); expect_res(0,1) = value_type(-2.129279455094817,0); expect_res(0,2) = value_type(0.000000000000000,0);
    expect_res(1,0) = value_type(-2.129279455094817,0); expect_res(1,1) = value_type( 2.352409615243247,0); expect_res(1,2) = value_type(0.000000000000000,0);
    expect_res(2,0) = value_type( 0.000000000000000,0); expect_res(2,1) = value_type( 0.000000000000000,0); expect_res(2,2) = value_type(1.000000000000000,0);

    BOOST_UBLASX_DEBUG_TRACE("res = " << res);
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect_res, nr, nc, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'expm' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( complex_dense_matrix );

    BOOST_UBLASX_TEST_END();
}
