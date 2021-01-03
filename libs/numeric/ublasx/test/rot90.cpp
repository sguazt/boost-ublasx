/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/rot90.cpp
 *
 * \brief Test suite for the \c rot90 operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/rot90.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(4);

    vector_type v(n);

    v(0) = 1;
    v(1) = 2;
    v(2) = 3;
    v(3) = 4;

    vector_type res;
    vector_type expect_res(n);

    // k=1
    res = ublasx::rot90(v, 1);
    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(v) = " << res );
    expect_res = v;
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );

    // k=2
    res = ublasx::rot90(v, 2);
    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(v) = " << res );
    expect_res.clear();
    for (size_type i = 0; i < n; ++i)
    {
        expect_res(n-i-1) = v(i);
    }
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );

    // k=3
    res = ublasx::rot90(v, 3);
    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(v) = " << res );
    expect_res.clear();
    for (size_type i = 0; i < n; ++i)
    {
        expect_res(n-i-1) = v(i);
    }
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );

    // k=4
    res = ublasx::rot90(v, 4);
    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(v,4) = " << res );
    expect_res = v;
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    matrix_type A(nr,nc);

    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    // k=1
    R = ublasx::rot90(A);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(A,1) = " << R );
    expect_R.resize(nc, nr, false);
    expect_R.clear();
    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(nc-c-1,r) = A(r,c);
        }
    }
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nc, nr, tol );

    // k=2
    R = ublasx::rot90(A, 2);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(A,2) = " << R );
    expect_R.resize(nr, nc, false);
    expect_R.clear();
    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(nr-r-1,nc-c-1) = A(r,c);
        }
    }
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );

    // k=3
    R = ublasx::rot90(A, 3);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(A,3) = " << R );
    expect_R.resize(nc, nr, false);
    expect_R.clear();
    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(c,nr-r-1) = A(r,c);
        }
    }
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nc, nr, tol );

    // k=4
    R = ublasx::rot90(A, 4);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "rot90(A,4) = " << R );
    expect_R = A;
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'rot90' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_real_vector );
//  BOOST_UBLASX_TEST_DO( test_complex_vector );
    BOOST_UBLASX_TEST_DO( test_real_matrix );
//  BOOST_UBLASX_TEST_DO( test_complex_matrix );

    BOOST_UBLASX_TEST_END();
}
