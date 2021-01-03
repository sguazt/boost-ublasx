/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/lu.cpp
 *
 * \brief Test suite for the LU decomposition.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/lu.hpp>
#include <cmath>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


static const double TOL(1.0e-5);


BOOST_UBLASX_TEST_DEF( lu_solve_square_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: LU solver - Square Matrix - Column Major" );

    BOOST_UBLASX_DEBUG_STREAM_SETFLAGS( std::ios::boolalpha );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type n(4);

    matrix_type A(n,n);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;

    vector_type b(n);

    b(0) = 2.0;
    b(1) = 3.0;
    b(2) = 1.0;
    b(3) = 0.5;

    vector_type expect(n);

    expect(0) =  1.339863;
    expect(1) =  0.198970;
    expect(2) =  4.699314;
    expect(3) = -1.677257;


    vector_type x(n);

    size_type res;
    res = ublasx::lu_solve(A, b, x);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "LU solver succeded?  " << static_cast<bool>(res == 0) );
    BOOST_UBLASX_DEBUG_TRACE( "Ax = b ==> x = " << x );

    BOOST_UBLASX_TEST_CHECK( res == 0 );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect, n, TOL );
}


BOOST_UBLASX_TEST_DEF( lu_solve_square_row_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: LU solver - Square Matrix - Row Major" );

    BOOST_UBLASX_DEBUG_STREAM_SETFLAGS( std::ios::boolalpha );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type n(4);

    matrix_type A(n,n);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;

    vector_type b(n);

    b(0) = 2.0;
    b(1) = 3.0;
    b(2) = 1.0;
    b(3) = 0.5;

    vector_type expect(n);

    expect(0) =  1.339863;
    expect(1) =  0.198970;
    expect(2) =  4.699314;
    expect(3) = -1.677257;


    vector_type x(n);

    size_type res;
    res = ublasx::lu_solve(A, b, x);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "LU solver succeded?  " << static_cast<bool>(res == 0) );
    BOOST_UBLASX_DEBUG_TRACE( "Ax = b ==> x = " << x );

    BOOST_UBLASX_TEST_CHECK( res == 0 );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect, n, TOL );
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( lu_solve_square_column_major );
    BOOST_UBLASX_TEST_DO( lu_solve_square_row_major );

    BOOST_UBLASX_TEST_END();
}
