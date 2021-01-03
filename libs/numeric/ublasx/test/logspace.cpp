/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/logspace.cpp
 *
 * \brief Test suite for the \c logspace operation.
 *
 * Copyright (c) 2015, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/logspace.hpp>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_increasing )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Increasing Sequence" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const value_type a(0);
    const value_type b(5);
    const size_type n(10);
    const value_type base(10);

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::logspace(a, b, n, base);
    expect_res(0) = 1.00000000000000e+00;
    expect_res(1) = 3.59381366380463e+00;
    expect_res(2) = 1.29154966501488e+01;
    expect_res(3) = 4.64158883361278e+01;
    expect_res(4) = 1.66810053720006e+02;
    expect_res(5) = 5.99484250318941e+02;
    expect_res(6) = 2.15443469003188e+03;
    expect_res(7) = 7.74263682681128e+03;
    expect_res(8) = 2.78255940220713e+04;
    expect_res(9) = 1.00000000000000e+05;

    BOOST_UBLASX_DEBUG_TRACE( "logspace(" << a << "," << b << "," << n << "," << base << ") = " << res );

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_decreasing )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Decreasing Sequence" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const value_type a(5);
    const value_type b(0);
    const size_type n(10);
    const value_type base(10);

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::logspace(a, b, n, base);
    expect_res(0) = 1.00000000000000e+05;
    expect_res(1) = 2.78255940220713e+04;
    expect_res(2) = 7.74263682681127e+03;
    expect_res(3) = 2.15443469003188e+03;
    expect_res(4) = 5.99484250318941e+02;
    expect_res(5) = 1.66810053720006e+02;
    expect_res(6) = 4.64158883361278e+01;
    expect_res(7) = 1.29154966501488e+01;
    expect_res(8) = 3.59381366380463e+00;
    expect_res(9) = 1.00000000000000e+00;

    BOOST_UBLASX_DEBUG_TRACE( "logspace(" << a << "," << b << "," << n << "," << base << ") = " << res );

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_increasing_single )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Increasing Singleton Sequence" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const value_type a(0);
    const value_type b(5);
    const size_type n(1);
    const value_type base(10);

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::logspace(a, b, n, base);
    expect_res(0) = 1.00000000000000e+05;

    BOOST_UBLASX_DEBUG_TRACE( "logspace(" << a << "," << b << "," << n << "," << base << ") = " << res );

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_decreasing_single )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Decreasing Singleton Sequence" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const value_type a(5);
    const value_type b(0);
    const size_type n(1);
    const value_type base(10);

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::logspace(a, b, n, base);
    expect_res(0) = 1.00000000000000e+00;

    BOOST_UBLASX_DEBUG_TRACE( "logspace(" << a << "," << b << "," << n << "," << base << ") = " << res );

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}

int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'logspace' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_increasing );
    BOOST_UBLASX_TEST_DO( test_decreasing );
    BOOST_UBLASX_TEST_DO( test_increasing_single );
    BOOST_UBLASX_TEST_DO( test_decreasing_single );

    BOOST_UBLASX_TEST_END();
}
