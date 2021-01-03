/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/linspace.cpp
 *
 * \brief Test suite for the \c linspace operation.
 *
 * Copyright (c) 2013, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/linspace.hpp>
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

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::linspace(a, b, n);
    expect_res(0) = 0;
    expect_res(1) = 0.555555555555555;
    expect_res(2) = 1.111111111111111;
    expect_res(3) = 1.666666666666667;
    expect_res(4) = 2.222222222222222;
    expect_res(5) = 2.777777777777778;
    expect_res(6) = 3.333333333333333;
    expect_res(7) = 3.888888888888889;
    expect_res(8) = 4.444444444444445;
    expect_res(9) = 5;

    BOOST_UBLASX_DEBUG_TRACE( "linspace(" << a << "," << b << "," << n << ") = " << res );

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

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::linspace(a, b, n);
    expect_res(0) = 5;
    expect_res(1) = 4.444444444444445;
    expect_res(2) = 3.888888888888889;
    expect_res(3) = 3.333333333333333;
    expect_res(4) = 2.777777777777778;
    expect_res(5) = 2.222222222222222;
    expect_res(6) = 1.666666666666667;
    expect_res(7) = 1.111111111111111;
    expect_res(8) = 0.555555555555555;
    expect_res(9) = 0;

    BOOST_UBLASX_DEBUG_TRACE( "linspace(" << a << "," << b << "," << n << ") = " << res );

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

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::linspace(a, b, n);
    expect_res(0) = 5;

    BOOST_UBLASX_DEBUG_TRACE( "linspace(" << a << "," << b << "," << n << ") = " << res );

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

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::linspace(a, b, n);
    expect_res(0) = 0;

    BOOST_UBLASX_DEBUG_TRACE( "linspace(" << a << "," << b << "," << n << ") = " << res );

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}

int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'linspace' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_increasing );
    BOOST_UBLASX_TEST_DO( test_decreasing );
    BOOST_UBLASX_TEST_DO( test_increasing_single );
    BOOST_UBLASX_TEST_DO( test_decreasing_single );

    BOOST_UBLASX_TEST_END();
}
