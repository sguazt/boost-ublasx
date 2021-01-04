/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 *  \file libs/numeric/ublasx/eye.cpp
 *
 *  \brief Test suite for the \c eye operation.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 *
 *  <hr/>
 *
 *  Copyright (c) 2009, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/eye.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( default_square_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Default Square Matrix" );

    const std::size_t n = 3;

    ublas::identity_matrix<> res = ublasx::eye(n);
    ublas::identity_matrix<> expect_res(n);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, n, n );
}

BOOST_UBLASX_TEST_DEF( real_square_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Square Matrix" );

    typedef double value_type;

    const std::size_t n = 3;

    ublas::identity_matrix<value_type> res = ublasx::eye<value_type>(n);
    ublas::identity_matrix<value_type> expect_res(n);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, n, n );
}

BOOST_UBLASX_TEST_DEF( complex_square_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex Square Matrix" );

    typedef std::complex<double> value_type;

    const std::size_t n = 3;

    ublas::identity_matrix<value_type> res = ublasx::eye<value_type>(n);
    ublas::identity_matrix<value_type> expect_res(n);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, n, n );
}

BOOST_UBLASX_TEST_DEF( default_rect_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Default Rectangular Matrix" );

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    ublas::identity_matrix<> res = ublasx::eye(nr, nc);
    ublas::identity_matrix<> expect_res(nr, nc);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, nr, nc );
}

BOOST_UBLASX_TEST_DEF( real_rect_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Rectangular Matrix" );

    typedef double value_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    ublas::identity_matrix<value_type> res = ublasx::eye<value_type>(nr, nc);
    ublas::identity_matrix<value_type> expect_res(nr, nc);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, nr, nc );
}

BOOST_UBLASX_TEST_DEF( complex_rect_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex Rectangular Matrix" );

    typedef std::complex<double> value_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    ublas::identity_matrix<value_type> res = ublasx::eye<value_type>(nr, nc);
    ublas::identity_matrix<value_type> expect_res(nr, nc);

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect_res, nr, nc );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'eye' operations");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( default_square_matrix );
    BOOST_UBLASX_TEST_DO( real_square_matrix );
    BOOST_UBLASX_TEST_DO( complex_square_matrix );
    BOOST_UBLASX_TEST_DO( default_rect_matrix );
    BOOST_UBLASX_TEST_DO( real_rect_matrix );
    BOOST_UBLASX_TEST_DO( complex_rect_matrix );

    BOOST_UBLASX_TEST_END();
}
