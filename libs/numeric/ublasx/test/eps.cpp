/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/eps.cpp.
 *
 * \brief Test suite for the \c eps operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/eps.hpp>
#include <cmath>
#include "libs/numeric/ublasx/test/utils.hpp"
#include <limits>


namespace ublasx = boost::numeric::ublasx;


const float float_tol = ::std::numeric_limits<float>::epsilon()*2.0;
const double double_tol = ::std::numeric_limits<double>::epsilon()*2.0;


BOOST_UBLASX_TEST_DEF( float_no_arg )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Float type - No arg");

    typedef float value_type;

    value_type eps = ublasx::eps<value_type>();
    value_type expect_eps = ::std::numeric_limits<value_type>::epsilon();

    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, float_tol);
}


BOOST_UBLASX_TEST_DEF( float_scalar_arg )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Float type - Scalar arg");

    typedef float value_type;

    value_type val;
    value_type eps;
    value_type expect_eps;


    val = 0;
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, float_tol);

    val = ::std::numeric_limits<value_type>::min();
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, float_tol);

    val = ::std::numeric_limits<value_type>::min()/static_cast<value_type>(2);
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, float_tol);

    val = ::std::numeric_limits<value_type>::infinity();
    eps = ublasx::eps<value_type>(val);
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK(std::isnan(eps));

    val = ::std::numeric_limits<value_type>::quiet_NaN();
    eps = ublasx::eps<value_type>(val);
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK(std::isnan(eps));
}


BOOST_UBLASX_TEST_DEF( double_no_arg )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double type - No arg");

    typedef double value_type;

    value_type eps = ublasx::eps<value_type>();
    value_type expect_eps = ::std::numeric_limits<value_type>::epsilon();

    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, double_tol);
}


BOOST_UBLASX_TEST_DEF( double_scalar_arg )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double type - Scalar arg");

    typedef double value_type;

    value_type val;
    value_type eps;
    value_type expect_eps;


    val = 0;
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, double_tol);

    val = ::std::numeric_limits<value_type>::min();
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, double_tol);

    val = ::std::numeric_limits<value_type>::min()/static_cast<value_type>(2);
    eps = ublasx::eps<value_type>(val);
    expect_eps = ::std::numeric_limits<value_type>::denorm_min();
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK_CLOSE(eps, expect_eps, double_tol);

    val = ::std::numeric_limits<value_type>::infinity();
    eps = ublasx::eps<value_type>(val);
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK(std::isnan(eps));

    val = ::std::numeric_limits<value_type>::quiet_NaN();
    eps = ublasx::eps<value_type>(val);
    BOOST_UBLASX_DEBUG_TRACE("val = " << val);
    BOOST_UBLASX_DEBUG_TRACE("eps = " << eps);
    BOOST_UBLASX_TEST_CHECK(std::isnan(eps));
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( float_no_arg )
    BOOST_UBLASX_TEST_DO( float_scalar_arg )
    BOOST_UBLASX_TEST_DO( double_no_arg )
    BOOST_UBLASX_TEST_DO( double_scalar_arg )

    BOOST_UBLASX_TEST_END();
}
