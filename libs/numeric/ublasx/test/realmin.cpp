/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/realmin.cpp.
 *
 * \brief Test suite for the \c realmin operation.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublasx/operation/realmin.hpp>
#include <cmath>
#include "libs/numeric/ublasx/test/utils.hpp"
#include <limits>


namespace ublasx = boost::numeric::ublasx;


const float float_tol = ::std::numeric_limits<float>::epsilon()*2.0;
const double double_tol = ::std::numeric_limits<double>::epsilon()*2.0;


BOOST_UBLASX_TEST_DEF( single_precision )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Single Precision");

    typedef float value_type;

    value_type res = ublasx::realmin<value_type>();
    value_type expect_res = ::std::numeric_limits<value_type>::min();

    BOOST_UBLASX_DEBUG_TRACE("res = " << res);
    BOOST_UBLASX_TEST_CHECK_CLOSE(res, expect_res, float_tol);
}

BOOST_UBLASX_TEST_DEF( double_precision )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Precision");

    typedef double value_type;

    value_type res = ublasx::realmin<value_type>();
    value_type expect_res = ::std::numeric_limits<value_type>::min();

    BOOST_UBLASX_DEBUG_TRACE("res = " << res);
    BOOST_UBLASX_TEST_CHECK_CLOSE(res, expect_res, float_tol);
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'realmin' operations");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( single_precision )
    BOOST_UBLASX_TEST_DO( double_precision )

    BOOST_UBLASX_TEST_END();
}
