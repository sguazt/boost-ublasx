/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/seq.cpp
 *
 * \brief Test suite for the \c seq operation.
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
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/seq.hpp>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const std::size_t tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_range )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Range");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const std::size_t n(3);

    vector_type res;
    vector_type expect_res;

    res = ublasx::seq(5, 5+n);
    expect_res = vector_type(n);
    expect_res(0) = 5;
    expect_res(1) = 6;
    expect_res(2) = 7;

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_slice_incr )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Slice - Increasing");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const std::size_t n(9);

    vector_type res;
    vector_type expect_res;

    res = ublasx::seq(4, 2, n);
    expect_res = vector_type(n);
    std::size_t v(4);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect_res(i) = v;
        v += 2;
    }

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_slice_decr )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Slice - Decreasing");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const std::size_t n(9);

    vector_type res;
    vector_type expect_res;

    res = ublasx::seq(4, -2, n);
    expect_res = vector_type(n);
    short v(4);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect_res(i) = v;
        v -= 2;
    }

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
    BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'seq' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_range );
    BOOST_UBLASX_TEST_DO( test_slice_incr );
    BOOST_UBLASX_TEST_DO( test_slice_decr );

    BOOST_UBLASX_TEST_END();
}
