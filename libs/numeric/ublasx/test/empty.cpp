/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/empty.cpp
 *
 * \brief Test suite for the \c empty operation.
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

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/empty.hpp>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( vector )
{
    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v;

    bool res;
    bool expect_res;

    // v not initialized
    expect_res = true;
    BOOST_UBLASX_DEBUG_TRACE( "v: " << v );
    res = ublasx::empty(v);
    BOOST_UBLASX_DEBUG_TRACE( "empty(v)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // v initialized with a zero size
    expect_res = true;
    v = vector_type(0);
    BOOST_UBLASX_DEBUG_TRACE( "v: " << v );
    res = ublasx::empty(v);
    BOOST_UBLASX_DEBUG_TRACE( "empty(v)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // v initialized with a non-zero size
    expect_res = false;
    v = vector_type(1);
    BOOST_UBLASX_DEBUG_TRACE( "v: " << v );
    res = ublasx::empty(v);
    BOOST_UBLASX_DEBUG_TRACE( "empty(v)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );
}


BOOST_UBLASX_TEST_DEF( matrix )
{
    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    matrix_type m;

    bool res;
    bool expect_res;

    // m not initialized
    expect_res = true;
    BOOST_UBLASX_DEBUG_TRACE( "v: " << m );
    res = ublasx::empty(m);
    BOOST_UBLASX_DEBUG_TRACE( "empty(m)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // m initialized with a zero row size
    expect_res = true;
    m = matrix_type(0,1);
    BOOST_UBLASX_DEBUG_TRACE( "v: " << m );
    res = ublasx::empty(m);
    BOOST_UBLASX_DEBUG_TRACE( "empty(m)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // m initialized with a zero column size
    expect_res = true;
    m = matrix_type(1,0);
    BOOST_UBLASX_DEBUG_TRACE( "m: " << m );
    res = ublasx::empty(m);
    BOOST_UBLASX_DEBUG_TRACE( "empty(m)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // m initialized with a zero size
    expect_res = true;
    m = matrix_type(0,0);
    BOOST_UBLASX_DEBUG_TRACE( "m: " << m );
    res = ublasx::empty(m);
    BOOST_UBLASX_DEBUG_TRACE( "empty(m)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );

    // m initialized with non-zero sizes
    expect_res = false;
    m = matrix_type(1,1);
    BOOST_UBLASX_DEBUG_TRACE( "m: " << m );
    res = ublasx::empty(m);
    BOOST_UBLASX_DEBUG_TRACE( "empty(m)? " << std::boolalpha << res );
    BOOST_UBLASX_TEST_CHECK( res == expect_res );
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( vector );
    BOOST_UBLASX_TEST_DO( matrix );
    BOOST_UBLASX_TEST_END();
}
