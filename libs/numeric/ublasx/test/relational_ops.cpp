/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/test/relational_opts.cpp
 *
 * \brief Test suite for matrix/vector relational operators.
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
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/relational_ops.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( equal_real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Vector - Equality");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const ::std::size_t n(4);

    vector_type u;
    vector_type v;

    bool res;
    bool expect;

    u = ublas::scalar_vector<value_type>(n, 1);
    v = ublas::scalar_vector<value_type>(n, 1);
    res = u == v;
    expect = true;
    BOOST_UBLASX_DEBUG_TRACE("u=" << u);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("u==v? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );

    u = ublas::scalar_vector<value_type>(n, 1);
    v = 2*ublas::scalar_vector<value_type>(n, 1);
    res = u == v;
    expect = false;
    BOOST_UBLASX_DEBUG_TRACE("u=" << u);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("u==v? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( not_equal_real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Vector - Inequality");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const ::std::size_t n(4);

    vector_type u;
    vector_type v;

    bool res;
    bool expect;

    u = ublas::scalar_vector<value_type>(n, 1);
    v = 2*ublas::scalar_vector<value_type>(n, 1);
    res = u != v;
    expect = true;
    BOOST_UBLASX_DEBUG_TRACE("u=" << u);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("u!=v? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );

    u = ublas::scalar_vector<value_type>(n, 1);
    v = ublas::scalar_vector<value_type>(n, 1);
    res = u != v;
    expect = false;
    BOOST_UBLASX_DEBUG_TRACE("u=" << u);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("u!=v? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( equal_real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Equality");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(4);

    matrix_type A;
    matrix_type B;

    bool res;
    bool expect;

    A = ublas::identity_matrix<value_type>(nr,nc);
    B = ublas::identity_matrix<value_type>(nr,nc);
    res = A == B;
    expect = true;
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("B=" << B);
    BOOST_UBLASX_DEBUG_TRACE("A==B? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );

    A = ublas::identity_matrix<value_type>(nr,nc);
    B = 2*ublas::identity_matrix<value_type>(nr,nc);
    res = A == B;
    expect = false;
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("B=" << B);
    BOOST_UBLASX_DEBUG_TRACE("A==B? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( not_equal_real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Inequality");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(4);

    matrix_type A;
    matrix_type B;

    bool res;
    bool expect;

    A = ublas::identity_matrix<value_type>(nr,nc);
    B = 2*ublas::identity_matrix<value_type>(nr,nc);
    res = A != B;
    expect = true;
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("B=" << B);
    BOOST_UBLASX_DEBUG_TRACE("A!=B? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );

    A = ublas::identity_matrix<value_type>(nr,nc);
    B = ublas::identity_matrix<value_type>(nr,nc);
    res = A != B;
    expect = false;
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("B=" << B);
    BOOST_UBLASX_DEBUG_TRACE("A!=B? " << std::boolalpha << res);
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( equal_real_vector );
    BOOST_UBLASX_TEST_DO( not_equal_real_vector );
    BOOST_UBLASX_TEST_DO( equal_real_matrix );
    BOOST_UBLASX_TEST_DO( not_equal_real_matrix );

    BOOST_UBLASX_TEST_END();
}
