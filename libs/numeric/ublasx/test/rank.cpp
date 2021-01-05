/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/rank.cpp
 *
 * \brief Test suite for the \c rank operation.
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
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/rank.hpp>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( rank_deficient )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Rank Deficient matrix");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const std::size_t m = 3;
    const std::size_t n = 3;

    matrix_type A(m,n);
    A(0,0) = 3; A(0,1) = 1; A(0,2) = 2;
    A(1,0) = 2; A(1,1) = 0; A(1,2) = 5;
    A(2,0) = 5; A(2,1) = 1; A(2,2) = 7;

    size_type r = ublasx::rank(A);
    size_type expect_r = n-1;
    BOOST_UBLASX_DEBUG_TRACE("A = " << A);
    BOOST_UBLASX_DEBUG_TRACE("rank = " << r);
    BOOST_UBLASX_TEST_CHECK( r == expect_r );
}


BOOST_UBLASX_TEST_DEF( full_rank )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Full Rank matrix");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const std::size_t m = 3;
    const std::size_t n = 3;

    matrix_type A(m,n);
    A(0,0) = 3; A(0,1) = 1; A(0,2) = 2;
    A(1,0) = 2; A(1,1) = 0; A(1,2) = 5;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 3;

    size_type r = ublasx::rank(A);
    size_type expect_r = n;
    BOOST_UBLASX_DEBUG_TRACE("A = " << A);
    BOOST_UBLASX_DEBUG_TRACE("rank = " << r);
    BOOST_UBLASX_TEST_CHECK( r == expect_r );
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( rank_deficient );
    BOOST_UBLASX_TEST_DO( full_rank );

    BOOST_UBLASX_TEST_END();
}
