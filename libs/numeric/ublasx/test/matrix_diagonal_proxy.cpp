/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/matrix_diagonal_proxy.cpp
 *
 * \brief Test suite for the matrix diagonal proxy.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/proxy/matrix_diagonal.hpp>
#include <cstddef>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


const double tol = 1e-5;

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_square_matrix_main_diag )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Square Matrix -- Main Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t n(5);

    matrix_type A(n,n);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4; A(0,4) =  5;
    A(1,0) =  6; A(1,1) =  7; A(1,2) =  8; A(1,3) =  9; A(1,4) = 10;
    A(2,0) = 11; A(2,1) = 12; A(2,2) = 13; A(2,3) = 14; A(2,4) = 15;
    A(3,0) = 16; A(3,1) = 17; A(3,2) = 18; A(3,3) = 19; A(3,4) = 20;
    A(4,0) = 21; A(4,1) = 22; A(4,2) = 23; A(4,3) = 24; A(4,4) = 25;

    ublasx::matrix_diagonal<matrix_type> D(A, 0);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "D = " << D );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(D) == n );
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( D(i), A(i,i), tol );
    }
}


BOOST_UBLASX_TEST_DEF( test_trans_matrix_expr_main_diag )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Matrix Expression -- Transpose -- Main Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> aux_matrix_type;
    typedef ublas::matrix_unary2_traits<
                aux_matrix_type,
                ublas::scalar_identity<value_type>
            >::result_type matrix_type;

    const std::size_t n(5);

    aux_matrix_type A(n,n);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4; A(0,4) =  5;
    A(1,0) =  6; A(1,1) =  7; A(1,2) =  8; A(1,3) =  9; A(1,4) = 10;
    A(2,0) = 11; A(2,1) = 12; A(2,2) = 13; A(2,3) = 14; A(2,4) = 15;
    A(3,0) = 16; A(3,1) = 17; A(3,2) = 18; A(3,3) = 19; A(3,4) = 20;
    A(4,0) = 21; A(4,1) = 22; A(4,2) = 23; A(4,3) = 24; A(4,4) = 25;

    ublasx::matrix_diagonal<matrix_type const> D(ublas::trans(A), 0);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "A' = " << ublas::trans(A) );
    BOOST_UBLASX_DEBUG_TRACE( "D = " << D );
    BOOST_UBLASX_TEST_CHECK( ublasx::size(D) == n );
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( D(i), A(i,i), tol );
    }
}


BOOST_UBLASX_TEST_DEF( test_uminus_matrix_expr_main_diag )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Matrix Expression -- Unary Minus -- Main Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> aux_matrix_type;
    typedef ublas::matrix_unary1_traits<
                aux_matrix_type,
                ublas::scalar_negate<value_type>
            >::result_type matrix_type;

    const std::size_t n(5);

    aux_matrix_type A(n,n);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4; A(0,4) =  5;
    A(1,0) =  6; A(1,1) =  7; A(1,2) =  8; A(1,3) =  9; A(1,4) = 10;
    A(2,0) = 11; A(2,1) = 12; A(2,2) = 13; A(2,3) = 14; A(2,4) = 15;
    A(3,0) = 16; A(3,1) = 17; A(3,2) = 18; A(3,3) = 19; A(3,4) = 20;
    A(4,0) = 21; A(4,1) = 22; A(4,2) = 23; A(4,3) = 24; A(4,4) = 25;

    ublasx::matrix_diagonal<matrix_type const> D(-A, 0);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "-A = " << -A );
    BOOST_UBLASX_DEBUG_TRACE( "D = " << D );
    BOOST_UBLASX_TEST_CHECK( ublasx::size(D) == n );
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( D(i), -A(i,i), tol );
    }
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_square_matrix_main_diag );
    BOOST_UBLASX_TEST_DO( test_trans_matrix_expr_main_diag );
    BOOST_UBLASX_TEST_DO( test_uminus_matrix_expr_main_diag );

    BOOST_UBLASX_TEST_END();
}
