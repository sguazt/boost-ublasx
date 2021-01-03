/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/cat.cpp
 *
 * \brief Tests for the family of \c cat operations.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <algorithm>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/cat.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"
#include <iostream>

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;

static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_rows_dense_matrix_column_major_same_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Rows - Dense Matrix - Column Major - Same Rows Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nc = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(A_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10;
    B(1,0) = 11; B(1,1) = 12; B(1,2) = 13; B(1,3) = 14;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr, A_nc+B_nc);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) = B(0,0); expect_R(0,4) = B(0,1); expect_R(0,5) = B(0,2); expect_R(0,6) = B(0,3);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) = B(1,0); expect_R(1,4) = B(1,1); expect_R(1,5) = B(1,2); expect_R(1,6) = B(1,3);

    R = ublasx::cat_rows(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_rows(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_rows(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_rows_dense_matrix_row_major_same_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Rows - Dense Matrix - Row Major - Same Rows Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nc = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(A_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10;
    B(1,0) = 11; B(1,1) = 12; B(1,2) = 13; B(1,3) = 14;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr, A_nc+B_nc);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) = B(0,0); expect_R(0,4) = B(0,1); expect_R(0,5) = B(0,2); expect_R(0,6) = B(0,3);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) = B(1,0); expect_R(1,4) = B(1,1); expect_R(1,5) = B(1,2); expect_R(1,6) = B(1,3);

    R = ublasx::cat_rows(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_rows(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_rows(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_rows_dense_matrix_column_major_diff_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Rows - Dense Matrix - Column Major - Different Rows Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 3;
    const size_type B_nc = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10;
    B(1,0) = 11; B(1,1) = 12; B(1,2) = 13; B(1,3) = 14;
    B(2,0) = 15; B(2,1) = 16; B(2,2) = 17; B(2,3) = 18;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(std::max(A_nr,B_nr), A_nc+B_nc);

    // cat_columns(A,B)
    R = ublasx::cat_rows(A, B);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) = B(0,0); expect_R(0,4) = B(0,1); expect_R(0,5) = B(0,2); expect_R(0,6) = B(0,3);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) = B(1,0); expect_R(1,4) = B(1,1); expect_R(1,5) = B(1,2); expect_R(1,6) = B(1,3);
    expect_R(2,0) =      0; expect_R(2,1) =      0; expect_R(2,2) =      0; expect_R(2,3) = B(2,0); expect_R(2,4) = B(2,1); expect_R(2,5) = B(2,2); expect_R(2,6) = B(2,3);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_rows(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_rows(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_rows_dense_matrix_row_major_diff_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Rows - Dense Matrix - Row Major - Different Rows Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 3;
    const size_type B_nc = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10;
    B(1,0) = 11; B(1,1) = 12; B(1,2) = 13; B(1,3) = 14;
    B(2,0) = 15; B(2,1) = 16; B(2,2) = 17; B(2,3) = 18;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(std::max(A_nr,B_nr), A_nc+B_nc);

    // cat_columns(A,B)
    R = ublasx::cat_rows(A, B);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) = B(0,0); expect_R(0,4) = B(0,1); expect_R(0,5) = B(0,2); expect_R(0,6) = B(0,3);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) = B(1,0); expect_R(1,4) = B(1,1); expect_R(1,5) = B(1,2); expect_R(1,6) = B(1,3);
    expect_R(2,0) =      0; expect_R(2,1) =      0; expect_R(2,2) =      0; expect_R(2,3) = B(2,0); expect_R(2,4) = B(2,1); expect_R(2,5) = B(2,2); expect_R(2,6) = B(2,3);
    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_rows(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_rows(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_columns_dense_matrix_column_major_same_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Columns - Dense Matrix - Row Major - Same Columns Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,A_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9;
    B(1,0) = 10; B(1,1) = 11; B(1,2) = 12;
    B(2,0) = 13; B(2,1) = 14; B(2,2) = 15;
    B(3,0) = 16; B(3,1) = 17; B(3,2) = 18;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr+B_nr, A_nc);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2);
    expect_R(2,0) = B(0,0); expect_R(2,1) = B(0,1); expect_R(2,2) = B(0,2);
    expect_R(3,0) = B(1,0); expect_R(3,1) = B(1,1); expect_R(3,2) = B(1,2);
    expect_R(4,0) = B(2,0); expect_R(4,1) = B(2,1); expect_R(4,2) = B(2,2);
    expect_R(5,0) = B(3,0); expect_R(5,1) = B(3,1); expect_R(5,2) = B(3,2);

    R = ublasx::cat_columns(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_columns(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_columns(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_columns_dense_matrix_column_major_diff_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Columns - Dense Matrix - Row Major - Different Columns Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 4;
    const size_type B_nc = 5;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10; B(0,4) = 11;
    B(1,0) = 12; B(1,1) = 13; B(1,2) = 14; B(1,3) = 15; B(1,4) = 16;
    B(2,0) = 17; B(2,1) = 18; B(2,2) = 19; B(2,3) = 20; B(2,4) = 21;
    B(3,0) = 22; B(3,1) = 23; B(3,2) = 24; B(3,3) = 25; B(3,4) = 26;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr+B_nr, std::max(A_nc,B_nc));
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) =      0; expect_R(0,4) =      0;
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) =      0; expect_R(1,4) =      0;
    expect_R(2,0) = B(0,0); expect_R(2,1) = B(0,1); expect_R(2,2) = B(0,2); expect_R(2,3) = B(0,3); expect_R(2,4) = B(0,4);
    expect_R(3,0) = B(1,0); expect_R(3,1) = B(1,1); expect_R(3,2) = B(1,2); expect_R(3,3) = B(1,3); expect_R(3,4) = B(1,4);
    expect_R(4,0) = B(2,0); expect_R(4,1) = B(2,1); expect_R(4,2) = B(2,2); expect_R(4,3) = B(2,3); expect_R(4,4) = B(2,4);
    expect_R(5,0) = B(3,0); expect_R(5,1) = B(3,1); expect_R(5,2) = B(3,2); expect_R(5,3) = B(3,3); expect_R(5,4) = B(3,4);

    R = ublasx::cat_columns(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_columns(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_columns(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_columns_dense_matrix_row_major_same_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Columns - Dense Matrix - Row Major - Same Columns Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 4;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,A_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9;
    B(1,0) = 10; B(1,1) = 11; B(1,2) = 12;
    B(2,0) = 13; B(2,1) = 14; B(2,2) = 15;
    B(3,0) = 16; B(3,1) = 17; B(3,2) = 18;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr+B_nr, A_nc);
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2);
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2);
    expect_R(2,0) = B(0,0); expect_R(2,1) = B(0,1); expect_R(2,2) = B(0,2);
    expect_R(3,0) = B(1,0); expect_R(3,1) = B(1,1); expect_R(3,2) = B(1,2);
    expect_R(4,0) = B(2,0); expect_R(4,1) = B(2,1); expect_R(4,2) = B(2,2);
    expect_R(5,0) = B(3,0); expect_R(5,1) = B(3,1); expect_R(5,2) = B(3,2);

    R = ublasx::cat_columns(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_columns(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_columns(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


BOOST_UBLASX_TEST_DEF( test_columns_dense_matrix_row_major_diff_dim )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Concatenate Columns - Dense Matrix - Row Major - Different Columns Number" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const size_type A_nr = 2;
    const size_type A_nc = 3;
    const size_type B_nr = 4;
    const size_type B_nc = 5;

    matrix_type A(A_nr,A_nc);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type B(B_nr,B_nc);
    B(0,0) =  7; B(0,1) =  8; B(0,2) =  9; B(0,3) = 10; B(0,4) = 11;
    B(1,0) = 12; B(1,1) = 13; B(1,2) = 14; B(1,3) = 15; B(1,4) = 16;
    B(2,0) = 18; B(2,1) = 19; B(2,2) = 20; B(2,3) = 21; B(2,4) = 22;
    B(3,0) = 23; B(3,1) = 24; B(3,2) = 25; B(3,3) = 26; B(3,4) = 27;

    matrix_type R;
    matrix_type expect_R;

    expect_R = matrix_type(A_nr+B_nr, std::max(A_nc,B_nc));
    expect_R(0,0) = A(0,0); expect_R(0,1) = A(0,1); expect_R(0,2) = A(0,2); expect_R(0,3) =      0; expect_R(0,4) =      0;
    expect_R(1,0) = A(1,0); expect_R(1,1) = A(1,1); expect_R(1,2) = A(1,2); expect_R(1,3) =      0; expect_R(1,4) =      0;
    expect_R(2,0) = B(0,0); expect_R(2,1) = B(0,1); expect_R(2,2) = B(0,2); expect_R(2,3) = B(0,3); expect_R(2,4) = B(0,4);
    expect_R(3,0) = B(1,0); expect_R(3,1) = B(1,1); expect_R(3,2) = B(1,2); expect_R(3,3) = B(1,3); expect_R(3,4) = B(1,4);
    expect_R(4,0) = B(2,0); expect_R(4,1) = B(2,1); expect_R(4,2) = B(2,2); expect_R(4,3) = B(2,3); expect_R(4,4) = B(2,4);
    expect_R(5,0) = B(3,0); expect_R(5,1) = B(3,1); expect_R(5,2) = B(3,2); expect_R(5,3) = B(3,3); expect_R(5,4) = B(3,4);

    R = ublasx::cat_columns(A, B);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "cat_columns(A,B) = " << R );
    BOOST_UBLASX_DEBUG_TRACE( "expect cat_columns(A,B) = " << expect_R );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(R) == ublasx::num_rows(expect_R) );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(R) == ublasx::num_columns(expect_R) );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, ublasx::num_rows(expect_R), ublasx::num_columns(expect_R), tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'cat' operations");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_columns_dense_matrix_column_major_same_dim );
    BOOST_UBLASX_TEST_DO( test_columns_dense_matrix_row_major_same_dim );
    BOOST_UBLASX_TEST_DO( test_columns_dense_matrix_column_major_diff_dim );
    BOOST_UBLASX_TEST_DO( test_columns_dense_matrix_row_major_diff_dim );
    BOOST_UBLASX_TEST_DO( test_rows_dense_matrix_column_major_same_dim );
    BOOST_UBLASX_TEST_DO( test_rows_dense_matrix_row_major_same_dim );
    BOOST_UBLASX_TEST_DO( test_rows_dense_matrix_column_major_diff_dim );
    BOOST_UBLASX_TEST_DO( test_rows_dense_matrix_row_major_diff_dim );

    BOOST_UBLASX_TEST_END();
}
