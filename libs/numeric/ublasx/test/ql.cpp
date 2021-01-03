/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/ql.cpp
 *
 * \brief Test suite for the QL factorization.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/ql.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double tol = 1.0e-5;


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_real_square_matrix_row_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Square Matrix - Row Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_square_matrix_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real -Square Matrix - Column Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_recth_matrix_row_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Horizontal Matrix - Row Major - Full Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93; A(0,4) =  0.15; A(0,5) = -0.02;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64; A(1,4) =  0.30; A(1,5) =  1.03;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66; A(2,4) =  0.15; A(2,5) = -1.43;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08; A(3,4) = -2.13; A(3,5) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_recth_matrix_column_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Horizontal Matrix - Column Major - Full Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93; A(0,4) =  0.15; A(0,5) = -0.02;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64; A(1,4) =  0.30; A(1,5) =  1.03;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66; A(2,4) =  0.15; A(2,5) = -1.43;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08; A(3,4) = -2.13; A(3,5) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_recth_matrix_row_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Horizontal Matrix - Row Major - Economic Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93; A(0,4) =  0.15; A(0,5) = -0.02;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64; A(1,4) =  0.30; A(1,5) =  1.03;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66; A(2,4) =  0.15; A(2,5) = -1.43;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08; A(3,4) = -2.13; A(3,5) =  0.50;


    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_recth_matrix_column_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Horizontal Matrix - Column Major - Economic Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93; A(0,4) =  0.15; A(0,5) = -0.02;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64; A(1,4) =  0.30; A(1,5) =  1.03;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66; A(2,4) =  0.15; A(2,5) = -1.43;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08; A(3,4) = -2.13; A(3,5) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_rectv_matrix_row_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Vertical Matrix - Row Major - Full Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_rectv_matrix_column_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Vertical Matrix - Column Major - Full Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_rectv_matrix_row_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Vertical Matrix - Row Major - Economic Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_rectv_matrix_column_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Rectangular Vertical Matrix - Column Major - Economic Mode");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_square_matrix_row_major_oo )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Square Matrix - Row Major - QL Decomposition object");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;

    ublasx::ql_decomposition<value_type> ql(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << ql.L() );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(ql.Q(), ql.L()) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ql.Q(), ql.L()), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_square_matrix_column_major_oo )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Square Matrix - Column Major - QL Decomposition object");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;

    ublasx::ql_decomposition<value_type> ql(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << ql.L() );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(ql.Q(), ql.L()) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ql.Q(), ql.L()), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_square_matrix_row_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Square Matrix - Row Major");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_square_matrix_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex -Square Matrix - Column Major");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_recth_matrix_row_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Horizontal Matrix - Row Major - Full Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.98, 1.98); A(0,2) = value_type( 0.62,-0.46); A(0,3) = value_type(-0.37, 0.38); A(0,4) = value_type( 0.83, 0.51); A(0,5) = value_type( 1.08,-0.28);
    A(1,0) = value_type(-0.03, 0.96); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type( 1.01, 0.02); A(1,3) = value_type( 0.19,-0.54); A(1,4) = value_type( 0.20, 0.01); A(1,5) = value_type( 0.20,-0.12);
    A(2,0) = value_type(-0.91, 2.06); A(2,1) = value_type(-0.66, 0.42); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-0.98,-0.36); A(2,4) = value_type(-0.17,-0.46); A(2,5) = value_type(-0.07, 1.23);
    A(3,0) = value_type(-0.05, 0.41); A(3,1) = value_type(-0.81, 0.56); A(3,2) = value_type(-1.11, 0.60); A(3,3) = value_type( 0.22,-0.20); A(3,4) = value_type( 1.47, 1.59); A(3,5) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_recth_matrix_column_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Horizontal Matrix - Column Major - Full Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.98, 1.98); A(0,2) = value_type( 0.62,-0.46); A(0,3) = value_type(-0.37, 0.38); A(0,4) = value_type( 0.83, 0.51); A(0,5) = value_type( 1.08,-0.28);
    A(1,0) = value_type(-0.03, 0.96); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type( 1.01, 0.02); A(1,3) = value_type( 0.19,-0.54); A(1,4) = value_type( 0.20, 0.01); A(1,5) = value_type( 0.20,-0.12);
    A(2,0) = value_type(-0.91, 2.06); A(2,1) = value_type(-0.66, 0.42); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-0.98,-0.36); A(2,4) = value_type(-0.17,-0.46); A(2,5) = value_type(-0.07, 1.23);
    A(3,0) = value_type(-0.05, 0.41); A(3,1) = value_type(-0.81, 0.56); A(3,2) = value_type(-1.11, 0.60); A(3,3) = value_type( 0.22,-0.20); A(3,4) = value_type( 1.47, 1.59); A(3,5) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_recth_matrix_row_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Horizontal Matrix - Row Major - Economic Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.98, 1.98); A(0,2) = value_type( 0.62,-0.46); A(0,3) = value_type(-0.37, 0.38); A(0,4) = value_type( 0.83, 0.51); A(0,5) = value_type( 1.08,-0.28);
    A(1,0) = value_type(-0.03, 0.96); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type( 1.01, 0.02); A(1,3) = value_type( 0.19,-0.54); A(1,4) = value_type( 0.20, 0.01); A(1,5) = value_type( 0.20,-0.12);
    A(2,0) = value_type(-0.91, 2.06); A(2,1) = value_type(-0.66, 0.42); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-0.98,-0.36); A(2,4) = value_type(-0.17,-0.46); A(2,5) = value_type(-0.07, 1.23);
    A(3,0) = value_type(-0.05, 0.41); A(3,1) = value_type(-0.81, 0.56); A(3,2) = value_type(-1.11, 0.60); A(3,3) = value_type( 0.22,-0.20); A(3,4) = value_type( 1.47, 1.59); A(3,5) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_recth_matrix_column_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Horizontal Matrix - Column Major - Economic Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(4);
    const std::size_t n(6);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.98, 1.98); A(0,2) = value_type( 0.62,-0.46); A(0,3) = value_type(-0.37, 0.38); A(0,4) = value_type( 0.83, 0.51); A(0,5) = value_type( 1.08,-0.28);
    A(1,0) = value_type(-0.03, 0.96); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type( 1.01, 0.02); A(1,3) = value_type( 0.19,-0.54); A(1,4) = value_type( 0.20, 0.01); A(1,5) = value_type( 0.20,-0.12);
    A(2,0) = value_type(-0.91, 2.06); A(2,1) = value_type(-0.66, 0.42); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-0.98,-0.36); A(2,4) = value_type(-0.17,-0.46); A(2,5) = value_type(-0.07, 1.23);
    A(3,0) = value_type(-0.05, 0.41); A(3,1) = value_type(-0.81, 0.56); A(3,2) = value_type(-1.11, 0.60); A(3,3) = value_type( 0.22,-0.20); A(3,4) = value_type( 1.47, 1.59); A(3,5) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_rectv_matrix_row_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Vertical Matrix - Row Major - Full Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06); A(0,3) = value_type(-0.05, 0.41);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42); A(1,3) = value_type(-0.81, 0.56);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-1.11, 0.60);
    A(3,0) = value_type(-0.37, 0.38); A(3,1) = value_type( 0.19,-0.54); A(3,2) = value_type(-0.98,-0.36); A(3,3) = value_type( 0.22,-0.20);
    A(4,0) = value_type( 0.83, 0.51); A(4,1) = value_type( 0.20, 0.01); A(4,2) = value_type(-0.17,-0.46); A(4,3) = value_type( 1.47, 1.59);
    A(5,0) = value_type( 1.08,-0.28); A(5,1) = value_type( 0.20,-0.12); A(5,2) = value_type(-0.07, 1.23); A(5,3) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_rectv_matrix_column_major_full )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Vertical Matrix - Column Major - Full Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06); A(0,3) = value_type(-0.05, 0.41);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42); A(1,3) = value_type(-0.81, 0.56);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-1.11, 0.60);
    A(3,0) = value_type(-0.37, 0.38); A(3,1) = value_type( 0.19,-0.54); A(3,2) = value_type(-0.98,-0.36); A(3,3) = value_type( 0.22,-0.20);
    A(4,0) = value_type( 0.83, 0.51); A(4,1) = value_type( 0.20, 0.01); A(4,2) = value_type(-0.17,-0.46); A(4,3) = value_type( 1.47, 1.59);
    A(5,0) = value_type( 1.08,-0.28); A(5,1) = value_type( 0.20,-0.12); A(5,2) = value_type(-0.07, 1.23); A(5,3) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, true);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_rectv_matrix_row_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Vertical Matrix - Row Major - Economic Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06); A(0,3) = value_type(-0.05, 0.41);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42); A(1,3) = value_type(-0.81, 0.56);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-1.11, 0.60);
    A(3,0) = value_type(-0.37, 0.38); A(3,1) = value_type( 0.19,-0.54); A(3,2) = value_type(-0.98,-0.36); A(3,3) = value_type( 0.22,-0.20);
    A(4,0) = value_type( 0.83, 0.51); A(4,1) = value_type( 0.20, 0.01); A(4,2) = value_type(-0.17,-0.46); A(4,3) = value_type( 1.47, 1.59);
    A(5,0) = value_type( 1.08,-0.28); A(5,1) = value_type( 0.20,-0.12); A(5,2) = value_type(-0.07, 1.23); A(5,3) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_rectv_matrix_column_major_econ )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Rectangular Vertical Matrix - Column Major - Economic Mode");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t m(6);
    const std::size_t n(4);

    matrix_type A(m,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06); A(0,3) = value_type(-0.05, 0.41);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42); A(1,3) = value_type(-0.81, 0.56);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17); A(2,3) = value_type(-1.11, 0.60);
    A(3,0) = value_type(-0.37, 0.38); A(3,1) = value_type( 0.19,-0.54); A(3,2) = value_type(-0.98,-0.36); A(3,3) = value_type( 0.22,-0.20);
    A(4,0) = value_type( 0.83, 0.51); A(4,1) = value_type( 0.20, 0.01); A(4,2) = value_type(-0.17,-0.46); A(4,3) = value_type( 1.47, 1.59);
    A(5,0) = value_type( 1.08,-0.28); A(5,1) = value_type( 0.20,-0.12); A(5,2) = value_type(-0.07, 1.23); A(5,3) = value_type( 0.26, 0.26);

    matrix_type Q;
    matrix_type L;

    ublasx::ql_decompose(A, Q, L, false);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << L );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(Q, L) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == m );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(L) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(L) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(Q, L), A, m, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_square_matrix_row_major_oo )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Square Matrix - Row Major - QL Decomposition object");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17);

    ublasx::ql_decomposition<value_type> ql(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << ql.L() );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(ql.Q(), ql.L()) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ql.Q(), ql.L()), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_square_matrix_column_major_oo )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex - Square Matrix - Column Major - QL Decomposition object");

    typedef std::complex<double> value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t n(3);

    matrix_type A(n,n);

    A(0,0) = value_type( 0.96,-0.81); A(0,1) = value_type(-0.03, 0.96); A(0,2) = value_type(-0.91, 2.06);
    A(1,0) = value_type(-0.98, 1.98); A(1,1) = value_type(-1.20, 0.19); A(1,2) = value_type(-0.66, 0.42);
    A(2,0) = value_type( 0.62,-0.46); A(2,1) = value_type( 1.01, 0.02); A(2,2) = value_type( 0.63,-0.17);

    ublasx::ql_decomposition<value_type> ql(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );
    BOOST_UBLASX_DEBUG_TRACE( "L = " << ql.L() );
    BOOST_UBLASX_DEBUG_TRACE( "Q*L = " << ublas::prod(ql.Q(), ql.L()) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.Q()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ql.L()) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ql.Q(), ql.L()), A, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_prod_left_notrans_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Q*C Product - Column Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t A_nr(6);
    const std::size_t A_nc(4);

    matrix_type A(A_nr,A_nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    const std::size_t C_nr(6);
    const std::size_t C_nc(2);

    matrix_type C(C_nr, C_nc);

    C(0,0) = -2.67; C(0,1) =  0.41;
    C(1,0) = -0.55; C(1,1) = -3.10;
    C(2,0) =  3.34; C(2,1) = -4.01;
    C(3,0) = -0.77; C(3,1) =  2.76;
    C(4,0) =  0.48; C(4,1) = -6.17;
    C(5,0) =  4.10; C(5,1) =  0.21;

    matrix_type X;

    ublasx::ql_decomposition<value_type> ql(A);

    X = ql.lprod(C);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "C = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "X = " << X );
    BOOST_UBLASX_DEBUG_TRACE( "Q*C = " << ublas::prod(ql.Q(true), C) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == A_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == C_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, ublas::prod(ql.Q(true), C), A_nr, C_nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_prod_left_trans_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - Q'*C Product - Column Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t A_nr(6);
    const std::size_t A_nc(4);

    matrix_type A(A_nr,A_nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    const std::size_t C_nr(6);
    const std::size_t C_nc(2);

    matrix_type C(C_nr, C_nc);

    C(0,0) = -2.67; C(0,1) =  0.41;
    C(1,0) = -0.55; C(1,1) = -3.10;
    C(2,0) =  3.34; C(2,1) = -4.01;
    C(3,0) = -0.77; C(3,1) =  2.76;
    C(4,0) =  0.48; C(4,1) = -6.17;
    C(5,0) =  4.10; C(5,1) =  0.21;

    matrix_type X;

    ublasx::ql_decomposition<value_type> ql(A);

    X = ql.tlprod(C);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "C = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "X = " << X );
    BOOST_UBLASX_DEBUG_TRACE( "Q'*C = " << ublas::prod(ublas::trans(ql.Q(true)), C) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == A_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == C_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, ublas::prod(ublas::trans(ql.Q(true)), C), A_nr, C_nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_prod_right_notrans_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - C*Q Product - Column Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t A_nr(6);
    const std::size_t A_nc(4);

    matrix_type A(A_nr,A_nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    const std::size_t C_nr(2);
    const std::size_t C_nc(6);

    matrix_type C(C_nr, C_nc);

    C(0,0) = -2.67; C(0,1) = -0.55; C(0,2) =  3.34; C(0,3) = -0.77; C(0,4) =  0.48; C(0,5) =  4.10;
    C(1,0) =  0.41; C(1,1) = -3.10; C(1,2) = -4.01; C(1,3) =  2.76; C(1,4) = -6.17; C(1,5) =  0.21;

    matrix_type X;

    ublasx::ql_decomposition<value_type> ql(A);

    X = ql.rprod(C);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "C = " << C );
    BOOST_UBLASX_DEBUG_TRACE( "X = " << X );
    BOOST_UBLASX_DEBUG_TRACE( "C*Q = " << ublas::prod(C, ql.Q(true)) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == C_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == C_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, ublas::prod(C, ql.Q(true)), C_nr, C_nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_prod_right_trans_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Real - C*Q' Product - Column Major");

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    const std::size_t A_nr(6);
    const std::size_t A_nc(4);

    matrix_type A(A_nr,A_nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;
    A(4,0) =  0.15; A(4,1) =  0.30; A(4,2) =  0.15; A(4,3) = -2.13;
    A(5,0) = -0.02; A(5,1) =  1.03; A(5,2) = -1.43; A(5,3) =  0.50;

    const std::size_t C_nr(2);
    const std::size_t C_nc(6);

    matrix_type C(C_nr, C_nc);

    C(0,0) = -2.67; C(0,1) = -0.55; C(0,2) =  3.34; C(0,3) = -0.77; C(0,4) =  0.48; C(0,5) =  4.10;
    C(1,0) =  0.41; C(1,1) = -3.10; C(1,2) = -4.01; C(1,3) =  2.76; C(1,4) = -6.17; C(1,5) =  0.21;

    matrix_type X;

    ublasx::ql_decomposition<value_type> ql(A);

    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );

    X = ql.trprod(C);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Q = " << ql.Q() );
    BOOST_UBLASX_DEBUG_TRACE( "C = " << C );
    BOOST_UBLASX_DEBUG_TRACE( "X = " << X );
    BOOST_UBLASX_DEBUG_TRACE( "C*Q' = " << ublas::prod(C, ublas::trans(ql.Q(true))) );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == C_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == C_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, ublas::prod(C, ublas::trans(ql.Q(true))), C_nr, C_nc, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: QL factorization");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_real_square_matrix_row_major );
    BOOST_UBLASX_TEST_DO( test_real_square_matrix_column_major );
    BOOST_UBLASX_TEST_DO( test_real_rectv_matrix_row_major_full );
    BOOST_UBLASX_TEST_DO( test_real_rectv_matrix_column_major_full );
    BOOST_UBLASX_TEST_DO( test_real_rectv_matrix_row_major_econ );
    BOOST_UBLASX_TEST_DO( test_real_rectv_matrix_column_major_econ );
    BOOST_UBLASX_TEST_DO( test_real_recth_matrix_row_major_full );
    BOOST_UBLASX_TEST_DO( test_real_recth_matrix_column_major_full );
    BOOST_UBLASX_TEST_DO( test_real_recth_matrix_row_major_econ );
    BOOST_UBLASX_TEST_DO( test_real_recth_matrix_column_major_econ );
    BOOST_UBLASX_TEST_DO( test_real_square_matrix_row_major_oo );
    BOOST_UBLASX_TEST_DO( test_real_square_matrix_column_major_oo );
    BOOST_UBLASX_TEST_DO( test_complex_square_matrix_row_major );
    BOOST_UBLASX_TEST_DO( test_complex_square_matrix_column_major );
    BOOST_UBLASX_TEST_DO( test_complex_rectv_matrix_row_major_full );
    BOOST_UBLASX_TEST_DO( test_complex_rectv_matrix_column_major_full );
    BOOST_UBLASX_TEST_DO( test_complex_rectv_matrix_row_major_econ );
    BOOST_UBLASX_TEST_DO( test_complex_rectv_matrix_column_major_econ );
    BOOST_UBLASX_TEST_DO( test_complex_recth_matrix_row_major_full );
    BOOST_UBLASX_TEST_DO( test_complex_recth_matrix_column_major_full );
    BOOST_UBLASX_TEST_DO( test_complex_recth_matrix_row_major_econ );
    BOOST_UBLASX_TEST_DO( test_complex_recth_matrix_column_major_econ );
    BOOST_UBLASX_TEST_DO( test_complex_square_matrix_row_major_oo );
    BOOST_UBLASX_TEST_DO( test_complex_square_matrix_column_major_oo );
    BOOST_UBLASX_TEST_DO( test_real_matrix_prod_left_notrans_column_major );
    BOOST_UBLASX_TEST_DO( test_real_matrix_prod_left_trans_column_major );
    BOOST_UBLASX_TEST_DO( test_real_matrix_prod_right_notrans_column_major );
    BOOST_UBLASX_TEST_DO( test_real_matrix_prod_right_trans_column_major );

    BOOST_UBLASX_TEST_END();
}
