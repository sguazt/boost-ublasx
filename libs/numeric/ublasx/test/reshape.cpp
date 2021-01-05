/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/reshape.cpp
 *
 * \brief Test suite for the \c reshape operation.
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
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/reshape.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1e-5;


BOOST_UBLASX_TEST_DEF( reshape_by_dim1_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<1>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<1>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_dim1_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<1>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<1>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_dim2_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<2>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<2>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_dim2_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<2>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<2>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_by_dim1_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace<1>(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<1>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_by_dim1_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace<1>(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<1>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_by_dim2_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace<2>(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<2>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_by_dim2_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace<2>(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<2>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


/*
 * BEGIN FIXME: Does Not Work!
 *
BOOST_UBLASX_TEST_DEF( reshape_by_tag_major_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<ublas::tag::major>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<major>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_tag_major_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<ublas::tag::major>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<major>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_tag_minor_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;


    matrix_type X;
    X = ublasx::reshape<ublas::tag::minor>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<minor>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_tag_minor_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<ublas::tag::minor>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<minor>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_tag_leading_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;


    matrix_type X;
    X = ublasx::reshape<ublas::tag::leading>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<leading>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_by_tag_leading_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) =  7; E(0,2) = 2; E(0,3) =  8; E(0,4) = 3; E(0,5) =  9;
    E(1,0) =  4; E(1,1) = 10; E(1,2) = 5; E(1,3) = 11; E(1,4) = 6; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape<ublas::tag::leading>(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape<leading>(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}
 * BEGIN FIXME: Does Not Work!
 *
 */


BOOST_UBLASX_TEST_DEF( reshape_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X;
    X = ublasx::reshape(A, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_col_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( inplace_reshape_row_major )
{
    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type nr(3);
    const size_type nc(4);
    const size_type new_nr(2);
    const size_type new_nc(6);

    matrix_type A(nr,nc);

    A(0,0) =  1; A(0,1) =  4; A(0,2) =  7; A(0,3) = 10;
    A(1,0) =  2; A(1,1) =  5; A(1,2) =  8; A(1,3) = 11;
    A(2,0) =  3; A(2,1) =  6; A(2,2) =  9; A(2,3) = 12;

    matrix_type E(new_nr,new_nc);

    E(0,0) =  1; E(0,1) = 3; E(0,2) = 5; E(0,3) = 7; E(0,4) =  9; E(0,5) = 11;
    E(1,0) =  2; E(1,1) = 4; E(1,2) = 6; E(1,3) = 8; E(1,4) = 10; E(1,5) = 12;

    matrix_type X(A);
    ublasx::reshape_inplace(X, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("A=" << A);
    BOOST_UBLASX_DEBUG_TRACE("reshape(A," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(A," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


BOOST_UBLASX_TEST_DEF( reshape_vec )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Reshape Vector");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

    const size_type n(6);

    vector_type v(n);

    v(0) =  1;
    v(1) =  2;
    v(2) =  3;
    v(3) =  4;
    v(4) =  5;
    v(5) =  6;

    size_type new_nr;
    size_type new_nc;
    matrix_type E;
    matrix_type X;


    // vector(n) => matrix(n,1)
    new_nr = 6;
    new_nc = 1;
    E = matrix_type(new_nr,new_nc);
    E(0,0) =  1;
    E(1,0) =  2;
    E(2,0) =  3;
    E(3,0) =  4;
    E(4,0) =  5;
    E(5,0) =  6;
    X = ublasx::reshape(v, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("reshape(v," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(v," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );


    // vector(n) => matrix(1,n)
    new_nr = 1;
    new_nc = 6;
    E = matrix_type(new_nr,new_nc);
    E(0,0) =  1; E(0,1) =  2; E(0,2) =  3; E(0,3) =  4; E(0,4) =  5; E(0,5) = 6;
    X = ublasx::reshape(v, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("reshape(v," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(v," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );


    // vector(n) => matrix(n1,n2)
    new_nr = 3;
    new_nc = 2;
    E = matrix_type(new_nr,new_nc);
    E(0,0) = 1; E(0,1) = 4;
    E(1,0) = 2; E(1,1) = 5;
    E(2,0) = 3; E(2,1) = 6;
    X = ublasx::reshape(v, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("reshape(v," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(v," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );


    // vector(n) => matrix(n1,n2)
    new_nr = 2;
    new_nc = 3;
    E = matrix_type(new_nr,new_nc);
    E(0,0) = 1; E(0,1) = 3; E(0,2) = 5;
    E(1,0) = 2; E(1,1) = 4; E(1,2) = 6;
    X = ublasx::reshape(v, new_nr, new_nc);
    BOOST_UBLASX_DEBUG_TRACE("v=" << v);
    BOOST_UBLASX_DEBUG_TRACE("reshape(v," << new_nr << "," << new_nc << ")=" << X);
    BOOST_UBLASX_DEBUG_TRACE("Expected reshape(v," << new_nr << "," << new_nc << ")=" << E);
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(X) == new_nr );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(X) == new_nc );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, new_nr, new_nc, tol );
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( reshape_by_dim1_col_major );
    BOOST_UBLASX_TEST_DO( reshape_by_dim1_row_major );
    BOOST_UBLASX_TEST_DO( reshape_by_dim2_col_major );
    BOOST_UBLASX_TEST_DO( reshape_by_dim2_row_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_by_dim1_col_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_by_dim1_row_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_by_dim2_col_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_by_dim2_row_major );


//FIXME: does not work
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_major_col_major );
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_major_row_major );
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_minor_col_major );
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_minor_row_major );
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_leading_col_major );
//  BOOST_UBLASX_TEST_DO( reshape_by_tag_leading_row_major );

    BOOST_UBLASX_TEST_DO( reshape_col_major );
    BOOST_UBLASX_TEST_DO( reshape_row_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_col_major );
    BOOST_UBLASX_TEST_DO( inplace_reshape_row_major );

    BOOST_UBLASX_TEST_DO( reshape_vec );

    BOOST_UBLASX_TEST_END();
}
