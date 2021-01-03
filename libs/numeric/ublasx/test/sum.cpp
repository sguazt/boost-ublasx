/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/sum.cpp
 *
 * \brief Test suite for the \c sum operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/sum.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <functional>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double tol = 1.0e-5;


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_vector_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Container" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::zero_vector<value_type> zero_vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    size_type n(5);

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type sum_all = v(0)+v(1)+v(2)+v(4);

    zero_vector_type z(n);

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // sum(z)
    expect = value_type(0);
    res = ublasx::sum(z);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // sum(v)
    expect = sum_all;
    res = ublasx::sum(v);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum<1>(v)
    vexpect = ublas::vector<value_type>(1, sum_all);
    vres = ublasx::sum<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    size_type n(5);

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type sum_all = v(0)+v(1)+v(2)+v(4);

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // sum(-v)
    expect = -sum_all;
    res = ublasx::sum(-v);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << -v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum<1>(-v)
    vexpect = ublas::vector<value_type>(1, -sum_all);
    vres = ublasx::sum<1>(-v);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << -v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    size_type n(5);

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type sum_all = v(0)+v(1)+v(2)+v(4);

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // sum(ref(v))
    expect = sum_all;
    res = ublasx::sum(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum<1>(ref(v))
    vexpect = ublas::vector<value_type>(1, sum_all);
    vres = ublasx::sum<1>(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << vector_reference_type(v) << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::zero_matrix<value_type> zero_matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 5;
    size_type nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    zero_matrix_type Z(nr, nc);

    value_type sum_all =   A(0,0)+A(0,1)+A(0,2)+A(0,3)
                         + A(1,0)+A(1,1)+A(1,2)+A(1,3)
                         + A(2,0)+A(2,1)+A(2,2)+A(2,3)
                         + A(3,0)+A(3,1)+A(3,2)+A(3,3)
                         + A(4,0)+A(4,1)+A(4,2)+A(4,3);

    vector_type sum_rows(nc);
    sum_rows(0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0);
    sum_rows(1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1);
    sum_rows(2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2);
    sum_rows(3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

    vector_type sum_cols(nr);
    sum_cols(0) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
    sum_cols(1) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
    sum_cols(2) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
    sum_cols(3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
    sum_cols(4) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // sum_all(Z)
    expect = value_type(0);
    res = ublasx::sum_all(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_all(" << Z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // sum(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::sum(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // sum_rows(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::sum_rows(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_rows(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // sum_columns(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::sum_columns(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_columns(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // sum<1>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::sum<1>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // sum<2>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::sum<2>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // sum<tag::major>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::sum_by_tag<ublas::tag::major>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::major>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // sum<tag::minor>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::sum_by_tag<ublas::tag::minor>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::minor>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // sum<tag::leading>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::sum_by_tag<ublas::tag::leading>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::leading>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // sum_all(A)
    expect = value_type(sum_all);
    res = ublasx::sum_all(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_all(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum(A)
    vexpect = sum_rows;
    vres = ublasx::sum(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_rows(A)
    vexpect = sum_rows;
    vres = ublasx::sum_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_columns(A)
    vexpect = sum_cols;
    vres = ublasx::sum_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<1>(A)
    vexpect = sum_rows;
    vres = ublasx::sum<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<2>(A)
    vexpect = sum_cols;
    vres = ublasx::sum<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::major>(A)
    vexpect = sum_rows;
    vres = ublasx::sum_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::minor>(A)
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::leading>(A)
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 5;
    size_type nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type sum_all =   A(0,0)+A(0,1)+A(0,2)+A(0,3)
                         + A(1,0)+A(1,1)+A(1,2)+A(1,3)
                         + A(2,0)+A(2,1)+A(2,2)+A(2,3)
                         + A(3,0)+A(3,1)+A(3,2)+A(3,3)
                         + A(4,0)+A(4,1)+A(4,2)+A(4,3);

    vector_type sum_rows(nc);
    sum_rows(0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0);
    sum_rows(1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1);
    sum_rows(2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2);
    sum_rows(3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

    vector_type sum_cols(nr);
    sum_cols(0) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
    sum_cols(1) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
    sum_cols(2) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
    sum_cols(3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
    sum_cols(4) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // sum_all(A)
    expect = value_type(sum_all);
    res = ublasx::sum_all(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_all(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum(A)
    vexpect = sum_rows;
    vres = ublasx::sum(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_rows(A)
    vexpect = sum_rows;
    vres = ublasx::sum_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_columns(A)
    vexpect = sum_cols;
    vres = ublasx::sum_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<1>(A)
    vexpect = sum_rows;
    vres = ublasx::sum<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<2>(A)
    vexpect = sum_cols;
    vres = ublasx::sum<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::major>(A)
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::minor>(A)
    vexpect = sum_rows;
    vres = ublasx::sum_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::leading>(A)
    vexpect = sum_rows;
    vres = ublasx::sum_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 5;
    size_type nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type sum_all =   A(0,0)+A(0,1)+A(0,2)+A(0,3)
                         + A(1,0)+A(1,1)+A(1,2)+A(1,3)
                         + A(2,0)+A(2,1)+A(2,2)+A(2,3)
                         + A(3,0)+A(3,1)+A(3,2)+A(3,3)
                         + A(4,0)+A(4,1)+A(4,2)+A(4,3);

    vector_type sum_rows(nc);
    sum_rows(0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0);
    sum_rows(1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1);
    sum_rows(2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2);
    sum_rows(3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

    vector_type sum_cols(nr);
    sum_cols(0) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
    sum_cols(1) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
    sum_cols(2) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
    sum_cols(3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
    sum_cols(4) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // sum_all(A')
    expect = value_type(sum_all);
    res = ublasx::sum_all(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_all(" << A << "') = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // sum(A')
    vexpect = sum_cols;
    vres = ublasx::sum(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_rows(A')
    vexpect = sum_cols;
    vres = ublasx::sum_rows(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_rows(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_columns(A')
    vexpect = sum_rows;
    vres = ublasx::sum_columns(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_columns(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<1>(A')
    vexpect = sum_cols;
    vres = ublasx::sum<1>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<2>(A')
    vexpect = sum_rows;
    vres = ublasx::sum<2>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum<2>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::major>(A')
    vexpect = sum_rows;
    vres = ublasx::sum_by_tag<ublasx::tag::major>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::major>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::minor>(A')
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::minor>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::minor>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::leading>(A')
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::leading>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::leading>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 5;
    size_type nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type sum_all =  A(0,0)+A(0,1)+A(0,2)+A(0,3)
                         + A(1,0)+A(1,1)+A(1,2)+A(1,3)
                         + A(2,0)+A(2,1)+A(2,2)+A(2,3)
                         + A(3,0)+A(3,1)+A(3,2)+A(3,3)
                         + A(4,0)+A(4,1)+A(4,2)+A(4,3);

    vector_type sum_rows(nc);
    sum_rows(0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0);
    sum_rows(1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1);
    sum_rows(2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2);
    sum_rows(3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

    vector_type sum_cols(nr);
    sum_cols(0) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
    sum_cols(1) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
    sum_cols(2) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
    sum_cols(3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
    sum_cols(4) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // sum_all(ref(A))
    expect = value_type(sum_all);
    res = ublasx::sum_all(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_all(reference(" << A << ")) = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // sum(ref(A))
    vexpect = sum_rows;
    vres = ublasx::sum(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_rows(ref(A))
    vexpect = sum_rows;
    vres = ublasx::sum_rows(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_rows(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum_columns(ref(A))
    vexpect = sum_cols;
    vres = ublasx::sum_columns(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_columns(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<1>(ref(A))
    vexpect = sum_rows;
    vres = ublasx::sum<1>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum<1>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<2>(ref(A))
    vexpect = sum_cols;
    vres = ublasx::sum<2>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum<2>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::major>(ref(A))
    vexpect = sum_rows;
    vres = ublasx::sum_by_tag<ublasx::tag::major>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::major>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::minor>(ref(A))
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::minor>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::minor>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );

    // sum<tag::leading>(ref(A))
    vexpect = sum_cols;
    vres = ublasx::sum_by_tag<ublasx::tag::leading>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "sum_by_tag<tag::leading>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, vexpect.size(), tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'sum' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_container );
    BOOST_UBLASX_TEST_DO( test_vector_expression );
    BOOST_UBLASX_TEST_DO( test_vector_reference );
    BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_col_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_matrix_expression );
    BOOST_UBLASX_TEST_DO( test_matrix_reference );

    BOOST_UBLASX_TEST_END();
}
