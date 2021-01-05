/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/max.cpp
 *
 * \brief Test suite for the \c max operation.
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

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/max.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double tol = 1.0e-5;

namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Vector" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // max(v)
    expect = value_type(1.023787);
    res = ublasx::max(v);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max<1>(v)
    vexpect = ublas::vector<value_type>(1, 1.023787);
    vres = ublasx::max<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( complex_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex Vector" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v(5);

    v(0) = value_type( 0.000000,-0.54000);
    v(1) = value_type(-0.108929, 2.43000);
    v(2) = value_type( 0.000000, 1.00030);
    v(3) = value_type(-0.050000, 1.00030);
    v(4) = value_type( 1.023787,-4.24959);

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // max(v)
    expect = value_type(1.023787,-4.24959);
    res = ublasx::max(v);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max<1>(v)
    vexpect = ublas::vector<value_type>(1, value_type( 1.023787,-4.24959));
    vres = ublasx::max<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( vector_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Vector Container" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::zero_vector<value_type> zero_vector_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    zero_vector_type z(5);

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // max(z)
    expect = value_type(0);
    res = ublasx::max(z);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // max(v)
    expect = value_type(1.023787);
    res = ublasx::max(v);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max<1>(v)
    vexpect = ublas::vector<value_type>(1, 1.023787);
    vres = ublasx::max<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // max(-v)
    expect = value_type(0);
    res = ublasx::max(-v);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << -v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max<1>(-v)
    vexpect = ublas::vector<value_type>(1, 0);
    vres = ublasx::max<1>(-v);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << -v << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type expect(0);
    value_type res(0);
    ublas::vector<value_type> vexpect(0);
    ublas::vector<value_type> vres(0);


    // max(ref(v))
    expect = value_type(1.023787);
    res = ublasx::max(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "max(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max<1>(ref(v))
    vexpect = ublas::vector<value_type>(1, 1.023787);
    vres = ublasx::max<1>(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << vector_reference_type(v) << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, 1, tol );
}


BOOST_UBLASX_TEST_DEF( real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    vector_type max_rows(nr);
    max_rows(0) = 0.798938;
    max_rows(1) = 0.891726;
    max_rows(2) = 0;
    max_rows(3) = 0.675382;
    max_rows(4) = 1.231751;

    vector_type max_cols(nc);
    max_cols(0) = 1.023787;
    max_cols(1) = 1.0;
    max_cols(2) = 1.231751;
    max_cols(3) = 1.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(A)
    expect = value_type(1.231751);
    res = ublasx::max(A);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max_rows(A)
    vexpect = max_rows;
    vres = ublasx::max_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(A)
    vexpect = max_cols;
    vres = ublasx::max_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(A)
    vexpect = max_rows;
    vres = ublasx::max<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(A)
    vexpect = max_cols;
    vres = ublasx::max<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(A)
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex Matrix" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = value_type( 0.000000,-1.000000); A(0,1) = value_type(0.274690, 1.231751); A(0,2) = value_type(0.090000, 0.108929); A(0,3) = value_type(0.798938,1.000000);
    A(1,0) = value_type( 0.108929, 0.450332); A(1,1) = value_type(0.000000, 1.400000); A(1,2) = value_type(0.891726, 1.023787); A(1,3) = value_type(0.000000,1.230000);
    A(2,0) = value_type(-0.500000, 0.500000); A(2,1) = value_type(0.000000, 2.100000); A(2,2) = value_type(0.090000,-1.230000); A(2,3) = value_type(0.000000,0.675382);
    A(3,0) = value_type( 0.000000,-0.500000); A(3,1) = value_type(0.675382,-1.230000); A(3,2) = value_type(0.090000, 1.231751); A(3,3) = value_type(0.450332,0.891726);
    A(4,0) = value_type( 1.023787, 0.798938); A(4,1) = value_type(1.000000, 0.891726); A(4,2) = value_type(1.231751, 0.000000); A(4,3) = value_type(1.000000,0.500000);

    vector_type max_rows(nr);
    max_rows(0) = value_type(0.798938, 1.000000);
    max_rows(1) = value_type(0.000000, 1.400000);
    max_rows(2) = value_type(0.000000, 2.100000);
    max_rows(3) = value_type(0.675382,-1.230000);
    max_rows(4) = value_type(1.000000, 0.891726);

    vector_type max_cols(nc);
    max_cols(0) = value_type(1.023787,0.798938);
    max_cols(1) = value_type(0.000000,2.100000);
    max_cols(2) = value_type(0.891726,1.023787);
    max_cols(3) = value_type(0.798938,1.000000);

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(A)
    expect = value_type(0.000000,2.100000);
    res = ublasx::max(A);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max_rows(A)
    vexpect = max_rows;
    vres = ublasx::max_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(A)
    vexpect = max_cols;
    vres = ublasx::max_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(A)
    vexpect = max_rows;
    vres = ublasx::max<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(A)
    vexpect = max_cols;
    vres = ublasx::max<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(A)
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::zero_matrix<value_type> zero_matrix_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    zero_matrix_type Z(nr, nc);

    vector_type max_rows(nr);
    max_rows(0) = 0.798938;
    max_rows(1) = 0.891726;
    max_rows(2) = 0;
    max_rows(3) = 0.675382;
    max_rows(4) = 1.231751;

    vector_type max_cols(nc);
    max_cols(0) = 1.023787;
    max_cols(1) = 1.0;
    max_cols(2) = 1.231751;
    max_cols(3) = 1.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(Z)
    expect = value_type(0);
    res = ublasx::max(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << Z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // max_rows(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::max_rows(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::max_columns(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::max<1>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::max<2>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::max_by_tag<ublasx::tag::major>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::max_by_tag<ublasx::tag::minor>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::max_by_tag<ublasx::tag::leading>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max(A)
    expect = value_type(1.231751);
    res = ublasx::max(A);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max_rows(A)
    vexpect = max_rows;
    vres = ublasx::max_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(A)
    vexpect = max_cols;
    vres = ublasx::max_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(A)
    vexpect = max_rows;
    vres = ublasx::max<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(A)
    vexpect = max_cols;
    vres = ublasx::max<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(A)
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    vector_type max_rows(nr);
    max_rows(0) = 0.798938;
    max_rows(1) = 0.891726;
    max_rows(2) = 0;
    max_rows(3) = 0.675382;
    max_rows(4) = 1.231751;

    vector_type max_cols(nc);
    max_cols(0) = 1.023787;
    max_cols(1) = 1.0;
    max_cols(2) = 1.231751;
    max_cols(3) = 1.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(A)
    expect = value_type(1.231751);
    res = ublasx::max(A);
    BOOST_UBLASX_DEBUG_TRACE( "max(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max_rows(A)
    vexpect = max_rows;
    vres = ublasx::max_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(A)
    vexpect = max_cols;
    vres = ublasx::max_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(A)
    vexpect = max_rows;
    vres = ublasx::max<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(A)
    vexpect = max_cols;
    vres = ublasx::max<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(A)
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::minor>(A)
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::leading>(A)
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );
}


BOOST_UBLASX_TEST_DEF( matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    vector_type max_rows(nr);
    max_rows(0) = 0.798938;
    max_rows(1) = 0.891726;
    max_rows(2) = 0;
    max_rows(3) = 0.675382;
    max_rows(4) = 1.231751;

    vector_type max_cols(nc);
    max_cols(0) = 1.023787;
    max_cols(1) = 1.0;
    max_cols(2) = 1.231751;
    max_cols(3) = 1.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(A')
    expect = value_type(1.231751);
    res = ublasx::max(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max(" << A << "') = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // max_rows(A')
    vexpect = max_cols;
    vres = ublasx::max_rows(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max_columns(A')
    vexpect = max_rows;
    vres = ublasx::max_columns(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<1>(A')
    vexpect = max_cols;
    vres = ublasx::max<1>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<2>(A')
    vexpect = max_rows;
    vres = ublasx::max<2>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::major>(A')
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::major>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(A')
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(A')
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


BOOST_UBLASX_TEST_DEF( matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;
    typedef ublas::vector<value_type> vector_type;

    std::size_t nr = 5;
    std::size_t nc = 4;

    matrix_type A(nr, nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    vector_type max_rows(nr);
    max_rows(0) = 0.798938;
    max_rows(1) = 0.891726;
    max_rows(2) = 0;
    max_rows(3) = 0.675382;
    max_rows(4) = 1.231751;

    vector_type max_cols(nc);
    max_cols(0) = 1.023787;
    max_cols(1) = 1.0;
    max_cols(2) = 1.231751;
    max_cols(3) = 1.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // max(ref(A))
    expect = value_type(1.231751);
    res = ublasx::max(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max(reference(" << A << ")) = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // max_rows(ref(A))
    vexpect = max_rows;
    vres = ublasx::max_rows(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_rows(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max_columns(ref(A))
    vexpect = max_cols;
    vres = ublasx::max_columns(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_columns(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<1>(ref(A))
    vexpect = max_rows;
    vres = ublasx::max<1>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max<1>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<2>(ref(A))
    vexpect = max_cols;
    vres = ublasx::max<2>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max<2>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::major>(ref(A))
    vexpect = max_rows;
    vres = ublasx::max_by_tag<ublasx::tag::major>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::major>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // max<tag::minor>(ref(A))
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::minor>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::minor>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // max<tag::leading>(ref(A))
    vexpect = max_cols;
    vres = ublasx::max_by_tag<ublasx::tag::leading>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "max_by_tag<tag::leading>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'max' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( real_vector );
    BOOST_UBLASX_TEST_DO( complex_vector );
    BOOST_UBLASX_TEST_DO( vector_container );
    BOOST_UBLASX_TEST_DO( vector_expression );
    BOOST_UBLASX_TEST_DO( vector_reference );
    BOOST_UBLASX_TEST_DO( real_matrix );
    BOOST_UBLASX_TEST_DO( complex_matrix );
    BOOST_UBLASX_TEST_DO( row_major_matrix_container );
    BOOST_UBLASX_TEST_DO( col_major_matrix_container );
    BOOST_UBLASX_TEST_DO( matrix_expression );
    BOOST_UBLASX_TEST_DO( matrix_reference );

    BOOST_UBLASX_TEST_END();
}
