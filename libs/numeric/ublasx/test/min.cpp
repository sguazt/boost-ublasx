/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/min.cpp
 *
 * \brief Test suite for the \c min operation.
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
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/min.hpp>
#include <boost/numeric/ublasx/tags.hpp>
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


    // min(v)
    expect = value_type(0);
    res = ublasx::min(v);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min<1>(v)
    vexpect = ublas::vector<value_type>(1, 0);
    vres = ublasx::min<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << v << ") = " << vres << " ==> " << vexpect );
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
    v(4) = value_type( 1.023787,-4.24959);;

    value_type expect;
    value_type res;
    ublas::vector<value_type> vexpect;
    ublas::vector<value_type> vres;


    // min(v)
    expect = value_type(0.00000,-0.54000);
    res = ublasx::min(v);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min<1>(v)
    vexpect = ublas::vector<value_type>(1, value_type(0.00000,-0.54000));
    vres = ublasx::min<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << v << ") = " << vres << " ==> " << vexpect );
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


    // min(z)
    expect = value_type(0);
    res = ublasx::min(z);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // min(v)
    expect = value_type(0);
    res = ublasx::min(v);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min<1>(v)
    vexpect = ublas::vector<value_type>(1, 0);
    vres = ublasx::min<1>(v);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << v << ") = " << vres << " ==> " << vexpect );
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


    // min(-v)
    expect = value_type(-1.023787);
    res = ublasx::min(-v);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << -v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min<1>(-v)
    vexpect = ublas::vector<value_type>(1, -1.023787);
    vres = ublasx::min<1>(-v);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << -v << ") = " << vres << " ==> " << vexpect );
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


    // min(ref(v))
    expect = value_type(0);
    res = ublasx::min(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "min(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min<1>(ref(v))
    vexpect = ublas::vector<value_type>(1, 0);
    vres = ublasx::min<1>(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << vector_reference_type(v) << ") = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = 0.0;
    min_rows(1) = 0.0;
    min_rows(2) = 0.0;
    min_rows(3) = 0.0;
    min_rows(4) = 1.0;

    vector_type min_cols(nc);
    min_cols(0) = 0.0;
    min_cols(1) = 0.0;
    min_cols(2) = 0.0;
    min_cols(3) = 0.0;

    value_type expect;
    value_type res;
    vector_type vexpect;
    vector_type vres;


    // min(A)
    expect = value_type(0);
    res = ublasx::min(A);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min_rows(A)
    vexpect = min_rows;
    vres = ublasx::min_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(A)
    vexpect = min_cols;
    vres = ublasx::min_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(A)
    vexpect = min_rows;
    vres = ublasx::min<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(A)
    vexpect = min_cols;
    vres = ublasx::min<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(A)
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = value_type(0.090000, 0.108929);
    min_rows(1) = value_type(0.108929, 0.450332);
    min_rows(2) = value_type(0.000000, 0.675382);
    min_rows(3) = value_type(0.000000,-0.500000);
    min_rows(4) = value_type(1.000000, 0.500000);

    vector_type min_cols(nc);
    min_cols(0) = value_type(0.108929,0.450332);
    min_cols(1) = value_type(0.274690,1.231751);
    min_cols(2) = value_type(0.090000,0.108929);
    min_cols(3) = value_type(0.000000,0.675382);

    value_type expect;
    value_type res;
    vector_type vexpect;
    vector_type vres;


    // min(A)
    expect = value_type(0.090000,0.108929);
    res = ublasx::min(A);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min_rows(A)
    vexpect = min_rows;
    vres = ublasx::min_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(A)
    vexpect = min_cols;
    vres = ublasx::min_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(A)
    vexpect = min_rows;
    vres = ublasx::min<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(A)
    vexpect = min_cols;
    vres = ublasx::min<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(A)
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = 0.0;
    min_rows(1) = 0.0;
    min_rows(2) = 0.0;
    min_rows(3) = 0.0;
    min_rows(4) = 1.0;

    vector_type min_cols(nc);
    min_cols(0) = 0.0;
    min_cols(1) = 0.0;
    min_cols(2) = 0.0;
    min_cols(3) = 0.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // min(Z)
    expect = value_type(0);
    res = ublasx::min(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << Z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // min_rows(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::min_rows(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::min_columns(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::min<1>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::min<2>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(Z)
    vexpect = ublas::vector<value_type>(nr, 0);
    vres = ublasx::min_by_tag<ublasx::tag::major>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::min_by_tag<ublasx::tag::minor>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(Z)
    vexpect = ublas::vector<value_type>(nc, 0);
    vres = ublasx::min_by_tag<ublasx::tag::leading>(Z);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << Z << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min(A)
    expect = value_type(0);
    res = ublasx::min(A);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min_rows(A)
    vexpect = min_rows;
    vres = ublasx::min_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(A)
    vexpect = min_cols;
    vres = ublasx::min_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(A)
    vexpect = min_rows;
    vres = ublasx::min<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(A)
    vexpect = min_cols;
    vres = ublasx::min<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(A)
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = 0.0;
    min_rows(1) = 0.0;
    min_rows(2) = 0.0;
    min_rows(3) = 0.0;
    min_rows(4) = 1.0;

    vector_type min_cols(nc);
    min_cols(0) = 0.0;
    min_cols(1) = 0.0;
    min_cols(2) = 0.0;
    min_cols(3) = 0.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // min(A)
    expect = value_type(0);
    res = ublasx::min(A);
    BOOST_UBLASX_DEBUG_TRACE( "min(" << A << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min_rows(A)
    vexpect = min_rows;
    vres = ublasx::min_rows(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(A)
    vexpect = min_cols;
    vres = ublasx::min_columns(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(A)
    vexpect = min_rows;
    vres = ublasx::min<1>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(A)
    vexpect = min_cols;
    vres = ublasx::min<2>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(A)
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::major>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::minor>(A)
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << A << ") = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::leading>(A)
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(A);
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << A << ") = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = 0.0;
    min_rows(1) = 0.0;
    min_rows(2) = 0.0;
    min_rows(3) = 0.0;
    min_rows(4) = 1.0;

    vector_type min_cols(nc);
    min_cols(0) = 0.0;
    min_cols(1) = 0.0;
    min_cols(2) = 0.0;
    min_cols(3) = 0.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // min(A')
    expect = value_type(0);
    res = ublasx::min(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min(" << A << "') = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );

    // min_rows(A')
    vexpect = min_cols;
    vres = ublasx::min_rows(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min_columns(A')
    vexpect = min_rows;
    vres = ublasx::min_columns(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<1>(A')
    vexpect = min_cols;
    vres = ublasx::min<1>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<2>(A')
    vexpect = min_rows;
    vres = ublasx::min<2>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::major>(A')
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::major>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(A')
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(" << A << "') = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(A')
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(" << A << "') = " << vres << " ==> " << vexpect );
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

    vector_type min_rows(nr);
    min_rows(0) = 0.0;
    min_rows(1) = 0.0;
    min_rows(2) = 0.0;
    min_rows(3) = 0.0;
    min_rows(4) = 1.0;

    vector_type min_cols(nc);
    min_cols(0) = 0.0;
    min_cols(1) = 0.0;
    min_cols(2) = 0.0;
    min_cols(3) = 0.0;

    value_type expect(0);
    value_type res(0);
    vector_type vexpect(0);
    vector_type vres(0);


    // min(ref(A))
    expect = value_type(0);
    res = ublasx::min(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min(reference(" << A << ")) = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // min_rows(ref(A))
    vexpect = min_rows;
    vres = ublasx::min_rows(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_rows(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min_columns(ref(A))
    vexpect = min_cols;
    vres = ublasx::min_columns(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_columns(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<1>(ref(A))
    vexpect = min_rows;
    vres = ublasx::min<1>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min<1>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<2>(ref(A))
    vexpect = min_cols;
    vres = ublasx::min<2>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min<2>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::major>(ref(A))
    vexpect = min_rows;
    vres = ublasx::min_by_tag<ublasx::tag::major>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::major>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nr, tol );

    // min<tag::minor>(ref(A))
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::minor>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::minor>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );

    // min<tag::leading>(ref(A))
    vexpect = min_cols;
    vres = ublasx::min_by_tag<ublasx::tag::leading>(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "min_by_tag<tag::leading>(reference(" << A << ")) = " << vres << " ==> " << vexpect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vres, vexpect, nc, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'min' operation");

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
