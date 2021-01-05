/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/which.cpp
 *
 * \brief Test the \c which operation.
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

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/which.hpp>
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
    typedef ublas::vector_traits<vector_type>::size_type size_type;
    typedef ublas::zero_vector<value_type> zero_vector_type;
    typedef ublas::vector<size_type> out_vector_type;

    size_type n = 5;
    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    zero_vector_type z(n);

    value_type val(0);
    out_vector_type expect(0);
    out_vector_type res(0);


    // which(z)
    expect = out_vector_type(0);
    res = ublasx::which(z);
    BOOST_UBLASX_DEBUG_TRACE( "which(" << z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 0 );

    // which(v)
    expect = out_vector_type(2);
    expect(0) = 1;
    expect(1) = 4;
    res = ublasx::which(v);
    BOOST_UBLASX_DEBUG_TRACE( "which(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 2 );

    // which(v, > .5)
    val = 0.5;
    expect = out_vector_type(1);
    expect(0) = 4;
    res = ublasx::which(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 1 );

    // which(v, > -.1)
    val = -0.1;
    expect = out_vector_type(n);
    expect(0) = 0;
    expect(1) = 1;
    expect(2) = 2;
    expect(3) = 3;
    expect(4) = 4;
    res = ublasx::which(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;
    typedef ublas::vector<size_type> out_vector_type;

    size_type n = 5;
    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type val(0);
    out_vector_type expect(0);
    out_vector_type res(0);


    // which(-v)
    expect = out_vector_type(2);
    expect(0) = 1;
    expect(1) = 4;
    res = ublasx::which(-v);
    BOOST_UBLASX_DEBUG_TRACE( "which(" << -v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 2 );

    // which(-v, > -.5)
    val = -0.5;
    expect = out_vector_type(4);
    expect(0) = 0;
    expect(1) = 1;
    expect(2) = 2;
    expect(3) = 3;
    res = ublasx::which(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << -v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 4 );

    // which(-v, > -1.5)
    val = -1.5;
    expect = out_vector_type(5);
    expect(0) = 0;
    expect(1) = 1;
    expect(2) = 2;
    expect(3) = 3;
    expect(4) = 4;
    res = ublasx::which(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << -v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;
    typedef ublas::vector<size_type> out_vector_type;

    size_type n = 5;
    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;


    value_type val(0);
    out_vector_type expect(0);
    out_vector_type res(0);

    // which(ref(v))
    expect = out_vector_type(2);
    expect(0) = 1;
    expect(1) = 4;
    res = ublasx::which(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 2 );

    // which(ref(v), > .5)
    val = 0.5;
    expect = out_vector_type(1);
    expect(0) = 4;
    res = ublasx::which(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << vector_reference_type(v) << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, 1 );

    // which(ref(v), > -.1)
    val = -0.1;
    expect = out_vector_type(n);
    expect(0) = 0;
    expect(1) = 1;
    expect(2) = 2;
    expect(3) = 3;
    expect(4) = 4;
    res = ublasx::which(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "which(" << vector_reference_type(v) << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


/*
BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::zero_matrix<value_type> zero_matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    zero_matrix_type Z(5, 4);

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(Z)
    expect = false;
    res = ublasx::all(Z);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << Z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A)
    expect = false;
    res = ublasx::all(A);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A, > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A, > -.1)
    val = -0.1;
    expect = true;
    res = ublasx::all(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(A)
    expect = false;
    res = ublasx::all(A);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A, > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A, > -.5)
    val = -0.5;
    expect = true;
    res = ublasx::all(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(A')
    expect = false;
    res = ublasx::all(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << "') = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A', > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(ublas::trans(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << "', > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(A', > -.5)
    val = -0.5;
    expect = true;
    res = ublasx::all(ublas::trans(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << A << "', > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;

    matrix_type A(5,4);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(ref(A))
    expect = false;
    res = ublasx::all(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "all(reference(" << A << ")) = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(ref(A), > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(matrix_reference_type(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(reference(" << A << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(ref(A), > -.5)
    val = -0.5;
    expect = true;
    res = ublasx::all(matrix_reference_type(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(reference(" << A << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}
*/


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'which' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_container );
    BOOST_UBLASX_TEST_DO( test_vector_expression );
    BOOST_UBLASX_TEST_DO( test_vector_reference );
//  BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
//  BOOST_UBLASX_TEST_DO( test_col_major_matrix_container );
//  BOOST_UBLASX_TEST_DO( test_matrix_expression );
//  BOOST_UBLASX_TEST_DO( test_matrix_reference );

    BOOST_UBLASX_TEST_END();
}
