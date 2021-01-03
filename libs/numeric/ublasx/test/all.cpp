/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/all.cpp
 *
 * \brief Test suite for the \c all operation.
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
#include <boost/numeric/ublasx/operation/all.hpp>
#include <functional>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_vector_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Container" );

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

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(z)
    expect = false;
    res = ublasx::all(z);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(v)
    expect = false;
    res = ublasx::all(v);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << v << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(v, > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(v, > -.1)
    val = -0.1;
    expect = true;
    res = ublasx::all(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // all(-v)
    expect = false;
    res = ublasx::all(-v);
    BOOST_UBLASX_DEBUG_TRACE( "all(" << -v << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(-v, > -.5)
    val = -0.5;
    expect = false;
    res = ublasx::all(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << -v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(-v, > -1.5)
    val = -1.5;
    expect = true;
    res = ublasx::all(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << -v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;

    vector_type v(5);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;


    value_type val(0);
    bool expect(false);
    bool res(false);

    // all(ref(v))
    expect = false;
    res = ublasx::all(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << vector_reference_type(v) << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(ref(v), > .5)
    val = 0.5;
    expect = false;
    res = ublasx::all(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << vector_reference_type(v) << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // all(ref(v), > -.1)
    val = -0.1;
    expect = true;
    res = ublasx::all(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "all(" << vector_reference_type(v) << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


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


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'all' operation");

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
