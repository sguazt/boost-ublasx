/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/any.cpp
 * \brief Test the \c any operation.
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
#include <boost/numeric/ublasx/operation/any.hpp>
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

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;

    zero_vector_type z(5);

    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(z)
    expect = false;
    res = ublasx::any(z);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(v)
    expect = true;
    res = ublasx::any(v);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << v << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(v, > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(v, > 1.5)
    val = 1.5;
    expect = false;
    res = ublasx::any(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(-v)
    expect = true;
    res = ublasx::any(-v);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << -v << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(-v, > -.5)
    val = -0.5;
    expect = true;
    res = ublasx::any(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << -v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(-v, > -.0)
    val = 0;
    expect = false;
    res = ublasx::any(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << -v << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    value_type val(0);
    bool expect(false);
    bool res(false);

    // any(ref(v))
    expect = true;
    res = ublasx::any(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << vector_reference_type(v) << ")) = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(ref(v), > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << vector_reference_type(v) << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(ref(v), > 1.5)
    val = 1.5;
    expect = false;
    res = ublasx::any(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << vector_reference_type(v) << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::zero_matrix<value_type> zero_matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;

    zero_matrix_type Z(5, 4);


    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(Z)
    expect = false;
    res = ublasx::any(Z);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << Z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A)
    expect = true;
    res = ublasx::any(A);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A, > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A, > 2.5)
    val = 2.5;
    expect = false;
    res = ublasx::any(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(A)
    expect = true;
    res = ublasx::any(A);
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A, > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A, > 2.5)
    val = 2.5;
    expect = false;
    res = ublasx::any(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(A')
    expect = true;
    res = ublasx::any(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << "') = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A', > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(ublas::trans(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << "', > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(A', > 2.5)
    val = 2.5;
    expect = false;
    res = ublasx::any(ublas::trans(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(" << A << "', > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;

    value_type val(0);
    bool expect(false);
    bool res(false);


    // any(ref(A))
    expect = true;
    res = ublasx::any(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << A << ")) = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(ref(A), > .5)
    val = 0.5;
    expect = true;
    res = ublasx::any(matrix_reference_type(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << A << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );

    // any(ref(A), > 2.5)
    val = 2.5;
    expect = false;
    res = ublasx::any(matrix_reference_type(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "any(reference(" << A << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK( res == expect );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'any' operation");

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
