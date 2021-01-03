/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/hold.cpp
 *
 * \brief Test the \c hold operation.
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
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/hold.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <cstddef>
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
    typedef ublas::vector<bool> out_vector_type;

    const std::size_t n = 5;

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    zero_vector_type z(n);

    value_type val(0);
    out_vector_type expect;
    out_vector_type res;


    // hold(z)
    BOOST_UBLASX_DEBUG_TRACE( "NOTE: Expect to fail cause ublas::vector_assign assume the value type is a floating point" );
    expect = out_vector_type(n, false);
    res = ublasx::hold(z);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << z << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );

    BOOST_UBLASX_DEBUG_TRACE( "HERE.2" );
    // hold(v)
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = v(i) != 0;
    }
    res = ublasx::hold(v);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );

    BOOST_UBLASX_DEBUG_TRACE( "HERE.3" );
    // hold(v, > .5)
    val = 0.5;
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = v(i) > val;
    }
    res = ublasx::hold(v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector<bool> out_vector_type;

    const std::size_t n = 5;

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;

    value_type val(0);
    out_vector_type expect;
    out_vector_type res;


    // hold(-v)
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = (-v(i)) != 0;
    }
    res = ublasx::hold(-v);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << -v << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );

    // hold(-v, > -.5)
    val = -0.5;
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = (-v(i)) > val;
    }
    res = ublasx::hold(-v, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << -v << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;
    typedef ublas::vector<bool> out_vector_type;

    const std::size_t n = 5;

    vector_type v(n);

    v(0) = 0.0;
    v(1) = 0.108929;
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 1.023787;


    value_type val(0);
    out_vector_type expect;
    out_vector_type res;

    // hold(ref(v))
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = v(i) != 0;
    }
    res = ublasx::hold(vector_reference_type(v));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );

    // which(ref(v), > .5)
    val = 0.5;
    expect = out_vector_type(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect(i) = v(i) > val;
    }
    res = ublasx::hold(vector_reference_type(v), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << vector_reference_type(v) << ", > " << val << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect, n );
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::zero_matrix<value_type, ublas::row_major> zero_matrix_type;
    typedef ublas::matrix<bool, ublas::row_major> out_matrix_type;

    const std::size_t nr(5);
    const std::size_t nc(4);

    matrix_type A(nr,nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    zero_matrix_type Z(nr, nc);

    value_type val(0);
    out_matrix_type expect;
    out_matrix_type res;


    // hold(Z)
    expect = out_matrix_type(nr, nc, false);
    res = ublasx::hold(Z);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << Z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );

    // hold(A)
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) != 0;
        }
    }
    res = ublasx::hold(A);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );

    // hold(A, > .5)
    val = 0.5;
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) > val;
        }
    }
    res = ublasx::hold(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );
}


BOOST_UBLASX_TEST_DEF( test_column_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::zero_matrix<value_type, ublas::column_major> zero_matrix_type;
    typedef ublas::matrix<bool, ublas::column_major> out_matrix_type;

    const std::size_t nr(5);
    const std::size_t nc(4);

    matrix_type A(nr,nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    zero_matrix_type Z(nr, nc);

    value_type val(0);
    out_matrix_type expect;
    out_matrix_type res;


    // hold(Z)
    expect = out_matrix_type(nr, nc, false);
    res = ublasx::hold(Z);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << Z << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );

    // hold(A)
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) != 0;
        }
    }
    res = ublasx::hold(A);
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );

    // hold(A, > .5)
    val = 0.5;
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) > val;
        }
    }
    res = ublasx::hold(A, ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << ", > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix<bool, ublas::column_major> out_matrix_type;

    const std::size_t nr(5);
    const std::size_t nc(4);

    matrix_type A(nr,nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type val(0);
    out_matrix_type expect;
    out_matrix_type res;


    // all(A')
    expect = out_matrix_type(nc, nr);
    for (std::size_t r = 0; r < nc; ++r)
    {
        for (std::size_t c = 0; c < nr; ++c)
        {
            expect(r,c) = A(c,r) != 0;
        }
    }
    res = ublasx::hold(ublas::trans(A));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << "') = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nc, nr );

    // hold(A', > .5)
    val = 0.5;
    expect = out_matrix_type(nc, nr);
    for (std::size_t r = 0; r < nc; ++r)
    {
        for (std::size_t c = 0; c < nr; ++c)
        {
            expect(r,c) = A(c,r) > val;
        }
    }
    res = ublasx::hold(ublas::trans(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(" << A << "', > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nc, nr );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;
    typedef ublas::matrix<bool, ublas::column_major> out_matrix_type;

    const std::size_t nr(5);
    const std::size_t nc(4);

    matrix_type A(nr,nc);

    A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
    A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
    A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

    value_type val(0);
    out_matrix_type expect;
    out_matrix_type res;


    // hold(ref(A))
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) != 0;
        }
    }
    res = ublasx::hold(matrix_reference_type(A));
    BOOST_UBLASX_DEBUG_TRACE( "hold(reference(" << A << ")) = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );

    // hold(ref(A), > .5)
    val = 0.5;
    expect = out_matrix_type(nr, nc);
    for (std::size_t r = 0; r < nr; ++r)
    {
        for (std::size_t c = 0; c < nc; ++c)
        {
            expect(r,c) = A(r,c) > val;
        }
    }
    res = ublasx::hold(matrix_reference_type(A), ::std::bind2nd(::std::greater<value_type>(), val));
    BOOST_UBLASX_DEBUG_TRACE( "hold(reference(" << A << "), > " << val << ") = " << ::std::boolalpha << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( res, expect, nr, nc );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'hold' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_container );
    BOOST_UBLASX_TEST_DO( test_vector_expression );
    BOOST_UBLASX_TEST_DO( test_vector_reference );
    BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_column_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_matrix_expression );
    BOOST_UBLASX_TEST_DO( test_matrix_reference );

    BOOST_UBLASX_TEST_END();
}
