/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/dot.cpp
 *
 * \brief Test suite for the \c dot operation.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/dot.hpp>
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

    size_type n(3);

    vector_type v1(n);
    v1(0) = 1;
    v1(1) = 2;
    v1(2) = 3;

    vector_type v2(n);
    v2(0) = 4;
    v2(1) = 5;
    v2(2) = 6;

    value_type expect(0);
    value_type res(0);

    // dot(v1,v2)
    expect = ublas::inner_prod(v1, v2);
    res = ublasx::dot(v1, v2);
    BOOST_UBLASX_DEBUG_TRACE( "dot(" << v1 << "," << v2 << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    size_type n(3);

    vector_type v1(n);
    v1(0) = 1;
    v1(1) = 2;
    v1(2) = 3;

    vector_type v2(n);
    v2(0) = 4;
    v2(1) = 5;
    v2(2) = 6;

    value_type expect(0);
    value_type res(0);

    // dot(-v1,-v2)
    expect = ublas::inner_prod(v1, v2);
    res = ublasx::dot(-v1, -v2);
    BOOST_UBLASX_DEBUG_TRACE( "dot(" << v1 << "," << v2 << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_reference<vector_type> vector_reference_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    size_type n(3);

    vector_type v1(n);
    v1(0) = 1;
    v1(1) = 2;
    v1(2) = 3;

    vector_type v2(n);
    v2(0) = 4;
    v2(1) = 5;
    v2(2) = 6;

    value_type expect(0);
    value_type res(0);

    // dot(ref(v1),ref(v2))
    expect = ublas::inner_prod(vector_reference_type(v1), vector_reference_type(v2));
    res = ublasx::dot(v1, v2);
    BOOST_UBLASX_DEBUG_TRACE( "dot(" << v1 << "," << v2 << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect, tol );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 3;
    size_type nc = 4;

    matrix_type A(nr, nc);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
    A(1,0) =  5; A(1,1) =  6; A(1,2) =  7; A(1,3) =  8;
    A(2,0) =  9; A(2,1) = 10; A(2,2) = 11; A(2,3) = 12;

    matrix_type B(nr, nc);
    B(0,0) = 13; B(0,1) = 14; B(0,2) = 15; B(0,3) = 16;
    B(1,0) = 17; B(1,1) = 18; B(1,2) = 19; B(1,3) = 20;
    B(2,0) = 21; B(2,1) = 22; B(2,2) = 23; B(2,3) = 24;

    vector_type dot_1(nc);
    for (size_type i = 0; i < nc; ++i)
    {
        dot_1(i) = ublas::inner_prod(ublas::column(A,i), ublas::column(B,i));
    }

    vector_type dot_2(nr);
    for (size_type i = 0; i < nr; ++i)
    {
        dot_2(i) = ublas::inner_prod(ublas::row(A,i), ublas::row(B,i));
    }


    vector_type expect;;
    vector_type res;;


    // dot<1>(A,B)
    expect = dot_1;
    res = ublasx::dot<1>(A, B);
    BOOST_UBLASX_DEBUG_TRACE( "dot<1>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );

    // dot<2>(A,B)
    expect = dot_2;
    res = ublasx::dot<2>(A, B);
    BOOST_UBLASX_DEBUG_TRACE( "dot<2>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 3;
    size_type nc = 4;

    matrix_type A(nr, nc);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
    A(1,0) =  5; A(1,1) =  6; A(1,2) =  7; A(1,3) =  8;
    A(2,0) =  9; A(2,1) = 10; A(2,2) = 11; A(2,3) = 12;

    matrix_type B(nr, nc);
    B(0,0) = 13; B(0,1) = 14; B(0,2) = 15; B(0,3) = 16;
    B(1,0) = 17; B(1,1) = 18; B(1,2) = 19; B(1,3) = 20;
    B(2,0) = 21; B(2,1) = 22; B(2,2) = 23; B(2,3) = 24;

    vector_type dot_1(nc);
    for (size_type i = 0; i < nc; ++i)
    {
        dot_1(i) = ublas::inner_prod(ublas::column(A,i), ublas::column(B,i));
    }

    vector_type dot_2(nr);
    for (size_type i = 0; i < nr; ++i)
    {
        dot_2(i) = ublas::inner_prod(ublas::row(A,i), ublas::row(B,i));
    }


    vector_type expect;;
    vector_type res;;


    // dot<1>(A,B)
    expect = dot_1;
    res = ublasx::dot<1>(A, B);
    BOOST_UBLASX_DEBUG_TRACE( "dot<1>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );

    // dot<2>(A,B)
    expect = dot_2;
    res = ublasx::dot<2>(A, B);
    BOOST_UBLASX_DEBUG_TRACE( "dot<2>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 3;
    size_type nc = 4;

    matrix_type A(nr, nc);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
    A(1,0) =  5; A(1,1) =  6; A(1,2) =  7; A(1,3) =  8;
    A(2,0) =  9; A(2,1) = 10; A(2,2) = 11; A(2,3) = 12;

    matrix_type B(nr, nc);
    B(0,0) = 13; B(0,1) = 14; B(0,2) = 15; B(0,3) = 16;
    B(1,0) = 17; B(1,1) = 18; B(1,2) = 19; B(1,3) = 20;
    B(2,0) = 21; B(2,1) = 22; B(2,2) = 23; B(2,3) = 24;

    vector_type dot_1(nr);
    for (size_type i = 0; i < nr; ++i)
    {
        dot_1(i) = ublas::inner_prod(ublas::row(A,i), ublas::row(B,i));
    }

    vector_type dot_2(nc);
    for (size_type i = 0; i < nc; ++i)
    {
        dot_2(i) = ublas::inner_prod(ublas::column(A,i), ublas::column(B,i));
    }


    vector_type expect;;
    vector_type res;;


    // dot<1>(A',B')
    expect = dot_1;
    res = ublasx::dot<1>(ublas::trans(A), ublas::trans(B));
    BOOST_UBLASX_DEBUG_TRACE( "dot<1>(" << ublas::trans(A) << "," << ublas::trans(B) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );

    // dot<2>(A',B')
    expect = dot_2;
    res = ublasx::dot<2>(ublas::trans(A), ublas::trans(B));
    BOOST_UBLASX_DEBUG_TRACE( "dot<2>(" << ublas::trans(A) << "," << ublas::trans(B) << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_reference<matrix_type> matrix_reference_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    size_type nr = 3;
    size_type nc = 4;

    matrix_type A(nr, nc);
    A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
    A(1,0) =  5; A(1,1) =  6; A(1,2) =  7; A(1,3) =  8;
    A(2,0) =  9; A(2,1) = 10; A(2,2) = 11; A(2,3) = 12;

    matrix_type B(nr, nc);
    B(0,0) = 13; B(0,1) = 14; B(0,2) = 15; B(0,3) = 16;
    B(1,0) = 17; B(1,1) = 18; B(1,2) = 19; B(1,3) = 20;
    B(2,0) = 21; B(2,1) = 22; B(2,2) = 23; B(2,3) = 24;

    vector_type dot_1(nc);
    for (size_type i = 0; i < nc; ++i)
    {
        dot_1(i) = ublas::inner_prod(ublas::column(A,i), ublas::column(B,i));
    }

    vector_type dot_2(nr);
    for (size_type i = 0; i < nr; ++i)
    {
        dot_2(i) = ublas::inner_prod(ublas::row(A,i), ublas::row(B,i));
    }


    vector_type expect;;
    vector_type res;;


    // dot<1>(ref(A),ref(B))
    expect = dot_1;
    res = ublasx::dot<1>(matrix_reference_type(A), matrix_reference_type(B));
    BOOST_UBLASX_DEBUG_TRACE( "dot<1>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );

    // dot<2>(ref(A),ref(B))
    expect = dot_2;
    res = ublasx::dot<2>(matrix_reference_type(A), matrix_reference_type(B));
    BOOST_UBLASX_DEBUG_TRACE( "dot<2>(" << A << "," << B << ") = " << res << " ==> " << expect );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, expect.size(), tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'dot' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_container );
    BOOST_UBLASX_TEST_DO( test_vector_expression );
    BOOST_UBLASX_TEST_DO( test_vector_reference );
    BOOST_UBLASX_TEST_DO( test_col_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
    BOOST_UBLASX_TEST_DO( test_matrix_expression );
    BOOST_UBLASX_TEST_DO( test_matrix_reference );

    BOOST_UBLASX_TEST_END();
}
