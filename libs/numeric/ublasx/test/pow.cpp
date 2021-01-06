/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/pow.cpp
 *
 * \brief Test suite for the \c pow operation.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2015, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/pow.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_real_vector_1 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector -> [vector .^ scalar]" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n = 5;
    const double exp = 3;

    vector_type v(n);

    v(0) = -1.9;
    v(1) = -0.2;
    v(2) =  3.4;
    v(3) =  5.6;
    v(4) =  7.0;


    vector_type res;
    vector_type expect_res(n);

    res = ublasx::pow(v, exp);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "pow(v," << exp << ") = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = std::pow(v(i), exp);
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_vector_2 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector -> [scalar .^ vector]" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n = 5;
    const double base = 10;

    vector_type v(n);

    v(0) = -1.9;
    v(1) = -0.2;
    v(2) =  3.4;
    v(3) =  5.6;
    v(4) =  7.0;


    vector_type res;
    vector_type expect_res(n);

    res = ublasx::pow(base, v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "pow(" << base << ",v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = std::pow(base, v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_vector_1 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vector -> [vector .^ scalar]" );

    typedef std::complex<double> in_value_type;
    typedef in_value_type out_value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<in_value_type> in_vector_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const size_type n = 4;
    const double exp = 3;

    in_vector_type v(n);

    v(0) = in_value_type(1,2);
    v(1) = in_value_type(2,3);
    v(2) = in_value_type(3,4);
    v(3) = in_value_type(4,5);

    out_vector_type res;
    out_vector_type expect_res(n);

    res = ublasx::pow(v, exp);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "pow(v, " << exp << ") = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = std::pow(v(i), exp);
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_vector_2 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vector -> [scalar .^ vector]" );

    typedef std::complex<double> in_value_type;
    typedef in_value_type out_value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<in_value_type> in_vector_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const size_type n = 4;
    const double base = 10;

    in_vector_type v(n);

    v(0) = in_value_type(1,2);
    v(1) = in_value_type(2,3);
    v(2) = in_value_type(3,4);
    v(3) = in_value_type(4,5);

    out_vector_type res;
    out_vector_type expect_res(n);

    res = ublasx::pow(base, v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "pow(" << base << ",v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        // Remember that c^{a+ib} == e^{log_{10}(c)*(a+ib)}
        expect_res(i) = std::exp(std::log(base)*v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_1 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix -> [matrix .^ scalar]" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr = 2;
    const size_type nc = 3;
    const double exp = 3;

    matrix_type A(nr,nc);

    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::pow(A, exp);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = std::pow(A(r,c), exp);
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix_2 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix -> [scalar .^ matrix]" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr = 2;
    const size_type nc = 3;
    const double base = 10;

    matrix_type A(nr,nc);

    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::pow(base, A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "pow(" << base << ",A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = std::pow(base, A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_1 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Matrix -> [matrix .^ scalar]" );

    typedef std::complex<double> in_value_type;
    typedef in_value_type out_value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<in_value_type> in_matrix_type;
    typedef ublas::matrix<out_value_type> out_matrix_type;

    const size_type nr = 2;
    const size_type nc = 3;
    const double exp = 3;

    in_matrix_type A(nr,nc);

    A(0,0) = in_value_type(1,2); A(0,1) = in_value_type(2,3); A(0,2) = in_value_type(3,4);
    A(1,0) = in_value_type(4,5); A(1,1) = in_value_type(5,6); A(1,2) = in_value_type(6,7);

    out_matrix_type R;
    out_matrix_type expect_R(nr,nc);

    R = ublasx::pow(A, exp);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = std::pow(A(r,c), exp);
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_2 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Matrix -> [scalar .^ matrix]" );

    typedef std::complex<double> in_value_type;
    typedef in_value_type out_value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<in_value_type> in_matrix_type;
    typedef ublas::matrix<out_value_type> out_matrix_type;

    const size_type nr = 2;
    const size_type nc = 3;
    const double base = 10;

    in_matrix_type A(nr,nc);

    A(0,0) = in_value_type(1,2); A(0,1) = in_value_type(2,3); A(0,2) = in_value_type(3,4);
    A(1,0) = in_value_type(4,5); A(1,1) = in_value_type(5,6); A(1,2) = in_value_type(6,7);

    out_matrix_type R;
    out_matrix_type expect_R(nr,nc);

    R = ublasx::pow(base, A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "pow(" << base << ",A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = std::exp(std::log(base)*A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'pow' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_real_vector_1 );
    BOOST_UBLASX_TEST_DO( test_real_vector_2 );
    BOOST_UBLASX_TEST_DO( test_complex_vector_1 );
    BOOST_UBLASX_TEST_DO( test_complex_vector_2 );
    BOOST_UBLASX_TEST_DO( test_real_matrix_1 );
    BOOST_UBLASX_TEST_DO( test_real_matrix_2 );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_1 );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_2 );

    BOOST_UBLASX_TEST_END();
}
