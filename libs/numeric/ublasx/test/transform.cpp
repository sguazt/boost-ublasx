/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/transform.cpp
 *
 * \brief Test suite for the \c transform operation.
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

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/transform.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <cmath>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


template <typename AT, typename RT>
static RT my_function(AT const& x)
{
    return ::std::abs(x);
}


template <typename AT, typename RT>
struct my_functor: public std::unary_function<AT, RT>
{
    RT operator()(AT const& x)
    {
        return ::std::abs(x);
    }
};


BOOST_UBLASX_TEST_DEF( test_vector_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Unary Function");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    const std::size_t n(4);

    vector_type v(n);
    v(0) = -1;
    v(1) =  2;
    v(2) =  3;
    v(3) = -4;

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::transform(v, boost::function<value_type (value_type)>(my_function<value_type,value_type>));

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = my_function<value_type,value_type>(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(res, expect_res, n, tol);
}


BOOST_UBLASX_TEST_DEF( test_vector_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Unary Functor");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublas::vector_traits<vector_type>::size_type size_type;

    const std::size_t n(4);

    vector_type v(n);
    v(0) = -1;
    v(1) =  2;
    v(2) =  3;
    v(3) = -4;

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::transform(v, my_functor<value_type,value_type>());

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = my_functor<value_type,value_type>()(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(res, expect_res, n, tol);
}


BOOST_UBLASX_TEST_DEF( test_matrix_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Unary Function");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const std::size_t nr(3);
    const std::size_t nc(2);

    matrix_type A(nr,nc);
    A(0,0) = -1; A(0,1) =  2;
    A(1,0) =  3; A(1,1) = -4;
    A(2,0) = -5; A(2,1) = -6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::transform(A, ::boost::function<value_type (value_type)>(my_function<value_type,value_type>));

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "R = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = my_function<value_type,value_type>(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(R, expect_R, nr, nc, tol);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Unary Functor");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::size_type size_type;

    const std::size_t nr(3);
    const std::size_t nc(2);

    matrix_type A(nr,nc);
    A(0,0) = -1; A(0,1) =  2;
    A(1,0) =  3; A(1,1) = -4;
    A(2,0) = -5; A(2,1) = -6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::transform(A, my_functor<value_type,value_type>());

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "R = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = my_functor<value_type,value_type>()(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(R, expect_R, nr, nc, tol);
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'transform' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_function );
    BOOST_UBLASX_TEST_DO( test_vector_functor );
    BOOST_UBLASX_TEST_DO( test_matrix_function );
    BOOST_UBLASX_TEST_DO( test_matrix_functor );

    BOOST_UBLASX_TEST_END();
}
