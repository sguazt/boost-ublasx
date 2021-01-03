/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/for_each.cpp
 *
 * \brief Test suite for the \c for_each operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/bind.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/for_each.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


template <typename T>
static void my_function(T const& x)
{
    BOOST_UBLASX_DEBUG_TRACE("x = " << x);
}


template <typename T>
struct my_functor: public std::unary_function<T,void>
{
    void operator()(T const& x) const
    {
        BOOST_UBLASX_DEBUG_TRACE("x = " << x);
    }
};


template <typename T>
void my_add(T const& x, T& s)
{
    s += x;
}


template <typename T>
struct my_adder
{
    void operator()(T const& x, T& s) const
    {
        s += x;
    }
};


BOOST_UBLASX_TEST_DEF( test_vector_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Function");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const std::size_t n(4);

    vector_type v(n);
    v(0) =  1;
    v(1) = -2;
    v(2) = -3;
    v(3) =  4;

    ublasx::for_each(v, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_vector_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Functor");

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;

    const std::size_t n(4);

    vector_type v(n);
    v(0) =  1;
    v(1) = -2;
    v(2) = -3;
    v(3) =  4;

    ublasx::for_each(v, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_vector_bound_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Bound Function");

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(4);

    vector_type v(n);
    v(0) =  1;
    v(1) = -2;
    v(2) = -3;
    v(3) =  4;

    value_type res(0);
    value_type expect_res(0);

    ublasx::for_each(v, boost::bind<void>(my_add<value_type>, _1, boost::ref(res)));

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res += v(i);
    }

    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_bound_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Vector - Bound Functor");

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(4);

    vector_type v(n);
    v(0) =  1;
    v(1) = -2;
    v(2) = -3;
    v(3) =  4;

    value_type res(0);
    value_type expect_res(0);

    ublasx::for_each(v, boost::bind<void>(my_adder<value_type>(), _1, boost::ref(res)));

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res += v(i);
    }

    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Function");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each(A, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Functor");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each(A, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_bound_function )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Bound Function");

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    value_type res(0);
    value_type expect_res(0);

    ublasx::for_each(A, boost::bind<void>(my_add<value_type>, _1, boost::ref(res)));

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_res += A(r,c);
        }
    }

    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_bound_functor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Bound Function");

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    value_type res(0);
    value_type expect_res(0);

    ublasx::for_each(A, boost::bind<void>(my_adder<value_type>(), _1, boost::ref(res)));

    BOOST_UBLASX_DEBUG_TRACE( "res = " << res );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_res += A(r,c);
        }
    }

    BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_function_dim1 )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Function - By Dimension: 1");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each<1>(A, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor_dim1 )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Functor - By Dimension: 1");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each<1>(A, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_function_dim2 )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Function - By Dimension: 2");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each<2>(A, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor_dim2 )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Functor - By Dimension: 2");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each<2>(A, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_function_dim_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Function - By Dimension: Major");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each_by_tag<ublasx::tag::major>(A, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor_dim_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Functor - By Dimension: Major");


    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each_by_tag<ublasx::tag::major>(A, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_function_dim_minor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Function - By Dimension: Minor");

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each_by_tag<ublasx::tag::minor>(A, my_function<value_type>);

    BOOST_UBLASX_TEST_CHECK(true);
}


BOOST_UBLASX_TEST_DEF( test_matrix_functor_dim_minor )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Matrix - Functor - By Dimension: Minor");


    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;

    const std::size_t nr(2);
    const std::size_t nc(3);

    matrix_type A(nr,nc);
    A(0,0) =  1; A(0,1) = -2; A(0,2) = -3;
    A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;

    ublasx::for_each_by_tag<ublasx::tag::minor>(A, my_functor<value_type>());

    BOOST_UBLASX_TEST_CHECK(true);
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'for_each' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_vector_function );
    BOOST_UBLASX_TEST_DO( test_vector_functor );
    BOOST_UBLASX_TEST_DO( test_matrix_function );
    BOOST_UBLASX_TEST_DO( test_matrix_functor );
    BOOST_UBLASX_TEST_DO( test_matrix_function_dim1 );
    BOOST_UBLASX_TEST_DO( test_matrix_functor_dim1 );
    BOOST_UBLASX_TEST_DO( test_matrix_function_dim2 );
    BOOST_UBLASX_TEST_DO( test_matrix_functor_dim2 );
    BOOST_UBLASX_TEST_DO( test_matrix_function_dim_major );
    BOOST_UBLASX_TEST_DO( test_matrix_functor_dim_major );
    BOOST_UBLASX_TEST_DO( test_matrix_function_dim_minor );
    BOOST_UBLASX_TEST_DO( test_matrix_functor_dim_minor );
    BOOST_UBLASX_TEST_DO( test_vector_bound_function );
    BOOST_UBLASX_TEST_DO( test_vector_bound_functor );
    BOOST_UBLASX_TEST_DO( test_matrix_bound_function );
    BOOST_UBLASX_TEST_DO( test_matrix_bound_functor );

    BOOST_UBLASX_TEST_END();
}
