/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/isfinite.cpp
 *
 * \brief Test suite for the \c isfinite operation.
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

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/isfinite.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


namespace detail { namespace /*<unnamed>*/ {

template <typename T>
inline
static int isfinite_impl(T x)
{
    return ::std::isfinite(x);
}

template <typename T>
inline
static int isfinite_impl(::std::complex<T> const& x)
{
    return ::std::isfinite(x.real()) && ::std::isfinite(x.imag());
}

}} // Namespace detail::<unnamed>


BOOST_UBLASX_TEST_DEF( test_real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector" );

    typedef double in_value_type;
    typedef int out_value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<in_value_type> in_vector_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const size_type n(4);

    in_vector_type v(n);

    v(0) =  1;
    v(1) =  std::numeric_limits<in_value_type>::quiet_NaN();
    v(2) =  std::numeric_limits<in_value_type>::infinity();
    v(3) = -std::numeric_limits<in_value_type>::infinity();

    out_vector_type res;
    out_vector_type expect_res(n);

    res = ublasx::isfinite(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "isfinite(v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = detail::isfinite_impl(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vector" );

    typedef double real_type;
    typedef std::complex<real_type> in_value_type;
    typedef int out_value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<in_value_type> in_vector_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const size_type n(9);

    in_vector_type v(n);

    v(0) = in_value_type(1,2);
    v(1) = in_value_type(1,std::numeric_limits<real_type>::quiet_NaN());
    v(2) = in_value_type(std::numeric_limits<real_type>::quiet_NaN(),1);
    v(3) = in_value_type(1,std::numeric_limits<real_type>::infinity());
    v(4) = in_value_type(std::numeric_limits<real_type>::infinity(),1);
    v(5) = in_value_type(1,-std::numeric_limits<real_type>::infinity());
    v(6) = in_value_type(-std::numeric_limits<real_type>::infinity(),1);
    v(7) = in_value_type(std::numeric_limits<real_type>::infinity(),std::numeric_limits<real_type>::quiet_NaN());
    v(8) = in_value_type(-std::numeric_limits<real_type>::infinity(),std::numeric_limits<real_type>::quiet_NaN());

    out_vector_type res;
    out_vector_type expect_res(n);

    res = ublasx::isfinite(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "isfinite(v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = detail::isfinite_impl(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix" );

    typedef double in_value_type;
    typedef int out_value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<in_value_type> in_matrix_type;
    typedef ublas::matrix<out_value_type> out_matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    in_matrix_type A(nr,nc);

    A(0,0) = 1; A(0,1) = std::numeric_limits<in_value_type>::quiet_NaN(); A(0,2) = 3;
    A(1,0) = std::numeric_limits<in_value_type>::infinity(); A(1,1) = 5; A(1,2) = -std::numeric_limits<in_value_type>::infinity();

    out_matrix_type R;
    out_matrix_type expect_R(nr,nc);

    R = ublasx::isfinite(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "isfinite(A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = detail::isfinite_impl(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Matrix" );

    typedef double real_type;
    typedef std::complex<real_type> in_value_type;
    typedef int out_value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<in_value_type> in_matrix_type;
    typedef ublas::matrix<out_value_type> out_matrix_type;

    const size_type nr(3);
    const size_type nc(3);

    in_matrix_type A(nr,nc);

    A(0,0) = in_value_type(1,2); A(0,1) = in_value_type(1,std::numeric_limits<real_type>::quiet_NaN()); A(0,2) = in_value_type(std::numeric_limits<real_type>::quiet_NaN(),1);
    A(1,0) = in_value_type(1,std::numeric_limits<real_type>::infinity()); A(1,1) = in_value_type(std::numeric_limits<real_type>::infinity(),1); A(1,2) = in_value_type(1,-std::numeric_limits<real_type>::infinity());
    A(2,0) = in_value_type(-std::numeric_limits<real_type>::infinity(),1); A(2,1) = in_value_type(std::numeric_limits<real_type>::infinity(),std::numeric_limits<real_type>::quiet_NaN()); A(2,2) = in_value_type(-std::numeric_limits<real_type>::infinity(),std::numeric_limits<real_type>::quiet_NaN());

    out_matrix_type R;
    out_matrix_type expect_R(nr,nc);

    R = ublasx::isfinite(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "isfinite(A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = detail::isfinite_impl(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'isfinite' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_real_vector );
    BOOST_UBLASX_TEST_DO( test_complex_vector );
    BOOST_UBLASX_TEST_DO( test_real_matrix );
    BOOST_UBLASX_TEST_DO( test_complex_matrix );

    BOOST_UBLASX_TEST_END();
}
