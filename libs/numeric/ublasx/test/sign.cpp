/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/sign.cpp
 *
 * \brief Test suite for the \c sign operation.
 *
 * \author comcon1 based on code of Marco Guazzone
 *
 * <hr/>
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/sign.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"
#include <limits>


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


static const double tol = std::numeric_limits<double>::epsilon()*2.0;


namespace detail { namespace /*<unnamed>*/ {

template <typename T>
inline
static T sign_impl(T x)
{
    if (std::isnan(x))
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

template <typename T>
inline
static std::complex<T> sign_impl(const std::complex<T>& x)
{
    T a = std::abs(x);
    return (a == 0) ? std::complex<T>(0,0) : x/std::abs(x);
}

template <typename T>
inline
static bool isnan_impl(T x)
{
    return std::isnan(x);
}

template <typename T>
inline
static bool isnan_impl(const std::complex<T>& x)
{
    return std::isnan(std::real(x)) || std::isnan(std::imag(x));
}

}} // Namespace detail::<unnamed>


BOOST_UBLASX_TEST_DEF( test_real_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(4);

    vector_type v(n);

    v(0) =  0.0;
    v(1) = -2.0;
    v(2) = -3.0;
    v(3) =  4.0;

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::sign(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "sign(v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = detail::sign_impl(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vector" );

    typedef double real_type;
    typedef std::complex<real_type> value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(4);

    vector_type v(n);

    v(0) = value_type( 0.0,  0.0);
    v(1) = value_type(-2.0,  2.0);
    v(2) = value_type(-2.0, -2.0);
    v(3) = value_type( 2.0, -2.0);

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::sign(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "sign(v) = " << res );

    for (size_type i = 0; i < n; ++i)
    {
        expect_res(i) = detail::sign_impl(v(i));
    }

    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    matrix_type A(nr,nc);

    A(0,0) =  0; A(0,1) = -2; A(0,2) = -3;
    A(1,0) = -4; A(1,1) =  5; A(1,2) =  6;

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::sign(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "sign(A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = detail::sign_impl(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Matrix" );

    typedef double real_type;
    typedef std::complex<real_type> value_type;
    typedef std::size_t size_type;
    typedef ublas::matrix<value_type> matrix_type;

    const size_type nr(2);
    const size_type nc(3);

    matrix_type A(nr,nc);

    A(0,0) = value_type( 0,-6); A(0,1) = value_type(-2, 4); A(0,2) = value_type(-3,-3);
    A(1,0) = value_type(-4, 2); A(1,1) = value_type( 5, 7); A(1,2) = value_type( 6,-1);

    matrix_type R;
    matrix_type expect_R(nr,nc);

    R = ublasx::sign(A);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "sign(A) = " << R );

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
            expect_R(r,c) = detail::sign_impl(A(r,c));
        }
    }

    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_special_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Special - Vector" );

    typedef double value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(3);

    vector_type v(n);

    v(0) =  std::numeric_limits<value_type>::quiet_NaN();
    v(1) =  std::numeric_limits<value_type>::infinity();
    v(2) = -std::numeric_limits<value_type>::infinity();

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::sign(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "sign(v) = " << res );

    // We don't use BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE() since by definition NaN != NaN
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res(0)) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res(1),  1, tol );
    BOOST_UBLASX_TEST_CHECK_CLOSE( res(2), -1, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_special_vector )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Special - Vector" );

    typedef double real_type;
    typedef std::complex<real_type> value_type;
    typedef std::size_t size_type;
    typedef ublas::vector<value_type> vector_type;

    const size_type n(13);

    vector_type v(n);

    v( 0) = value_type( 1, std::numeric_limits<real_type>::quiet_NaN());
    v( 1) = value_type(-1, std::numeric_limits<real_type>::quiet_NaN());
    v( 2) = value_type( std::numeric_limits<real_type>::quiet_NaN(),  1);
    v( 3) = value_type( std::numeric_limits<real_type>::quiet_NaN(),-1);
    v( 4) = value_type( std::numeric_limits<real_type>::quiet_NaN(), std::numeric_limits<real_type>::quiet_NaN());
    v( 5) = value_type( std::numeric_limits<real_type>::infinity(), std::numeric_limits<real_type>::quiet_NaN());
    v( 6) = value_type(-std::numeric_limits<real_type>::infinity(), std::numeric_limits<real_type>::quiet_NaN());
    v( 7) = value_type( std::numeric_limits<real_type>::quiet_NaN(), std::numeric_limits<real_type>::infinity());
    v( 8) = value_type( std::numeric_limits<real_type>::quiet_NaN(),-std::numeric_limits<real_type>::infinity());
    v( 9) = value_type( std::numeric_limits<real_type>::infinity(), std::numeric_limits<real_type>::infinity());
    v(10) = value_type( std::numeric_limits<real_type>::infinity(),-std::numeric_limits<real_type>::infinity());
    v(11) = value_type(-std::numeric_limits<real_type>::infinity(), std::numeric_limits<real_type>::infinity());
    v(12) = value_type(-std::numeric_limits<real_type>::infinity(),-std::numeric_limits<real_type>::infinity());

    vector_type res;
    vector_type expect_res(n);

    res = ublasx::sign(v);

    BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
    BOOST_UBLASX_DEBUG_TRACE( "sign(v) = " << res );

    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 0)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 1)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 2)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 3)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 4)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 5)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 6)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 7)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 8)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res( 9)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res(10)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res(11)) );
    BOOST_UBLASX_TEST_CHECK( detail::isnan_impl(res(12)) );
}


int main()
{

    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'sign' operation");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_real_vector );
    BOOST_UBLASX_TEST_DO( test_complex_vector );
    BOOST_UBLASX_TEST_DO( test_real_matrix );
    BOOST_UBLASX_TEST_DO( test_complex_matrix );
    BOOST_UBLASX_TEST_DO( test_real_special_vector );
    BOOST_UBLASX_TEST_DO( test_complex_special_vector );

    BOOST_UBLASX_TEST_END();
}
