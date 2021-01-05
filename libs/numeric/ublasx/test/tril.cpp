/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/tril.cpp
 *
 * \brief Test suite for the lower-triangular view operation.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/tril.hpp>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1e-5;


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(n,n);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(n,n);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(n,n,value_type(1));

    for (::std::ptrdiff_t k = n-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < (n-k-1); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(n,n,value_type(1));

    for (::std::ptrdiff_t k = n-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < (n-k-1); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(n, n);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < n; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < n; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(n, n);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < n; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < n; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < nr; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < nr; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k == 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k > 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < ::std::min(nr,nc+k-1); ++i)
        {
            E(i,i+1-k) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k < 0" );

    typedef double value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < ::std::min(nr,nc+k-1); ++i)
        {
            E(i,i+1-k) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(n,n);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(n,n);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(n,n,value_type(1));

    for (::std::ptrdiff_t k = n-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < (n-k-1); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(n,n,value_type(1));

    for (::std::ptrdiff_t k = n-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < (n-k-1); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        ////BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(n, n);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < n; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < n; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t n(4);

    matrix_type A(n,n, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(n, n);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < n; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < n; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        ////BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < nr; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        ////BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(4);
    const ::std::size_t nc(6);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < nr; ++i)
        {
            E(i,static_cast< ::std::ptrdiff_t >(i+1-k)) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_keq0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k == 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr,nc);
    E(0,0) = 
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
    BOOST_UBLASX_DEBUG_TRACE( "tril(A)=" << X );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    X = ublasx::tril(A);
    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_kgt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k > 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::scalar_matrix<value_type,ublas::lower>(nr,nc,value_type(1));

    for (::std::ptrdiff_t k = nc-1; k >= 0; --k)
    {
        X = ublasx::tril(A, k);

        for (::std::size_t i = 0; i < ::std::min(nr,static_cast< ::std::size_t >(::std::max(static_cast<int>(nc-k-1),0))); ++i)
        {
            E(i,i+k+1) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << k << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < ::std::min(nr,nc+k-1); ++i)
        {
            E(i,i+1-k) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_klt0 )
{
    BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k < 0" );

    typedef double real_type;
    typedef ::std::complex<real_type> value_type;
    typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

    const ::std::size_t nr(6);
    const ::std::size_t nc(4);

    matrix_type A(nr,nc, value_type(1));


    matrix_type X;
    matrix_type E;

    E = ublasx::triangular_matrix<value_type,ublas::lower>(nr, nc);
    E(0,0) =
    E(1,0) = E(1,1) =
    E(2,0) = E(2,1) = E(2,2) =
    E(3,0) = E(3,1) = E(3,2) = E(3,3) =
    E(4,0) = E(4,1) = E(4,2) = E(4,3) =
    E(5,0) = E(5,1) = E(5,2) = E(5,3) = value_type(1);

    for (::std::size_t k = 0; k < nr; ++k)
    {
        X = ublasx::tril(A, -static_cast< ::std::ptrdiff_t >(k));

        for (::std::size_t i = k-1; k > 0 && i < ::std::min(nr,nc+k-1); ++i)
        {
            E(i,i+1-k) = value_type(0);
        }

        //BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
        BOOST_UBLASX_DEBUG_TRACE( "tril(A," << -static_cast< ::std::ptrdiff_t >(k) << ")=" << X );
        BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
    }
}


int main()
{
    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_keq0 );
    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_kgt0 );
    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_klt0 );
    BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_klt0 );

    BOOST_UBLASX_TEST_END();
}
