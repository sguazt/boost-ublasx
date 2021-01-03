/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 *  \file libs/numeric/ublasx/diag.cpp
 *
 *  \brief Test suite for the \c diag operation.
 *
 *  Copyright (c) 2009, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/container/generalized_diagonal_matrix.hpp>
#include <boost/numeric/ublasx/operation/diag.hpp>
#include <cmath>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double TOL = 1.0e-5; ///< Tolerance for real number comparison.

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


//@{ View //////////////////////////////////////////////////////////////////////


BOOST_UBLASX_TEST_DEF( test_main_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Main Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix,ix) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix,ix), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_up1_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Upper Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, 1);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix,ix+1) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix,ix+1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_up2_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Upper Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, 2);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix,ix+2) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix,ix+2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_up3_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Upper Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, 3);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix,ix+3) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix,ix+3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_low1_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Lower Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, -1);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix+1,ix) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix+1,ix), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_low2_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Lower Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, -2);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix+2,ix) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix+2,ix), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_low3_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Lower Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, -3);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix+3,ix) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix+3,ix), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_low4_diagonal_view )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Fourth Lower Diagonal -- View" );

    typedef double value_type;
    typedef ublas::matrix<value_type> matrix_type;
    typedef ublasx::matrix_diagonal<matrix_type> diagonal_type;

    matrix_type A(5,4);

    A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
    A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
    A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
    A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
    A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


    diagonal_type D = ublasx::diag(A, -4);

    for (
        diagonal_type::const_iterator it = D.begin();
        it != D.end();
        ++it
    ) {
        matrix_type::size_type ix(it.index());

        BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix << ") = " << *it << " ==> " << A(ix+4,ix) );
        BOOST_UBLASX_TEST_CHECK_CLOSE( *it, A(ix+4,ix), TOL );
    }
}


//@} View //////////////////////////////////////////////////////////////////////

//@{ Creation //////////////////////////////////////////////////////////////////


BOOST_UBLASX_TEST_DEF( test_main_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Main Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << v.size() );
    BOOST_UBLASX_TEST_CHECK( D.size1() == v.size() );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << v.size() );
    BOOST_UBLASX_TEST_CHECK( D.size2() == v.size() );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 0 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 0 );

    // Check elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up1_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Upper Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    //typedef ublas::banded_matrix<value_type> matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 1);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+1) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+1) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+1) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+1) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 1 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 1 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+1) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+1) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up2_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Upper Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    //typedef ublas::banded_matrix<value_type> matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 2);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+2) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+2) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+2) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+2) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 2 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 2 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+2) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+2) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up3_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Upper Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 3);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+3) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+3) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+3) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+3) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 3 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 3 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+3) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+3) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low1_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Lower Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, -1);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+1) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+1) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+1) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+1) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -1 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -1 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+1) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+1) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low2_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Lower Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, -2);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+2) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+2) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+2) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+2) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -2 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -2 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+2) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+2) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low3_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Lower Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, -3);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+3) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+3) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+3) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+3) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -3 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -3 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+3) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+3) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low4_diagonal_create )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Fourth Lower Diagonal -- Create" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, -4);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << (v.size()+4) );
    BOOST_UBLASX_TEST_CHECK( D.size1() == (v.size()+4) );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << (v.size()+4) );
    BOOST_UBLASX_TEST_CHECK( D.size2() == (v.size()+4) );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -4 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -4 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+4) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+4) ? v(col) : value_type(0)), TOL );
        }
    }
}


//@} Creation //////////////////////////////////////////////////////////////////

//@{ Rectangular Creation //////////////////////////////////////////////////////


BOOST_UBLASX_TEST_DEF( test_main_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Main Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, 0);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 0 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 0 );

    // Check elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up1_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Upper Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, 1);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 1 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 1 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+1) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+1) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up2_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Upper Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, 2);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 2 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 2 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+2) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+2) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_up3_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Upper Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, 3);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << 3 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == 3 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << ((row+3) == col ? v(row) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, ((row+3) == col ? v(row) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low1_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST First Lower Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, -1);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -1 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -1 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+1) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+1) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low2_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Second Lower Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublasx::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, -2);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -2 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -2 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+2) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+2) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low3_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Third Lower Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, -3);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -3 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -3 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+3) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+3) ? v(col) : value_type(0)), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_low4_diagonal_create_rect )
{
    BOOST_UBLASX_DEBUG_TRACE( "TEST Fourth Lower Diagonal -- Create Rectangular" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(5);

    v(0) = 0.555950;
    v(1) = 0.108929;
    v(2) = 0.948014;
    v(3) = 0.023787;
    v(4) = 1.023787;


    matrix_type D = ublasx::diag(v, 5, 4, -4);

    // Check dimensions
    BOOST_UBLASX_DEBUG_TRACE( "D.size1() = " << D.size1() << " ==> " << 5 );
    BOOST_UBLASX_TEST_CHECK( D.size1() == 5 );
    BOOST_UBLASX_DEBUG_TRACE( "D.size2() = " << D.size2() << " ==> " << 4 );
    BOOST_UBLASX_TEST_CHECK( D.size2() == 4 );
    BOOST_UBLASX_DEBUG_TRACE( "D.offset() = " << D.offset() << " ==> " << -4 );
    BOOST_UBLASX_TEST_CHECK( D.offset() == -4 );

    // Check Elements
    for (
        matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "diag(v)(" << row << "," << col << ") = " << *col_it << " ==> " << (row == (col+4) ? v(col) : 0) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (row == (col+4) ? v(col) : value_type(0)), TOL );
        }
    }
}


//@} Rectangular Creation //////////////////////////////////////////////////////


//@{ Double Creation ///////////////////////////////////////////////////////////


///FIXME: does not work since matrix_diagonal is not a really vector
//BOOST_UBLASX_TEST_DEF( test_main_diagonal_double_create )
//{
//  BOOST_UBLASX_DEBUG_TRACE( "TEST Main Diagonal -- Double Creation" );
//
//  typedef double value_type;
//  typedef ublas::matrix<value_type> in_matrix_type;
//  typedef ublasx::generalized_diagonal_matrix<value_type> out_matrix_type;
//
//  in_matrix_type A(5,4);
//
//  A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
//  A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
//  A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
//  A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
//  A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;
//
//
//  out_matrix_type D = ublasx::diag(ublasx::diag(A));
//
//  for (
//      out_matrix_type::const_iterator1 it1 = D.begin1();
//      it1 != D.end1();
//      ++it1
//  ) {
//      for (
//          out_matrix_type::const_iterator2 it2 = it1.begin();
//          it2 != it1.end();
//          ++it2
//      ) {
//          out_matrix_type::size_type ix1(it2.index1());
//          out_matrix_type::size_type ix2(it2.index2());
//
//          BOOST_UBLASX_DEBUG_TRACE( "diag(A)(" << ix1 << "," << ix2 << ") = " << *it2 << " ==> " << A(ix1,ix1) );
//          BOOST_UBLASX_TEST_CHECK_CLOSE( *it2, A(ix1,ix2), TOL );
//      }
//  }
//}


//@} Double Creation ///////////////////////////////////////////////////////////


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'diag' operations");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_main_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_up1_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_up2_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_up3_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_low1_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_low2_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_low3_diagonal_view );
    BOOST_UBLASX_TEST_DO( test_low4_diagonal_view );

    BOOST_UBLASX_TEST_DO( test_main_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_up1_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_up2_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_up3_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_low1_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_low2_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_low3_diagonal_create );
    BOOST_UBLASX_TEST_DO( test_low4_diagonal_create );

    BOOST_UBLASX_TEST_DO( test_main_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_up1_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_up2_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_up3_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_low1_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_low2_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_low3_diagonal_create_rect );
    BOOST_UBLASX_TEST_DO( test_low4_diagonal_create_rect );

//  BOOST_UBLASX_TEST_DO( test_main_diagonal_double_create );

    BOOST_UBLASX_TEST_END();
}
