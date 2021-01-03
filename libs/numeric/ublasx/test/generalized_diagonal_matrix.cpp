/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 *  \file libs/numeric/ublasx/generalied_diagonal_matrix.cpp
 *
 *  \brief Test suite for the \c generalied_diagonal_matrix matrix container.
 *
 *  Copyright (c) 2009, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/container/generalized_diagonal_matrix.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double TOL(1.0e-5); ///< Tolerance for real numbers comparison.

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;

//@{ Generalized Diagonal Matrix -- Construction ///////////////////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;

    BOOST_UBLASX_DEBUG_TRACE( "A(0,0) " << A(0,0) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,0), 0.555950, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,1) " << A(1,1) << " ==> " << 0.830123 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,1), 0.830123, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,2) " << A(2,2) << " ==> " << 0.216504 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,2), 0.216504, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,3) " << A(3,3) << " ==> " << 0.450332 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,3), 0.450332, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,1) " << A(0,1) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,1), 0.274690, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,2) " << A(1,2) << " ==> " << 0.891726 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,2), 0.891726, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,3) " << A(2,3) << " ==> " << 0.883152 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,3), 0.883152, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,2) " << A(0,2) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,2), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,3) " << A(1,3) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,3), 0.895283, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,3) " << A(0,3) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,3), 0.798938, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(1,0) " << A(1,0) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,0), 0.108929, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,1) " << A(2,1) << " ==> " << 0.973234 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,1), 0.973234, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,2) " << A(3,2) << " ==> " << 0.231751 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,2), 0.231751, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(2,0) " << A(2,0) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,0), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,1) " << A(3,1) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,1), 0.675382, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(3,0) " << A(3,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,0), 0.023787, TOL );
}


//@} Generalized Diagonal Matrix -- Construction ///////////////////////////////

//@{ Generalized Diagonal Matrix -- Column-major construction //////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;

    BOOST_UBLASX_DEBUG_TRACE( "A(0,0) " << A(0,0) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,0), 0.555950, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,1) " << A(1,1) << " ==> " << 0.830123 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,1), 0.830123, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,2) " << A(2,2) << " ==> " << 0.216504 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,2), 0.216504, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,3) " << A(3,3) << " ==> " << 0.450332 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,3), 0.450332, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, 1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,1) " << A(0,1) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,1), 0.274690, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,2) " << A(1,2) << " ==> " << 0.891726 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,2), 0.891726, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,3) " << A(2,3) << " ==> " << 0.883152 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,3), 0.883152, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, 2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,2) " << A(0,2) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,2), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,3) " << A(1,3) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,3), 0.895283, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, 3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,3) " << A(0,3) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,3), 0.798938, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, -1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(1,0) " << A(1,0) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,0), 0.108929, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,1) " << A(2,1) << " ==> " << 0.973234 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,1), 0.973234, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,2) " << A(3,2) << " ==> " << 0.231751 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,2), 0.231751, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(2,0) " << A(2,0) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,0), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,1) " << A(3,1) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,1), 0.675382, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type, ublas::column_major> matrix_type;

    matrix_type A(4, -3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(3,0) " << A(3,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,0), 0.023787, TOL );
}


//@} Generalized Diagonal Matrix -- Column-major construction //////////////////

//@{ Generalized Diagonal Matrix -- Rectangular construction ///////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,0);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332; /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,0) " << A(0,0) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,0), 0.555950, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,1) " << A(1,1) << " ==> " << 0.830123 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,1), 0.830123, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,2) " << A(2,2) << " ==> " << 0.216504 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,2), 0.216504, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,3) " << A(3,3) << " ==> " << 0.450332 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,3), 0.450332, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(3,4) = 0.555950; /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,1) " << A(0,1) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,1), 0.274690, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,2) " << A(1,2) << " ==> " << 0.891726 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,2), 0.891726, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,3) " << A(2,3) << " ==> " << 0.883152 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,3), 0.883152, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,4) " << A(3,4) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,4), 0.555950, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(2,4) = 0.555950; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(3,5) = 0.274690; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,2) " << A(0,2) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,2), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,3) " << A(1,3) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,3), 0.895283, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,4) " << A(2,4) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,4), 0.555950, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,5) " << A(3,5) << " ==> " << 0.274690 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,5), 0.274690, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(1,4) = 0.540605; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(2,5) = 0.895283; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(3,6) = 0.555950;

    BOOST_UBLASX_DEBUG_TRACE( "A(0,3) " << A(0,3) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,3), 0.798938, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,4) " << A(1,4) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,4), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,5) " << A(2,5) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,5), 0.895283, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,6) " << A(3,6) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,6), 0.555950, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up4_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Fourth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,4);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(0,4) = 0.798938; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(1,5) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(2,6) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,4) " << A(0,4) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,4), 0.798938, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,5) " << A(1,5) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,5), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,6) " << A(2,6) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,6), 0.895283, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up5_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Fifth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,5);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(0,5) = 0.798938; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(1,6) = 0.540605;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,5) " << A(0,5) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,5), 0.798938, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,6) " << A(1,6) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,6), 0.540605, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_up6_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Sixth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,6);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            A(0,6) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,6) " << A(0,6) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,6), 0.798938, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,-1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(1,0) " << A(1,0) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,0), 0.108929, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,1) " << A(2,1) << " ==> " << 0.973234 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,1), 0.973234, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,2) " << A(3,2) << " ==> " << 0.231751 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,2), 0.231751, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,-2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(2,0) " << A(2,0) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,0), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,1) " << A(3,1) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,1), 0.675382, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_hrect_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4,7,-3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(3,0) " << A(3,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,0), 0.023787, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,0);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,0) " << A(0,0) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,0), 0.555950, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,1) " << A(1,1) << " ==> " << 0.830123 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,1), 0.830123, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,2) " << A(2,2) << " ==> " << 0.216504 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,2), 0.216504, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,3) " << A(3,3) << " ==> " << 0.450332 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,3), 0.450332, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,1) " << A(0,1) << " ==> " << 0.555950 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,1), 0.274690, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,2) " << A(1,2) << " ==> " << 0.891726 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,2), 0.891726, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,3) " << A(2,3) << " ==> " << 0.883152 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,3), 0.883152, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,2) " << A(0,2) << " ==> " << 0.540605 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,2), 0.540605, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(1,3) " << A(1,3) << " ==> " << 0.895283 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,3), 0.895283, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(0,3) " << A(0,3) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(0,3), 0.798938, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(4,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(1,0) " << A(1,0) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(1,0), 0.108929, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(2,1) " << A(2,1) << " ==> " << 0.973234 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,1), 0.973234, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,2) " << A(3,2) << " ==> " << 0.231751 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,2), 0.231751, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(4,3) " << A(4,3) << " ==> " << 0.798938 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(4,3), 0.798938, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 0.108929; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(5,3) = 0.973234;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(2,0) " << A(2,0) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(2,0), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(3,1) " << A(3,1) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,1), 0.675382, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(4,2) " << A(4,2) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(4,2), 0.108929, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(5,3) " << A(5,3) << " ==> " << 0.973234 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(5,3), 0.973234, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(4,1) = 0.948014; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(5,2) = 0.675382; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(6,3) = 0.108929;

    BOOST_UBLASX_DEBUG_TRACE( "A(3,0) " << A(3,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(3,0), 0.023787, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(4,1) " << A(4,1) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(4,1), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(5,2) " << A(5,2) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(5,2), 0.675382, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(6,3) " << A(6,3) << " ==> " << 0.108929 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(6,3), 0.108929, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low4_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Fourth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-4);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(4,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(5,1) = 0.948014; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(6,2) = 0.675382; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(4,0) " << A(4,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(4,0), 0.023787, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(5,1) " << A(5,1) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(5,1), 0.948014, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(6,2) " << A(6,2) << " ==> " << 0.675382 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(6,2), 0.675382, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low5_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Fifth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-5);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(5,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(6,1) = 0.948014; /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(5,0) " << A(5,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(5,0), 0.023787, TOL );
    BOOST_UBLASX_DEBUG_TRACE( "A(6,1) " << A(6,1) << " ==> " << 0.948014 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(6,1), 0.948014, TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_vrect_low6_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Sixth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(7,4,-6);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(6,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A(6,0) " << A(6,0) << " ==> " << 0.023787 );
    BOOST_UBLASX_TEST_CHECK_CLOSE( A(6,0), 0.023787, TOL );
}


//@} Generalized Diagonal Matrix -- Rectangular construction ///////////////////

//@{ Generalized Diagonal Matrix -- Row-by-column iteration ////////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator1 row_it = A.begin1();
        row_it != A.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *col_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Matrix -- Row-by-column iteration ////////////////////

//@{ Generalized Diagonal Matrix -- Column-by-row iteration ////////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, 3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix_traits<matrix_type>::difference_type difference_type;

    matrix_type A(4, -3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    for (
        matrix_type::const_iterator2 col_it = A.begin2();
        col_it != A.end2();
        ++col_it
    ) {
        for (
            matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            matrix_type::size_type row(row_it.index1());
            matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "A(" << row << "," << col << ") " << *row_it << " ==> " << A(row,col) );
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == A.offset() );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Matrix -- Column-by-row iteration ////////////////////

//@{ Generalized Diagonal Matrix -- Copy-construction //////////////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4);

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,0) " << B(0,0) << " ==> " << A(0,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,0), A(0,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,1) " << B(1,1) << " ==> " << A(1,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,1), A(1,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,2) " << B(2,2) << " ==> " << A(2,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,2), A(2,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,3) " << B(3,3) << " ==> " << A(3,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,3), A(3,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 1);

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,1) " << B(0,1) << " ==> " << A(0,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,1), A(0,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,2) " << B(1,2) << " ==> " << A(1,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,2), A(1,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,3) " << B(2,3) << " ==> " << A(2,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,3), A(2,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 2);

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,2) " << B(0,2) << " ==> " << A(0,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,2), A(0,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,3) " << B(1,3) << " ==> " << A(1,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,3), A(1,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, 3);

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,3) " << B(0,3) << " ==> " << A(0,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,3), A(0,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -1);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(1,0) " << B(1,0) << " ==> " << A(1,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,0), A(1,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,1) " << B(2,1) << " ==> " << A(2,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,1), A(2,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,2) " << B(3,2) << " ==> " << A(3,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,2), A(3,2), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(2,0) " << B(2,0) << " ==> " << A(2,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,0), A(2,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,1) " << B(3,1) << " ==> " << A(3,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,1), A(3,1), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(4, -3);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */

    matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(3,0) " << B(3,0) << " ==> " << A(3,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,0), A(3,0), TOL );
}


//@} Generalized Diagonal Matrix -- Copy-construction //////////////////////////

//@{ Generalized Diagonal Matrix -- Matrix-copy-construction ///////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    A(0,0) = 0.555950; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(1,1) = 0.830123; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(2,2) = 0.216504; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(3,3) = 0.450332;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    result_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,0) " << B(0,0) << " ==> " << A(0,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,0), A(0,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,1) " << B(1,1) << " ==> " << A(1,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,1), A(1,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,2) " << B(2,2) << " ==> " << A(2,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,2), A(2,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,3) " << B(3,3) << " ==> " << A(3,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,3), A(3,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            A(0,1) = 0.274690; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(1,2) = 0.891726; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(2,3) = 0.883152;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    result_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,1) " << B(0,1) << " ==> " << A(0,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,1), A(0,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,2) " << B(1,2) << " ==> " << A(1,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,2), A(1,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,3) " << B(2,3) << " ==> " << A(2,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,3), A(2,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            A(0,2) = 0.540605; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(1,3) = 0.895283;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    result_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,2) " << B(0,2) << " ==> " << A(0,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,2), A(0,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,3) " << B(1,3) << " ==> " << A(1,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,3), A(1,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            /* 0 */            A(0,3) = 0.798938;
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */

    result_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,3) " << B(0,3) << " ==> " << A(0,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,3), A(0,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(1,0) = 0.108929; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(2,1) = 0.973234; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(3,2) = 0.231751; /* 0 */
    /* 0 */            /* 0 */            /* 0 */            A(4,3) = 1.450332;

    result_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "B(1,0) " << B(1,0) << " ==> " << A(1,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,0), A(1,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,1) " << B(2,1) << " ==> " << A(2,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,1), A(2,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,2) " << B(3,2) << " ==> " << A(3,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,2), A(3,2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(4,3) " << B(4,3) << " ==> " << A(4,3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(4,3), A(4,3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    result_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "B(2,0) " << B(2,0) << " ==> " << A(2,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,0), A(2,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,1) " << B(3,1) << " ==> " << A(3,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,1), A(3,1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(4,2) " << B(4,2) << " ==> " << A(4,2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(4,2), A(4,2), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(3,0) = 0.023787; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(4,1) = 1.675382; /* 0 */            /* 0 */

    result_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "B(3,0) " << B(3,0) << " ==> " << A(3,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,0), A(3,0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(4,1) " << B(4,1) << " ==> " << A(4,1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(4,1), A(4,1), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low4_diagonal_mat_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Fourth Lower Diagonal -- Matrix-Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> source_matrix_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type;

    source_matrix_type A(5, 4, value_type(0));

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(4,0) = 1.023787; /* 0 */            /* 0 */            /* 0 */

    result_matrix_type B(A, -4);

    BOOST_UBLASX_DEBUG_TRACE( "B(4,0) " << B(4,0) << " ==> " << A(4,0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(4,0), A(4,0), TOL );
}


//@} Generalized Diagonal Matrix -- Matrix-copy-construction ///////////////////

//@{ Generalized Diagonal Matrix -- Vector-copy-construction ///////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_main_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Main Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(4);

    v(0) = 0.555950;
    v(1) = 0.830123;
    v(2) = 0.216504;
    v(3) = 0.450332;

    matrix_type B(v);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,0) " << B(0,0) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,0), v(0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,1) " << B(1,1) << " ==> " << v(1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,1), v(1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,2) " << B(2,2) << " ==> " << v(2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,2), v(2), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,3) " << B(3,3) << " ==> " << v(3) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,3), v(3), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up1_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Upper Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(3);

    v(0) = 0.274690;
    v(1) = 0.891726;
    v(2) = 0.883152;

    matrix_type B(v, 1);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,1) " << B(0,1) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,1), v(0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,2) " << B(1,2) << " ==> " << v(1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,2), v(1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,3) " << B(2,3) << " ==> " << v(2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,3), v(2), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up2_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Upper Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(2);

    v(0) = 0.540605;
    v(1) = 0.895283;

    matrix_type B(v, 2);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,2) " << B(0,2) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,2), v(0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(1,3) " << B(1,3) << " ==> " << v(1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,3), v(1), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_up3_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Upper Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(3);

    v(0) = 0.798938;

    matrix_type B(v, 3);

    BOOST_UBLASX_DEBUG_TRACE( "B(0,3) " << B(0,3) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(0,3), v(0), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low1_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- First Lower Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(3);

    v(0) = 0.108929;
    v(1) = 0.973234;
    v(2) = 0.231751;

    matrix_type B(v, -1);

    BOOST_UBLASX_DEBUG_TRACE( "B(1,0) " << B(1,0) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(1,0), v(0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(2,1) " << B(2,1) << " ==> " << v(1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,1), v(1), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,2) " << B(3,2) << " ==> " << v(2) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,2), v(2), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low2_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Second Lower Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(2);

    v(0) = 0.948014;
    v(1) = 0.675382;

    matrix_type B(v, -2);

    BOOST_UBLASX_DEBUG_TRACE( "B(2,0) " << B(2,0) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(2,0), v(0), TOL );
    BOOST_UBLASX_DEBUG_TRACE( "B(3,1) " << B(3,1) << " ==> " << v(1) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,1), v(1), TOL );
}


BOOST_UBLASX_TEST_DEF( test_gdm_low3_diagonal_vec_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Third Lower Diagonal -- Vector-Copy-Construction" );

    typedef double value_type;
    typedef ublas::vector<value_type> vector_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    vector_type v(1);

    v(0) = 0.023787;

    matrix_type B(v, -3);

    BOOST_UBLASX_DEBUG_TRACE( "B(3,0) " << B(3,0) << " ==> " << v(0) );
    BOOST_UBLASX_TEST_CHECK_CLOSE( B(3,0), v(0), TOL );
}


//@} Generalized Diagonal Matrix -- Vector-copy-construction ///////////////////

//@{ Generalized Diagonal Matrix -- Matrix operations //////////////////////////


BOOST_UBLASX_TEST_DEF( test_gdm_op_transpose )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Transpose" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;

    matrix_type A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type C(4, 5, 2);
    C = ublas::trans(A);

    for (
        matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            matrix_type::size_type row(col_it.index1());
            matrix_type::size_type col(col_it.index2());

            const matrix_type::value_type a = static_cast<matrix_type const&>(A)(col,row); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << a );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, a, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_sum_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal + Dense" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type1;
    typedef ublas::matrix<value_type> matrix_type2;
    typedef ublas::matrix<value_type> result_matrix_type;

    matrix_type1 A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type2 B(5,4);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    result_matrix_type C = A + B;

    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const matrix_type1::value_type a = static_cast<matrix_type1 const&>(A)(row,col); // FIXME: This type of cast is very boring!!
            const matrix_type2::value_type b = B(row,col);

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << (a+b) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (a+b), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_diff_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal - Dense" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type1;
    typedef ublas::matrix<value_type> matrix_type2;
    typedef ublas::matrix<value_type> result_matrix_type;

    matrix_type1 A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type2 B(5,4);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    result_matrix_type C = A - B;

    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const matrix_type1::value_type a = static_cast<matrix_type1 const&>(A)(row,col); // FIXME: This type of cast is very boring!!
            const matrix_type2::value_type b = B(row,col);

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << (a-b) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (a-b), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_prod )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal * Generalized Diagonal => Dense" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef ublas::matrix<value_type> result_matrix_type;

    matrix_type A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type B(4, 3, 1);

    /* 0 */            B(0,1) = 0.274690; /* 0 */
    /* 0 */            /* 0 */            B(1,2) = 0.891726;
    /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */

    result_matrix_type T(5, 3, 0);

    /* 0 */            /* 0 */            /* 0 */
    T(1,0) = 0;        /* 0 */            /* 0 */
    /* 0 */            T(2,1) = 0.260410; /* 0 */
    /* 0 */            /* 0 */            T(3,2) = 0.602256;
    /* 0 */            /* 0 */            /* 0 */

    result_matrix_type C; //(5, 3, -1);
    C = ublas::prod(A, B);

    BOOST_UBLASX_TEST_CHECK( C.size1() == T.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == T.size2() );
    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            //const matrix_type::value_type t = static_cast<matrix_type const&>(T)(row,col); // FIXME: This type of cast is very boring!!
            const result_matrix_type::value_type t = T(row,col);

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << t );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, t, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_prod_bis )
{
    //NOTE: This test only works if before assigned the result of the product
    // you build the result matrix with the right structure, that is:
    //       (size1, size2, offset).

    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal * Generalized Diagonal => Generalized Diagonal" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type;
    typedef matrix_type result_matrix_type;

    matrix_type A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type B(4, 3, 1);

    /* 0 */            B(0,1) = 0.274690; /* 0 */
    /* 0 */            /* 0 */            B(1,2) = 0.891726;
    /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */

    result_matrix_type T(5, 3, -1);

    /* 0 */            /* 0 */            /* 0 */
    T(1,0) = 0;        /* 0 */            /* 0 */
    /* 0 */            T(2,1) = 0.260410; /* 0 */
    /* 0 */            /* 0 */            T(3,2) = 0.602256;
    /* 0 */            /* 0 */            /* 0 */

    result_matrix_type C(5, 3, -1);
    C = ublas::prod(A, B);

    BOOST_UBLASX_TEST_CHECK( C.size1() == T.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == T.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == T.offset() );
    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const result_matrix_type::value_type t = static_cast<result_matrix_type const&>(T)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << t );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, t, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_element_prod_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal .* Dense" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type1;
    typedef ublas::matrix<value_type> matrix_type2;
    typedef ublas::matrix<value_type> result_matrix_type;

    matrix_type1 A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type2 B(5,4);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    result_matrix_type C = ublas::element_prod(A, B);

    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const matrix_type1::value_type a = static_cast<matrix_type1 const&>(A)(row,col); // FIXME: This type of cast is very boring!!
            const matrix_type2::value_type b = B(row,col);

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << (a*b) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (a*b), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gdm_op_element_div_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Matrix -- Operations -- Generalized Diagonal ./ Dense" );

    typedef double value_type;
    typedef ublasx::generalized_diagonal_matrix<value_type> matrix_type1;
    typedef ublas::matrix<value_type> matrix_type2;
    typedef ublas::matrix<value_type> result_matrix_type;

    matrix_type1 A(5, 4, -2);

    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    /* 0 */            /* 0 */            /* 0 */            /* 0 */
    A(2,0) = 0.948014; /* 0 */            /* 0 */            /* 0 */
    /* 0 */            A(3,1) = 0.675382; /* 0 */            /* 0 */
    /* 0 */            /* 0 */            A(4,2) = 1.231751; /* 0 */

    matrix_type2 B(5,4);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    result_matrix_type C = ublas::element_div(A, B);

    for (
        result_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const matrix_type1::value_type a = static_cast<matrix_type1 const&>(A)(row,col); // FIXME: This type of cast is very boring!!
            const matrix_type2::value_type b = B(row,col);

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") " << *col_it << " ==> " << (a/b) );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, (a/b), TOL );
        }
    }
}


//@} Generalized Diagonal Matrix -- Matrix operations //////////////////////////


//@{ Generalized Diagonal Adaptor -- Construction //////////////////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i), A(i,i), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+1), A(i,i+1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+2), A(i,i+2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+3), A(i,i+3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 1; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-1), A(i,i-1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 2; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-2), A(i,i-2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 3; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-3), A(i,i-3), TOL );
    }
}


//@} Generalized Diagonal Adaptor -- Construction //////////////////////////////

//@{ Generalized Diagonal Adaptor -- Column-major construction /////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i), A(i,i), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+1), A(i,i+1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+2), A(i,i+2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (n-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+3), A(i,i+3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 1; i < (n-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-1), A(i,i-1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    for (std::size_t i = 2; i < (n-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-2), A(i,i-2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_col_major )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Column Major" );

    typedef double value_type;
    typedef ublas::matrix<value_type, ublas::column_major> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 3; i < (n-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-3), A(i,i-3), TOL );
    }
}


//@} Generalized Diagonal Adaptor -- Column-major construction /////////////////

//@{ Generalized Diagonal Adaptor -- Rectangular construction //////////////////


BOOST_UBLASX_TEST_DEF( test_gda_hrect_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i), A(i,i), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+1), A(i,i+1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+2), A(i,i+2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+3), A(i,i+3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up4_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nr-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+4), A(i,i+4), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up5_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fifth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 5);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nr-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+5), A(i,i+5), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_up6_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Sixth Upper Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, 6);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nr-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+6), A(i,i+6), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 1; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-1), A(i,i-1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 2; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-2), A(i,i-2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_hrect_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Horizontal Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 4;
    const std::size_t nc = 7;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60; A(0,5) =  3.31; A(0,6) = -4.81;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04; A(1,5) =  5.29; A(1,6) =  3.55;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89; A(2,5) =  8.20; A(2,6) = -1.51;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08; A(3,4) = -7.66; A(3,5) = -7.33; A(3,6) =  6.18;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 3; i < nr; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-3), A(i,i-3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_main_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < nc; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i), A(i,i), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_up1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nc-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+1), A(i,i+1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_up2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nc-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+2), A(i,i+2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_up3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 0; i < (nc-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i+3), A(i,i+3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low1_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 1; i < nc; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-1), A(i,i-1), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low2_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 2; i < nc; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-2), A(i,i-2), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low3_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 3; i < nc; ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-3), A(i,i-3), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low4_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 4; i < (nc-1); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-4), A(i,i-4), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low5_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fifth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -5);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 5; i < (nc-2); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-5), A(i,i-5), TOL );
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_vrect_low6_diagonal )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Sixth Lower Diagonal -- Vertical Rectangular Matrix" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;

    const std::size_t nr = 7;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;
    A(5,0) =  3.31; A(5,1) =  5.29; A(5,2) =  8.20; A(5,3) = -7.33;
    A(6,0) = -4.81; A(6,1) =  3.55; A(6,2) = -1.51; A(6,3) =  6.18;

    adapter_matrix_type B(A, -6);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (std::size_t i = 6; i < (nc-3); ++i)
    {
        BOOST_UBLASX_TEST_CHECK_CLOSE( B(i,i-6), A(i,i-6), TOL );
    }
}


//@} Generalized Diagonal Adaptor -- Rectangular construction //////////////////

//@{ Generalized Diagonal Adaptor -- Row-by-column iteration ///////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up4_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Upper Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_hrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Horizontal Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low4_diagonal_vrect_row_col_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Lower Diagonal -- Vertical Rectangular Matrix -- Row-Col Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator1 row_it = B.begin1();
        row_it != B.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Adaptor -- Row-by-column iteration ///////////////////

//@{ Generalized Diagonal Adaptor -- Column-by-row iteration ///////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up4_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Upper Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, 4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_hrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Horizontal Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 3;
    const std::size_t nc = 5;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25; A(0,4) = -4.60;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14; A(1,4) = -7.04;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35; A(2,4) = -3.89;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, 1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, 2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -1);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -2);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -3);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low4_diagonal_vrect_col_row_iteration )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Fourth Lower Diagonal -- Vertical Rectangular Matrix -- Col-Row Iteration" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 3;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89;

    adapter_matrix_type B(A, -4);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );

    BOOST_UBLASX_TEST_CHECK( B.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( B.size2() == A.size2() );
    for (
        adapter_matrix_type::const_iterator2 col_it = B.begin2();
        col_it != B.end2();
        ++col_it
    ) {
        for (
            adapter_matrix_type::const_iterator1 row_it = col_it.begin();
            row_it != col_it.end();
            ++row_it
        ) {
            adapter_matrix_type::size_type row(row_it.index1());
            adapter_matrix_type::size_type col(row_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "B(" << row << "," << col << ") = " << *row_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == B.offset() );
            // double-check: *row_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *row_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Adaptor -- Column-by-row iteration ///////////////////

//@{ Generalized Diagonal Adaptor -- Copy-construction /////////////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_copy )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Copy-Construction" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    adapter_matrix_type C(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Adaptor -- Copy-construction /////////////////////////

//@{ Generalized Diagonal Adaptor -- Copy-assignement //////////////////////////


BOOST_UBLASX_TEST_DEF( test_gda_main_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Main Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up1_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Upper Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 1);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up2_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Upper Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 2);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_up3_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Upper Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, 3);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low1_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- First Lower Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -1);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low2_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Second Lower Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -2);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_low3_diagonal_assign )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Third Lower Diagonal -- Copy-Assignement" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t n = 4;

    adaptee_matrix_type A(n, n);

    A(0,0) = -0.57; A(0,1) = -1.28; A(0,2) = -0.39; A(0,3) =  0.25;
    A(1,0) = -1.93; A(1,1) =  1.08; A(1,2) = -0.31; A(1,3) = -2.14;
    A(2,0) =  2.30; A(2,1) =  0.24; A(2,2) =  0.40; A(2,3) = -0.35;
    A(3,0) = -1.93; A(3,1) =  0.64; A(3,2) = -0.66; A(3,3) =  0.08;

    adapter_matrix_type B(A, -3);

    adapter_matrix_type C(A);
    C = B;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(row,col) << " == " << A(row,col));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(row,col) == A(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(row,col), TOL );
        }
    }
}


//@} Generalized Diagonal Adaptor -- Copy-assignement //////////////////////////

//@{ Generalized Diagonal Adaptor -- Matrix operations /////////////////////////


BOOST_UBLASX_TEST_DEF( test_gda_op_transpose )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Transpose" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix_traits<adapter_matrix_type>::difference_type difference_type;

    const std::size_t nr = 5;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    adapter_matrix_type B(A, -2);

    //FIXME: The use of the auxiliary matrix At is needed in order to assign to C the right matrix structure (i.e., #rows, #columns)
    adaptee_matrix_type At(nc, nr);
    adapter_matrix_type C(At, 2);
    C = ublas::trans(B);

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );

    BOOST_UBLASX_TEST_CHECK( C.size1() == B.size2() );
    BOOST_UBLASX_TEST_CHECK( C.size2() == B.size1() );
    BOOST_UBLASX_TEST_CHECK( C.offset() == -B.offset() );
    for (
        adapter_matrix_type::const_iterator1 row_it = C.begin1();
        row_it != C.end1();
        ++row_it
    ) {
        for (
            adapter_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            adapter_matrix_type::size_type row(col_it.index1());
            adapter_matrix_type::size_type col(col_it.index2());

            BOOST_UBLASX_DEBUG_TRACE( "C(" << row << "," << col << ") = " << *col_it << " ==> " << B(col,row) << " == " << A(col,row));
            BOOST_UBLASX_TEST_CHECK( difference_type(col-row) == C.offset() );
            // double-check: *col_it == B(col,row) == A(col,row)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, B(col,row), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, A(col,row), TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_op_sum_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Generalized Diagonal + Dense" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix<value_type> dense_matrix_type;

    const std::size_t nr = 5;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    dense_matrix_type B(nr, nc);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    adapter_matrix_type C(A, -2);

    dense_matrix_type D = B + C;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );
    BOOST_UBLASX_DEBUG_TRACE( "D " << D );

    BOOST_UBLASX_TEST_CHECK( D.size1() == nr );
    BOOST_UBLASX_TEST_CHECK( D.size2() == nc );
    for (
        dense_matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            dense_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            dense_matrix_type::size_type row(col_it.index1());
            dense_matrix_type::size_type col(col_it.index2());

            const dense_matrix_type::value_type b = B(row,col);
            const dense_matrix_type::value_type c = static_cast<adapter_matrix_type const&>(C)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "D(" << row << "," << col << ") = " << *col_it << " ==> " << D(row,col) << " == " << (b+c) );
            // double-check: *col_it == D(row,col) == B(row,col)+C(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, D(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, b+c, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_op_diff_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Generalized Diagonal - Dense" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix<value_type> dense_matrix_type;

    const std::size_t nr = 5;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    dense_matrix_type B(nr, nc);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;

    adapter_matrix_type C(A, -2);

    dense_matrix_type D = B - C;

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );
    BOOST_UBLASX_DEBUG_TRACE( "D " << D );

    BOOST_UBLASX_TEST_CHECK( D.size1() == A.size1() );
    BOOST_UBLASX_TEST_CHECK( D.size2() == A.size2() );
    for (
        dense_matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            dense_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            dense_matrix_type::size_type row(col_it.index1());
            dense_matrix_type::size_type col(col_it.index2());

            const dense_matrix_type::value_type b = B(row,col);
            const dense_matrix_type::value_type c = static_cast<adapter_matrix_type const&>(C)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "D(" << row << "," << col << ") = " << *col_it << " ==> " << D(row,col) << " == " << (b-c) );
            // double-check: *col_it == D(row,col) == B(row,col)-C(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, D(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, b-c, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_op_prod )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Generalized Diagonal * Generalized Diagonal" );

    typedef double value_type;
    typedef ublas::matrix<value_type> adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef ublas::matrix<value_type> dense_matrix_type;
    //typedef ublasx::generalized_diagonal_matrix<value_type> result_matrix_type; // DON'T WORK
    typedef dense_matrix_type result_matrix_type;

    adaptee_matrix_type A(5, 4);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    adaptee_matrix_type B(4, 3);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751;

    adapter_matrix_type C(A, -2);
    adapter_matrix_type D(B, 1);

    result_matrix_type E = ublas::prod(C, D);

    //result_matrix_type T(5, 3, -1); // To be used when result is a banded matrix --> DON'T WORK
    result_matrix_type T(5, 3, 0);

    /* 0 */              /* 0 */               /* 0 */
    T(1,0) = 0.00000000; /* 0 */               /* 0 */
    /* 0 */              T(2,1) = -0.10712910; /* 0 */
    /* 0 */              /* 0 */               T(3,2) = -1.90829364;
    /* 0 */              /* 0 */               /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );
    BOOST_UBLASX_DEBUG_TRACE( "D " << D );
    BOOST_UBLASX_DEBUG_TRACE( "E " << E );

    BOOST_UBLASX_TEST_CHECK( E.size1() == T.size1() );
    BOOST_UBLASX_TEST_CHECK( E.size2() == T.size2() );
    for (
        result_matrix_type::const_iterator1 row_it = E.begin1();
        row_it != E.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const result_matrix_type::value_type t = static_cast<result_matrix_type const&>(T)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "E(" << row << "," << col << ") = " << *col_it << " ==> " << E(row,col) << " == " << t );
            // double-check: *col_it == D(row,col) == B(row,col)+C(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, E(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, t, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_op_element_prod_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Generalized Diagonal .* Dense" );

    typedef double value_type;
    typedef ublas::matrix<value_type> dense_matrix_type;
    typedef dense_matrix_type adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef dense_matrix_type result_matrix_type;

    const std::size_t nr = 5;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    adaptee_matrix_type B(nr, nc);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;


    adapter_matrix_type C(A, -2);

    result_matrix_type D = ublas::element_prod(C, B);

    result_matrix_type T(5, 4, 0);

    /* 0 */               /* 0 */               /* 0 */               /* 0 */
    /* 0 */               /* 0 */               /* 0 */               /* 0 */
    T(2,0) = -0.36972546; /* 0 */               /* 0 */               /* 0 */
    /* 0 */               T(3,1) = -1.44531748; /* 0 */               /* 0 */
    /* 0 */               /* 0 */               T(4,2) = -4.79150750; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );
    BOOST_UBLASX_DEBUG_TRACE( "D " << D );

    BOOST_UBLASX_TEST_CHECK( D.size1() == T.size1() );
    BOOST_UBLASX_TEST_CHECK( D.size2() == T.size2() );
    //BOOST_UBLASX_TEST_CHECK( D.offset() == T.offset() );
    for (
        result_matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const result_matrix_type::value_type t = static_cast<result_matrix_type const&>(T)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "E(" << row << "," << col << ") = " << *col_it << " ==> " << D(row,col) << " == " << t );
            // double-check: *col_it == D(row,col) == T(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, D(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, t, TOL );
        }
    }
}


BOOST_UBLASX_TEST_DEF( test_gda_op_element_div_dense )
{
    BOOST_UBLASX_DEBUG_TRACE( "Generalized Diagonal Adaptor -- Operations -- Generalized Diagonal ./ Dense" );

    typedef double value_type;
    typedef ublas::matrix<value_type> dense_matrix_type;
    typedef dense_matrix_type adaptee_matrix_type;
    typedef ublasx::generalized_diagonal_adaptor<adaptee_matrix_type> adapter_matrix_type;
    typedef dense_matrix_type result_matrix_type;

    const std::size_t nr = 5;
    const std::size_t nc = 4;

    adaptee_matrix_type A(nr, nc);

    A(0,0) = -0.57; A(0,1) = -1.93; A(0,2) =  2.30; A(0,3) = -1.93;
    A(1,0) = -1.28; A(1,1) =  1.08; A(1,2) =  0.24; A(1,3) =  0.64;
    A(2,0) = -0.39; A(2,1) = -0.31; A(2,2) =  0.40; A(2,3) = -0.66;
    A(3,0) =  0.25; A(3,1) = -2.14; A(3,2) = -0.35; A(3,3) =  0.08;
    A(4,0) = -4.60; A(4,1) = -7.04; A(4,2) = -3.89; A(4,3) = -7.66;

    adaptee_matrix_type B(nr, nc);

    B(0,0) = 0.555950; B(0,1) = 0.274690; B(0,2) = 0.540605; B(0,3) = 0.798938;
    B(1,0) = 0.108929; B(1,1) = 0.830123; B(1,2) = 0.891726; B(1,3) = 0.895283;
    B(2,0) = 0.948014; B(2,1) = 0.973234; B(2,2) = 0.216504; B(2,3) = 0.883152;
    B(3,0) = 0.023787; B(3,1) = 0.675382; B(3,2) = 0.231751; B(3,3) = 0.450332;
    B(4,0) = 1.023787; B(4,1) = 1.675382; B(4,2) = 1.231751; B(4,3) = 1.450332;


    adapter_matrix_type C(A, -2);

    result_matrix_type D = ublas::element_div(C, B);

    result_matrix_type T(5, 4, 0);

    /* 0 */                /* 0 */               /* 0 */               /* 0 */
    /* 0 */                /* 0 */               /* 0 */               /* 0 */
    T(2,0) = -0.411386330; /* 0 */               /* 0 */               /* 0 */
    /* 0 */                T(3,1) = -3.16857719; /* 0 */               /* 0 */
    /* 0 */                /* 0 */               T(4,2) = -3.15810838; /* 0 */

    BOOST_UBLASX_DEBUG_TRACE( "A " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B " << B );
    BOOST_UBLASX_DEBUG_TRACE( "C " << C );
    BOOST_UBLASX_DEBUG_TRACE( "D " << D );

    BOOST_UBLASX_TEST_CHECK( D.size1() == T.size1() );
    BOOST_UBLASX_TEST_CHECK( D.size2() == T.size2() );
    //BOOST_UBLASX_TEST_CHECK( D.offset() == T.offset() );
    for (
        result_matrix_type::const_iterator1 row_it = D.begin1();
        row_it != D.end1();
        ++row_it
    ) {
        for (
            result_matrix_type::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it
        ) {
            result_matrix_type::size_type row(col_it.index1());
            result_matrix_type::size_type col(col_it.index2());

            const result_matrix_type::value_type t = static_cast<result_matrix_type const&>(T)(row,col); // FIXME: This type of cast is very boring!!

            BOOST_UBLASX_DEBUG_TRACE( "D(" << row << "," << col << ") = " << *col_it << " ==> " << D(row,col) << " == " << t );
            // double-check: *col_it == D(row,col) == T(row,col)
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, D(row,col), TOL );
            BOOST_UBLASX_TEST_CHECK_CLOSE( *col_it, t, TOL );
        }
    }
}


//@} Generalized Diagonal Adaptor -- Matrix operations /////////////////////////


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: Generalized Diagonal Matrix and Adaptor classes");

    BOOST_UBLASX_TEST_BEGIN();

    // Generalized Diagonal Matrix -- Matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal );

    // Generalized Diagonal Matrix -- Column-major matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_col_major );

    // Generalized Diagonal Matrix -- Rectangular matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up4_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up5_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_up6_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_hrect_low3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low4_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low5_diagonal );
    BOOST_UBLASX_TEST_DO( test_gdm_vrect_low6_diagonal );

    // Generalized Diagonal Matrix -- Matrix row-by-column iteration tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_row_col_iteration );

    // Generalized Diagonal Matrix -- Matrix column-by-row iteration tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_col_row_iteration );

    // Generalized Diagonal Matrix -- Matrix copy-construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_copy );

    // Generalized Diagonal Matrix -- Matrix matrix-copy-construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_mat_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low4_diagonal_mat_copy );

    // Generalized Diagonal Matrix -- Matrix vector-copy-construction tests
    BOOST_UBLASX_TEST_DO( test_gdm_main_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up1_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up2_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_up3_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low1_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low2_diagonal_vec_copy );
    BOOST_UBLASX_TEST_DO( test_gdm_low3_diagonal_vec_copy );

    // Generalized Diagonal Matrix -- Matrix operations
    BOOST_UBLASX_TEST_DO( test_gdm_op_transpose );
    BOOST_UBLASX_TEST_DO( test_gdm_op_sum_dense );
    BOOST_UBLASX_TEST_DO( test_gdm_op_diff_dense );
    BOOST_UBLASX_TEST_DO( test_gdm_op_prod );
    BOOST_UBLASX_TEST_DO( test_gdm_op_prod_bis );
    BOOST_UBLASX_TEST_DO( test_gdm_op_element_prod_dense );
    BOOST_UBLASX_TEST_DO( test_gdm_op_element_div_dense );

    // Generalized Diagonal Adaptor -- Matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal );

    // Generalized Diagonal Adaptor -- Column-major matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_col_major );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_col_major );

    // Generalized Diagonal Adaptor -- Rectangular matrix construction tests
    BOOST_UBLASX_TEST_DO( test_gda_hrect_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up4_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up5_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_up6_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_hrect_low3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_main_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_up1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_up2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_up3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low1_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low2_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low3_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low4_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low5_diagonal );
    BOOST_UBLASX_TEST_DO( test_gda_vrect_low6_diagonal );

    // Generalized Diagonal Adaptor -- Matrix row-by-column iteration tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up4_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_hrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_vrect_row_col_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low4_diagonal_vrect_row_col_iteration );

    // Generalized Diagonal Adaptor -- Matrix column-by-row iteration tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up4_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_hrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_vrect_col_row_iteration );
    BOOST_UBLASX_TEST_DO( test_gda_low4_diagonal_vrect_col_row_iteration );

    // Generalized Diagonal Adaptor -- Matrix copy-construction tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_copy );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_copy );

    // Generalized Diagonal Adaptor -- Matrix copy-assignement tests
    BOOST_UBLASX_TEST_DO( test_gda_main_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_up1_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_up2_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_up3_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_low1_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_low2_diagonal_assign );
    BOOST_UBLASX_TEST_DO( test_gda_low3_diagonal_assign );

    // Generalized Diagonal Adaptor -- Matrix operations
    BOOST_UBLASX_TEST_DO( test_gda_op_transpose );
    BOOST_UBLASX_TEST_DO( test_gda_op_sum_dense );
    BOOST_UBLASX_TEST_DO( test_gda_op_diff_dense );
    BOOST_UBLASX_TEST_DO( test_gda_op_prod );
    BOOST_UBLASX_TEST_DO( test_gda_op_element_prod_dense );
    BOOST_UBLASX_TEST_DO( test_gda_op_element_div_dense );

    BOOST_UBLASX_TEST_END();
}
