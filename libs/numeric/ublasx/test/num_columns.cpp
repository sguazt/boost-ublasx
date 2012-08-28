/**
 * \file libs/numeric/ublasx/num_columns.cpp
 *
 * \brief Test suite for the \c num_columns operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	BOOST_UBLASX_DEBUG_TRACE( "num_columns(A) = " << ublasx::num_columns(A) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(A) == A.size2() );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	BOOST_UBLASX_DEBUG_TRACE( "num_columns(A) = " << ublasx::num_columns(A) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(A) == A.size2() );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	BOOST_UBLASX_DEBUG_TRACE( "num_columns(A') = " << ublasx::num_columns(ublas::trans(A)) << " ==> " << ublas::trans(A).size2() );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ublas::trans(A)) == ublas::trans(A).size2() );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ublas::matrix_reference<matrix_type> matrix_reference_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	BOOST_UBLASX_DEBUG_TRACE( "num_columns(reference(A)) = " << ublasx::num_columns(matrix_reference_type(A)) << " ==> " << matrix_reference_type(A).size2() );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(matrix_reference_type(A)) == matrix_reference_type(A).size2() );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'num_columns' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
	BOOST_UBLASX_TEST_DO( test_col_major_matrix_container );
	BOOST_UBLASX_TEST_DO( test_matrix_expression );
	BOOST_UBLASX_TEST_DO( test_matrix_reference );

	BOOST_UBLASX_TEST_END();
}
