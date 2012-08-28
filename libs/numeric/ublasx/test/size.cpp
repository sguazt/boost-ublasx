/**
 * \file libs/numeric/ublasx/test/size.cpp
 *
 * \brief Test the \c size operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */


#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_vector_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Container" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;

	vector_type v(5);

	v(0) = 0.555950;
	v(1) = 0.108929;
	v(2) = 0.948014;
	v(3) = 0.023787;
	v(4) = 1.023787;


	// size(v)
	BOOST_UBLASX_DEBUG_TRACE( "size(v) = " << ublasx::size(v) << " ==> " << v.size() );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(v) == v.size() );

	// size<1>(v)
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(v) = " << (ublasx::size<1>(v)) << " ==> " << v.size() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(v) == v.size()) );

	// [NOT_COMPILE]: this should *correctly* cause a compilation error
	// size<2>(v)
	//BOOST_UBLASX_DEBUG_TRACE( "size<2>(v) = " << (ublasx::size<vector_type,2>(v)) << " ==> " << v.size() );
	//BOOST_UBLASX_TEST_CHECK( (ublasx::size<2>(v) == v.size()) );
	// [/NOT_COMPILE]
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;

	vector_type v(5);

	v(0) = 0.555950;
	v(1) = 0.108929;
	v(2) = 0.948014;
	v(3) = 0.023787;
	v(4) = 1.023787;


	// size(-v)
	BOOST_UBLASX_DEBUG_TRACE( "size(-v) = " << ublasx::size(-v) << " ==> " << (-v).size() );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(-v) == (-v).size() );

	// size<1>(-v)
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(-v) = " << (ublasx::size<1>(-v)) << " ==> " << (-v).size() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(-v) == (-v).size()) );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector_reference<vector_type> vector_reference_type;

	vector_type v(5);

	v(0) = 0.555950;
	v(1) = 0.108929;
	v(2) = 0.948014;
	v(3) = 0.023787;
	v(4) = 1.023787;


	// size(reference(v)
	BOOST_UBLASX_DEBUG_TRACE( "size(reference(v)) = " << ublasx::size(vector_reference_type(v)) << " ==> " << vector_reference_type(v).size() );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(vector_reference_type(v)) == vector_reference_type(v).size() );

	// size<1>(reference(v))
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(reference(v)) = " << (ublasx::size<1>(vector_reference_type(v))) << " ==> " << vector_reference_type(v).size() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(vector_reference_type(v)) == vector_reference_type(v).size()) );
}


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


	// [NOT_COMPILE]
	// size(A)
	//BOOST_UBLASX_DEBUG_TRACE( "size(A) = " << ublasx::size(A) << " ==> " << A.size1() );
	//BOOST_UBLASX_TEST_CHECK( ublasx::size(A) == A.size1() );
	// [/NOT_COMPILE]

	// size<1>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(A) = " << (ublasx::size<1>(A)) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(A) == A.size1()) );

	// size<2>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<2>(A) = " << (ublasx::size<2>(A)) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<2>(A) == A.size2()) );

	// size<major>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<major>(A) = " << (ublasx::size<ublasx::tag::major>(A)) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::major>(A) == A.size1()) );

	// size<minor>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<minor>(A) = " << (ublasx::size<ublasx::tag::minor>(A)) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::minor>(A) == A.size2()) );

	// size<leading>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<leading>(A) = " << (ublasx::size<ublasx::tag::leading>(A)) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::leading>(A) == A.size2()) );
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


	// size<1>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(A) = " << (ublasx::size<1>(A)) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(A) == A.size1()) );

	// size<2>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<2>(A) = " << (ublasx::size<2>(A)) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<2>(A) == A.size2()) );

	// size<major>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<major>(A) = " << (ublasx::size<ublasx::tag::major>(A)) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::major>(A) == A.size2()) );

	// size<minor>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<minor>(A) = " << (ublasx::size<ublasx::tag::minor>(A)) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::minor>(A) == A.size1()) );

	// size<leading>(A)
	BOOST_UBLASX_DEBUG_TRACE( "size<leading>(A) = " << (ublasx::size<ublasx::tag::leading>(A)) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::leading>(A) == A.size1()) );
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


	// size<1>(A')
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(A') = " << (ublasx::size<1>(ublas::trans(A))) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(ublas::trans(A)) == A.size2()) );

	// size<2>(A')
	BOOST_UBLASX_DEBUG_TRACE( "size<2>(A') = " << (ublasx::size<2>(ublas::trans(A))) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<2>(ublas::trans(A)) == A.size1()) );

	// size<major>(A') [A is row-major => A' column-major, and viceversa]
	BOOST_UBLASX_DEBUG_TRACE( "size<major>(A') = " << (ublasx::size<ublas::tag::major>(ublas::trans(A))) << " ==> " << A.size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::major>(ublas::trans(A)) == A.size1()) );

	// size<minor>(A')  [A is row-major => A' column-major, and viceversa]
	BOOST_UBLASX_DEBUG_TRACE( "size<minor>(A') = " << (ublasx::size<ublasx::tag::minor>(ublas::trans(A))) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::minor>(ublas::trans(A)) == A.size2()) );

	// size<leading>(A')  [A row-major => A' column-major, and viceversa]
	BOOST_UBLASX_DEBUG_TRACE( "size<leading>(A') = " << (ublasx::size<ublasx::tag::leading>(ublas::trans(A))) << " ==> " << A.size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::leading>(ublas::trans(A)) == A.size2()) );
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


	// size<1>(reference(A))
	BOOST_UBLASX_DEBUG_TRACE( "size<1>(reference(A)) = " << (ublasx::size<1>(matrix_reference_type(A))) << " ==> " << matrix_reference_type(A).size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<1>(matrix_reference_type(A)) == matrix_reference_type(A).size1()) );

	// size<2>(reference(A))
	BOOST_UBLASX_DEBUG_TRACE( "size<2>(reference(A)) = " << (ublasx::size<2>(matrix_reference_type(A))) << " ==> " << matrix_reference_type(A).size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<2>(matrix_reference_type(A)) == matrix_reference_type(A).size2()) );

	// size<major>(reference(A))
	BOOST_UBLASX_DEBUG_TRACE( "size<major>(reference(A) = " << (ublasx::size<ublasx::tag::major>(matrix_reference_type(A))) << " ==> " << matrix_reference_type(A).size1() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::major>(matrix_reference_type(A)) == matrix_reference_type(A).size1()) );

	// size<minor>(reference(A))
	BOOST_UBLASX_DEBUG_TRACE( "size<minor>(reference(A)) = " << (ublasx::size<ublasx::tag::minor>(matrix_reference_type(A))) << " ==> " << matrix_reference_type(A).size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::minor>(matrix_reference_type(A)) == matrix_reference_type(A).size2()) );

	// size<leading>(reference(A))
	BOOST_UBLASX_DEBUG_TRACE( "size<leading>(reference(A)) = " << (ublasx::size<ublasx::tag::leading>(matrix_reference_type(A))) << " ==> " << matrix_reference_type(A).size2() );
	BOOST_UBLASX_TEST_CHECK( (ublasx::size<ublasx::tag::leading>(matrix_reference_type(A)) == matrix_reference_type(A).size2()) );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'size' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_vector_container );
	BOOST_UBLASX_TEST_DO( test_vector_expression );
	BOOST_UBLASX_TEST_DO( test_vector_reference );
	BOOST_UBLASX_TEST_DO( test_row_major_matrix_container );
	BOOST_UBLASX_TEST_DO( test_col_major_matrix_container );
	BOOST_UBLASX_TEST_DO( test_matrix_expression );
	BOOST_UBLASX_TEST_DO( test_matrix_reference );

	BOOST_UBLASX_TEST_END();
}
