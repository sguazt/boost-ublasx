/**
 * \file libs/numeric/ublasx/test/pow.cpp
 *
 * \brief Test suite for the \c pow operation.
 *
 * Copyright (c) 2015, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
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


BOOST_UBLASX_TEST_DEF( test_real_matrix_positive_exponent )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Matrix - Positive Exponent" );

	typedef double value_type;
	typedef std::size_t size_type;
	typedef ublas::matrix<value_type> matrix_type;

	const size_type n = 2;
	const double exp = 3;

	matrix_type A(n,n);

	A(0,0) = 1; A(0,1) = 2;
	A(1,0) = 4; A(1,1) = 5;

	matrix_type R;
	matrix_type expect_R;

	R = ublasx::pow(A, exp);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

	expect_R = A;
	for (size_type i = 0; i < (exp-1); ++i)
	{
		expect_R = ublas::prod(expect_R, A);
	}

	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, n, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_real_matrix_negative_exponent )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Matrix - Negative Exponent" );

	typedef double value_type;
	typedef std::size_t size_type;
	typedef ublas::matrix<value_type> matrix_type;

	const size_type n = 2;
	const double exp = -3;

	matrix_type A(n,n);

	A(0,0) = 1; A(0,1) = 2;
	A(1,0) = 4; A(1,1) = 5;

	matrix_type R;
	matrix_type expect_R;

	R = ublasx::pow(A, exp);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

	matrix_type invA(n,n);
	invA(0,0) = -5.0/3.0; invA(0,1) =  2.0/3.0;
	invA(1,0) =  4.0/3.0; invA(1,1) = -1.0/3.0;

	expect_R = invA;
	for (size_type i = 0; i < (-exp-1); ++i)
	{
		expect_R = ublas::prod(expect_R, invA);
	}


	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, n, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_real_matrix_zero_exponent )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real Matrix - Zero Exponent" );

	typedef double value_type;
	typedef std::size_t size_type;
	typedef ublas::matrix<value_type> matrix_type;

	const size_type n = 2;
	const double exp = 0;

	matrix_type A(n,n);

	A(0,0) = 1; A(0,1) = 2;
	A(1,0) = 4; A(1,1) = 5;

	matrix_type R;
	matrix_type expect_R = ublas::identity_matrix<value_type>(n);

	R = ublasx::pow(A, exp);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, n, n, tol );
}

BOOST_UBLASX_TEST_DEF( test_complex_matrix_positive_exponent )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex Matrix - Positive Exponent" );

	typedef std::complex<double> in_value_type;
	typedef in_value_type out_value_type;
	typedef std::size_t size_type;
	typedef ublas::matrix<in_value_type> in_matrix_type;
	typedef ublas::matrix<out_value_type> out_matrix_type;

	const size_type n = 2;
	const int exp = 3;

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(1,2); A(0,1) = in_value_type(2,3);
	A(1,0) = in_value_type(4,5); A(1,1) = in_value_type(5,6);

	out_matrix_type R;
	out_matrix_type expect_R;

	R = ublasx::pow(A, exp);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "pow(A, " << exp << ") = " << R );

	expect_R = A;
	for (size_type i = 0; i < (exp-1); ++i)
	{
		expect_R = ublas::prod(expect_R, A);
	}

	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, n, n, tol );
}


int main()
{

	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'pow' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_real_matrix_positive_exponent );
	BOOST_UBLASX_TEST_DO( test_real_matrix_negative_exponent );
	BOOST_UBLASX_TEST_DO( test_real_matrix_zero_exponent );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_positive_exponent );

	BOOST_UBLASX_TEST_END();
}
