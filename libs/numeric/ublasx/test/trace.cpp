/**
 * \file libs/numeric/ublasx/test/trace.cpp
 *
 * \brief Test suite for the \c trace operation.
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
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/trace.hpp>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( real_square_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Square Matrix - Column-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
	A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( real_square_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Square Matrix - Row-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
	A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Square Matrix - Column-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type(1,-5); A(0,1) = value_type(2, 3); A(0,2) = value_type(3,-9);
	A(1,0) = value_type(4, 4); A(1,1) = value_type(5,-2); A(1,2) = value_type(6, 8);
	A(2,0) = value_type(7,-3); A(2,1) = value_type(8, 1); A(2,2) = value_type(9,-7);

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Square Matrix - Row-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type(1,-5); A(0,1) = value_type(2, 3); A(0,2) = value_type(3,-9);
	A(1,0) = value_type(4, 4); A(1,1) = value_type(5,-2); A(1,2) = value_type(6, 8);
	A(2,0) = value_type(7,-3); A(2,1) = value_type(8, 1); A(2,2) = value_type(9,-7);

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( real_recth_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Rectangular Horizontal Matrix - Column-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;
	const std::size_t m = 5;

	matrix_type A(n,m);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4; A(0,4) =  5;
	A(1,0) =  6; A(1,1) =  7; A(1,2) =  8; A(1,3) =  9; A(1,4) = 10;
	A(2,0) = 11; A(2,1) = 12; A(2,2) = 13; A(2,3) = 14; A(2,4) = 15;

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( real_recth_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Rectangular Horizontal Matrix - Row-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;
	const std::size_t m = 5;

	matrix_type A(n,m);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4; A(0,4) =  5;
	A(1,0) =  6; A(1,1) =  7; A(1,2) =  8; A(1,3) =  9; A(1,4) = 10;
	A(2,0) = 11; A(2,1) = 12; A(2,2) = 13; A(2,3) = 14; A(2,4) = 15;


	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_recth_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Rectangular Horizontal Matrix - Column-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;
	const std::size_t m = 5;

	matrix_type A(n,m);
	A(0,0) = value_type( 1,-11); A(0,1) = value_type( 2, 12); A(0,2) = value_type( 3,-13); A(0,3) = value_type( 4, 14); A(0,4) = value_type( 5, -15);
	A(1,0) = value_type( 6,  6); A(1,1) = value_type( 7,- 7); A(1,2) = value_type( 8,  8); A(1,3) = value_type( 9,- 9); A(1,4) = value_type(10,  10);
	A(2,0) = value_type(11,- 1); A(2,1) = value_type(12,  2); A(2,2) = value_type(13,- 3); A(2,3) = value_type(14,  4); A(2,4) = value_type(15, - 5);

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_recth_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Rectangular Horizontal Matrix - Row-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;
	const std::size_t m = 5;

	matrix_type A(n,m);
	A(0,0) = value_type( 1,-11); A(0,1) = value_type( 2, 12); A(0,2) = value_type( 3,-13); A(0,3) = value_type( 4, 14); A(0,4) = value_type( 5,-15);
	A(1,0) = value_type( 6,  6); A(1,1) = value_type( 7,- 7); A(1,2) = value_type( 8,  8); A(1,3) = value_type( 9,- 9); A(1,4) = value_type(10, 10);
	A(2,0) = value_type(11,- 1); A(2,1) = value_type(12,  2); A(2,2) = value_type(13,- 3); A(2,3) = value_type(14,  4); A(2,4) = value_type(15,- 5);

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( real_rectv_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Rectangular Vertical Matrix - Column-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 5;
	const std::size_t m = 3;

	matrix_type A(n,m);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;
	A(4,0) = 13; A(4,1) = 14; A(4,2) = 15;

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( real_rectv_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Type - Rectangular Vertical Matrix - Row-Major");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 5;
	const std::size_t m = 3;

	matrix_type A(n,m);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;
	A(4,0) = 13; A(4,1) = 14; A(4,2) = 15;

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_rectv_col_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Rectangular Vertical Matrix - Column-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 5;
	const std::size_t m = 3;

	matrix_type A(n,m);
	A(0,0) = value_type( 1,-13); A(0,1) = value_type( 2, 14); A(0,2) = value_type( 3,-15);
	A(1,0) = value_type( 4, 10); A(1,1) = value_type( 5,-11); A(1,2) = value_type( 6, 12);
	A(2,0) = value_type( 7,- 7); A(2,1) = value_type( 8,  8); A(2,2) = value_type( 9,- 9);
	A(3,0) = value_type(10,  4); A(3,1) = value_type(11,- 5); A(3,2) = value_type(12,  6);
	A(4,0) = value_type(13,- 1); A(4,1) = value_type(14,  2); A(4,2) = value_type(15,  3);

	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


BOOST_UBLASX_TEST_DEF( complex_rectv_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Type - Vertical Horizontal Matrix - Row-Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 5;
	const std::size_t m = 3;

	matrix_type A(n,m);
	A(0,0) = value_type( 1,-13); A(0,1) = value_type( 2, 14); A(0,2) = value_type( 3,-15);
	A(1,0) = value_type( 4, 10); A(1,1) = value_type( 5,-11); A(1,2) = value_type( 6, 12);
	A(2,0) = value_type( 7,- 7); A(2,1) = value_type( 8,  8); A(2,2) = value_type( 9,- 9);
	A(3,0) = value_type(10,  4); A(3,1) = value_type(11,- 5); A(3,2) = value_type(12,  6);
	A(4,0) = value_type(13,- 1); A(4,1) = value_type(14,  2); A(4,2) = value_type(15,  3);


	value_type expect_tr = A(0,0)+A(1,1)+A(2,2);
	value_type tr = ublasx::trace(A);
	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "tr(A) = " << tr );
	BOOST_UBLASX_TEST_CHECK_CLOSE( tr, expect_tr, tol );
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( real_square_col_major );
	BOOST_UBLASX_TEST_DO( real_square_row_major );
	BOOST_UBLASX_TEST_DO( complex_square_col_major );
	BOOST_UBLASX_TEST_DO( complex_square_row_major );
	BOOST_UBLASX_TEST_DO( real_recth_col_major );
	BOOST_UBLASX_TEST_DO( real_recth_row_major );
	BOOST_UBLASX_TEST_DO( complex_recth_col_major );
	BOOST_UBLASX_TEST_DO( complex_recth_row_major );
	BOOST_UBLASX_TEST_DO( real_rectv_col_major );
	BOOST_UBLASX_TEST_DO( real_rectv_row_major );
	BOOST_UBLASX_TEST_DO( complex_rectv_col_major );
	BOOST_UBLASX_TEST_DO( complex_rectv_row_major );

	BOOST_UBLASX_TEST_END();
}
