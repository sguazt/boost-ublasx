/**
 * \file libs/numeric/ublasx/test/cond.cpp
 *
 * \brief Test suite for the \c cond operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

//#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/symmetric.hpp>
//#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublasx/operation/cond.hpp>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include "libs/numeric/ublasx/test/utils.hpp"

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( norm_1_real_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_1(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_1(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_1(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_1(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_1(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_1(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_1(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_1(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_1(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_1(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_1(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_1(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_2_real_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Real Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 4;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 49026176493774648;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_real_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Real Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 4;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 49026176493774648;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_complex_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Complex Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 4;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 20994351988002824;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_complex_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Complex Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 4;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 20994351988002824;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_real_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Real Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type Well(nr,nc);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;
	Well(3,0) =  1; Well(3,1) =  2; Well(3,2) = -1;

	matrix_type Ill(nr,nc);
	Ill(0,0) =  1; Ill(0,1) =  2; Ill(0,2) =  3;
	Ill(1,0) =  4; Ill(1,1) =  5; Ill(1,2) =  6;
	Ill(2,0) =  7; Ill(2,1) =  8; Ill(2,2) =  9;
	Ill(3,0) = 10; Ill(3,1) = 11; Ill(3,2) = 12;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 3.41990480101429;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 14259982749169812;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_real_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Real Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type Well(nr,nc);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;
	Well(3,0) =  1; Well(3,1) =  2; Well(3,2) = -1;

	matrix_type Ill(nr,nc);
	Ill(0,0) =  1; Ill(0,1) =  2; Ill(0,2) =  3;
	Ill(1,0) =  4; Ill(1,1) =  5; Ill(1,2) =  6;
	Ill(2,0) =  7; Ill(2,1) =  8; Ill(2,2) =  9;
	Ill(3,0) = 10; Ill(3,1) = 11; Ill(3,2) = 12;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 3.41990480101429;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 14259982749169812;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_complex_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Complex Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type Well(nr,nc);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);
	Well(3,0) = value_type( 1, 0); Well(3,1) = value_type( 2,-1); Well(3,2) = value_type(-1, 2);

	matrix_type Ill(nr,nc);
	Ill(0,0) = value_type( 1,10); Ill(0,1) = value_type( 2,14); Ill(0,2) = value_type( 3,18);
	Ill(1,0) = value_type( 4,11); Ill(1,1) = value_type( 5,15); Ill(1,2) = value_type( 6,19);
	Ill(2,0) = value_type( 7,12); Ill(2,1) = value_type( 8,16); Ill(2,2) = value_type( 9,20);
	Ill(3,0) = value_type(10,13); Ill(3,1) = value_type(11,17); Ill(3,2) = value_type(12,21);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 3.67416702058981;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 37955084752566112;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_2_complex_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 2-Norm - Complex Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type Well(nr,nc);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);
	Well(3,0) = value_type( 1, 0); Well(3,1) = value_type( 2,-1); Well(3,2) = value_type(-1, 2);

	matrix_type Ill(nr,nc);
	Ill(0,0) = value_type( 1,10); Ill(0,1) = value_type( 2,14); Ill(0,2) = value_type( 3,18);
	Ill(1,0) = value_type( 4,11); Ill(1,1) = value_type( 5,15); Ill(1,2) = value_type( 6,19);
	Ill(2,0) = value_type( 7,12); Ill(2,1) = value_type( 8,16); Ill(2,2) = value_type( 9,20);
	Ill(3,0) = value_type(10,13); Ill(3,1) = value_type(11,17); Ill(3,2) = value_type(12,21);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 3.67416702058981;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_2(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = 37955084752566112;
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_2(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
}


BOOST_UBLASX_TEST_DEF( norm_inf_real_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: inf-Norm - Real Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_inf(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_inf(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_inf_real_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: inf-Norm - Real Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_inf(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_inf(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_inf_complex_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_inf(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_inf(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_inf_complex_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_inf(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	res = ublasx::cond_inf(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_inf_real_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_inf(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_inf_real_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_inf(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_inf_complex_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_inf(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_inf_complex_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_inf(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_real_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Real Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5.25;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_frobenius(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	//res = 456177073660509760;
	res = ublasx::cond_frobenius(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_real_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Real Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) =  2; Well(0,1) = -1; Well(0,2) =  0;
	Well(1,0) = -1; Well(1,1) =  3; Well(1,2) = -1;
	Well(2,0) =  0; Well(2,1) = -1; Well(2,2) =  2;

	matrix_type Ill(n,n);
	Ill(0,0) = 1; Ill(0,1) = 2; Ill(0,2) = 3;
	Ill(1,0) = 4; Ill(1,1) = 5; Ill(1,2) = 6;
	Ill(2,0) = 7; Ill(2,1) = 8; Ill(2,2) = 9;

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5.25;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_frobenius(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	//res = 456177073660509760;
	res = ublasx::cond_frobenius(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_complex_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-norm - Complex Square Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5.25;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_frobenius(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	//res = 54418634865903768;
	res = ublasx::cond_frobenius(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_complex_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Complex Square Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef real_type result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type Well(n,n);
	Well(0,0) = value_type( 2, 2); Well(0,1) = value_type(-1,-1); Well(0,2) = value_type( 0, 0);
	Well(1,0) = value_type(-1,-1); Well(1,1) = value_type( 3, 3); Well(1,2) = value_type(-1,-1);
	Well(2,0) = value_type( 0, 0); Well(2,1) = value_type(-1,-1); Well(2,2) = value_type( 2, 2);

	matrix_type Ill(n,n);
	Ill(0,0) = value_type(1,10); Ill(0,1) = value_type(2,13); Ill(0,2) = value_type(3,16);
	Ill(1,0) = value_type(4,11); Ill(1,1) = value_type(5,14); Ill(1,2) = value_type(6,17);
	Ill(2,0) = value_type(7,12); Ill(2,1) = value_type(8,15); Ill(2,2) = value_type(9,18);

	result_type res;
	result_type expect_res;

	// Well-conditioned matrix
	expect_res = 5.25;
	BOOST_UBLASX_DEBUG_TRACE("Well-conditioned Matrix = " << Well);
	res = ublasx::cond_frobenius(Well);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );

	// Ill-conditioned matrix
	expect_res = std::numeric_limits<result_type>::infinity();
	BOOST_UBLASX_DEBUG_TRACE("Ill-conditioned Matrix = " << Ill);
	//res = 54418634865903768;
	res = ublasx::cond_frobenius(Ill);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);
	BOOST_UBLASX_TEST_CHECK( std::isinf(res) );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_real_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Real Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_frobenius(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_real_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Real Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) =  1; A(0,1) =  2; A(0,2) =  3;
	A(1,0) =  4; A(1,1) =  5; A(1,2) =  6;
	A(2,0) =  7; A(2,1) =  8; A(2,2) =  9;
	A(3,0) = 10; A(3,1) = 11; A(3,2) = 12;

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_frobenius(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_complex_rectangular_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Complex Rectangular Dense Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_frobenius(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


BOOST_UBLASX_TEST_DEF( norm_frobenius_complex_rectangular_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: frobenius-Norm - Complex Rectangular Dense Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t nr = 4;
	const std::size_t nc = 3;

	matrix_type A(nr,nc);
	A(0,0) = value_type( 1,13); A(0,1) = value_type( 2,17); A(0,2) = value_type( 3,21);
	A(1,0) = value_type( 4,14); A(1,1) = value_type( 5,18); A(1,2) = value_type( 6,22);
	A(2,0) = value_type( 7,15); A(2,1) = value_type( 8,19); A(2,2) = value_type( 9,23);
	A(3,0) = value_type(10,16); A(3,1) = value_type(11,20); A(3,2) = value_type(12,24);

	bool res;

	// Condition number for rectangular matrix only available with the 2-norm
	BOOST_UBLASX_DEBUG_TRACE("Matrix = " << A);
	try
	{
		ublasx::cond_frobenius(A);
		res = false;
	}
	catch (...)
	{
		res = true;
	}
	BOOST_UBLASX_TEST_CHECK( res );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'cond' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( norm_1_real_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_rectangular_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_rectangular_dense_matrix_row_major );

	BOOST_UBLASX_TEST_DO( norm_2_real_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_2_real_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_2_complex_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_2_complex_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_2_real_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_2_real_rectangular_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_2_complex_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_2_complex_rectangular_dense_matrix_row_major );

	BOOST_UBLASX_TEST_DO( norm_inf_real_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_inf_real_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_inf_complex_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_inf_complex_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_inf_real_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_inf_real_rectangular_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_inf_complex_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_inf_complex_rectangular_dense_matrix_row_major );

	BOOST_UBLASX_TEST_DO( norm_frobenius_real_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_real_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_complex_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_complex_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_real_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_real_rectangular_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_complex_rectangular_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_frobenius_complex_rectangular_dense_matrix_row_major );

	BOOST_UBLASX_TEST_END();
}
