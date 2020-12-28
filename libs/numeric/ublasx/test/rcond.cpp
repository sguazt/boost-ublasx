/**
 * \file libs/numeric/ublasx/test/rcond.cpp
 *
 * \brief Test suite for the \c rcond operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

//ATTENTION: test fails
//TODO: fix it

#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublasx/operation/rcond.hpp>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"

namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( norm_1_real_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Square Dense Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
	A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 1.5420e-18; // Computed with matlab R2008a, octave 3.2.4 and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-18 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Square Dense Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
	A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 1.5420e-18; // Computed with matlab R2008a, octave 3.2.4 and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-18 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_upper_triangular_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Upper Triangular Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::upper,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	            A(1,1) = 4; A(1,2) = 5;
	                        A(2,2) = 6;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.07142857; // Computed with matlab R2008a, octave 3.2.4 and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_upper_triangular_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Upper Triangular Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::upper,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
	            A(1,1) = 4; A(1,2) = 5;
	                        A(2,2) = 6;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.07142857; // Computed with matlab R2008a, octave 3.2.4 and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_lower_triangular_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Lower Triangular Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::lower,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1;
	A(1,0) = 2; A(1,1) = 3;
	A(2,0) = 4; A(2,1) = 5; A(2,2) = 6;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.0703125; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_lower_triangular_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Lower Triangular Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::lower,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = 1;
	A(1,0) = 2; A(1,1) = 3;
	A(2,0) = 4; A(2,1) = 5; A(2,2) = 6;

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.0703125; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_banded_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Banded Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::banded_matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n,1,2);
	A(0,0) = -0.23; A(0,1) = 2.54; A(0,2) = -3.66;         /* 0 */ // 1st row
	A(1,0) = -6.98; A(1,1) = 2.46; A(1,2) = -2.73; A(1,3) = -2.13; // 2nd row
	        /* 0 */ A(2,1) = 2.56; A(2,2) =  2.46; A(2,3) =  4.07; // 3rd row
	        /* 0 */        /* 0 */ A(3,2) = -4.78; A(3,3) = -3.82; // 4th row


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.017728; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_banded_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Banded Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::banded_matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n,1,2);
	A(0,0) = -0.23; A(0,1) = 2.54; A(0,2) = -3.66;         /* 0 */ // 1st row
	A(1,0) = -6.98; A(1,1) = 2.46; A(1,2) = -2.73; A(1,3) = -2.13; // 2nd row
	        /* 0 */ A(2,1) = 2.56; A(2,2) =  2.46; A(2,3) =  4.07; // 3rd row
	        /* 0 */        /* 0 */ A(3,2) = -4.78; A(3,3) = -3.82; // 4th row

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.017728; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_lower_symmetric_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Lower Symmetric Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::symmetric_matrix<value_type,ublas::lower,ublas::column_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) =  2.07;   /* 3.87*/      /* 4.20*/      /*-1.15*/
	A(1,0) =  3.87; A(1,1) = -0.21;  /* 1.87*/      /* 0.63*/
	A(2,0) =  4.20; A(2,1) = 1.87; A(2,2) = 1.15;   /* 2.06*/
	A(3,0) = -1.15; A(3,1) = 0.63; A(3,2) = 2.06; A(3,3) = -1.81;


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.01321232; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_lower_symmetric_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Lower Symmetric Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::symmetric_matrix<value_type,ublas::lower,ublas::row_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) =  2.07;   /* 3.87*/      /* 4.20*/      /*-1.15*/
	A(1,0) =  3.87; A(1,1) = -0.21;  /* 1.87*/      /* 0.63*/
	A(2,0) =  4.20; A(2,1) = 1.87; A(2,2) = 1.15;   /* 2.06*/
	A(3,0) = -1.15; A(3,1) = 0.63; A(3,2) = 2.06; A(3,3) = -1.81;


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.01321232; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_upper_symmetric_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Upper Symmetric Matrix - Column Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::symmetric_matrix<value_type,ublas::upper,ublas::column_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) =  2.07; A(0,1) =  3.87; A(0,2) = 4.20; A(0,3) = -1.15;
	A(1,0) =  3.87; A(1,1) = -0.21; A(1,2) = 1.87; A(1,3) =  0.63;
	  /* 4.20*/       /* 1.87*/     A(2,2) = 1.15; A(2,3) =  2.06;
	  /*-1.15*/       /* 0.63*/       /* 2.06*/    A(3,3) = -1.81;


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.01321232; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_upper_symmetric_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Upper Symmetric Matrix - Row Major");

	typedef double value_type;
	typedef double result_type;
	typedef ublas::symmetric_matrix<value_type,ublas::upper,ublas::row_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) =  2.07; A(0,1) =  3.87; A(0,2) = 4.20; A(0,3) = -1.15;
	A(1,0) =  3.87; A(1,1) = -0.21; A(1,2) = 1.87; A(1,3) =  0.63;
	  /* 4.20*/       /* 1.87*/     A(2,2) = 1.15; A(2,3) =  2.06;
	  /*-1.15*/       /* 0.63*/       /* 2.06*/    A(3,3) = -1.81;


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.01321232; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_lower_hermitian_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Lower Hermitian Matrix - Column Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::hermitian_matrix<value_type,ublas::lower,ublas::column_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) = value_type(-1.36, 0.00);
	A(1,0) = value_type( 1.58,-0.90); A(1,1) = value_type(-8.87, 0.00);
	A(2,0) = value_type( 2.21, 0.21); A(2,1) = value_type(-1.84, 0.03); A(2,2) = value_type(-4.63, 0.00);
	A(3,0) = value_type( 3.91,-1.50); A(3,1) = value_type(-1.78,-1.18); A(3,2) = value_type( 0.11,-0.11); A(3,3) = value_type(-1.84, 0.00);


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.028470197472865; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_real_lower_hermitian_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Lower Hermitian Matrix - Row Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::hermitian_matrix<value_type,ublas::lower,ublas::row_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n);
	A(0,0) = value_type(-1.36, 0.00);
	A(1,0) = value_type( 1.58,-0.90); A(1,1) = value_type(-8.87, 0.00);
	A(2,0) = value_type( 2.21, 0.21); A(2,1) = value_type(-1.84, 0.03); A(2,2) = value_type(-4.63, 0.00);
	A(3,0) = value_type( 3.91,-1.50); A(3,1) = value_type(-1.78,-1.18); A(3,2) = value_type( 0.11,-0.11); A(3,3) = value_type(-1.84, 0.00);


	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.028470197472865; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_square_dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Column Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2); A(0,1) = value_type( 3, 4); A(0,2) = value_type( 5, 6);
	A(1,0) = value_type( 7, 8); A(1,1) = value_type( 9,10); A(1,2) = value_type(11,12);
	A(2,0) = value_type(13,14); A(2,1) = value_type(15,16); A(2,2) = value_type(17,18);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	//expect_res = 4.3758e-18; // Computed with matlab R2008a and octave 3.2.4 and R 2.11.1
	expect_res = 1.136408e-18; // Computed with R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-18 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_square_dense_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Square Dense Matrix - Row Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2); A(0,1) = value_type( 3, 4); A(0,2) = value_type( 5, 6);
	A(1,0) = value_type( 7, 8); A(1,1) = value_type( 9,10); A(1,2) = value_type(11,12);
	A(2,0) = value_type(13,14); A(2,1) = value_type(15,16); A(2,2) = value_type(17,18);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	//expect_res = 4.375805911678970e-18; // Computed with matlab R2008a and octave 3.2.4
	expect_res = 1.136408e-18; // Computed with R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-18 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_upper_triangular_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Upper Triangular Matrix - Column Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::upper,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2); A(0,1) = value_type( 3, 4); A(0,2) = value_type( 5, 6);
	                            A(1,1) = value_type( 7, 8); A(1,2) = value_type( 9,10);
	                                                        A(2,2) = value_type(11,12);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.059560668674847; // Computed with matlab R2008a, octave 3.2.4
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_upper_triangular_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Real Upper Triangular Matrix - Row Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::upper,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2); A(0,1) = value_type( 3, 4); A(0,2) = value_type( 5, 6);
	                            A(1,1) = value_type( 7, 8); A(1,2) = value_type( 9,10);
	                                                        A(2,2) = value_type(11,12);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.059560668674847; // Computed with matlab R2008a and octave 3.2.4
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_lower_triangular_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Lower Triangular Matrix - Column Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::lower,ublas::column_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2);
	A(1,0) = value_type( 3, 4); A(1,1) = value_type( 5, 6);
	A(2,0) = value_type( 7, 8); A(2,1) = value_type( 9,10); A(2,2) = value_type(11,12);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.059544953702088; // Computed with matlab R2008a, octave 3.2.4
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_lower_triangular_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex lower Triangular Matrix - Row Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::triangular_matrix<value_type,ublas::lower,ublas::row_major> matrix_type;

	const std::size_t n = 3;

	matrix_type A(n,n);
	A(0,0) = value_type( 1, 2);
	A(1,0) = value_type( 3, 4); A(1,1) = value_type( 5, 6);
	A(2,0) = value_type( 7, 8); A(2,1) = value_type( 9,10); A(2,2) = value_type(11,12);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.059544953702088; // Computed with matlab R2008a and octave 3.2.4
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-7 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_banded_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Banded Matrix - Column Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::banded_matrix<value_type,ublas::column_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n,1,2);
	A(0,0) = value_type(-1.65, 2.26); A(0,1) = value_type(-2.05,-0.85); A(0,2) = value_type( 0.97,-2.84);
	A(1,0) = value_type( 0.00, 6.30); A(1,1) = value_type(-1.48,-1.75); A(1,2) = value_type(-3.99, 4.01); A(1,3) = value_type( 0.59,-0.48);
			                          A(2,1) = value_type(-0.77, 2.83); A(2,2) = value_type(-1.06, 1.94); A(2,3) = value_type( 3.33,-1.04);
							                                            A(3,2) = value_type( 4.48,-1.09); A(3,3) = value_type(-0.46,-1.72);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.009594414793018; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


BOOST_UBLASX_TEST_DEF( norm_1_complex_banded_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: 1-Norm - Complex Banded Matrix - Row Major");

	typedef std::complex<double> value_type;
	typedef double result_type;
	typedef ublas::banded_matrix<value_type,ublas::row_major> matrix_type;

	const std::size_t n = 4;

	matrix_type A(n,n,1,2);
	A(0,0) = value_type(-1.65, 2.26); A(0,1) = value_type(-2.05,-0.85); A(0,2) = value_type( 0.97,-2.84);
	A(1,0) = value_type( 0.00, 6.30); A(1,1) = value_type(-1.48,-1.75); A(1,2) = value_type(-3.99, 4.01); A(1,3) = value_type( 0.59,-0.48);
			                          A(2,1) = value_type(-0.77, 2.83); A(2,2) = value_type(-1.06, 1.94); A(2,3) = value_type( 3.33,-1.04);
							                                            A(3,2) = value_type( 4.48,-1.09); A(3,3) = value_type(-0.46,-1.72);

	result_type res;
	result_type expect_res;

	res = ublasx::rcond(A);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("res = " << res);

	expect_res = 0.009594414793018; // Computed with matlab R2008a, octave 3.2.4, and R 2.11.1
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, 1.0e-6 );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'rcond' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( norm_1_real_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_upper_triangular_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_upper_triangular_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_lower_triangular_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_lower_triangular_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_banded_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_banded_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_lower_symmetric_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_lower_symmetric_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_upper_symmetric_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_upper_symmetric_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_square_dense_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_square_dense_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_upper_triangular_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_upper_triangular_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_lower_triangular_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_lower_triangular_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_banded_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_banded_matrix_row_major );
	BOOST_UBLASX_TEST_DO( norm_1_complex_lower_hermitian_matrix_column_major );
	BOOST_UBLASX_TEST_DO( norm_1_real_lower_hermitian_matrix_row_major );

	BOOST_UBLASX_TEST_END();
}
