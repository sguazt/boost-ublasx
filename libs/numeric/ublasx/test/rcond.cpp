/**
 * \file libs/numeric/ublasx/test/rcond.cpp
 *
 * \brief Test suite for the \c rcond operation.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

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


static const double tol = 1.0e-5;


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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 2 3; 4 5 6; 7 8 9]
	// rcond(A)
	// ```

	expect_res = 1.541976423090495e-18;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 2 3; 4 5 6; 7 8 9]
	// rcond(A)
	// ```

	expect_res = 1.541976423090495e-18;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 2 3; 0 4 5; 0 0 6]
	// rcond(A)
	// ```

	expect_res = 0.071428571428571;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 2 3; 0 4 5; 0 0 6]
	// rcond(A)
	// ```

	expect_res = 0.071428571428571;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 0 0; 2 3 0; 4 5 6]
	// rcond(A)
	// ```

	expect_res = 0.0703125;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1 0 0; 2 3 0; 4 5 6]
	// rcond(A)
	// ```

	expect_res = 0.0703125;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-0.23 2.54 -3.66 0; -6.98 2.46 -2.73 -2.13; 0 2.56 2.46 4.07; 0 0 -4.78 -3.82]
	// rcond(A)
	// ```

	expect_res = 0.017727735801114;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-0.23 2.54 -3.66 0; -6.98 2.46 -2.73 -2.13; 0 2.56 2.46 4.07; 0 0 -4.78 -3.82]
	// rcond(A)
	// ```

	expect_res = 0.017727735801114;
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[2.07 3.87 4.20, -1.15; 3.87 -0.21 1.87 0.63; 4.20 1.87 1.15 2.06; -1.15 0.63 2.06 -1.81]
	// rcond(A)
	// ```

	expect_res = 0.013212321296670;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[2.07 3.87 4.20, -1.15; 3.87 -0.21 1.87 0.63; 4.20 1.87 1.15 2.06; -1.15 0.63 2.06 -1.81]
	// rcond(A)
	// ```

	expect_res = 0.013212321296670;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[2.07 3.87 4.20, -1.15; 3.87 -0.21 1.87 0.63; 4.20 1.87 1.15 2.06; -1.15 0.63 2.06 -1.81]
	// rcond(A)
	// ```

	expect_res = 0.013212321296670;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[2.07 3.87 4.20, -1.15; 3.87 -0.21 1.87 0.63; 4.20 1.87 1.15 2.06; -1.15 0.63 2.06 -1.81]
	// rcond(A)
	// ```

	expect_res = 0.013212321296670;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-1.36+0i 1.58+0.9i 2.21-0.21i 3.91+1.5i; 1.58-0.9i -8.87+0i -1.84-0.03i -1.78+1.18i; 2.21+0.21i -1.84+0.03i -4.63+0i 0.11+0.11i; 3.91-1.5i -1.78-1.18i 0.11-0.11i -1.84+0i]
	// rcond(A)
	// ```

	expect_res = 0.149720039067262;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-1.36+0i 1.58+0.9i 2.21-0.21i 3.91+1.5i; 1.58-0.9i -8.87+0i -1.84-0.03i -1.78+1.18i; 2.21+0.21i -1.84+0.03i -4.63+0i 0.11+0.11i; 3.91-1.5i -1.78-1.18i 0.11-0.11i -1.84+0i]
	// rcond(A)
	// ```

	expect_res = 0.149720039067262;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 3+4i 5+6i; 7+8i 9+10i 11+12i; 13+14i 15+16i 17+18i]
	// rcond(A)
	// ```

    // NOTE: the result obtained with MATLAB (1.347314580577711e-17) may differ
    // from the one obtained with Octave (1.285698858022688e-17), which in turn
    // may differ from the one obtained with the `cond_1` function (1.13641e-18).
    // Since all these results are very large numbers, probably the best
    // expected result to use is +inf.
	//expect_res = 1.347314580577711e-17; // MATLAB
	//expect_res = 1.285698858022688e-17; // Octave (also R 4.0.3)
    expect_res = std::numeric_limits<result_type>::infinity();

	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 3+4i 5+6i; 7+8i 9+10i 11+12i; 13+14i 15+16i 17+18i]
	// rcond(A)
	// ```

    // NOTE: the result obtained with MATLAB (1.347314580577711e-17) may differ
    // from the one obtained with Octave (1.285698858022688e-17), which in turn
    // may differ from the one obtained with the `cond_1` function (1.13641e-18).
    // Since all these results are very large numbers, probably the best
    // expected result to use is +inf.
	//expect_res = 1.347314580577711e-17; // MATLAB
	//expect_res = 1.285698858022688e-17; // Octave (also R 4.0.3)
    expect_res = std::numeric_limits<result_type>::infinity();

	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 3+4i 5+6i; 0 7+8i 9+10i; 0 0 11+12i]
	// rcond(A)
	// ```

	expect_res = 0.059560668674847;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 3+4i 5+6i; 0 7+8i 9+10i; 0 0 11+12i]
	// rcond(A)
	// ```

	expect_res = 0.059560668674847;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 0 0; 3+4i 5+6i 0; 7+8i 9+10i 11+12i]
	// rcond(A)
	// ```

	expect_res = 0.059544953702088;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[1+2i 0 0; 3+4i 5+6i 0; 7+8i 9+10i 11+12i]
	// rcond(A)
	// ```

	expect_res = 0.059544953702088;
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-1.65+2.26i -2.05-0.85i 0.97-2.84i 0; 0+6.30i -1.48-1.75i -3.99+4.01i 0.59-0.48i; 0 -0.77+2.83i -1.06+1.94i 3.33-1.04i; 0 0 4.48-1.09i -0.46-1.72i]
	// rcond(A)
	// ```

	expect_res = 0.009594414793018;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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

	// Results obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	// ```octave
	// A=[-1.65+2.26i -2.05-0.85i 0.97-2.84i 0; 0+6.30i -1.48-1.75i -3.99+4.01i 0.59-0.48i; 0 -0.77+2.83i -1.06+1.94i 3.33-1.04i; 0 0 4.48-1.09i -0.46-1.72i]
	// rcond(A)
	// ```

	expect_res = 0.009594414793018;
	BOOST_UBLASX_TEST_CHECK_CLOSE( res, expect_res, tol );
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
