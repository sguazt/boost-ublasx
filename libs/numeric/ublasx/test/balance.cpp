/**
 * \file libw/numeric/ublasx/operation/balance.cpp
 *
 * \brief Test case for matrix balance operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail&gt;
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/balance.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1e-5;


BOOST_UBLASX_TEST_DEF( col_major_double_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - Scale and Permute");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> balance(A)

	// Expected balanced matrix
	expect(0,0) = -0.40000; expect(0,1) = 1.60000; expect(0,2) =  3.80000; expect(0,3) = -0.75000;
	expect(1,0) =  0.00000; expect(1,1) = 0.20000; expect(1,2) =  0.91000; expect(1,3) =  4.31250;
	expect(2,0) =  0.00000; expect(2,1) = 0.91000; expect(2,2) =  5.14000; expect(2,3) = -4.10000;
	expect(3,0) =  0.00000; expect(3,1) = 2.80000; expect(3,2) = -2.64000; expect(3,3) =  0.66000;

	res = ublasx::balance(A);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_double_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - Scale and Permute");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35,0 0.66]
	//   >> balance(A)

	// Expected balanced matrix
	expect(0,0) = -0.40000; expect(0,1) = 1.60000; expect(0,2) = 3.80000; expect(0,3) = -0.75000;
	expect(1,0) =  0.00000; expect(1,1) = 0.20000; expect(1,2) = 0.91000; expect(1,3) =  4.31250;
	expect(2,0) =  0.00000; expect(2,1) = 0.91000; expect(2,2) = 5.14000; expect(2,3) = -4.10000;
	expect(3,0) =  0.00000; expect(3,1) = 2.80000; expect(3,2) = -2.6400; expect(3,3) = 0.660000;

	res = ublasx::balance(A);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_complex_double_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Complex (Double Precision) - Scale and Permute");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) = value_type( 1.50,-2.75); A(0,1) = value_type( 0.00, 0.00); A(0,2) = value_type( 0.00, 0.00); A(0,3) = value_type( 0.00, 0.00);
	A(1,0) = value_type(-8.06,-1.24); A(1,1) = value_type(-2.50,-0.50); A(1,2) = value_type( 0.00, 0.00); A(1,3) = value_type(-0.75, 0.50);
	A(2,0) = value_type(-2.09, 7.56); A(2,1) = value_type( 1.39, 3.97); A(2,2) = value_type(-1.25, 0.75); A(2,3) = value_type(-4.82,-5.67);
	A(3,0) = value_type( 6.18, 9.79); A(3,1) = value_type(-0.92,-0.62); A(3,2) = value_type( 0.00, 0.00); A(3,3) = value_type(-2.50,-0.50);

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[1.5-2.75i 0+0i 0+0i 0+0i; -8.06-1.24i -2.5-0.5i 0+0i -0.75+0.5i; -2.09+7.56i 1.39+3.97i -1.25+0.75i -4.82-5.67i; 6.18+9.79i -0.92-0.62i 0+0i -2.5-0.5i]
	//   >> balance(A)

	// Expected balanced matrix
	expect(0,0) = value_type(-1.25000, 0.75000); expect(0,1) = value_type( 1.39000, 3.97000); expect(0,2) = value_type(-4.82000,-5.67000); expect(0,3) = value_type(-2.09000, 7.56000);
	expect(1,0) = value_type( 0.00000, 0.00000); expect(1,1) = value_type(-2.50000,-0.50000); expect(1,2) = value_type(-0.75000, 0.50000); expect(1,3) = value_type(-8.06000,-1.24000);
	expect(2,0) = value_type( 0.00000, 0.00000); expect(2,1) = value_type(-0.92000,-0.62000); expect(2,2) = value_type(-2.50000,-0.50000); expect(2,3) = value_type( 6.18000, 9.79000);
	expect(3,0) = value_type( 0.00000, 0.00000); expect(3,1) = value_type( 0.00000, 0.00000); expect(3,2) = value_type( 0.00000, 0.00000); expect(3,3) = value_type( 1.50000,-2.75000);

	res = ublasx::balance(A);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_complex_double_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Complex (Double Precision) - Scale and Permute");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) = value_type( 1.50,-2.75); A(0,1) = value_type( 0.00, 0.00); A(0,2) = value_type( 0.00, 0.00); A(0,3) = value_type( 0.00, 0.00);
	A(1,0) = value_type(-8.06,-1.24); A(1,1) = value_type(-2.50,-0.50); A(1,2) = value_type( 0.00, 0.00); A(1,3) = value_type(-0.75, 0.50);
	A(2,0) = value_type(-2.09, 7.56); A(2,1) = value_type( 1.39, 3.97); A(2,2) = value_type(-1.25, 0.75); A(2,3) = value_type(-4.82,-5.67);
	A(3,0) = value_type( 6.18, 9.79); A(3,1) = value_type(-0.92,-0.62); A(3,2) = value_type( 0.00, 0.00); A(3,3) = value_type(-2.50,-0.50);

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[1.5-2.75i 0+0i 0+0i 0+0i; -8.06-1.24i -2.5-0.5i 0+0i -0.75+0.5i; -2.09+7.56i 1.39+3.97i -1.25+0.75i -4.82-5.67i; 6.18+9.79i -0.92-0.62i 0+0i -2.5-0.5i]
	//   >> balance(A)

	// Expected balanced matrix
	expect(0,0) = value_type(-1.25000, 0.75000); expect(0,1) = value_type( 1.39000, 3.97000); expect(0,2) = value_type(-4.82000,-5.67000); expect(0,3) = value_type(-2.09000, 7.56000);
	expect(1,0) = value_type( 0.00000, 0.00000); expect(1,1) = value_type(-2.50000,-0.50000); expect(1,2) = value_type(-0.75000, 0.50000); expect(1,3) = value_type(-8.06000,-1.24000);
	expect(2,0) = value_type( 0.00000, 0.00000); expect(2,1) = value_type(-0.92000,-0.62000); expect(2,2) = value_type(-2.50000,-0.50000); expect(2,3) = value_type( 6.18000, 9.79000);
	expect(3,0) = value_type( 0.00000, 0.00000); expect(3,1) = value_type( 0.00000, 0.00000); expect(3,2) = value_type( 0.00000, 0.00000); expect(3,3) = value_type( 1.50000,-2.75000);

	res = ublasx::balance(A);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_double_both_balmat )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - Scale and Permute - Balancing Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	matrix_type balancing_res(n,n);
	matrix_type balanced_expect(n,n);
	matrix_type balancing_expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [DD, AA] = balance(A)
	// where DD -> balancing_expect and AA -> balanced_expect

	// Expected balanced matrix
	balanced_expect(0,0) = -0.40000; balanced_expect(0,1) = 1.60000; balanced_expect(0,2) =  3.80000; balanced_expect(0,3) = -0.75000;
	balanced_expect(1,0) =  0.00000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.91000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.00000; balanced_expect(2,1) = 0.91000; balanced_expect(2,2) =  5.14000; balanced_expect(2,3) = -4.10000;
	balanced_expect(3,0) =  0.00000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) = -2.64000; balanced_expect(3,3) =  0.66000;

	// Expected balancing matrix
	balancing_expect(0,0) =  0.00000; balancing_expect(0,1) = 0.00000; balancing_expect(0,2) =  2.00000; balancing_expect(0,3) =  0.00000;
	balancing_expect(1,0) =  0.00000; balancing_expect(1,1) = 2.00000; balancing_expect(1,2) =  0.00000; balancing_expect(1,3) =  0.00000;
	balancing_expect(2,0) =  1.00000; balancing_expect(2,1) = 0.00000; balancing_expect(2,2) =  0.00000; balancing_expect(2,3) =  0.00000;
	balancing_expect(3,0) =  0.00000; balancing_expect(3,1) = 0.00000; balancing_expect(3,2) =  0.00000; balancing_expect(3,3) =  0.25000;

	balanced_res = ublasx::balance(A, balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Matrix=" << balancing_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balancing_res, balancing_expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_double_both_balmat )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - Scale and Permute - Balancing Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	matrix_type balancing_res(n,n);
	matrix_type balanced_expect(n,n);
	matrix_type balancing_expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [DD, AA] = balance(A)
	// where DD -> balancing_expect and AA -> balanced_expect

	// Expected balanced matrix
	balanced_expect(0,0) = -0.40000; balanced_expect(0,1) = 1.60000; balanced_expect(0,2) =  3.80000; balanced_expect(0,3) = -0.75000;
	balanced_expect(1,0) =  0.00000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.91000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.00000; balanced_expect(2,1) = 0.91000; balanced_expect(2,2) =  5.14000; balanced_expect(2,3) = -4.10000;
	balanced_expect(3,0) =  0.00000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) = -2.64000; balanced_expect(3,3) =  0.66000;

	// Expected balancing matrix
	balancing_expect(0,0) =  0.00000; balancing_expect(0,1) = 0.00000; balancing_expect(0,2) =  2.00000; balancing_expect(0,3) =  0.00000;
	balancing_expect(1,0) =  0.00000; balancing_expect(1,1) = 2.00000; balancing_expect(1,2) =  0.00000; balancing_expect(1,3) =  0.00000;
	balancing_expect(2,0) =  1.00000; balancing_expect(2,1) = 0.00000; balancing_expect(2,2) =  0.00000; balancing_expect(2,3) =  0.00000;
	balancing_expect(3,0) =  0.00000; balancing_expect(3,1) = 0.00000; balancing_expect(3,2) =  0.00000; balancing_expect(3,3) =  0.25000;

	balanced_res = ublasx::balance(A, balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Matrix=" << balancing_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balancing_res, balancing_expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_double_both_balpermvec )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - Scale and Permute - Balancing and Permutation Vector");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector<size_type> size_vector_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	vector_type balancing_res(n);
	size_vector_type permuting_res(n);
	matrix_type balanced_expect(n,n);
	vector_type balancing_expect(n);
	size_vector_type permuting_expect(n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [D, P, AA] = balance(A)
	// where D -> balancing_expect, (P-1) -> permuting_expect, and AA -> balanced_exect.

	// Expected balanced matrix
	balanced_expect(0,0) = -0.40000; balanced_expect(0,1) = 1.60000; balanced_expect(0,2) =  3.80000; balanced_expect(0,3) = -0.75000;
	balanced_expect(1,0) =  0.00000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.91000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.00000; balanced_expect(2,1) = 0.91000; balanced_expect(2,2) =  5.14000; balanced_expect(2,3) = -4.10000;
	balanced_expect(3,0) =  0.00000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) = -2.64000; balanced_expect(3,3) =  0.66000;

	// Expected scaling vector
	balancing_expect(0) = 1.00000;
	balancing_expect(1) = 2.00000;
	balancing_expect(2) = 2.00000;
	balancing_expect(3) = 0.25000;

	// Expected permutation vector
	permuting_expect(0) = 2;
	permuting_expect(1) = 1;
	permuting_expect(2) = 0;
	permuting_expect(3) = 3;

	balanced_res = ublasx::balance(A, balancing_res, permuting_res);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Vector=" << balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("Permuting Vector=" << permuting_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( balancing_res, balancing_expect, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( permuting_res, permuting_expect, n );
}


BOOST_UBLASX_TEST_DEF( row_major_double_both_balpermvec )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - Scale and Permute - Balancing and Permutation Vector");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector<size_type> size_vector_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	vector_type balancing_res(n);
	size_vector_type permuting_res(n);
	matrix_type balanced_expect(n,n);
	vector_type balancing_expect(n);
	size_vector_type permuting_expect(n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [D, P, AA] = balance(A)
	// where D -> balancing_expect, (P-1) -> permuting_expect, and AA -> balanced_exect.

	// Expected balanced matrix
	balanced_expect(0,0) = -0.40000; balanced_expect(0,1) = 1.60000; balanced_expect(0,2) =  3.80000; balanced_expect(0,3) = -0.75000;
	balanced_expect(1,0) =  0.00000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.91000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.00000; balanced_expect(2,1) = 0.91000; balanced_expect(2,2) =  5.14000; balanced_expect(2,3) = -4.10000;
	balanced_expect(3,0) =  0.00000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) = -2.64000; balanced_expect(3,3) =  0.66000;

	// Expected scaling vector
	balancing_expect(0) = 1.00000;
	balancing_expect(1) = 2.00000;
	balancing_expect(2) = 2.00000;
	balancing_expect(3) = 0.25000;

	// Expected permutation vector
	permuting_expect(0) = 2;
	permuting_expect(1) = 1;
	permuting_expect(2) = 0;
	permuting_expect(3) = 3;

	balanced_res = ublasx::balance(A, balancing_res, permuting_res);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Vector=" << balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("Permuting Vector=" << permuting_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( balancing_res, balancing_expect, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( permuting_res, permuting_expect, n );
}


BOOST_UBLASX_TEST_DEF( col_major_double_noperm )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - No Permute");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> balance(A, 'noperm')

	// Expected balanced matrix
	expect(0,0) =  5.14000; expect(0,1) = 0.91000; expect(0,2) =  0.00000; expect(0,3) = -4.10000;
	expect(1,0) =  0.91000; expect(1,1) = 0.20000; expect(1,2) =  0.00000; expect(1,3) =  4.31250;
	expect(2,0) =  0.47500; expect(2,1) = 0.20000; expect(2,2) = -0.40000; expect(2,3) = -0.09375;
	expect(3,0) = -2.64000; expect(3,1) = 2.80000; expect(3,2) =  0.00000; expect(3,3) =  0.66000;

	res = ublasx::balance(A, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_double_noperm )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - No Permute");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> balance(A, 'noperm')

	// Expected balanced matrix
	expect(0,0) =  5.14000; expect(0,1) = 0.91000; expect(0,2) =  0.00000; expect(0,3) = -4.10000;
	expect(1,0) =  0.91000; expect(1,1) = 0.20000; expect(1,2) =  0.00000; expect(1,3) =  4.31250;
	expect(2,0) =  0.47500; expect(2,1) = 0.20000; expect(2,2) = -0.40000; expect(2,3) = -0.09375;
	expect(3,0) = -2.64000; expect(3,1) = 2.80000; expect(3,2) =  0.00000; expect(3,3) =  0.66000;

	res = ublasx::balance(A, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_complex_double_noperm )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Complex (Double Precision) - No Permute");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) = value_type( 1.50,-2.75); A(0,1) = value_type( 0.00, 0.00); A(0,2) = value_type( 0.00, 0.00); A(0,3) = value_type( 0.00, 0.00);
	A(1,0) = value_type(-8.06,-1.24); A(1,1) = value_type(-2.50,-0.50); A(1,2) = value_type( 0.00, 0.00); A(1,3) = value_type(-0.75, 0.50);
	A(2,0) = value_type(-2.09, 7.56); A(2,1) = value_type( 1.39, 3.97); A(2,2) = value_type(-1.25, 0.75); A(2,3) = value_type(-4.82,-5.67);
	A(3,0) = value_type( 6.18, 9.79); A(3,1) = value_type(-0.92,-0.62); A(3,2) = value_type( 0.00, 0.00); A(3,3) = value_type(-2.50,-0.50);

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[1.5-2.75i 0+0i 0+0i 0+0i; -8.06-1.24i -2.5-0.5i 0+0i -0.75+0.5i; -2.09+7.56i 1.39+3.97i -1.25+0.75i -4.82-5.67i; 6.18+9.79i -0.92-0.62i 0+0i -2.5-0.5i]
	//   >> balance(A, 'noperm')

	// Expected balanced matrix
	expect(0,0) = value_type( 1.500000,-2.750000); expect(0,1) = value_type( 0.000000, 0.000000); expect(0,2) = value_type( 0.000000, 0.000000); expect(0,3) = value_type( 0.000000, 0.000000);
	expect(1,0) = value_type(-2.015000,-0.310000); expect(1,1) = value_type(-2.500000,-0.500000); expect(1,2) = value_type( 0.000000, 0.000000); expect(1,3) = value_type(-0.750000, 0.500000);
	expect(2,0) = value_type(-0.130625, 0.472500); expect(2,1) = value_type( 0.347500, 0.992500); expect(2,2) = value_type(-1.250000, 0.750000); expect(2,3) = value_type(-1.205000,-1.417500);
	expect(3,0) = value_type( 1.545000, 2.447500); expect(3,1) = value_type(-0.920000,-0.620000); expect(3,2) = value_type( 0.000000, 0.000000); expect(3,3) = value_type(-2.500000,-0.500000);

	res = ublasx::balance(A, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_complex_double_noperm )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Complex (Double Precision) - No Permute");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) = value_type( 1.50,-2.75); A(0,1) = value_type( 0.00, 0.00); A(0,2) = value_type( 0.00, 0.00); A(0,3) = value_type( 0.00, 0.00);
	A(1,0) = value_type(-8.06,-1.24); A(1,1) = value_type(-2.50,-0.50); A(1,2) = value_type( 0.00, 0.00); A(1,3) = value_type(-0.75, 0.50);
	A(2,0) = value_type(-2.09, 7.56); A(2,1) = value_type( 1.39, 3.97); A(2,2) = value_type(-1.25, 0.75); A(2,3) = value_type(-4.82,-5.67);
	A(3,0) = value_type( 6.18, 9.79); A(3,1) = value_type(-0.92,-0.62); A(3,2) = value_type( 0.00, 0.00); A(3,3) = value_type(-2.50,-0.50);

	matrix_type res(n,n);
	matrix_type expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[1.5-2.75i 0+0i 0+0i 0+0i; -8.06-1.24i -2.5-0.5i 0+0i -0.75+0.5i; -2.09+7.56i 1.39+3.97i -1.25+0.75i -4.82-5.67i; 6.18+9.79i -0.92-0.62i 0+0i -2.5-0.5i]
	//   >> balance(A, 'noperm')

	// Expected balanced matrix
	expect(0,0) = value_type( 1.500000,-2.750000); expect(0,1) = value_type( 0.000000, 0.000000); expect(0,2) = value_type( 0.000000, 0.000000); expect(0,3) = value_type( 0.000000, 0.000000);
	expect(1,0) = value_type(-2.015000,-0.310000); expect(1,1) = value_type(-2.500000,-0.500000); expect(1,2) = value_type( 0.000000, 0.000000); expect(1,3) = value_type(-0.750000, 0.500000);
	expect(2,0) = value_type(-0.130625, 0.472500); expect(2,1) = value_type( 0.347500, 0.992500); expect(2,2) = value_type(-1.250000, 0.750000); expect(2,3) = value_type(-1.205000,-1.417500);
	expect(3,0) = value_type( 1.545000, 2.447500); expect(3,1) = value_type(-0.920000,-0.620000); expect(3,2) = value_type( 0.000000, 0.000000); expect(3,3) = value_type(-2.500000,-0.500000);

	res = ublasx::balance(A, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_double_noperm_balmat )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - No Permute - Balancing Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	matrix_type balancing_res(n,n);
	matrix_type balanced_expect(n,n);
	matrix_type balancing_expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [DD, AA] = balance(A, 'noperm')
	// where DD -> balancing_expect and AA -> balanced_expect

	// Expected balanced matrix
	balanced_expect(0,0) =  5.14000; balanced_expect(0,1) = 0.91000; balanced_expect(0,2) =  0.00000; balanced_expect(0,3) = -4.10000;
	balanced_expect(1,0) =  0.91000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.00000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.47500; balanced_expect(2,1) = 0.20000; balanced_expect(2,2) = -0.40000; balanced_expect(2,3) = -0.09375;
	balanced_expect(3,0) = -2.64000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) =  0.00000; balanced_expect(3,3) =  0.66000;

	// Expected balancing matrix
	balancing_expect(0,0) = 2.00000; balancing_expect(0,1) = 0.00000; balancing_expect(0,2) = 0.00000; balancing_expect(0,3) = 0.00000;
	balancing_expect(1,0) = 0.00000; balancing_expect(1,1) = 2.00000; balancing_expect(1,2) = 0.00000; balancing_expect(1,3) = 0.00000;
	balancing_expect(2,0) = 0.00000; balancing_expect(2,1) = 0.00000; balancing_expect(2,2) = 8.00000; balancing_expect(2,3) = 0.00000;
	balancing_expect(3,0) = 0.00000; balancing_expect(3,1) = 0.00000; balancing_expect(3,2) = 0.00000; balancing_expect(3,3) = 0.25000;

	balanced_res = ublasx::balance(A, balancing_res, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Matrix=" << balancing_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balancing_res, balancing_expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_double_noperm_balmat )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - No Permute - Balancing Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	matrix_type balancing_res(n,n);
	matrix_type balanced_expect(n,n);
	matrix_type balancing_expect(n,n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [DD, AA] = balance(A, 'noperm')
	// where DD -> balancing_expect and AA -> balanced_expect

	// Expected balanced matrix
	balanced_expect(0,0) =  5.14000; balanced_expect(0,1) = 0.91000; balanced_expect(0,2) =  0.00000; balanced_expect(0,3) = -4.10000;
	balanced_expect(1,0) =  0.91000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.00000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.47500; balanced_expect(2,1) = 0.20000; balanced_expect(2,2) = -0.40000; balanced_expect(2,3) = -0.09375;
	balanced_expect(3,0) = -2.64000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) =  0.00000; balanced_expect(3,3) =  0.66000;

	// Expected balancing matrix
	balancing_expect(0,0) = 2.00000; balancing_expect(0,1) = 0.00000; balancing_expect(0,2) = 0.00000; balancing_expect(0,3) = 0.00000;
	balancing_expect(1,0) = 0.00000; balancing_expect(1,1) = 2.00000; balancing_expect(1,2) = 0.00000; balancing_expect(1,3) = 0.00000;
	balancing_expect(2,0) = 0.00000; balancing_expect(2,1) = 0.00000; balancing_expect(2,2) = 8.00000; balancing_expect(2,3) = 0.00000;
	balancing_expect(3,0) = 0.00000; balancing_expect(3,1) = 0.00000; balancing_expect(3,2) = 0.00000; balancing_expect(3,3) = 0.25000;

	balanced_res = ublasx::balance(A, balancing_res, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Matrix=" << balancing_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balancing_res, balancing_expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_double_noperm_balpermvec )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-Major Matrix - Double Precision - No Permute - Balancing and Permutation Vector");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector<size_type> size_vector_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	vector_type balancing_res(n);
	size_vector_type permuting_res(n);
	matrix_type balanced_expect(n,n);
	vector_type balancing_expect(n);
	size_vector_type permuting_expect(n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [DD, AA] = balance(A, 'noperm')
    //   >> [D, P, AA] = balance(A)
    // where D -> balancing_expect, (P-1) -> permuting_expect, and AA -> balanced_exect.

	// Expected balanced matrix
	balanced_expect(0,0) =  5.14000; balanced_expect(0,1) = 0.91000; balanced_expect(0,2) =  0.00000; balanced_expect(0,3) = -4.10000;
	balanced_expect(1,0) =  0.91000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.00000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.47500; balanced_expect(2,1) = 0.20000; balanced_expect(2,2) = -0.40000; balanced_expect(2,3) = -0.09375;
	balanced_expect(3,0) = -2.64000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) =  0.00000; balanced_expect(3,3) =  0.66000;

	// Expected scaling vector
	balancing_expect(0) = 2.00000;
	balancing_expect(1) = 2.00000;
	balancing_expect(2) = 8.00000;
	balancing_expect(3) = 0.25000;

	// Expected permutation vector
	permuting_expect(0) = 0;
	permuting_expect(1) = 1;
	permuting_expect(2) = 2;
	permuting_expect(3) = 3;

	balanced_res = ublasx::balance(A, balancing_res, permuting_res, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Vector=" << balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("Permuting Vector=" << permuting_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( balancing_res, balancing_expect, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( permuting_res, permuting_expect, n );
}


BOOST_UBLASX_TEST_DEF( row_major_double_noperm_balpermvec )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-Major Matrix - Double Precision - No Permute - Balancing and Permutation Vector");

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;
	typedef typename ublas::matrix_traits<matrix_type>::size_type size_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector<size_type> size_vector_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  5.14; A(0,1) = 0.91; A(0,2) =  0.00; A(0,3) = -32.80;
	A(1,0) =  0.91; A(1,1) = 0.20; A(1,2) =  0.00; A(1,3) =  34.50;
	A(2,0) =  1.90; A(2,1) = 0.80; A(2,2) = -0.40; A(2,3) =  -3.00;
	A(3,0) = -0.33; A(3,1) = 0.35; A(3,2) =  0.00; A(3,3) =   0.66;

	matrix_type balanced_res(n,n);
	vector_type balancing_res(n);
	size_vector_type permuting_res(n);
	matrix_type balanced_expect(n,n);
	vector_type balancing_expect(n);
	size_vector_type permuting_expect(n);

	// Result obtained with:
	// - MATLAB 2017a
	// - Octave 5.2.0
	// on Fedora 33 x86_64, kernel 5.9.16-200, gcc 10.2.1, glibc 2.32, LAPACK 3.9.0
	//   >> A=[5.14 0.91 0 -32.8; 0.91 0.2 0 34.5; 1.9 0.8 -0.4 -3; -0.33 0.35 0 0.66]
	//   >> [D, P, AA] = balance(A, 'noperm')
	// where D -> balancing_expect, (P-1) -> permuting_expect, and AA -> balanced_exect.

	// Expected balanced matrix
	balanced_expect(0,0) =  5.14000; balanced_expect(0,1) = 0.91000; balanced_expect(0,2) =  0.00000; balanced_expect(0,3) = -4.10000;
	balanced_expect(1,0) =  0.91000; balanced_expect(1,1) = 0.20000; balanced_expect(1,2) =  0.00000; balanced_expect(1,3) =  4.31250;
	balanced_expect(2,0) =  0.47500; balanced_expect(2,1) = 0.20000; balanced_expect(2,2) = -0.40000; balanced_expect(2,3) = -0.09375;
	balanced_expect(3,0) = -2.64000; balanced_expect(3,1) = 2.80000; balanced_expect(3,2) =  0.00000; balanced_expect(3,3) =  0.66000;

	// Expected scaling vector
	balancing_expect(0) = 2.00000;
	balancing_expect(1) = 2.00000;
	balancing_expect(2) = 8.00000;
	balancing_expect(3) = 0.25000;

	// Expected permutation vector
	permuting_expect(0) = 0;
	permuting_expect(1) = 1;
	permuting_expect(2) = 2;
	permuting_expect(3) = 3;

	balanced_res = ublasx::balance(A, balancing_res, permuting_res, true, false);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Balanced A=" << balanced_res);
	BOOST_UBLASX_DEBUG_TRACE("Balancing Vector=" << balancing_res);
	BOOST_UBLASX_DEBUG_TRACE("Permuting Vector=" << permuting_res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( balanced_res, balanced_expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( balancing_res, balancing_expect, n, tol );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( permuting_res, permuting_expect, n );
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( col_major_double_both );
	BOOST_UBLASX_TEST_DO( row_major_double_both );
	BOOST_UBLASX_TEST_DO( col_major_complex_double_both );
	BOOST_UBLASX_TEST_DO( row_major_complex_double_both );
	BOOST_UBLASX_TEST_DO( col_major_double_both_balmat );
	BOOST_UBLASX_TEST_DO( row_major_double_both_balmat );
	BOOST_UBLASX_TEST_DO( col_major_double_both_balpermvec );
	BOOST_UBLASX_TEST_DO( row_major_double_both_balpermvec );

	BOOST_UBLASX_TEST_DO( col_major_double_noperm );
	BOOST_UBLASX_TEST_DO( row_major_double_noperm );
	BOOST_UBLASX_TEST_DO( col_major_complex_double_noperm );
	BOOST_UBLASX_TEST_DO( row_major_complex_double_noperm );
	BOOST_UBLASX_TEST_DO( col_major_double_noperm_balmat );
	BOOST_UBLASX_TEST_DO( row_major_double_noperm_balmat );
	BOOST_UBLASX_TEST_DO( col_major_double_noperm_balpermvec );
	BOOST_UBLASX_TEST_DO( row_major_double_noperm_balpermvec );

	BOOST_UBLASX_TEST_END();
}
