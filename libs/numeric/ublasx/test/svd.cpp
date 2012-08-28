/**
 * \file libs/numeric/ublasx/test/svd.cpp
 *
 * \brief Test suite for the SVD.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <algorithm>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/svd.hpp>
#include <complex>
#include <cstddef>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( singular_values_real_column_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Singular Values - Real Matrix - Column Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	vector_type expect_s(std::min(n,m));
	expect_s(0) = 9.996627661356916;
	expect_s(1) = 3.683101373968637;
	expect_s(2) = 1.356928726274717;
	expect_s(3) = 0.500044099129892;


	vector_type s = ublasx::svd_values(A);
	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("s = " << s);
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(s, expect_s, std::min(m,n), tol);
}


BOOST_UBLASX_TEST_DEF( singular_values_real_row_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Singular Values - Real Matrix - Row Major");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	vector_type expect_s(std::min(n,m));
	expect_s(0) = 9.996627661356916;
	expect_s(1) = 3.683101373968637;
	expect_s(2) = 1.356928726274717;
	expect_s(3) = 0.500044099129892;


	vector_type s = ublasx::svd_values(A);
	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("s = " << s);
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(s, expect_s, std::min(m,n), tol);
}


BOOST_UBLASX_TEST_DEF( singular_values_complex_column_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Singular Values - Complex Matrix - Column Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);


	vector_type expect_s(std::min(n,m));
	expect_s(0) = 3.999423572044701;
	expect_s(1) = 3.000270074501588;
	expect_s(2) = 1.994428215493926;
	expect_s(3) = 0.999473193570071;


	vector_type s = ublasx::svd_values(A);
	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("s = " << s);
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(s, expect_s, std::min(m,n), tol);
}


BOOST_UBLASX_TEST_DEF( singular_values_complex_row_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Singular Values - Complex Matrix - Row Major");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);


	vector_type expect_s(std::min(n,m));
	expect_s(0) = 3.999423572044701;
	expect_s(1) = 3.000270074501588;
	expect_s(2) = 1.994428215493926;
	expect_s(3) = 0.999473193570071;


	vector_type s = ublasx::svd_values(A);
	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("s = " << s);
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(s, expect_s, std::min(m,n), tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_real_column_major_matrix_full_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Real Matrix - Column Major - Full Mode");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, true);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	matrix_type X;
	X = ublas::prod(svd.U(), svd.S());
	X = ublas::prod(X, svd.VH());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(A, X, n, m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_real_row_major_matrix_full_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Real Matrix - Row Major - Full Mode");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, true);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	matrix_type X;
	X = ublas::prod(svd.U(), svd.S());
	X = ublas::prod(X, svd.VH());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(A, X, n, m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_complex_column_major_matrix_full_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Complex Matrix - Column Major - Full Mode");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);

	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, true);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^H = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	matrix_type X;
	X = ublas::prod(svd.U(), svd.S());
	X = ublas::prod(X, svd.VH());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(A, X, n, m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_complex_row_major_matrix_full_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Complex Matrix - Row Major - Full Mode");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);

	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, true);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^H = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	matrix_type X;
	X = ublas::prod(svd.U(), svd.S());
	X = ublas::prod(X, svd.VH());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(A, X, n, m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_real_column_major_matrix_eco_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Real Matrix - Column Major - Economic Mode");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	ublasx::svd_decomposition<value_type> expect_svd;
	expect_svd = ublasx::svd_decompose(A, true);
	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, false);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.U(), expect_svd.U(), n, std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.S(), expect_svd.S(), std::min(n,m), std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.VH(), expect_svd.VH(), std::min(n,m), m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_real_row_major_matrix_eco_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Real Matrix - Row Major - Economic Mode");

	typedef double real_type;
	typedef real_type value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(6);
	const std::size_t m(4);

	matrix_type A(n,m);
	A(0,0) =  2.27; A(0,1) = -1.54; A(0,2) =  1.15; A(0,3) = -1.94;
	A(1,0) =  0.28; A(1,1) = -1.67; A(1,2) =  0.94; A(1,3) = -0.78;
	A(2,0) = -0.48; A(2,1) = -3.09; A(2,2) =  0.99; A(2,3) = -0.21;
	A(3,0) =  1.07; A(3,1) =  1.22; A(3,2) =  0.79; A(3,3) =  0.63;
	A(4,0) = -2.35; A(4,1) =  2.93; A(4,2) = -1.45; A(4,3) =  2.30;
	A(5,0) =  0.62; A(5,1) = -7.39; A(5,2) =  1.03; A(5,3) = -2.57;

	ublasx::svd_decomposition<value_type> expect_svd;
	expect_svd = ublasx::svd_decompose(A, true);
	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, false);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.U(), expect_svd.U(), n, std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.S(), expect_svd.S(), std::min(n,m), std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.VH(), expect_svd.VH(), std::min(n,m), m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_complex_column_major_matrix_eco_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Complex Matrix - Column Major - Economic Mode");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);


	ublasx::svd_decomposition<value_type> expect_svd;
	expect_svd = ublasx::svd_decompose(A, true);
	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, false);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.U(), expect_svd.U(), n, std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.S(), expect_svd.S(), std::min(n,m), std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.VH(), expect_svd.VH(), std::min(n,m), m, tol);
}


BOOST_UBLASX_TEST_DEF( svd_oo_complex_row_major_matrix_eco_mode )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: SVD decomposition class - Complex Matrix - Row Major - Economic Mode");

	typedef double real_type;
	typedef std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<real_type> vector_type;


	const std::size_t n(4);
	const std::size_t m(6);

	matrix_type A(n,m);
	A(0,0) = value_type( 0.96, 0.81); A(0,1) = value_type(-0.98,-1.98); A(0,2) = value_type( 0.62, 0.46); A(0,3) = value_type(-0.37,-0.38); A(0,4) = value_type( 0.83,-0.51); A(0,5) = value_type( 1.08, 0.28);
	A(1,0) = value_type(-0.03,-0.96); A(1,1) = value_type(-1.20,-0.19); A(1,2) = value_type( 1.01,-0.02); A(1,3) = value_type( 0.19, 0.54); A(1,4) = value_type( 0.20,-0.01); A(1,5) = value_type( 0.20, 0.12);
	A(2,0) = value_type(-0.91,-2.06); A(2,1) = value_type(-0.66,-0.42); A(2,2) = value_type( 0.63, 0.17); A(2,3) = value_type(-0.98, 0.36); A(2,4) = value_type(-0.17, 0.46); A(2,5) = value_type(-0.07,-1.23);
	A(3,0) = value_type(-0.05,-0.41); A(3,1) = value_type(-0.81,-0.56); A(3,2) = value_type(-1.11,-0.60); A(3,3) = value_type( 0.22, 0.20); A(3,4) = value_type( 1.47,-1.59); A(3,5) = value_type( 0.26,-0.26);


	ublasx::svd_decomposition<value_type> expect_svd;
	expect_svd = ublasx::svd_decompose(A, true);
	ublasx::svd_decomposition<value_type> svd;
	svd = ublasx::svd_decompose(A, false);

	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	BOOST_UBLASX_DEBUG_TRACE("U = " << svd.U());
	BOOST_UBLASX_DEBUG_TRACE("S = " << svd.S());
	BOOST_UBLASX_DEBUG_TRACE("V^T = " << svd.VH());
	BOOST_UBLASX_DEBUG_TRACE("V = " << svd.V());
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.U(), expect_svd.U(), n, std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.S(), expect_svd.S(), std::min(n,m), std::min(n,m), tol);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(svd.VH(), expect_svd.VH(), std::min(n,m), m, tol);
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( singular_values_real_column_major_matrix );
	BOOST_UBLASX_TEST_DO( singular_values_real_row_major_matrix );
	BOOST_UBLASX_TEST_DO( singular_values_complex_column_major_matrix );
	BOOST_UBLASX_TEST_DO( singular_values_complex_row_major_matrix );
	BOOST_UBLASX_TEST_DO( svd_oo_real_column_major_matrix_full_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_real_row_major_matrix_full_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_complex_column_major_matrix_full_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_complex_row_major_matrix_full_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_real_column_major_matrix_eco_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_real_row_major_matrix_eco_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_complex_column_major_matrix_eco_mode );
	BOOST_UBLASX_TEST_DO( svd_oo_complex_row_major_matrix_eco_mode );

	BOOST_UBLASX_TEST_END();
}
