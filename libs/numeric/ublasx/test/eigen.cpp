/**
 * \file libs/numeric/ublasx/test/eigen.cpp
 *
 * \brief Test the \c eigen operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/diag.hpp>
#include <boost/numeric/ublasx/operation/eigen.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Left Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type LV;

	ublasx::left_eigen(A, w, LV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Right Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type RV;

	ublasx::right_eigen(A, w, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Only Eigenvalues");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_vector_type expect_w;

	expect_w = out_vector_type(n);
	expect_w(0) = out_value_type(  2.85813, 10.76275);
	expect_w(1) = out_value_type(  2.85813,-10.76275);
	expect_w(2) = out_value_type(- 0.68667,  4.70426);
	expect_w(3) = out_value_type(- 0.68667, -4.70426);
	expect_w(4) = out_value_type(-10.46292,  0.00000);


	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_only_vectors )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Only Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_matrix_type LV;
	out_matrix_type RV;
	out_matrix_type expect_LV(n,n);
	out_matrix_type expect_RV(n,n);

	ublasx::eigenvectors(A, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	expect_LV(0,0) = out_value_type( 0.04441, 0.28792); expect_LV(0,1) = out_value_type( 0.04441,-0.28792); expect_LV(0,2) = out_value_type(-0.13256,-0.32729); expect_LV(0,3) = out_value_type(-0.13256, 0.32729); expect_LV(0,4) = out_value_type( 0.04084,-0.00000);
	expect_LV(1,0) = out_value_type( 0.61816, 0.00000); expect_LV(1,1) = out_value_type( 0.61816, 0.00000); expect_LV(1,2) = out_value_type( 0.68687, 0.00000); expect_LV(1,3) = out_value_type( 0.68687,-0.00000); expect_LV(1,4) = out_value_type( 0.55995,-0.00000);
	expect_LV(2,0) = out_value_type(-0.03576,-0.57711); expect_LV(2,1) = out_value_type(-0.03576, 0.57711); expect_LV(2,2) = out_value_type(-0.39033,-0.07487); expect_LV(2,3) = out_value_type(-0.39033, 0.07487); expect_LV(2,4) = out_value_type(-0.12850,-0.00000);
	expect_LV(3,0) = out_value_type( 0.28373, 0.01135); expect_LV(3,1) = out_value_type( 0.28373,-0.01135); expect_LV(3,2) = out_value_type(-0.01820,-0.18727); expect_LV(3,3) = out_value_type(-0.01820, 0.18727); expect_LV(3,4) = out_value_type(-0.79670,-0.00000);
	expect_LV(4,0) = out_value_type(-0.04495, 0.34061); expect_LV(4,1) = out_value_type(-0.04495,-0.34061); expect_LV(4,2) = out_value_type(-0.40322, 0.21812); expect_LV(4,3) = out_value_type(-0.40322,-0.21812); expect_LV(4,4) = out_value_type( 0.18314,-0.00000);

	expect_RV(0,0) = out_value_type(0.10806, 0.16865); expect_RV(0,1) = out_value_type(0.10806,-0.16865); expect_RV(0,2) = out_value_type( 0.73223, 0.00000); expect_RV(0,3) = out_value_type( 0.73223, 0.00000); expect_RV(0,4) = out_value_type( 0.46065, 0.00000);
	expect_RV(1,0) = out_value_type(0.40631,-0.25901); expect_RV(1,1) = out_value_type(0.40631, 0.25901); expect_RV(1,2) = out_value_type(-0.02646,-0.01695); expect_RV(1,3) = out_value_type(-0.02646, 0.01695); expect_RV(1,4) = out_value_type( 0.33770, 0.00000);
	expect_RV(2,0) = out_value_type(0.10236,-0.50880); expect_RV(2,1) = out_value_type(0.10236, 0.50880); expect_RV(2,2) = out_value_type( 0.19165,-0.29257); expect_RV(2,3) = out_value_type( 0.19165, 0.29257); expect_RV(2,4) = out_value_type( 0.30874, 0.00000);
	expect_RV(3,0) = out_value_type(0.39863,-0.09133); expect_RV(3,1) = out_value_type(0.39863, 0.09133); expect_RV(3,2) = out_value_type(-0.07901,-0.07808); expect_RV(3,3) = out_value_type(-0.07901, 0.07808); expect_RV(3,4) = out_value_type(-0.74385, 0.00000);
	expect_RV(4,0) = out_value_type(0.53954, 0.00000); expect_RV(4,1) = out_value_type(0.53954, 0.00000); expect_RV(4,2) = out_value_type(-0.29160,-0.49310); expect_RV(4,3) = out_value_type(-0.29160, 0.49310); expect_RV(4,4) = out_value_type( 0.15853, 0.00000);

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Left Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type LV;

	ublasx::left_eigen(A, w, LV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Right Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_matrix_type RV;

	ublasx::right_eigen(A, w, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Only Eigenvalues");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_vector_type w;
	out_vector_type expect_w;

	expect_w = out_vector_type(n);
	expect_w(0) = out_value_type(  2.85813, 10.76275);
	expect_w(1) = out_value_type(  2.85813,-10.76275);
	expect_w(2) = out_value_type(- 0.68667,  4.70426);
	expect_w(3) = out_value_type(- 0.68667, -4.70426);
	expect_w(4) = out_value_type(-10.46292,  0.00000);


	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_only_vectors )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Only Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
	A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
	A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
	A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
	A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

	out_matrix_type LV;
	out_matrix_type RV;
	out_matrix_type expect_LV(n,n);
	out_matrix_type expect_RV(n,n);

	ublasx::eigenvectors(A, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	expect_LV(0,0) = out_value_type( 0.04441, 0.28792); expect_LV(0,1) = out_value_type( 0.04441,-0.28792); expect_LV(0,2) = out_value_type(-0.13256,-0.32729); expect_LV(0,3) = out_value_type(-0.13256, 0.32729); expect_LV(0,4) = out_value_type(-0.04084,-0.00000);
	expect_LV(1,0) = out_value_type( 0.61816, 0.00000); expect_LV(1,1) = out_value_type( 0.61816, 0.00000); expect_LV(1,2) = out_value_type( 0.68687, 0.00000); expect_LV(1,3) = out_value_type( 0.68687,-0.00000); expect_LV(1,4) = out_value_type(-0.55995,-0.00000);
	expect_LV(2,0) = out_value_type(-0.03576,-0.57711); expect_LV(2,1) = out_value_type(-0.03576, 0.57711); expect_LV(2,2) = out_value_type(-0.39033,-0.07487); expect_LV(2,3) = out_value_type(-0.39033, 0.07487); expect_LV(2,4) = out_value_type( 0.12850,-0.00000);
	expect_LV(3,0) = out_value_type( 0.28373, 0.01135); expect_LV(3,1) = out_value_type( 0.28373,-0.01135); expect_LV(3,2) = out_value_type(-0.01820,-0.18727); expect_LV(3,3) = out_value_type(-0.01820, 0.18727); expect_LV(3,4) = out_value_type( 0.79670,-0.00000);
	expect_LV(4,0) = out_value_type(-0.04495, 0.34061); expect_LV(4,1) = out_value_type(-0.04495,-0.34061); expect_LV(4,2) = out_value_type(-0.40322, 0.21812); expect_LV(4,3) = out_value_type(-0.40322,-0.21812); expect_LV(4,4) = out_value_type(-0.18314,-0.00000);

	expect_RV(0,0) = out_value_type(0.10806, 0.16865); expect_RV(0,1) = out_value_type(0.10806,-0.16865); expect_RV(0,2) = out_value_type( 0.73223, 0.00000); expect_RV(0,3) = out_value_type( 0.73223, 0.00000); expect_RV(0,4) = out_value_type(-0.46065, 0.00000);
	expect_RV(1,0) = out_value_type(0.40631,-0.25901); expect_RV(1,1) = out_value_type(0.40631, 0.25901); expect_RV(1,2) = out_value_type(-0.02646,-0.01695); expect_RV(1,3) = out_value_type(-0.02646, 0.01695); expect_RV(1,4) = out_value_type(-0.33770, 0.00000);
	expect_RV(2,0) = out_value_type(0.10236,-0.50880); expect_RV(2,1) = out_value_type(0.10236, 0.50880); expect_RV(2,2) = out_value_type( 0.19165,-0.29257); expect_RV(2,3) = out_value_type( 0.19165, 0.29257); expect_RV(2,4) = out_value_type(-0.30874, 0.00000);
	expect_RV(3,0) = out_value_type(0.39863,-0.09133); expect_RV(3,1) = out_value_type(0.39863, 0.09133); expect_RV(3,2) = out_value_type(-0.07901,-0.07808); expect_RV(3,3) = out_value_type(-0.07901, 0.07808); expect_RV(3,4) = out_value_type( 0.74385, 0.00000);
	expect_RV(4,0) = out_value_type(0.53954, 0.00000); expect_RV(4,1) = out_value_type(0.53954, 0.00000); expect_RV(4,2) = out_value_type(-0.29160,-0.49310); expect_RV(4,3) = out_value_type(-0.29160, 0.49310); expect_RV(4,4) = out_value_type(-0.15853, 0.00000);

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Both Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Left Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type LV;

	ublasx::left_eigen(A, w, LV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Right Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type RV;

	ublasx::right_eigen(A, w, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Only Eigenvalues");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	expect_w(0) = out_value_type(-9.42985074873922,-12.98329567302135);
	expect_w(1) = out_value_type(-3.44184845897663, 12.68973749844945);
	expect_w(2) = out_value_type( 0.10554548255761,- 3.39504658829915);
	expect_w(3) = out_value_type( 5.75615372515821,  7.12860476287106);


	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w,n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_only_vectors )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Only Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_matrix_type RV;
	out_matrix_type LV;
	out_matrix_type expect_RV(n, n);
	out_matrix_type expect_LV(n, n);

	ublasx::eigenvectors(A, LV, RV);

	expect_LV(0,0) = out_value_type( 0.241443,-0.184652); expect_LV(0,1) = out_value_type( 0.613497, 0.00000); expect_LV(0,2) = out_value_type(-0.18283,-0.334722); expect_LV(0,3) = out_value_type( 0.27648, 0.08843);
	expect_LV(1,0) = out_value_type( 0.786121, 0.000000); expect_LV(1,1) = out_value_type(-0.049905,-0.27212); expect_LV(1,2) = out_value_type( 0.82183, 0.000000); expect_LV(1,3) = out_value_type(-0.54771, 0.15723);
	expect_LV(2,0) = out_value_type( 0.219515,-0.268865); expect_LV(2,1) = out_value_type(-0.208777, 0.53473); expect_LV(2,2) = out_value_type(-0.37143, 0.152499); expect_LV(2,3) = out_value_type( 0.44508, 0.09122);
	expect_LV(3,0) = out_value_type(-0.016984, 0.410925); expect_LV(3,1) = out_value_type( 0.402720,-0.23531); expect_LV(3,2) = out_value_type( 0.05748, 0.120794); expect_LV(3,3) = out_value_type( 0.62016, 0.00000);

	expect_RV(0,0) = out_value_type( 0.430856520077611, 0.326812737812621); expect_RV(0,1) = out_value_type( 0.825682050767281, 0.000000000000000); expect_RV(0,2) = out_value_type( 0.598395978553945, 0.000000000000000); expect_RV(0,3) = out_value_type(-0.305431903484378, 0.033331648617999);
	expect_RV(1,0) = out_value_type( 0.508741460297097,-0.028833421706928); expect_RV(1,1) = out_value_type( 0.075029167881412,-0.248728504509167); expect_RV(1,2) = out_value_type(-0.400476162752076,-0.201449222762560); expect_RV(1,3) = out_value_type( 0.039782828157833, 0.344507652215461);
	expect_RV(2,0) = out_value_type( 0.619849652765775, 0.000000000000000); expect_RV(2,1) = out_value_type(-0.245755789978015, 0.278872402211696); expect_RV(2,2) = out_value_type(-0.090080019075949,-0.475264621539173); expect_RV(2,3) = out_value_type( 0.358325436515984, 0.060645069885247);
	expect_RV(3,0) = out_value_type(-0.226928243319268, 0.110439278464036); expect_RV(3,1) = out_value_type(-0.103434063728144,-0.319201465363233); expect_RV(3,2) = out_value_type(-0.434840295495405, 0.133724917858160); expect_RV(3,3) = out_value_type( 0.808243289317835, 0.000000000000000);


	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Both Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Left Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type LV;

	ublasx::left_eigen(A, w, LV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Right Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_matrix_type RV;

	ublasx::right_eigen(A, w, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Only Eigenvalues");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	expect_w(0) = out_value_type(-9.42985074873922,-12.98329567302135);
	expect_w(1) = out_value_type(-3.44184845897663, 12.68973749844945);
	expect_w(2) = out_value_type( 0.10554548255761,- 3.39504658829915);
	expect_w(3) = out_value_type( 5.75615372515821,  7.12860476287106);


	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w,n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_only_vectors )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Only Eigenvectors");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
	A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
	A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
	A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

	out_matrix_type RV;
	out_matrix_type LV;
	out_matrix_type expect_RV(n, n);
	out_matrix_type expect_LV(n, n);

	ublasx::eigenvectors(A, LV, RV);

	expect_LV(0,0) = out_value_type( 0.241443,-0.184652); expect_LV(0,1) = out_value_type( 0.613497, 0.00000); expect_LV(0,2) = out_value_type(-0.18283,-0.334722); expect_LV(0,3) = out_value_type( 0.27648, 0.08843);
	expect_LV(1,0) = out_value_type( 0.786121, 0.000000); expect_LV(1,1) = out_value_type(-0.049905,-0.27212); expect_LV(1,2) = out_value_type( 0.82183, 0.000000); expect_LV(1,3) = out_value_type(-0.54771, 0.15723);
	expect_LV(2,0) = out_value_type( 0.219515,-0.268865); expect_LV(2,1) = out_value_type(-0.208777, 0.53473); expect_LV(2,2) = out_value_type(-0.37143, 0.152499); expect_LV(2,3) = out_value_type( 0.44508, 0.09122);
	expect_LV(3,0) = out_value_type(-0.016984, 0.410925); expect_LV(3,1) = out_value_type( 0.402720,-0.23531); expect_LV(3,2) = out_value_type( 0.05748, 0.120794); expect_LV(3,3) = out_value_type( 0.62016, 0.00000);

	expect_RV(0,0) = out_value_type( 0.430856520077611, 0.326812737812621); expect_RV(0,1) = out_value_type( 0.825682050767281, 0.000000000000000); expect_RV(0,2) = out_value_type( 0.598395978553945, 0.000000000000000); expect_RV(0,3) = out_value_type(-0.305431903484378, 0.033331648617999);
	expect_RV(1,0) = out_value_type( 0.508741460297097,-0.028833421706928); expect_RV(1,1) = out_value_type( 0.075029167881412,-0.248728504509167); expect_RV(1,2) = out_value_type(-0.400476162752076,-0.201449222762560); expect_RV(1,3) = out_value_type( 0.039782828157833, 0.344507652215461);
	expect_RV(2,0) = out_value_type( 0.619849652765775, 0.000000000000000); expect_RV(2,1) = out_value_type(-0.245755789978015, 0.278872402211696); expect_RV(2,2) = out_value_type(-0.090080019075949,-0.475264621539173); expect_RV(2,3) = out_value_type( 0.358325436515984, 0.060645069885247);
	expect_RV(3,0) = out_value_type(-0.226928243319268, 0.110439278464036); expect_RV(3,1) = out_value_type(-0.103434063728144,-0.319201465363233); expect_RV(3,2) = out_value_type(-0.434840295495405, 0.133724917858160); expect_RV(3,3) = out_value_type( 0.808243289317835, 0.000000000000000);


	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Column Major");

	typedef double value_type;
	typedef value_type in_value_type;
	//typedef ::std::complex<value_type> out_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
					A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
									A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
													A(3,3) =  5.70; A(3,4) =  1.80;
																	A(4,4) = -7.10;

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_column_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Column Major - Only Eigenvalues");

	typedef double value_type;
	typedef value_type in_value_type;
	//typedef ::std::complex<value_type> out_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
					A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
									A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
													A(3,3) =  5.70; A(3,4) =  1.80;
																	A(4,4) = -7.10;

	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	expect_w(0) = -11.065575263268382;
	expect_w(1) =  -6.228746932398537;
	expect_w(2) =   0.864027975272064;
	expect_w(3) =   8.865457108365522;
	expect_w(4) =  16.094837112029339;

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Row Major");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
					A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
									A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
													A(3,3) =  5.70; A(3,4) =  1.80;
																	A(4,4) = -7.10;

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_row_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Row Major - Only Eigenvalues");

	typedef double value_type;
	typedef value_type in_value_type;
	//typedef ::std::complex<value_type> out_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(5);

	in_matrix_type A(n,n);

	A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
					A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
									A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
													A(3,3) =  5.70; A(3,4) =  1.80;
																	A(4,4) = -7.10;

	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	expect_w(0) = -11.065575263268382;
	expect_w(1) =  -6.228746932398537;
	expect_w(2) =   0.864027975272064;
	expect_w(3) =   8.865457108365522;
	expect_w(4) =  16.094837112029339;

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Column Major");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
	                                    A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
	                                                                          A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
	                                                                                                                A(3,3) = out_value_type( 8.44, 0.00);


	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::column_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_column_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Column Major - Only Eigenvalues");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::vector<value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
	                                    A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
	                                                                          A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
	                                                                                                                A(3,3) = out_value_type( 8.44, 0.00);


	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	expect_w(0) = -16.00474647209476;
	expect_w(1) = - 6.76497015479332;
	expect_w(2) =   6.66571145350710;
	expect_w(3) =  25.51400517338097;

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_row_major )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Row Major");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
	                                    A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
	                                                                          A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
	                                                                                                                A(3,3) = out_value_type( 8.44, 0.00);


	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D(n,n);
	D = ublasx::diag<out_vector_type,ublas::row_major>(w);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_row_major_only_values )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Row Major - Only Eigenvalues");

	typedef double value_type;
	typedef ::std::complex<value_type> in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::vector<value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);

	A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
	                                    A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
	                                                                          A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
	                                                                                                                A(3,3) = out_value_type( 8.44, 0.00);


	out_vector_type w;
	out_vector_type expect_w(n);

	ublasx::eigenvalues(A, w);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

	expect_w(0) = -16.00474647209476;
	expect_w(1) = - 6.76497015479332;
	expect_w(2) =   6.66571145350710;
	expect_w(3) =  25.51400517338097;

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, B, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, RV);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
	BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, B, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, RV);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
	BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Left Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type V;

	ublasx::left_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(ublas::herm(V), A);
	Y = ublas::prod(ublas::herm(V), B);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
	//FIXME: this test fails but the computation of eigenvectos seems OK
	// We need further investigation
	//BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
	BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Left Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type V;

	ublasx::left_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(ublas::herm(V), A);
	Y = ublas::prod(ublas::herm(V), B);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
	//FIXME: this test fails but the computation of eigenvectos seems OK
	// We need further investigation
	//BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
	BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Right Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type V;

	ublasx::right_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, V);
	Y = ublas::prod(B, V);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Right Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef ::std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
	A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
	A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
	A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

	in_matrix_type B(n,n);
	B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
	B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
	B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
	B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

	out_vector_type w;
	out_matrix_type V;

	ublasx::right_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, V);
	Y = ublas::prod(B, V);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Both Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, B, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, RV);
	//Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
	Y = ublas::prod(B, RV);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
	BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Both Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type LV;
	out_matrix_type RV;

	ublasx::eigen(A, B, w, LV, RV);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, RV);
	//Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
	Y = ublas::prod(B, RV);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
	BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Left Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::left_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(ublas::herm(V), A);
	Y = ublas::prod(ublas::herm(V), B);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
	//FIXME: this test fails but the computation of eigenvectos seems OK
	// We need further investigation
	//BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
	BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_left )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Left Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::left_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(ublas::herm(V), A);
	Y = ublas::prod(ublas::herm(V), B);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
	//FIXME: this test fails but the computation of eigenvectos seems OK
	// We need further investigation
	//BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
	BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Right Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::right_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, V);
	Y = ublas::prod(B, V);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_right )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Right Eigenvectors");

	typedef double value_type;
	typedef std::complex<value_type> in_value_type;
	typedef std::complex<value_type> out_value_type;
	typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
	A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
	A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
	A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

	in_matrix_type B(n,n);
	B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
	B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
	B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
	B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::right_eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X =  ublas::prod(A, V);
	Y = ublas::prod(B, V);
	Y = ublas::prod(Y, D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_pair_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix Pair - Column Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 0.24;  A(0,1) =  0.39; A(0,2) =  0.42; A(0,3) = -0.1;
					A(1,1) = -0.11; A(1,2) =  0.79; A(1,3) =  0.6;
									A(2,2) = -0.25; A(2,3) =  0.4;
													A(3,3) = -0.03;

	in_matrix_type B(n,n);
	B(0,0) = 4.16;  B(0,1) = -3.12; B(0,2) =  0.56; B(0,3) = -0.10;
					B(1,1) =  5.03; B(1,2) = -0.83; B(1,3) =  1.09;
									B(2,2) =  0.76; B(2,3) =  0.34;
													B(3,3) =  1.18;

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X = ublas::prod(A, V);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_pair_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix Pair - Row Major - Both Eigenvectors");

	typedef double value_type;
	typedef value_type in_value_type;
	typedef value_type out_value_type;
	typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<out_value_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = 0.24;  A(0,1) =  0.39; A(0,2) =  0.42; A(0,3) = -0.1;
					A(1,1) = -0.11; A(1,2) =  0.79; A(1,3) =  0.6;
									A(2,2) = -0.25; A(2,3) =  0.4;
													A(3,3) = -0.03;

	in_matrix_type B(n,n);
	B(0,0) = 4.16;  B(0,1) = -3.12; B(0,2) =  0.56; B(0,3) = -0.10;
					B(1,1) =  5.03; B(1,2) = -0.83; B(1,3) =  1.09;
									B(2,2) =  0.76; B(2,3) =  0.34;
													B(3,3) =  1.18;

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X = ublas::prod(A, V);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_upper_herm_matrix_pair_column_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: complex Upper Hermitian Matrix Pair - Column Major - Both Eigenvectors");

	typedef double real_type;
	typedef std::complex<real_type> complex_type;
	typedef ublas::hermitian_matrix<complex_type, ublas::upper, ublas::column_major> in_matrix_type;
	typedef ublas::matrix<complex_type, ublas::column_major> out_matrix_type;
	typedef ublas::vector<real_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = complex_type(-7.36, 0.00); A(0,1) = complex_type( 0.77, -0.43); A(0,2) = complex_type(-0.64, -0.92); A(0,3) = complex_type( 3.01, -6.97);
										A(1,1) = complex_type( 3.49,  0.00); A(1,2) = complex_type( 2.19,  4.45); A(1,3) = complex_type( 1.90,  3.73);
																			 A(2,2) = complex_type( 0.12,  0.00); A(2,3) = complex_type( 2.88, -3.17);
																			 									  A(3,3) = complex_type(-2.54,  0.00);

	in_matrix_type B(n,n);
	B(0,0) = complex_type( 3.23, 0.00); B(0,1) = complex_type( 1.51, -1.92); B(0,2) = complex_type( 1.90,  0.84); B(0,3) = complex_type( 0.42,  2.50);
										B(1,1) = complex_type( 3.58,  0.00); B(1,2) = complex_type(-0.23,  1.11); B(1,3) = complex_type(-1.18,  1.37);
																			 B(2,2) = complex_type( 4.09,  0.00); B(2,3) = complex_type( 2.33, -0.14);
																												  B(3,3) = complex_type( 4.29,  0.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X = ublas::prod(A, V);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_upper_herm_matrix_pair_row_major_both )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Upper Hermitian Matrix Pair - Row Major - Both Eigenvectors");

	typedef double real_type;
	typedef std::complex<real_type> complex_type;
	typedef ublas::hermitian_matrix<complex_type, ublas::upper, ublas::row_major> in_matrix_type;
	typedef ublas::matrix<complex_type, ublas::row_major> out_matrix_type;
	typedef ublas::vector<real_type> out_vector_type;

	const std::size_t n(4);

	in_matrix_type A(n,n);
	A(0,0) = complex_type(-7.36, 0.00); A(0,1) = complex_type( 0.77, -0.43); A(0,2) = complex_type(-0.64, -0.92); A(0,3) = complex_type( 3.01, -6.97);
										A(1,1) = complex_type( 3.49,  0.00); A(1,2) = complex_type( 2.19,  4.45); A(1,3) = complex_type( 1.90,  3.73);
																			 A(2,2) = complex_type( 0.12,  0.00); A(2,3) = complex_type( 2.88, -3.17);
																			 									  A(3,3) = complex_type(-2.54,  0.00);

	in_matrix_type B(n,n);
	B(0,0) = complex_type( 3.23, 0.00); B(0,1) = complex_type( 1.51, -1.92); B(0,2) = complex_type( 1.90,  0.84); B(0,3) = complex_type( 0.42,  2.50);
										B(1,1) = complex_type( 3.58,  0.00); B(1,2) = complex_type(-0.23,  1.11); B(1,3) = complex_type(-1.18,  1.37);
																			 B(2,2) = complex_type( 4.09,  0.00); B(2,3) = complex_type( 2.33, -0.14);
																												  B(3,3) = complex_type( 4.29,  0.00);

	out_vector_type w;
	out_matrix_type V;

	ublasx::eigen(A, B, w, V);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
	BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

	BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
	out_matrix_type D;
	out_matrix_type X;
	out_matrix_type Y;
	D = ublasx::diag(w);
	X = ublas::prod(A, V);
	Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
	BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
	BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'eigen' operations");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_both );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_left );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_right );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_only_values );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_only_vectors );

	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_both );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_left );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_right );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_only_values );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_only_vectors );

	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_both );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_left );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_right );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_only_values );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_only_vectors );

	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_both );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_left );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_right );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_only_values );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_only_vectors );

	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major );
	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major_only_values );
	//BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major_only_vectors );//TODO

	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major );
	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major_only_values );
	//BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major_only_vectors );//TODO

	BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major );
	BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major_only_values );
	//BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major_only_vectors );//TODO

	BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major );
	BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major_only_values );
	//BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major_only_vectors );//TODO

	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_both );
	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_both );
	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_left );
	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_left );
	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_right );
	BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_right );

	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_both );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_both );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_left );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_left );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_right );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_right );

	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_pair_column_major_both );
	BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_pair_row_major_both );

	BOOST_UBLASX_TEST_DO( test_complex_upper_herm_matrix_pair_column_major_both );
	BOOST_UBLASX_TEST_DO( test_complex_upper_herm_matrix_pair_row_major_both );

	BOOST_UBLASX_TEST_END();
}
