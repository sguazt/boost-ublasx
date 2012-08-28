/**
 * \file libs/numeric/ublasx/test/lsq.cpp
 *
 * \brief Test the \c lsq operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/lsq.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_lls_qr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - LLS - QR Method");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(nr);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_qr(A, b);
	expect_x(0) = -0.79974;
	expect_x(1) = -3.28796;
	expect_x(2) = -7.47498;
	expect_x(3) =  4.93927;
	expect_x(4) =  0.76783;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_lls_qr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - LLS - QR Method");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(6);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_qr(A, b);
	expect_x(0) = -0.799744726899358;
	expect_x(1) = -3.287963505993538;
	expect_x(2) = -7.474984265142480;
	expect_x(3) =  4.939273145125775;
	expect_x(4) =  0.767833440867089;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_lls_qr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - LLS - QR Method");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_qr(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_lls_qr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - LLS - QR Method");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_qr(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_lls_svd )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - LLS - SVD Method");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(nr);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_svd(A, b);
	expect_x(0) = -0.79974;
	expect_x(1) = -3.28796;
	expect_x(2) = -7.47498;
	expect_x(3) =  4.93927;
	expect_x(4) =  0.76783;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_lls_svd )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - LLS - SVD Method");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(6);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_svd(A, b);
	expect_x(0) = -0.799744726899358;
	expect_x(1) = -3.287963505993538;
	expect_x(2) = -7.474984265142480;
	expect_x(3) =  4.939273145125775;
	expect_x(4) =  0.767833440867089;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_lls_svd )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - LLS - SVD Method");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_svd(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_lls_svd )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - LLS - SVD Method");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq_svd(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_lls )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - LLS");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(nr);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq(A, b);
	expect_x(0) = -0.79974;
	expect_x(1) = -3.28796;
	expect_x(2) = -7.47498;
	expect_x(3) =  4.93927;
	expect_x(4) =  0.76783;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_lls )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - LLS");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(5);
	const ::std::size_t n(6);

	matrix_type A(nr,nc);
	A(0,0) = -0.09; A(0,1) =  0.14; A(0,2) = -0.46; A(0,3) =  0.68; A(0,4) =  1.29;
	A(1,0) = -1.56; A(1,1) =  0.20; A(1,2) =  0.29; A(1,3) =  1.09; A(1,4) =  0.51;
	A(2,0) = -1.48; A(2,1) = -0.43; A(2,2) =  0.89; A(2,3) = -0.71; A(2,4) = -0.96;
	A(3,0) = -1.09; A(3,1) =  0.84; A(3,2) =  0.77; A(3,3) =  2.11; A(3,4) = -1.27;
	A(4,0) =  0.08; A(4,1) =  0.55; A(4,2) = -1.13; A(4,3) =  0.14; A(4,4) =  1.74;
	A(5,0) = -1.59; A(5,1) = -0.72; A(5,2) =  1.06; A(5,3) =  1.24; A(5,4) =  0.34;

	vector_type b(n);
	b(0) =  7.4;
	b(1) =  4.2;
	b(2) = -8.3;
	b(3) =  1.8;
	b(4) =  8.6;
	b(5) =  2.1;

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq(A, b);
	expect_x(0) = -0.799744726899358;
	expect_x(1) = -3.287963505993538;
	expect_x(2) = -7.474984265142480;
	expect_x(3) =  4.939273145125775;
	expect_x(4) =  0.767833440867089;


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_lls )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - LLS");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_lls )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - LLS");

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t nr(5);
	const ::std::size_t nc(4);
	const ::std::size_t n(5);

	matrix_type A(nr,nc);
	A(0,0) = value_type( 0.47,-0.34); A(0,1) = value_type(-0.40, 0.54); A(0,2) = value_type( 0.60, 0.01); A(0,3) = value_type( 0.80,-1.02);
	A(1,0) = value_type(-0.32,-0.23); A(1,1) = value_type(-0.05, 0.20); A(1,2) = value_type(-0.26,-0.44); A(1,3) = value_type(-0.43, 0.17);
	A(2,0) = value_type( 0.35,-0.60); A(2,1) = value_type(-0.52,-0.34); A(2,2) = value_type( 0.87,-0.11); A(2,3) = value_type(-0.34,-0.09);
	A(3,0) = value_type( 0.89, 0.71); A(3,1) = value_type(-0.45,-0.45); A(3,2) = value_type(-0.02,-0.57); A(3,3) = value_type( 1.14,-0.78);
	A(4,0) = value_type(-0.19, 0.06); A(4,1) = value_type( 0.11,-0.85); A(4,2) = value_type( 1.44, 0.80); A(4,3) = value_type( 0.07, 1.14);

	vector_type b(n);
	b(0) = value_type(-1.08,-2.59);
	b(1) = value_type(-2.61,-1.49);
	b(2) = value_type( 3.13,-3.61);
	b(3) = value_type( 7.33,-8.01);
	b(4) = value_type( 9.12, 7.63);

	vector_type x;
	vector_type expect_x(nc);

	x = ublasx::llsq(A, b);
	expect_x(0) = value_type(18.79221131415766,  9.58842519277362);
	expect_x(1) = value_type(19.15428710640874,  2.12745817492880);
	expect_x(2) = value_type( 2.79395045513666, 10.27260222931818);
	expect_x(3) = value_type( 7.14260392345630,-11.39648999358683);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "b = " << b );
    BOOST_UBLASX_DEBUG_TRACE( "min_x ||Ax-b||_2 --> x = " << x );
	BOOST_UBLASX_TEST_CHECK( ublasx::size(x) == ublasx::size(expect_x) );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( x, expect_x, nc, tol );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'llsq' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_lls_qr );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_lls_qr );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_lls_qr );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_lls_qr );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_lls_svd );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_lls_svd );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_lls_svd );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_lls_svd );
	BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_lls );
	BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_lls );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_lls );
	BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_lls );

	BOOST_UBLASX_TEST_END();
}
