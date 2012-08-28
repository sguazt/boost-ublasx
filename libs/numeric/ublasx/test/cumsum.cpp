/**
 * \file libs/numeric/ublasx/cumsum.cpp
 *
 * \brief Test suite for the \c cumsum operation.
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
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/cumsum.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <functional>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double tol = 1.0e-5;


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_vector_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Container" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::zero_vector<value_type> zero_vector_type;
	typedef ublas::vector_traits<vector_type>::size_type size_type;

	size_type n(5);

	vector_type v(n);

	v(0) = 0.0;
	v(1) = 0.108929;
	v(2) = 0.0;
	v(3) = 0.0;
	v(4) = 1.023787;

	zero_vector_type z(n);

	vector_type expect(n, 0);
	vector_type res(n, 0);


	// cumsum(z)
	expect = vector_type(n, 0);
	res = ublasx::cumsum(z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );

	// cumsum(v)
	expect = vector_type(n);
	expect(0) = v(0);
	expect(1) = v(0)+v(1);
	expect(2) = v(0)+v(1)+v(2);
	expect(3) = v(0)+v(1)+v(2)+v(3);
	expect(4) = v(0)+v(1)+v(2)+v(3)+v(4);
	res = ublasx::cumsum(v);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << v << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );

	// cumsum<1>(v)
	expect = vector_type(n);
	expect(0) = v(0);
	expect(1) = v(0)+v(1);
	expect(2) = v(0)+v(1)+v(2);
	expect(3) = v(0)+v(1)+v(2)+v(3);
	expect(4) = v(0)+v(1)+v(2)+v(3)+v(4);
	res = ublasx::cumsum<1>(v);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << v << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_expression )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Expression" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector_traits<vector_type>::size_type size_type;

	size_type n(5);

	vector_type v(n);

	v(0) = 0.0;
	v(1) = 0.108929;
	v(2) = 0.0;
	v(3) = 0.0;
	v(4) = 1.023787;

	vector_type expect(n, 0);
	vector_type res(n, 0);


	// cumsum(-v)
	expect = vector_type(n);
	expect(0) = -v(0);
	expect(1) = -(v(0)+v(1));
	expect(2) = -(v(0)+v(1)+v(2));
	expect(3) = -(v(0)+v(1)+v(2)+v(3));
	expect(4) = -(v(0)+v(1)+v(2)+v(3)+v(4));
	res = ublasx::cumsum(-v);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << -v << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );

	// cumsum<1>(-v)
	expect = vector_type(n);
	expect(0) = -v(0);
	expect(1) = -(v(0)+v(1));
	expect(2) = -(v(0)+v(1)+v(2));
	expect(3) = -(v(0)+v(1)+v(2)+v(3));
	expect(4) = -(v(0)+v(1)+v(2)+v(3)+v(4));
	res = ublasx::cumsum<1>(-v);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << -v << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_vector_reference )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Reference" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::vector_reference<vector_type> vector_reference_type;
	typedef ublas::vector_traits<vector_type>::size_type size_type;

	size_type n(5);

	vector_type v(n);

	v(0) = 0.0;
	v(1) = 0.108929;
	v(2) = 0.0;
	v(3) = 0.0;
	v(4) = 1.023787;

	vector_type expect(n, 0);
	vector_type res(n, 0);


	// cumsum(ref(v))
	expect = vector_type(n);
	expect(0) = v(0);
	expect(1) = v(0)+v(1);
	expect(2) = v(0)+v(1)+v(2);
	expect(3) = v(0)+v(1)+v(2)+v(3);
	expect(4) = v(0)+v(1)+v(2)+v(3)+v(4);
	res = ublasx::cumsum(vector_reference_type(v));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );

	// cumsum<1>(ref(v))
	expect = vector_type(n);
	expect(0) = v(0);
	expect(1) = v(0)+v(1);
	expect(2) = v(0)+v(1)+v(2);
	expect(3) = v(0)+v(1)+v(2)+v(3);
	expect(4) = v(0)+v(1)+v(2)+v(3)+v(4);
	res = ublasx::cumsum<1>(vector_reference_type(v));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << vector_reference_type(v) << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Container" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublas::zero_matrix<value_type> zero_matrix_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::matrix_traits<matrix_type>::size_type size_type;

	size_type nr = 5;
	size_type nc = 4;

	matrix_type A(nr, nc);

	A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
	A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
	A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

	zero_matrix_type Z(nr,nc);

	matrix_type cumsum_rows(nr,nc);
	cumsum_rows(0,0) = A(0,0); cumsum_rows(0,1) = A(0,1); cumsum_rows(0,2) = A(0,2); cumsum_rows(0,3) = A(0,3);
	cumsum_rows(1,0) = A(0,0)+A(1,0); cumsum_rows(1,1) = A(0,1)+A(1,1); cumsum_rows(1,2) = A(0,2)+A(1,2); cumsum_rows(1,3) = A(0,3)+A(1,3);
	cumsum_rows(2,0) = A(0,0)+A(1,0)+A(2,0); cumsum_rows(2,1) = A(0,1)+A(1,1)+A(2,1); cumsum_rows(2,2) = A(0,2)+A(1,2)+A(2,2); cumsum_rows(2,3) = A(0,3)+A(1,3)+A(2,3);
	cumsum_rows(3,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0); cumsum_rows(3,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1); cumsum_rows(3,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2); cumsum_rows(3,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3);
	cumsum_rows(4,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0); cumsum_rows(4,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1); cumsum_rows(4,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2); cumsum_rows(4,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

	matrix_type cumsum_cols(nr,nc);
	cumsum_cols(0,0) = A(0,0); cumsum_cols(0,1) = A(0,0)+A(0,1); cumsum_cols(0,2) = A(0,0)+A(0,1)+A(0,2); cumsum_cols(0,3) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
	cumsum_cols(1,0) = A(1,0); cumsum_cols(1,1) = A(1,0)+A(1,1); cumsum_cols(1,2) = A(1,0)+A(1,1)+A(1,2); cumsum_cols(1,3) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
	cumsum_cols(2,0) = A(2,0); cumsum_cols(2,1) = A(2,0)+A(2,1); cumsum_cols(2,2) = A(2,0)+A(2,1)+A(2,2); cumsum_cols(2,3) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
	cumsum_cols(3,0) = A(3,0); cumsum_cols(3,1) = A(3,0)+A(3,1); cumsum_cols(3,2) = A(3,0)+A(3,1)+A(3,2); cumsum_cols(3,3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
	cumsum_cols(4,0) = A(4,0); cumsum_cols(4,1) = A(4,0)+A(4,1); cumsum_cols(4,2) = A(4,0)+A(4,1)+A(4,2); cumsum_cols(4,3) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

	matrix_type expect(nr,nc);
	matrix_type res(nr,nc);


	// cumsum(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_rows(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_rows(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_columns(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_columns(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<1>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum<1>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<2>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum<2>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::major>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::major>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::minor>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::minor>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::leading>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::leading>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum(A)
	expect = cumsum_rows;
	res = ublasx::cumsum(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_rows(A)
	expect = cumsum_rows;
	res = ublasx::cumsum_rows(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_columns(A)
	expect = cumsum_cols;
	res = ublasx::cumsum_columns(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<1>(A)
	expect = cumsum_rows;
	res = ublasx::cumsum<1>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<2>(A)
	expect = cumsum_cols;
	res = ublasx::cumsum<2>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<2>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::major>(A)
	expect = cumsum_rows;
	res = ublasx::cumsum_by_tag<ublasx::tag::major>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::minor>(A)
	expect = cumsum_cols;
	res = ublasx::cumsum_by_tag<ublasx::tag::minor>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::leading>(A)
	expect = cumsum_cols;
	res = ublasx::cumsum_by_tag<ublasx::tag::leading>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_container )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Container" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublas::zero_matrix<value_type> zero_matrix_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::matrix_traits<matrix_type>::size_type size_type;

	size_type nr = 5;
	size_type nc = 4;

	matrix_type A(nr, nc);

	A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
	A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
	A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

	zero_matrix_type Z(nr,nc);

	matrix_type cumsum_rows(nr,nc);
	cumsum_rows(0,0) = A(0,0); cumsum_rows(0,1) = A(0,1); cumsum_rows(0,2) = A(0,2); cumsum_rows(0,3) = A(0,3);
	cumsum_rows(1,0) = A(0,0)+A(1,0); cumsum_rows(1,1) = A(0,1)+A(1,1); cumsum_rows(1,2) = A(0,2)+A(1,2); cumsum_rows(1,3) = A(0,3)+A(1,3);
	cumsum_rows(2,0) = A(0,0)+A(1,0)+A(2,0); cumsum_rows(2,1) = A(0,1)+A(1,1)+A(2,1); cumsum_rows(2,2) = A(0,2)+A(1,2)+A(2,2); cumsum_rows(2,3) = A(0,3)+A(1,3)+A(2,3);
	cumsum_rows(3,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0); cumsum_rows(3,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1); cumsum_rows(3,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2); cumsum_rows(3,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3);
	cumsum_rows(4,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0); cumsum_rows(4,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1); cumsum_rows(4,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2); cumsum_rows(4,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

	matrix_type cumsum_cols(nr,nc);
	cumsum_cols(0,0) = A(0,0); cumsum_cols(0,1) = A(0,0)+A(0,1); cumsum_cols(0,2) = A(0,0)+A(0,1)+A(0,2); cumsum_cols(0,3) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
	cumsum_cols(1,0) = A(1,0); cumsum_cols(1,1) = A(1,0)+A(1,1); cumsum_cols(1,2) = A(1,0)+A(1,1)+A(1,2); cumsum_cols(1,3) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
	cumsum_cols(2,0) = A(2,0); cumsum_cols(2,1) = A(2,0)+A(2,1); cumsum_cols(2,2) = A(2,0)+A(2,1)+A(2,2); cumsum_cols(2,3) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
	cumsum_cols(3,0) = A(3,0); cumsum_cols(3,1) = A(3,0)+A(3,1); cumsum_cols(3,2) = A(3,0)+A(3,1)+A(3,2); cumsum_cols(3,3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
	cumsum_cols(4,0) = A(4,0); cumsum_cols(4,1) = A(4,0)+A(4,1); cumsum_cols(4,2) = A(4,0)+A(4,1)+A(4,2); cumsum_cols(4,3) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

	matrix_type expect(nr,nc);
	matrix_type res(nr,nc);


	// cumsum(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_rows(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_rows(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_columns(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_columns(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<1>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum<1>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<2>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum<2>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::major>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::major>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::minor>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::minor>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::leading>(Z)
	expect = matrix_type(nr, nc, 0);
	res = ublasx::cumsum_by_tag<ublas::tag::leading>(Z);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(" << Z << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum(A)
	expect = cumsum_rows;
	res = ublasx::cumsum(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_rows(A)
	expect = cumsum_rows;
	res = ublasx::cumsum_rows(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_columns(A)
	expect = cumsum_cols;
	res = ublasx::cumsum_columns(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<1>(A)
	expect = cumsum_rows;
	res = ublasx::cumsum<1>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<2>(A)
	expect = cumsum_cols;
	res = ublasx::cumsum<2>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<2>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::major>(A)
	expect = cumsum_cols;
	res = ublasx::cumsum_by_tag<ublasx::tag::major>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::minor>(A)
	expect = cumsum_rows;
	res = ublasx::cumsum_by_tag<ublasx::tag::minor>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::leading>(A)
	expect = cumsum_rows;
	res = ublasx::cumsum_by_tag<ublasx::tag::leading>(A);
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(" << A << ") = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_expression )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Expression" );

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::matrix_traits<matrix_type>::size_type size_type;

	size_type nr = 5;
	size_type nc = 4;

	matrix_type A(nr, nc);

	A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
	A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
	A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

	matrix_type cumsum_rows(nr,nc);
	cumsum_rows(0,0) = A(0,0); cumsum_rows(0,1) = A(0,1); cumsum_rows(0,2) = A(0,2); cumsum_rows(0,3) = A(0,3);
	cumsum_rows(1,0) = A(0,0)+A(1,0); cumsum_rows(1,1) = A(0,1)+A(1,1); cumsum_rows(1,2) = A(0,2)+A(1,2); cumsum_rows(1,3) = A(0,3)+A(1,3);
	cumsum_rows(2,0) = A(0,0)+A(1,0)+A(2,0); cumsum_rows(2,1) = A(0,1)+A(1,1)+A(2,1); cumsum_rows(2,2) = A(0,2)+A(1,2)+A(2,2); cumsum_rows(2,3) = A(0,3)+A(1,3)+A(2,3);
	cumsum_rows(3,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0); cumsum_rows(3,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1); cumsum_rows(3,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2); cumsum_rows(3,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3);
	cumsum_rows(4,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0); cumsum_rows(4,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1); cumsum_rows(4,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2); cumsum_rows(4,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

	matrix_type cumsum_cols(nr,nc);
	cumsum_cols(0,0) = A(0,0); cumsum_cols(0,1) = A(0,0)+A(0,1); cumsum_cols(0,2) = A(0,0)+A(0,1)+A(0,2); cumsum_cols(0,3) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
	cumsum_cols(1,0) = A(1,0); cumsum_cols(1,1) = A(1,0)+A(1,1); cumsum_cols(1,2) = A(1,0)+A(1,1)+A(1,2); cumsum_cols(1,3) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
	cumsum_cols(2,0) = A(2,0); cumsum_cols(2,1) = A(2,0)+A(2,1); cumsum_cols(2,2) = A(2,0)+A(2,1)+A(2,2); cumsum_cols(2,3) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
	cumsum_cols(3,0) = A(3,0); cumsum_cols(3,1) = A(3,0)+A(3,1); cumsum_cols(3,2) = A(3,0)+A(3,1)+A(3,2); cumsum_cols(3,3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
	cumsum_cols(4,0) = A(4,0); cumsum_cols(4,1) = A(4,0)+A(4,1); cumsum_cols(4,2) = A(4,0)+A(4,1)+A(4,2); cumsum_cols(4,3) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

	matrix_type expect;
	matrix_type res;


	// cumsum(A')
	expect = ublas::trans(cumsum_cols);
	res = ublasx::cumsum(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum_rows(A')
	expect = ublas::trans(cumsum_cols);
	res = ublasx::cumsum_rows(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum_columns(A')
	expect = ublas::trans(cumsum_rows);
	res = ublasx::cumsum_columns(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum<1>(A')
	expect = ublas::trans(cumsum_cols);
	res = ublasx::cumsum<1>(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum<2>(A')
	expect = ublas::trans(cumsum_rows);
	res = ublasx::cumsum<2>(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<2>(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum<tag::major>(A')
	expect = ublas::trans(cumsum_rows);
	res = ublasx::cumsum_by_tag<ublasx::tag::major>(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum<tag::minor>(A')
	expect = ublas::trans(cumsum_cols);
	res = ublasx::cumsum_by_tag<ublasx::tag::minor>(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );

	// cumsum<tag::leading>(A')
	expect = ublas::trans(cumsum_cols);
	res = ublasx::cumsum_by_tag<ublasx::tag::leading>(ublas::trans(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(" << A << "') = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nc, nr, tol );
}


BOOST_UBLASX_TEST_DEF( test_matrix_reference )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Matrix Reference" );

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ublas::matrix_reference<matrix_type> matrix_reference_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::matrix_traits<matrix_type>::size_type size_type;

	size_type nr = 5;
	size_type nc = 4;

	matrix_type A(nr, nc);

	A(0,0) = 0.0;      A(0,1) = 0.274690; A(0,2) = 0.0;      A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.0;      A(1,2) = 0.891726; A(1,3) = 0.0;
	A(2,0) = 0.0;      A(2,1) = 0.0;      A(2,2) = 0.0;      A(2,3) = 0.0;
	A(3,0) = 0.0;      A(3,1) = 0.675382; A(3,2) = 0.0;      A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.0;      A(4,2) = 1.231751; A(4,3) = 1.0;

	matrix_type cumsum_rows(nr,nc);
	cumsum_rows(0,0) = A(0,0); cumsum_rows(0,1) = A(0,1); cumsum_rows(0,2) = A(0,2); cumsum_rows(0,3) = A(0,3);
	cumsum_rows(1,0) = A(0,0)+A(1,0); cumsum_rows(1,1) = A(0,1)+A(1,1); cumsum_rows(1,2) = A(0,2)+A(1,2); cumsum_rows(1,3) = A(0,3)+A(1,3);
	cumsum_rows(2,0) = A(0,0)+A(1,0)+A(2,0); cumsum_rows(2,1) = A(0,1)+A(1,1)+A(2,1); cumsum_rows(2,2) = A(0,2)+A(1,2)+A(2,2); cumsum_rows(2,3) = A(0,3)+A(1,3)+A(2,3);
	cumsum_rows(3,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0); cumsum_rows(3,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1); cumsum_rows(3,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2); cumsum_rows(3,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3);
	cumsum_rows(4,0) = A(0,0)+A(1,0)+A(2,0)+A(3,0)+A(4,0); cumsum_rows(4,1) = A(0,1)+A(1,1)+A(2,1)+A(3,1)+A(4,1); cumsum_rows(4,2) = A(0,2)+A(1,2)+A(2,2)+A(3,2)+A(4,2); cumsum_rows(4,3) = A(0,3)+A(1,3)+A(2,3)+A(3,3)+A(4,3);

	matrix_type cumsum_cols(nr,nc);
	cumsum_cols(0,0) = A(0,0); cumsum_cols(0,1) = A(0,0)+A(0,1); cumsum_cols(0,2) = A(0,0)+A(0,1)+A(0,2); cumsum_cols(0,3) = A(0,0)+A(0,1)+A(0,2)+A(0,3);
	cumsum_cols(1,0) = A(1,0); cumsum_cols(1,1) = A(1,0)+A(1,1); cumsum_cols(1,2) = A(1,0)+A(1,1)+A(1,2); cumsum_cols(1,3) = A(1,0)+A(1,1)+A(1,2)+A(1,3);
	cumsum_cols(2,0) = A(2,0); cumsum_cols(2,1) = A(2,0)+A(2,1); cumsum_cols(2,2) = A(2,0)+A(2,1)+A(2,2); cumsum_cols(2,3) = A(2,0)+A(2,1)+A(2,2)+A(2,3);
	cumsum_cols(3,0) = A(3,0); cumsum_cols(3,1) = A(3,0)+A(3,1); cumsum_cols(3,2) = A(3,0)+A(3,1)+A(3,2); cumsum_cols(3,3) = A(3,0)+A(3,1)+A(3,2)+A(3,3);
	cumsum_cols(4,0) = A(4,0); cumsum_cols(4,1) = A(4,0)+A(4,1); cumsum_cols(4,2) = A(4,0)+A(4,1)+A(4,2); cumsum_cols(4,3) = A(4,0)+A(4,1)+A(4,2)+A(4,3);

	matrix_type expect;
	matrix_type res;


	// cumsum(ref(A))
	expect = cumsum_rows;
	res = ublasx::cumsum(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_rows(ref(A))
	expect = cumsum_rows;
	res = ublasx::cumsum_rows(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_rows(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum_columns(ref(A))
	expect = cumsum_cols;
	res = ublasx::cumsum_columns(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_columns(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<1>(ref(A))
	expect = cumsum_rows;
	res = ublasx::cumsum<1>(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<1>(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<2>(ref(A))
	expect = cumsum_cols;
	res = ublasx::cumsum<2>(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum<2>(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::major>(ref(A))
	expect = cumsum_rows;
	res = ublasx::cumsum_by_tag<ublasx::tag::major>(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::major>(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::minor>(ref(A))
	expect = cumsum_cols;
	res = ublasx::cumsum_by_tag<ublasx::tag::minor>(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::minor>(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );

	// cumsum<tag::leading>(ref(A))
	expect = cumsum_cols;
	res = ublasx::cumsum_by_tag<ublasx::tag::leading>(matrix_reference_type(A));
	BOOST_UBLASX_DEBUG_TRACE( "cumsum_by_tag<tag::leading>(reference(" << A << ")) = " << res << " ==> " << expect );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'cumsum' operation");

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
