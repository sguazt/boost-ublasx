/**
 * \file libs/numeric/ublasx/test/rep.cpp
 *
 * \brief Test suite for the \c rep operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/rep.hpp>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1e-5;


BOOST_UBLASX_TEST_DEF( rep_matrix_col_major )
{
	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(2);
	const ::std::size_t nc(2);
	const ::std::size_t out_nr(2*3);
	const ::std::size_t out_nc(2*4);

	matrix_type A(ublas::identity_matrix<value_type>(nr,nc));

	matrix_type E(out_nr, out_nc);
	ublas::subrange(E, 0, 2, 0, 2) = A; ublas::subrange(E, 0, 2, 2, 4) = A; ublas::subrange(E, 0, 2, 4, 6) = A; ublas::subrange(E, 0, 2, 6, 8) = A;
	ublas::subrange(E, 2, 4, 0, 2) = A; ublas::subrange(E, 2, 4, 2, 4) = A; ublas::subrange(E, 2, 4, 4, 6) = A; ublas::subrange(E, 2, 4, 6, 8) = A;
	ublas::subrange(E, 4, 6, 0, 2) = A; ublas::subrange(E, 4, 6, 2, 4) = A; ublas::subrange(E, 4, 6, 4, 6) = A; ublas::subrange(E, 4, 6, 6, 8) = A;

	matrix_type X;

	X = ublasx::rep(A, 3, 4);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Expected rep(A, " << out_nr << ", " << out_nc << ")=" << E);
	BOOST_UBLASX_DEBUG_TRACE("X=" << X);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, out_nr, out_nc, tol );
}


BOOST_UBLASX_TEST_DEF( rep_matrix_row_major )
{
	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(2);
	const ::std::size_t nc(2);
	const ::std::size_t out_nr(2*3);
	const ::std::size_t out_nc(2*4);

	matrix_type A(ublas::identity_matrix<value_type>(nr,nc));

	matrix_type E(out_nr, out_nc);
	ublas::subrange(E, 0, 2, 0, 2) = A; ublas::subrange(E, 0, 2, 2, 4) = A; ublas::subrange(E, 0, 2, 4, 6) = A; ublas::subrange(E, 0, 2, 6, 8) = A;
	ublas::subrange(E, 2, 4, 0, 2) = A; ublas::subrange(E, 2, 4, 2, 4) = A; ublas::subrange(E, 2, 4, 4, 6) = A; ublas::subrange(E, 2, 4, 6, 8) = A;
	ublas::subrange(E, 4, 6, 0, 2) = A; ublas::subrange(E, 4, 6, 2, 4) = A; ublas::subrange(E, 4, 6, 4, 6) = A; ublas::subrange(E, 4, 6, 6, 8) = A;

	matrix_type X;

	X = ublasx::rep(A, 3, 4);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("Expected rep(A, " << out_nr << ", " << out_nc << ")=" << E);
	BOOST_UBLASX_DEBUG_TRACE("X=" << X);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, out_nr, out_nc, tol );
}


BOOST_UBLASX_TEST_DEF( rep_vector )
{
	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;
	typedef ublas::matrix<value_type> matrix_type;

	const ::std::size_t n(3);
	const ::std::size_t out_nr(3*3);
	const ::std::size_t out_nc(1*4);

	vector_type v(n);
	v(0) = 1;
	v(1) = 2;
	v(2) = 3;

	matrix_type E(out_nr, out_nc);
	E(0,0) = 1; E(0,1) = 1; E(0,2) = 1; E(0,3) = 1;
	E(1,0) = 2; E(1,1) = 2; E(1,2) = 2; E(1,3) = 2;
	E(2,0) = 3; E(2,1) = 3; E(2,2) = 3; E(2,3) = 3;
	E(3,0) = 1; E(3,1) = 1; E(3,2) = 1; E(3,3) = 1;
	E(4,0) = 2; E(4,1) = 2; E(4,2) = 2; E(4,3) = 2;
	E(5,0) = 3; E(5,1) = 3; E(5,2) = 3; E(5,3) = 3;
	E(6,0) = 1; E(6,1) = 1; E(6,2) = 1; E(6,3) = 1;
	E(7,0) = 2; E(7,1) = 2; E(7,2) = 2; E(7,3) = 2;
	E(8,0) = 3; E(8,1) = 3; E(8,2) = 3; E(8,3) = 3;

	matrix_type X;

	X = ublasx::rep(v, out_nr, out_nc);
	BOOST_UBLASX_DEBUG_TRACE("v=" << v);
	BOOST_UBLASX_DEBUG_TRACE("Expected rep(v, " << out_nr << ", " << out_nc << ")=" << E);
	BOOST_UBLASX_DEBUG_TRACE("X=" << X);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, out_nr, out_nc, tol );
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( rep_matrix_col_major );
	BOOST_UBLASX_TEST_DO( rep_matrix_row_major );
	BOOST_UBLASX_TEST_DO( rep_vector );

	BOOST_UBLASX_TEST_END();
}
