/**
 * \file libs/numeric/ublasx/test/sign.cpp
 *
 * \brief Test suite for the \c sign operation.
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author comcon1 based on code of Marco Guazzone
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/sign.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-15;


BOOST_UBLASX_TEST_DEF( test_real_vector )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vector" );

	typedef double value_type;
	typedef std::size_t size_type;
	typedef ublas::vector<value_type> vector_type;

	const size_type n(4);

	vector_type v(n);

	v(0) =  0.0;
	v(1) = -2.0;
	v(2) = -3.0;
	v(3) =  4.0;

	vector_type res;
	vector_type expect_res(n);

	res = ublasx::sign(v);

	BOOST_UBLASX_DEBUG_TRACE( "v = " << v );
	BOOST_UBLASX_DEBUG_TRACE( "abs(v) = " << res );

	for (size_type i = 0; i < n; ++i)
	{
		expect_res(i) = v(i) >= 0 ? +1.0 : -1.0;
	}

	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect_res, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Matrix" );

	typedef double value_type;
	typedef std::size_t size_type;
	typedef ublas::matrix<value_type> matrix_type;

	const size_type nr(2);
	const size_type nc(3);

	matrix_type A(nr,nc);

	A(0,0) =  0; A(0,1) = -2; A(0,2) = -3;
	A(1,0) = -4; A(1,1) =  5; A(1,2) =  6;

	matrix_type R;
	matrix_type expect_R(nr,nc);

	R = ublasx::sign(A);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "abs(A) = " << R );

	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			expect_R(r,c) = A(r,c) >= 0 ? +1.0 : -1.0;
		}
	}

	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( R, expect_R, nr, nc, tol );
}


int main()
{

	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'sign' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_real_vector );
	BOOST_UBLASX_TEST_DO( test_real_matrix );

	BOOST_UBLASX_TEST_END();
}
