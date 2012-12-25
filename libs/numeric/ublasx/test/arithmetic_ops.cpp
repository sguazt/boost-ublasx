/**
 * \file boost/numeric/ublasx/test/arithmetic_opts.cpp
 *
 * \brief Test suite for matrix/vector arithmetic operators.
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/arithmetic_ops.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;


static const double tol(1e-5);


BOOST_UBLASX_TEST_DEF( scalar_div_real_vector )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Scalar divides Real Vector");

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;

	const ::std::size_t n(4);

	value_type c(2);
	vector_type v(n);

	vector_type res;
	vector_type expect(n);

	v(0) = 1;
	v(1) = 2;
	v(2) = 3;
	v(3) = 4;

	for (::std::size_t i = 0; i < n; ++i)
	{
		expect(i) = c/v(i);
	}

	res = c / v;

	BOOST_UBLASX_DEBUG_TRACE("c=" << c);
	BOOST_UBLASX_DEBUG_TRACE("v=" << v);
	BOOST_UBLASX_DEBUG_TRACE("c / v? " << res);
	BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( res, expect, n, tol );
}


BOOST_UBLASX_TEST_DEF( scalar_div_real_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Scalar divides Real Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;

	const ::std::size_t nr(3);
	const ::std::size_t nc(4);

	value_type c(2);
	matrix_type A(nr,nc);

	matrix_type res;
	matrix_type expect(nr,nc);

	A(0,0) = 1; A(0,1) = 4; A(0,2) = 7; A(0,3) = 10;
	A(1,0) = 2; A(1,1) = 5; A(1,2) = 8; A(1,3) = 11;
	A(2,0) = 3; A(2,1) = 6; A(2,2) = 9; A(2,3) = 12;

	for (::std::size_t i = 0; i < nr; ++i)
	{
		for (::std::size_t j = 0; j < nc; ++j)
		{
			expect(i,j) = c/A(i,j);
		}
	}

	res = c / A;

	BOOST_UBLASX_DEBUG_TRACE("c=" << c);
	BOOST_UBLASX_DEBUG_TRACE("A=" << A);
	BOOST_UBLASX_DEBUG_TRACE("c / A? " << res);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( res, expect, nr, nc, tol );
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( scalar_div_real_vector );
	BOOST_UBLASX_TEST_DO( scalar_div_real_matrix );

	BOOST_UBLASX_TEST_END();
}
