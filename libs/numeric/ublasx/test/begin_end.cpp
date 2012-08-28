/**
 * \file libs/numeric/ublasx/begin_end.hpp
 *
 * \brief Test suite for the \c begin and \c end operations.
 *
 * Copyright (c) 2009, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <cmath>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/begin.hpp>
#include <boost/numeric/ublasx/operation/end.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <boost/numeric/ublasx/traits/const_iterator_type.hpp>
#include <boost/numeric/ublasx/traits/iterator_type.hpp>
#include <iostream>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double TOL(1.0e-5); ///< Used for comparing two real numbers.


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;

BOOST_UBLASX_TEST_DEF( test_vector_iteration )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Iteration" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;

	vector_type v(5);

	v(0) = 0.555950;
	v(1) = 0.108929;
	v(2) = 0.948014;
	v(3) = 0.023787;
	v(4) = 1.023787;


	vector_type::size_type ix = 0;
	for (
		ublasx::iterator_type<vector_type>::type it = ublasx::begin<vector_type>(v);
		it != ublasx::end<vector_type>(v);
		++it
	) {
		BOOST_UBLASX_DEBUG_TRACE( "*it = " << *it << " ==> " << v(ix) );
		BOOST_UBLASX_TEST_CHECK_CLOSE( *it, v(ix), TOL );
		++ix;
	}
}


BOOST_UBLASX_TEST_DEF( test_vector_const_iteration )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Vector Const Iteration" );

	typedef double value_type;
	typedef ublas::vector<value_type> vector_type;

	vector_type v(5);

	v(0) = 0.555950;
	v(1) = 0.108929;
	v(2) = 0.948014;
	v(3) = 0.023787;
	v(4) = 1.023787;


	vector_type::size_type ix = 0;
	for (
		ublasx::const_iterator_type<vector_type>::type it = ublasx::begin<vector_type>(v);
		it != ublasx::end<vector_type>(v);
		++it
	) {
		BOOST_UBLASX_DEBUG_TRACE( "*it = " << *it << " ==> " << v(ix) );
		BOOST_UBLASX_TEST_CHECK_CLOSE( *it, v(ix), TOL );
		++ix;
	}
}


BOOST_UBLASX_TEST_DEF( test_row_major_matrix_iteration )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Row-major Matrix Iteration" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;
	typedef ublasx::iterator_type<matrix_type, ublasx::tag::major>::type outer_iterator_type;
	typedef ublasx::iterator_type<matrix_type, ublasx::tag::minor>::type inner_iterator_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	matrix_type::size_type row(0);
	for (
		outer_iterator_type outer_it = ublasx::begin<ublasx::tag::major>(A);
		outer_it != ublasx::end<ublasx::tag::major>(A);
		++outer_it
	) {
		matrix_type::size_type col(0);

		for (
			inner_iterator_type inner_it = ublasx::begin(outer_it);
			inner_it != ublasx::end(outer_it);
			++inner_it
		) {
			BOOST_UBLASX_DEBUG_TRACE( "*it = " << *inner_it << " ==> " << A(row,col) );
			BOOST_UBLASX_TEST_CHECK_CLOSE( *inner_it, A(row,col), TOL );

			++col;
		}

		++row;
	}
}


BOOST_UBLASX_TEST_DEF( test_col_major_matrix_iteration )
{
	BOOST_UBLASX_DEBUG_TRACE( "TEST Column-major Matrix Iteration" );

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ublasx::iterator_type<matrix_type, ublasx::tag::major>::type outer_iterator_type;
	typedef ublasx::iterator_type<matrix_type, ublasx::tag::minor>::type inner_iterator_type;

	matrix_type A(5,4);

	A(0,0) = 0.555950; A(0,1) = 0.274690; A(0,2) = 0.540605; A(0,3) = 0.798938;
	A(1,0) = 0.108929; A(1,1) = 0.830123; A(1,2) = 0.891726; A(1,3) = 0.895283;
	A(2,0) = 0.948014; A(2,1) = 0.973234; A(2,2) = 0.216504; A(2,3) = 0.883152;
	A(3,0) = 0.023787; A(3,1) = 0.675382; A(3,2) = 0.231751; A(3,3) = 0.450332;
	A(4,0) = 1.023787; A(4,1) = 1.675382; A(4,2) = 1.231751; A(4,3) = 1.450332;


	matrix_type::size_type col(0);
	for (
		outer_iterator_type outer_it = ublasx::begin<ublasx::tag::major>(A);
		outer_it != ublasx::end<ublasx::tag::major>(A);
		++outer_it
	) {
		matrix_type::size_type row(0);

		for (
			inner_iterator_type inner_it = ublasx::begin(outer_it);
			inner_it != ublasx::end(outer_it);
			++inner_it
		) {
			BOOST_UBLASX_DEBUG_TRACE( "*it = " << *inner_it << " ==> " << A(row,col) );
			BOOST_UBLASX_TEST_CHECK_CLOSE( *inner_it, A(row,col), TOL );

			++row;
		}

		++col;
	}
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'begin'/'end' operations");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_vector_iteration );
	BOOST_UBLASX_TEST_DO( test_vector_const_iteration );
	BOOST_UBLASX_TEST_DO( test_row_major_matrix_iteration );
	BOOST_UBLASX_TEST_DO( test_col_major_matrix_iteration );

	BOOST_UBLASX_TEST_END();
}
