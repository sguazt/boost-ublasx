/**
 * \file libs/numeric/ublasx/layout_type.cpp
 *
 * \brief Test suite for the \c layout_type type traits.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/mpl/assert.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/type_traits/is_same.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( dense_matrix_column_major )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Dense Matrix - Column Major." );

	// NOTE: value types don't care. Just use different types to make sure that they are not considered
	typedef double value_type1;
	typedef std::complex<double> value_type2;

	typedef ublas::matrix<value_type1, ublas::column_major> matrix_type1;
	typedef ublas::matrix<value_type2, ublasx::layout_type<matrix_type1>::type> matrix_type2;

	// To check at compile time, uncomment this
	BOOST_MPL_ASSERT((
		   boost::is_same<
			  ublas::matrix_traits<matrix_type1>::orientation_category,
			  ublas::matrix_traits<matrix_type2>::orientation_category
			>
	));

	// To check at runtime time, uncomment this
	bool flag = boost::is_same<
				  ublas::matrix_traits<matrix_type1>::orientation_category,
				  ublas::matrix_traits<matrix_type2>::orientation_category
				>::value;
	BOOST_UBLASX_TEST_CHECK( flag );
	//if (!flag)
	//{
	//	BOOST_UBLASX_DEBUG_TACE( "Different layout storage." );
	//}
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'layout_type' type traits");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( dense_matrix_column_major );

	BOOST_UBLASX_TEST_END();
}
