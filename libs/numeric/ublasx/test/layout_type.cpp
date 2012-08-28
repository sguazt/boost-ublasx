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


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'layout_type' type traits");

	typedef double value_type;

	typedef ublas::matrix<value_type, ublas::column_major> matrix_type1;

	typedef ublas::matrix<std::complex<double>, ublasx::layout_type<matrix_type1>::type> matrix_type2;
//	typedef ublas::matrix<std::complex<double>, ublas::row_major> matrix_type2;
//	typedef ublas::matrix<std::complex<double>, ublas::column_major> matrix_type2;

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
	if (!flag)
	{
		std::cerr << "Different layout storage." << std::endl;
	}
}
