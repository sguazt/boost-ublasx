/**
 * \file boost/numeric/ublasx/traits/is_vector.hpp
 *
 * \brief Traits type for determining if a given type is a vector.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_TRAITS_IS_VECTOR_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_IS_VECTOR_HPP

#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_base_of.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <typename T>
struct is_vector: ::boost::integral_constant<bool,false>
//						bool,
//						typename ::boost::is_base_of<
//								vector_expression<T>,
//								T
//							>::value
//					>
{
	// empty
};

template <typename T>
struct is_vector< vector_expression<T> >: ::boost::integral_constant<bool,true>
{
	// empty
};

}}} // boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_IS_VECTOR_HPP
