/**
 * \file boost/numeric/ublasx/traits/is_dimension_tag.hpp
 *
 * \brief Traits type for determining if a given type is a dimension tag.
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

#ifndef BOOST_NUMERIC_UBLASX_TRAITS_IS_DIMENSION_TAG_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_IS_DIMENSION_TAG_HPP

#include <boost/numeric/ublas/tags.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_base_of.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <typename T>
struct is_dimension_tag: ::boost::integral_constant<bool,false>
{
};

template <typename T>
struct is_dimension_tag< tag::dimension_tag<T> >: ::boost::integral_constant<bool,true>
//								typename ::boost::is_base_of<
//											tag::dimension_tag<T>,
//											T
//										>::type
//							>
{
	// empty
};

}}} // boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_IS_DIMENSION_TAG_HPP
