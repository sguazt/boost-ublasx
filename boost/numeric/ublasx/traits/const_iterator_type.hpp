/**
 * \file boost/numeric/ublasx/traits/const_iterator_type.hpp
 *
 * \brief Const iterator to a given container type.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef BOOST_NUMERIC_UBLASX_TRAITS_CONST_ITERATOR_TYPE_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_CONST_ITERATOR_TYPE_HPP


#include <boost/numeric/ublasx/tags.hpp>


namespace boost { namespace numeric { namespace ublasx {

/// A const iterator for the given container type over the given dimension.
// Accepted to be included in Boost.uBLAS; use this.
using ::boost::numeric::ublas::const_iterator_type;

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_CONST_ITERATOR_TYPE_HPP
