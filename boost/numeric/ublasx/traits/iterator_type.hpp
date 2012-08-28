/**
 * \file boost/numeric/ublasx/traits/iterator_type.hpp
 *
 * \brief Iterator to a given container type.
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */


#ifndef BOOST_NUMERIC_UBLASX_TRAITS_ITERATOR_TYPE_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_ITERATOR_TYPE_HPP


#include <boost/numeric/ublasx/tags.hpp>


namespace boost { namespace numeric { namespace ublasx {

/// An iterator for the given container type over the given dimension.
// Accepted to be included in Boost.uBLAS; use this.
using ::boost::numeric::ublas::iterator_type;

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_ITERATOR_TYPE_HPP
