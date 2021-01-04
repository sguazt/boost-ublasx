/**
 * \file boost/numeric/ublasx/traits/layout_type.hpp
 *
 * \brief Traits type for determining the layout type of a matrix expression.
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

#ifndef BOOST_NUMERIC_UBLASX_TRAITS_LAYOUT_TYPE_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_LAYOUT_TYPE_HPP


#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/traits.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail {

// Fall-back case for unknown_orientation_tag and possible future tags.
template <typename OrientationT>
struct orientation_to_layout_type
{
	typedef row_major type;
};

template <>
struct orientation_to_layout_type<column_major_tag>
{
	typedef column_major type;
};

template <>
struct orientation_to_layout_type<row_major_tag>
{
	typedef row_major type;
};


template <typename MatrixT>
struct layout_type_impl
{
	typedef typename orientation_to_layout_type<typename matrix_traits<MatrixT>::orientation_category>::type type;
};

} // Namespace detail


template <typename MatrixT>
struct layout_type
{
	typedef typename detail::layout_type_impl<MatrixT>::type type;
};

template <typename MatrixT>
struct layout_type< matrix_expression<MatrixT> >
{
	typedef typename detail::layout_type_impl<MatrixT>::type type;
};

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_LAYOUT_TYPE_HPP
