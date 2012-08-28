/**
 * \file boost/numeric/ublasx/traits/array_type.hpp
 *
 * \brief Traits type for determining the array type of a matrix expression.
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_TRAITS_ARRAY_TYPE_HPP
#define BOOST_NUMERIC_UBLASX_TRAITS_ARRAY_TYPE_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_cv.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {

template <typename MatrixT>
struct array_type_impl
{
	typedef typename MatrixT::array_type type;
};

template <typename MatrixExprT>
struct array_type_impl< matrix_expression<MatrixExprT> >
{
	typedef typename MatrixExprT::array_type type;
};

}} // Namespace detail::<unnamed>


template <typename MatrixT>
struct array_type
{
//	typedef typename MatrixT::array_type type;
	typedef typename detail::array_type_impl<
							typename ::boost::remove_const<
								typename ::boost::remove_cv<MatrixT>::type
							>::type
						>::type type;
};

//template <typename MatrixExprT>
//struct array_type< matrix_expression<MatrixExprT> >
//{
//	typedef typename MatrixExprT::array_type type;
//};

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_TRAITS_ARRAY_TYPE_HPP
