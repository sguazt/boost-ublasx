/**
 * \file boost/numeric/ublasx/operation/for_each.hpp
 *
 * The \c for_each operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_FOR_EACH_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_FOR_EACH_HPP


#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/tags.hpp>
#include <cstddef>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail {

template <std::size_t Dim>
struct for_each_by_dim_impl;

template <>
struct for_each_by_dim_impl<1>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type r = 0; r < nr; ++r)
		{
			for (size_type c = 0; c < nc; ++c)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_dim_impl<2>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type c = 0; c < nc; ++c)
		{
			for (size_type r = 0; r < nr; ++r)
			{
				f(me()(r,c));
			}
		}
	}
};


template <typename TagT, typename OrientationT>
struct for_each_by_tag_impl;

template <>
struct for_each_by_tag_impl<tag::major, row_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type r = 0; r < nr; ++r)
		{
			for (size_type c = 0; c < nc; ++c)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_tag_impl<tag::major, column_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type c = 0; c < nc; ++c)
		{
			for (size_type r = 0; r < nr; ++r)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_tag_impl<tag::minor, row_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type c = 0; c < nc; ++c)
		{
			for (size_type r = 0; r < nr; ++r)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_tag_impl<tag::minor, column_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type r = 0; r < nr; ++r)
		{
			for (size_type c = 0; c < nc; ++c)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_tag_impl<tag::leading, row_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type c = 0; c < nc; ++c)
		{
			for (size_type r = 0; r < nr; ++r)
			{
				f(me()(r,c));
			}
		}
	}
};

template <>
struct for_each_by_tag_impl<tag::leading, column_major_tag>
{
	template <typename MatrixExprT, typename UnaryFunctorT>
	static void apply(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
	{
		typedef typename matrix_traits<MatrixExprT>::size_type size_type;

		size_type nr = num_rows(me);
		size_type nc = num_columns(me);
		for (size_type r = 0; r < nr; ++r)
		{
			for (size_type c = 0; c < nc; ++c)
			{
				f(me()(r,c));
			}
		}
	}
};


} // Namespace detail


/**
 * \brief Apply a function to a vector expression.
 *
 * \tparam VectorExprT The type of input vector expression.
 * \tparam UnaryFunctorT The type of the function to be applied.
 *
 * \param ve The input vector expression.
 * \param f The unary function to be applied to the vector expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT, typename UnaryFunctorT>
void for_each(vector_expression<VectorExprT> const& ve, UnaryFunctorT f)
{
	typedef typename vector_traits<VectorExprT>::size_type size_type;

	size_type n = size(ve);
	for (size_type i = 0; i < n; ++i)
	{
		f(ve()(i));
	}
}


/**
 * \brief Apply a function to a matrix expression.
 *
 * \tparam MatrixExprT The type of input matrix expression.
 * \tparam UnaryFunctorT The type of the function to be applied.
 *
 * \param me The input matrix expression.
 * \param f The unary function to be applied to the matrix expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT, typename UnaryFunctorT>
void for_each(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
{
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;

	size_type nr = num_rows(me);
	size_type nc = num_columns(me);
	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			f(me()(r,c));
		}
	}
}


/**
 * \brief Apply a function to a matrix expression along the given dimension.
 *
 * \tparam Dim The dimension to follow when applying the function.
 *  Valid values are: 1 (by rows), and 2 (by columns).
 * \tparam MatrixExprT The type of input matrix expression.
 * \tparam UnaryFunctorT The type of the function to be applied.
 *
 * \param me The input matrix expression.
 * \param f The unary function to be applied to the matrix expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <std::size_t Dim, typename MatrixExprT, typename UnaryFunctorT>
void for_each(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
{
	detail::for_each_by_dim_impl<Dim>::template apply(me, f);
}


/**
 * \brief Apply a function to a matrix expression along the given dimension.
 *
 * \tparam TagT The dimension to follow when applying the function.
 *  Valid values are \c tag::major, \c tag::minor, and \c tag::leading.
 * \tparam MatrixExprT The type of input matrix expression.
 * \tparam UnaryFunctorT The type of the function to be applied.
 *
 * \param me The input matrix expression.
 * \param f The unary function to be applied to the matrix expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename TagT, typename MatrixExprT, typename UnaryFunctorT>
void for_each_by_tag(matrix_expression<MatrixExprT> const& me, UnaryFunctorT f)
{
	detail::for_each_by_tag_impl<TagT, typename matrix_traits<MatrixExprT>::orientation_category>::template apply(me, f);
}


}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_FOR_EACH_HPP
