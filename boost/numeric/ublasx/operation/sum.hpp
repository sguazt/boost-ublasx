/**
 * \file boost/numeric/ublasx/operation/sum.hpp
 *
 * \brief The \c sum operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_SUM_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_SUM_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublasx/operation/begin.hpp>
#include <boost/numeric/ublasx/operation/end.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <cstddef>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

//XXX: already implemented in vector_expression.hpp
///**
// * \brief Compute the sum of the elements of the given vector expression.
// * \tparam VectorExprT The type of the vector expression.
// * \param ve The vector expression whose elements are summed up.
// * \return The sum of the elements of the vector expression.
// *
// * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
// */
//template <typename VectorExprT>
//typename vector_traits<VectorExprT>::value_type sum(vector_expression<VectorExprT> const& ve);
using ::boost::numeric::ublas::sum;


/**
 * \brief Compute the sum of the elements of the given matrix expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param ve The matrix expression whose elements are summed up.
 * \return The sum of the elements of the matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
typename matrix_traits<MatrixExprT>::value_type sum_all(matrix_expression<MatrixExprT> const& me);


/**
 * \brief Compute the sum of the elements over each column of the given matrix
 *  expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are summed up by row.
 * \return A vector containing the sum of the elements over each column in the
 *  given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> sum(matrix_expression<MatrixExprT> const& me);


/**
 * \brief Compute the sum of the elements over each column in the given matrix
 *  expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are summed up by row.
 * \return A vector containing the sum of the elements over each column in the
 *  given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> sum_rows(matrix_expression<MatrixExprT> const& me);


/**
 * \brief Compute the sum of the elements over each row in the given matrix
 *  expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are summed up by column.
 * \return A vector containing the sum of the elements over each row in the
 *  given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> sum_columns(matrix_expression<MatrixExprT> const& me);


/**
 * \brief Compute the sum of the elements of the given matrix expression along
 *  the given dimension tag.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are summed up by the given
 *  dimension.
 * \return A vector containing the sum of the elements along the given dimension
 *  in the given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename TagT, typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> sum_by_tag(matrix_expression<MatrixExprT> const& me);

//@} Declarations


namespace detail { namespace /*<unnamed>*/ {

//@{ Declarations

/**
 * \brief Auxiliary class for computing the sum of the elements along the given
 *  dimension for a container of the given category.
 * \tparam Dim The dimension number (starting from 1).
 * \tparam CategoryT The category type (e.g., vector_tag).
 */
template < ::std::size_t Dim, typename CategoryT>
struct sum_by_dim_impl;

/**
 * \brief Auxiliary class for computing the sum of the elements along the given
 *  dimension tag for a container of the given category.
 * \tparam TagT The dimension tag type (e.g., tag::major).
 * \tparam CategoryT The category type (e.g., vector_tag).
 * \tparam OrientationT The orientation category type (e.g., row_major_tag).
 */
template <typename TagT, typename CategoryT, typename OrientationT>
struct sum_by_tag_impl;

//@} Declarations


//@{ Definitions

template <>
struct sum_by_dim_impl<1, vector_tag>
{
	template <typename VectorExprT>
	BOOST_UBLAS_INLINE
	static vector<typename vector_traits<VectorExprT>::value_type> apply(vector_expression<VectorExprT> const& ve)
	{
		typedef typename vector_traits<VectorExprT>::value_type value_type;

		vector<value_type> res(1);

		res(0) = sum(ve);

		return res;
	}
};


template <>
struct sum_by_dim_impl<1, matrix_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_rows(me);
	}
};


template <>
struct sum_by_dim_impl<2, matrix_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_columns(me);
	}
};


template <>
struct sum_by_tag_impl<tag::major, matrix_tag, row_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_rows(me);
	}
};


template <>
struct sum_by_tag_impl<tag::minor, matrix_tag, row_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_columns(me);
	}
};


template <>
struct sum_by_tag_impl<tag::leading, matrix_tag, row_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_columns(me);
	}
};


template <>
struct sum_by_tag_impl<tag::major, matrix_tag, column_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_columns(me);
	}
};


template <>
struct sum_by_tag_impl<tag::minor, matrix_tag, column_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_rows(me);
	}
};


template <>
struct sum_by_tag_impl<tag::leading, matrix_tag, column_major_tag>
{
	template <typename MatrixExprT>
	BOOST_UBLAS_INLINE
	static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
	{
		return sum_rows(me);
	}
};


template <typename TagT>
struct sum_by_tag_impl<TagT, matrix_tag, unknown_orientation_tag>: sum_by_tag_impl<TagT, matrix_tag, row_major_tag>
{
	// Empty
};

//@} Definitions

}} // Namespace detail::<unnamed>


//@{ Definitions

//XXX: already implemented in vector_expression.hpp
//template <typename VectorExprT>
//BOOST_UBLAS_INLINE
//typename vector_traits<VectorExprT>::value_type sum(vector_expression<VectorExprT> const& ve)
//{
//	typedef typename vector_traits<VectorExprT>::const_iterator iterator_type;;
//	typedef typename vector_traits<VectorExprT>::value_type value_type;
//
//	iterator_type it_end = end(ve);
//	value_type s = 0;
//
//	for (iterator_type it = begin(ve); it != it_end; ++it)
//	{
//		s += *it;
//	}
//
//	return s;
//}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::value_type sum_all(matrix_expression<MatrixExprT> const& me)
{
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;

	size_type nr = num_rows(me);
	size_type nc = num_columns(me);

	value_type s = 0;
	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			s += me()(r,c);
		}
	}

	return s;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> sum(matrix_expression<MatrixExprT> const& me)
{
	return sum_rows(me);
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> sum_rows(matrix_expression<MatrixExprT> const& me)
{
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;

	size_type nr = num_rows(me);
	size_type nc = num_columns(me);

	vector<value_type> s(nc);
	size_type j = 0;
	for (size_type c = 0; c < nc; ++c)
	{
		//s(j++) = sum(column(me, c)); //FIXME: don't work

		value_type cs = 0;
		for (size_type r = 0; r < nr; ++r)
		{
			cs += me()(r,c);
		}
		s(j++) = cs;
	}

	return s;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> sum_columns(matrix_expression<MatrixExprT> const& me)
{
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;

	size_type nr = num_rows(me);
	size_type nc = num_columns(me);

	vector<value_type> s(nr);
	size_type j = 0;
	for (size_type r = 0; r < nr; ++r)
	{
		//s(j++) = sum(row(me, r)); // FIXME don't work

		value_type rs = 0;
		for (size_type c = 0; c < nc; ++c)
		{
			rs += me()(r,c);
		}
		s(j++) = rs;
	}

	return s;
}


template <size_t Dim, typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> sum(vector_expression<VectorExprT> const& ve)
{
	return detail::sum_by_dim_impl<Dim, vector_tag>::template apply(ve);
}


template <size_t Dim, typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> sum(matrix_expression<MatrixExprT> const& me)
{
	return detail::sum_by_dim_impl<Dim, matrix_tag>::template apply(me);
}


template <typename TagT, typename MatrixExprT>
//template <typename MatrixExprT, typename TagT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> sum_by_tag(matrix_expression<MatrixExprT> const& me)
{
	return detail::sum_by_tag_impl<TagT, matrix_tag, typename matrix_traits<MatrixExprT>::orientation_category>::template apply(me);
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_SUM_HPP
