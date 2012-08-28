/**
 * \file boost/numeric/ublasx/operation/any.hpp
 *
 * \brief The \c any operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ANY_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ANY_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
//#include <boost/numeric/ublas/operation/begin.hpp>
//#include <boost/numeric/ublas/operation/end.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
//#include <boost/numeric/ublasx/tags.hpp>
#include <boost/numeric/ublas/traits.hpp>
//#include <boost/numeric/ublasx/traits/const_iterator_type.hpp>
#include <functional>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

/**
 * \brief Tell if there is at least one element of the given vector expression
 *  which satisfies the given unary predicate.
 * \tparam VectorExprT The type of the vector expression.
 * \tparam UnaryPredicateT The type of the unary predicate functor.
 * \param ve The vector expression over which check for the predicate.
 * \param p The unary predicate functor: must accept one argument and return
 *  a boolean value.
 * \return \c true if there exists at least one element of the given vector
 *  expression which satisfies the given predicate; \c false otherwise.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT, typename UnaryPredicateT>
bool any(vector_expression<VectorExprT> const& ve, UnaryPredicateT p);

/**
 * \brief Tell if the given vector expression contains at least one non-zero
 *  element.
 * \tparam VectorExprT The type of the vector expression.
 * \param ve The vector expression over which check for existence of non-zero
 *  elements.
 * \return \c true if the given vector expression contains at least one non-zero
 *  element; \c false otherwise.
 *
 * \note The test for zero equality is done in the strong sense, that is by not
 *  using any tolerance.
 *  For checking for "weak" zero equality between a given tolerance use the
 *  two-argument version of this function with an appropriate predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT>
bool any(vector_expression<VectorExprT> const& ve);

/**
 * \brief Tell if there is at least one element of the given matrix expression
 *  which satisfies the given unary predicate.
 * \tparam MatrixExprT The type of the matrix expression.
 * \tparam UnaryPredicateT The type of the unary predicate functor.
 * \param me The matrix expression over which check for the predicate.
 * \param p The unary predicate functor: must accept one argument and return
 *  a boolean value.
 * \return \c true if there exists at least one element of the given matrix
 *  expression which satisfies the given predicate; \c false otherwise.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT, typename UnaryPredicateT>
bool any(matrix_expression<MatrixExprT> const& me, UnaryPredicateT p);

/**
 * \brief Tell if the given matrix expression contains at least one non-zero
 *  element.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which check for existence of non-zero
 *  elements.
 * \return \c true if the given matrix expression contains at least one non-zero
 *  element; \c false otherwise.
 *
 * \note The test for zero equality is done in the strong sense, that is by not
 *  using any tolerance.
 *  For checking for "weak" zero equality between a given tolerance use the
 *  two-argument version of this function with an appropriate predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
bool any(matrix_expression<MatrixExprT> const& me);

//@} Declarations


//@{ Definitions

template <typename VectorExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
bool any(vector_expression<VectorExprT> const& ve, UnaryPredicateT p)
{
	// Implementation Note:
	//  Currently (06-2010) there are only iterators that skip zero-valued
	//  elements.
	//  So the iteration is done through plain indices instead of iterators.

//	typedef typename vector_traits<VectorExprT>::const_iterator iterator_type;
//
//	iterator_type it_end = end(ve);
//	for (iterator_type it = begin(ve); it != it_end; ++it)
//	{
//		if (p(*it))
//		{
//			return true;
//		}
//	}
//

	typedef typename vector_traits<VectorExprT>::size_type size_type;

	size_type n = size(ve);
	for (size_type i = 0; i < n; ++i)
	{
		if (p(ve()(i)))
		{
			return true;
		}
	}

	return false;
}


template <typename VectorExprT>
BOOST_UBLAS_INLINE
bool any(vector_expression<VectorExprT> const& ve)
{
	typedef typename vector_traits<VectorExprT>::value_type value_type;

	return any(ve, ::std::bind2nd(::std::not_equal_to<value_type>(), 0));
}


template <typename MatrixExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
bool any(matrix_expression<MatrixExprT> const& me, UnaryPredicateT p)
{
	// Implementation Note:
	//  Currently (06-2010) there are only iterators that skip zero-valued
	//  elements.
	//  So the iteration is done through plain indices instead of iterators.

//	typedef typename const_iterator_type<MatrixExprT,tag::major>::type maj_iterator_type;
//	typedef typename const_iterator_type<MatrixExprT,tag::minor>::type min_iterator_type;
//
//	maj_iterator_type maj_it_end = end<tag::major>(me);
//	for (maj_iterator_type maj_it = begin<tag::major>(me); maj_it != maj_it_end; ++maj_it)
//	{
//		min_iterator_type min_it_end = end(maj_it);
//		for (min_iterator_type min_it = begin(maj_it); min_it != min_it_end; ++min_it)
//		{
//			if (p(*min_it))
//			{
//				return true;
//			}
//		}
//	}

	typedef typename matrix_traits<MatrixExprT>::size_type size_type;

	size_type nr = num_rows(me);
	size_type nc = num_columns(me);
	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			if (p(me()(r,c)))
			{
				return true;
			}
		}
	}

	return false;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
bool any(matrix_expression<MatrixExprT> const& me)
{
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;

	return any(me, ::std::bind2nd(::std::not_equal_to<value_type>(), value_type(0)));
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ANY_HPP
