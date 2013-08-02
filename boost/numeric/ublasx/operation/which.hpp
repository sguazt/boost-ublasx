/**
 * \file boost/numeric/ublasx/operation/which.hpp
 *
 * \brief Find the positions of the elments of a given container which satisfy
 *  a given unary predicate.
 *
 * \todo Implementation for matrix expressions.
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_WHICH_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_WHICH_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <functional>


//TODO: implement the 'which' operation for matrix containers

namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

/**
 * \brief Find the positions of the elments of the given vector expression
 *  which satisfy the given unary predicate.
 * \tparam VectorExprT The type of the vector expression.
 * \tparam UnaryPredicateT The type of the unary predicate functor.
 * \param ve The vector expression over which check for the predicate.
 * \param p The unary predicate functor: must accept one argument and return
 *  a boolean value.
 * \return A vector of positions of the elements of the given vector
 *  expression which satisfy the given predicate; an empty vector, if no element
 *  satisfies the given predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT, typename UnaryPredicateT>
vector<typename vector_traits<VectorExprT>::size_type> which(vector_expression<VectorExprT> const& ve, UnaryPredicateT p);

/**
 * \brief Find the positions of the non-zero elments of the given vector
 *  expression.
 * \tparam VectorExprT The type of the vector expression.
 * \param ve The vector expression over which check for existence of non-zero
 *  elements.
 * \return A vector of positions of the non-zero elements of the given vector
 *  expression; an empty vector, if no non-zero element is found.
 *
 * \note The test for zero equality is done in the strong sense, that is by not
 *  using which tolerance.
 *  For checking for "weak" zero equality between a given tolerance use the
 *  two-argument version of this function with an appropriate predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT>
vector<typename vector_traits<VectorExprT>::size_type> which(vector_expression<VectorExprT> const& ve);

//@} Declarations


//@{ Definitions

template <typename VectorExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::size_type> which(vector_expression<VectorExprT> const& ve, UnaryPredicateT p)
{
	typedef typename vector_traits<VectorExprT>::size_type size_type;

	vector<size_type> res;
	size_type n = size(ve);
	size_type j = 0;
	for (size_type i = 0; i < n; ++i)
	{
		if (p(ve()(i)))
		{
			res.resize(res.size()+1);
			res(j++) = i;
		}
	}

	return res;
}


template <typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::size_type> which(vector_expression<VectorExprT> const& ve)
{
	typedef typename vector_traits<VectorExprT>::value_type value_type;

    return which(ve, ::std::bind2nd(::std::not_equal_to<value_type>(), 0));
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_WHICH_HPP
