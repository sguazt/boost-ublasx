/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/find.hpp
 *
 * \brief Find the elments of a given container which satisfy a given unary
 *  predicate.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompfinding file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_FIND_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_FIND_HPP

//TODO: Implementation for matrix expressions.

#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <functional>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

/**
 * \brief Find the elments of the given vector expression which satisfy the
 *  given unary predicate.
 * \tparam VectorExprT The type of the vector expression.
 * \tparam UnaryPredicateT The type of the unary predicate functor.
 * \param ve The vector expression over which check for the predicate.
 * \param p The unary predicate functor: must accept one argument and return
 * a boolean value.
 * \return A vector of the elements of the given vector expression which satisfy
 *  the given predicate; if no element satisfies the given predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT, typename UnaryPredicateT>
vector<typename vector_traits<VectorExprT>::value_type> find(vector_expression<VectorExprT> const& ve, UnaryPredicateT p);

/**
 * \brief Find the non-zero elments of the given vector expression.
 * \tparam VectorExprT The type of the vector expression.
 * \param ve The vector expression over which check for existence of non-zero
 *  elements.
 * \return \c true if the given vector expression contains at least one non-zero
 *  element; \c false otherwise.
 *
 * \note The test for zero equality is done in the strong sense, that is by not
 *  using find tolerance.
 *  For checking for "weak" zero equality between a given tolerance use the
 *  two-argument version of this function with an appropriate predicate.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT>
vector<typename vector_traits<VectorExprT>::value_type> find(vector_expression<VectorExprT> const& ve);

//@} Declarations


//@{ Definitions

template <typename VectorExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> find(vector_expression<VectorExprT> const& ve, UnaryPredicateT p)
{
    typedef typename vector_traits<VectorExprT>::size_type size_type;
    typedef typename vector_traits<VectorExprT>::value_type value_type;

    vector<value_type> res;
    size_type n = size(ve);
    size_type j = 0;
    for (size_type i = 0; i < n; ++i)
    {
        if (p(ve()(i)))
        {
            res.resize(res.size()+1);
            res(j++) = ve()(i);
        }
    }

    return res;
}


template <typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> find(vector_expression<VectorExprT> const& ve)
{
    typedef typename vector_traits<VectorExprT>::value_type value_type;

    return find(ve, ::std::bind2nd(::std::not_equal_to<value_type>(), 0));
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_FIND_HPP
