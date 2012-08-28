/**
 * \file boost/numeric/ublasx/operation/hold.hpp
 *
 * \brief Apply a given unary predicate to each element of a given container.
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_HOLD_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_HOLD_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <functional>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_hold_functor_traits
{
	typedef VectorExprT input_expression_type;
	typedef typename vector_traits<input_expression_type>::value_type signature_argument_type;
	typedef bool signature_result_type;
	typedef vector_unary_functor_traits<
				input_expression_type,
				signature_result_type (signature_argument_type)
			> unary_functor_expression_type;
	typedef typename unary_functor_expression_type::result_type result_type;
	typedef typename unary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT>
struct matrix_hold_functor_traits
{
	typedef MatrixExprT input_expression_type;
	typedef typename matrix_traits<input_expression_type>::value_type signature_argument_type;
	typedef bool signature_result_type;
	typedef matrix_unary_functor_traits<
				input_expression_type,
				signature_result_type (signature_argument_type)
			> unary_functor_expression_type;
	typedef typename unary_functor_expression_type::result_type result_type;
	typedef typename unary_functor_expression_type::expression_type expression_type;
};

bool nz(double v);
bool nz(double v)
{
	return v != 0;
}

} // Namespace detail



/**
 * \brief Test what elements of the given vector expression are different from
 *  zero.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A boolean vector expression such that the i-th element is \c true if
 *  and only if the i-th element of \a ve is different from zero.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_hold_functor_traits<VectorExprT>::result_type hold(vector_expression<VectorExprT> const& ve)
{
	typedef typename detail::vector_hold_functor_traits<VectorExprT>::expression_type expression_type;
	typedef typename detail::vector_hold_functor_traits<VectorExprT>::signature_argument_type value_type;

	return expression_type(
				ve(),
				//::std::bind2nd(::std::not_equal_to<value_type>(), 0)
				&detail::nz
		);
}


/**
 * \brief Test what elements of the given matrix expression are different from
 *  zero.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param ve The input matrix expression.
 * \return A boolean matrix expression such that the (i,j)-th element is \c true
 *  if and only if the (i,j)-th element of \a me is different from zero.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_hold_functor_traits<MatrixExprT>::result_type hold(matrix_expression<MatrixExprT> const& me)
{
	typedef typename detail::matrix_hold_functor_traits<MatrixExprT>::expression_type expression_type;
	typedef typename detail::matrix_hold_functor_traits<MatrixExprT>::signature_argument_type value_type;

	return expression_type(
				me(),
				::std::bind2nd(::std::not_equal_to<value_type>(), 0)
		);
}


/**
 * \brief Applies the given unary predicate to the given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 * \tparam UnaryPredicateT The type of the unary predicate (can either be a
 *  pointer to a function or a class which overloads \c operator()).
 *
 * \param ve The input vector expression.
 * \param pred Unary predicate taking an element in the range as argument \a ve,
 *  and returning a value indicating the falsehood (with \c false, or a zero
 *  value) or truth (\c true, or non-zero) of some condition applied to it. It
 *  can either be a pointer to a function or an object whose class overloads
 *  \c operator().
 * \return A vector expression representing the application of \a pred to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
typename detail::vector_hold_functor_traits<VectorExprT>::result_type hold(vector_expression<VectorExprT> const& ve, UnaryPredicateT pred)
{
	typedef typename detail::vector_hold_functor_traits<VectorExprT>::expression_type expression_type;

	return expression_type(ve(), pred);
}


/**
 * \brief Applies the given unary predicate to the given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam UnaryPredicateT The type of the unary predicate (can either be a
 *  pointer to a function or a class which overloads \c operator()).
 *
 * \param me The input matrix expression.
 * \param pred Unary predicate taking an element in the range as argument \a me,
 *  and returning a value indicating the falsehood (with \c false, or a zero
 *  value) or truth (\c true, or non-zero) of some condition applied to it. It
 *  can either be a pointer to a function or an object whose class overloads
 *  \c operator().
 * \return A matrix expression representing the application of \a pred to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT, typename UnaryPredicateT>
BOOST_UBLAS_INLINE
typename detail::matrix_hold_functor_traits<MatrixExprT>::result_type hold(matrix_expression<MatrixExprT> const& me, UnaryPredicateT pred)
{
	typedef typename detail::matrix_hold_functor_traits<MatrixExprT>::expression_type expression_type;

	return expression_type(me(), pred);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_HOLD_HPP
