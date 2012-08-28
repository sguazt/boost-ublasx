/**
 * \file boost/numeric/ublasx/operation/sqr.hpp
 *
 * \brief Compute the squared of each element of a vector or matrix expression.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_SQR_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_SQR_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_sqr_functor_traits
{
	typedef VectorExprT input_expression_type;
	typedef typename vector_traits<input_expression_type>::value_type signature_argument_type;
	typedef signature_argument_type signature_result_type;
	typedef vector_unary_functor_traits<
				input_expression_type,
				signature_result_type (signature_argument_type)
			> unary_functor_expression_type;
	typedef typename unary_functor_expression_type::result_type result_type;
	typedef typename unary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT>
struct matrix_sqr_functor_traits
{
	typedef MatrixExprT input_expression_type;
	typedef typename matrix_traits<input_expression_type>::value_type signature_argument_type;
	typedef signature_argument_type signature_result_type;
	typedef matrix_unary_functor_traits<
				input_expression_type,
				signature_result_type (signature_argument_type)
			> unary_functor_expression_type;
	typedef typename unary_functor_expression_type::result_type result_type;
	typedef typename unary_functor_expression_type::expression_type expression_type;
};


namespace /*<unnamed>*/ {

template <typename T>
BOOST_UBLAS_INLINE
T sqr_impl(T x)
{
	return x*x;
}

} // Namespace <unnamed>

} // Namespace detail


/**
 * \brief Compute the squared of each element of a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A matrix expression where each element of \a ve has been multiplied by
 *  itself.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_sqr_functor_traits<VectorExprT>::result_type sqr(vector_expression<VectorExprT> const& ve)
{
	typedef typename detail::vector_sqr_functor_traits<VectorExprT>::expression_type expression_type;
	typedef typename detail::vector_sqr_functor_traits<VectorExprT>::signature_argument_type signature_argument_type;
	typedef typename detail::vector_sqr_functor_traits<VectorExprT>::signature_result_type signature_result_type;

	return expression_type(ve(), detail::sqr_impl<signature_argument_type>);
//	return expression_type(ve(), detail::sqr_impl<signature_result_type>);
//	typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
//	fun_ptr_type ptr_sqr_fun(&detail::sqr_impl); 
//	return expression_type(ve(), ptr_sqr_fun);
}


/**
 * \brief Compute the squared of each element of a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression where each element of \a me has been multiplied by
 *  itself.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_sqr_functor_traits<MatrixExprT>::result_type sqr(matrix_expression<MatrixExprT> const& me)
{
	typedef typename detail::matrix_sqr_functor_traits<MatrixExprT>::expression_type expression_type;
	typedef typename detail::matrix_sqr_functor_traits<MatrixExprT>::signature_argument_type signature_argument_type;
	typedef typename detail::matrix_sqr_functor_traits<MatrixExprT>::signature_result_type signature_result_type;

	return expression_type(me(), detail::sqr_impl<signature_argument_type>);
//	return expression_type(me(), detail::sqr_impl<signature_result_type>(signature_argument_type));
//	typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
//	fun_ptr_type ptr_sqr_fun(&detail::sqr_impl); 
//	return expression_type(me(), ptr_sqr_fun);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_SQR_HPP
