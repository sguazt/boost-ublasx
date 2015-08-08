/**
 * \file boost/numeric/ublasx/operation/element_pow.hpp
 *
 * \brief Apply the \c std::pow function to each element of a vector or a matrix
 *  expression.
 *
 * Copyright (c) 2015, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ELEMENT_POW_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ELEMENT_POW_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_binary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_binary_functor.hpp>
#include <cmath>
#include <complex>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT, typename Arg2T>
struct vector_element_pow_functor_traits
{
	typedef VectorExprT input_expression_type;
	typedef typename vector_traits<input_expression_type>::value_type signature_argument1_type;
	typedef Arg2T signature_argument2_type;
	//typedef signature_argument_type signature_result_type;
	typedef typename promote_traits<
				signature_argument1_type,
				signature_argument2_type
			>::promote_type signature_result_type;
	typedef vector_binary_functor1_traits<
				input_expression_type,
				Arg2T,
				signature_result_type (signature_argument1_type, signature_argument2_type)
			> binary_functor_expression_type;
	typedef typename binary_functor_expression_type::result_type result_type;
	typedef typename binary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT, typename Arg2T>
struct matrix_element_pow_functor_traits
{
	typedef MatrixExprT input_expression_type;
	typedef typename matrix_traits<input_expression_type>::value_type signature_argument1_type;
	typedef Arg2T signature_argument2_type;
	//typedef signature_argument_type signature_result_type;
	typedef typename promote_traits<
				signature_argument1_type,
				signature_argument2_type
			>::promote_type signature_result_type;
	typedef matrix_binary_functor1_traits<
				input_expression_type,
				Arg2T,
				signature_result_type (signature_argument1_type, signature_argument2_type)
			> binary_functor_expression_type;
	typedef typename binary_functor_expression_type::result_type result_type;
	typedef typename binary_functor_expression_type::expression_type expression_type;
};


// Wrappers to the std::pow function to avoid compiler errors

template <typename T1, typename T2>
BOOST_UBLAS_INLINE
typename promote_traits<T1,T2>::promote_type element_pow(T1 x, T2 y)
{
    return ::std::pow(x, y);
}

template <typename T1, typename T2>
BOOST_UBLAS_INLINE
std::complex<T1> element_pow(std::complex<T1> const& x, T2 y)
{
    return ::std::pow(x, y);
}

} // Namespace detail


/**
 * \brief Applies the \c std::pow function to a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \param p The exponent.
 * \return A vector expression representing the application of \c std::pow to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT, typename T>
BOOST_UBLAS_INLINE
typename detail::vector_element_pow_functor_traits<VectorExprT,T>::result_type element_pow(vector_expression<VectorExprT> const& ve, T p)
{
	typedef typename detail::vector_element_pow_functor_traits<VectorExprT,T>::expression_type expression_type;
	typedef typename detail::vector_element_pow_functor_traits<VectorExprT,T>::signature_argument1_type signature_argument1_type;
	typedef typename detail::vector_element_pow_functor_traits<VectorExprT,T>::signature_argument2_type signature_argument2_type;
	typedef typename detail::vector_element_pow_functor_traits<VectorExprT,T>::signature_result_type signature_result_type;

//	return expression_type(ve(), detail::element_pow<signature_result_type>);
//	signature_result_type (*)(ptr_element_pow_fun)(signature_argument_type)(BOOST_NUMERIC_UBLASX_OPERATION_POW_NS_::element_pow); 
	typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
	fun_ptr_type ptr_element_pow_fun(&detail::element_pow); 
	return expression_type(ve(), p, ptr_element_pow_fun);
}


/**
 * \brief Applies the \c std::pow function to a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \param p The exponent.
 * \return A matrix expression representing the application of \c std::pow to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT, typename T>
BOOST_UBLAS_INLINE
typename detail::matrix_element_pow_functor_traits<MatrixExprT,T>::result_type element_pow(matrix_expression<MatrixExprT> const& me, T p)
{
	typedef typename detail::matrix_element_pow_functor_traits<MatrixExprT,T>::expression_type expression_type;
	typedef typename detail::matrix_element_pow_functor_traits<MatrixExprT,T>::signature_argument1_type signature_argument1_type;
	typedef typename detail::matrix_element_pow_functor_traits<MatrixExprT,T>::signature_argument2_type signature_argument2_type;
	typedef typename detail::matrix_element_pow_functor_traits<MatrixExprT,T>::signature_result_type signature_result_type;

//	return expression_type(me(), detail::element_pow<signature_result_type>(signature_argument_type));
	typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
	fun_ptr_type ptr_element_pow_fun(&detail::element_pow); 
	return expression_type(me(), p, ptr_element_pow_fun);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ELEMENT_POW_HPP
