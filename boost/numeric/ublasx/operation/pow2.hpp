/**
 * \file boost/numeric/ublasx/operation/pow2.hpp
 *
 * \brief Apply the \c std::pow2 function to a vector or matrix expression.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_POW2_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_POW2_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <cmath>
#include <complex>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_pow2_functor_traits
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
struct matrix_pow2_functor_traits
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


//namespace /*<unnamed>*/ {

/// Auxiliary function used to replace ::std::pow2 when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
T pow2(T x)
{
	return ::std::pow(2,x);
}

/// Auxiliary function used to replace ::std::pow2 when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
::std::complex<T> pow2(::std::complex<T> x)
{
	// Use the complex exponentiation formula.
	// See:
	// - http://en.wikipedia.org/wiki/Exponentiation#Powers_of_complex_numbers
	// - http://mathworld.wolfram.com/ComplexExponentiation.html

	T c0 = ::std::pow(2,x.real());
	T c1 = x.imag()*::std::log(2);
	return ::std::complex<T>(c0*::std::cos(c1),c0*::std::sin(c1));
}

//} // Namespace <unnamed>

} // Namespace detail


/**
 * \brief Applies the \c std::pow2 function to a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::pow2 to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_pow2_functor_traits<VectorExprT>::result_type pow2(vector_expression<VectorExprT> const& ve)
{
	typedef typename detail::vector_pow2_functor_traits<VectorExprT>::expression_type expression_type;
	typedef typename detail::vector_pow2_functor_traits<VectorExprT>::signature_argument_type signature_argument_type;
	typedef typename detail::vector_pow2_functor_traits<VectorExprT>::signature_result_type signature_result_type;

//	return expression_type(ve(), detail::pow2<signature_result_type>);
//	signature_result_type (*)(ptr_pow2_fun)(signature_argument_type)(BOOST_NUMERIC_UBLASX_OPERATION_POW2_NS_::pow2); 
	typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
	fun_ptr_type ptr_pow2_fun(&detail::pow2); 
	return expression_type(ve(), ptr_pow2_fun);
}


/**
 * \brief Applies the \c std::pow2 function to a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::pow2 to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_pow2_functor_traits<MatrixExprT>::result_type pow2(matrix_expression<MatrixExprT> const& me)
{
	typedef typename detail::matrix_pow2_functor_traits<MatrixExprT>::expression_type expression_type;
	typedef typename detail::matrix_pow2_functor_traits<MatrixExprT>::signature_argument_type signature_argument_type;
	typedef typename detail::matrix_pow2_functor_traits<MatrixExprT>::signature_result_type signature_result_type;

//	return expression_type(me(), detail::pow2<signature_result_type>(signature_argument_type));
	typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
	fun_ptr_type ptr_pow2_fun(&detail::pow2); 
	return expression_type(me(), ptr_pow2_fun);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_POW2_HPP
