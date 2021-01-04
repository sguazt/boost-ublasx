/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/round.hpp
 *
 * \brief Compute the interger nearest to each element of a vector or matrix
 *  expression.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ROUND_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ROUND_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <cmath>
#include <complex>


//#if __cplusplus > 199711L
// C++0x has ::std::round
//#   define BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_ ::std
//#else
// Use customer-made round function
#   define BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_ detail
//#endif // __cpluplus


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_round_functor_traits
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
struct matrix_round_functor_traits
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


/// Auxiliary function used to replace ::std::round when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
T round(T x)
{
    return (x > 0.0) ? ::std::floor(x + 0.5) : ::std::ceil(x - 0.5);
}

/// Auxiliary function used to replace ::std::round when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
::std::complex<T> round(::std::complex<T> x)
{
    T r = (x.real() > 0.0) ? ::std::floor(x.real() + 0.5) : ::std::ceil(x.real() - 0.5);
    T i = (x.imag() > 0.0) ? ::std::floor(x.imag() + 0.5) : ::std::ceil(x.imag() - 0.5);

    return ::std::complex<T>(r,i);
}

} // Namespace detail


/**
 * \brief Applies the \c std::round function to a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::round to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_round_functor_traits<VectorExprT>::result_type round(vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_round_functor_traits<VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_round_functor_traits<VectorExprT>::signature_argument_type signature_argument_type;
    typedef typename detail::vector_round_functor_traits<VectorExprT>::signature_result_type signature_result_type;

//  return expression_type(ve(), BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_::round<signature_result_type>);
//  signature_result_type (*)(ptr_round_fun)(signature_argument_type)(BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_::round); 
    typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
    fun_ptr_type ptr_round_fun(&BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_::round); 
    return expression_type(ve(), ptr_round_fun);
}


/**
 * \brief Applies the \c std::round function to a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::round to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_round_functor_traits<MatrixExprT>::result_type round(matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_round_functor_traits<MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_round_functor_traits<MatrixExprT>::signature_argument_type signature_argument_type;
    typedef typename detail::matrix_round_functor_traits<MatrixExprT>::signature_result_type signature_result_type;

//  return expression_type(me(), BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_::round<signature_result_type>(signature_argument_type));
    typedef signature_result_type(*fun_ptr_type)(signature_argument_type);
    fun_ptr_type ptr_round_fun(&BOOST_NUMERIC_UBLASX_OPERATION_ROUND_NS_::round); 
    return expression_type(me(), ptr_round_fun);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ROUND_HPP
