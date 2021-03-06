/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/log.hpp
 *
 * \brief Compute the natural logarithm to each element of a vector or matrix
 *  expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_LOG_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_LOG_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <cmath>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_log_functor_traits
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
struct matrix_log_functor_traits
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


/// Auxiliary function used to replace ::std::log2 when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
T log(T x)
{
    return ::std::log(x);
}

} // Namespace detail


/**
 * \brief Applies the \c std::log function to each element of a given vector
 *  expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::log to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_log_functor_traits<VectorExprT>::result_type log(vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_log_functor_traits<VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_log_functor_traits<VectorExprT>::signature_result_type signature_result_type;

    return expression_type(ve(), detail::log<signature_result_type>);
}


/**
 * \brief Applies the \c std::log function to each element of a given matrix
 *  expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::log to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_log_functor_traits<MatrixExprT>::result_type log(matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_log_functor_traits<MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_log_functor_traits<MatrixExprT>::signature_result_type signature_result_type;

    return expression_type(me(), detail::log<signature_result_type>);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_LOG_HPP
