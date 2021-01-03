/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/isfinite.hpp
 *
 * \brief Apply the \c std::isfinite function to a vector or matrix expression.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ISFINITE_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ISFINITE_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <boost/type_traits/is_complex.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <complex>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_isfinite_functor_traits
{
    typedef VectorExprT input_expression_type;
    typedef typename vector_traits<input_expression_type>::value_type signature_argument_type;
    typedef int signature_result_type;
    typedef vector_unary_functor_traits<
                input_expression_type,
                signature_result_type (signature_argument_type)
            > unary_functor_expression_type;
    typedef typename unary_functor_expression_type::result_type result_type;
    typedef typename unary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT>
struct matrix_isfinite_functor_traits
{
    typedef MatrixExprT input_expression_type;
    typedef typename matrix_traits<input_expression_type>::value_type signature_argument_type;
    typedef int signature_result_type;
    typedef matrix_unary_functor_traits<
                input_expression_type,
                signature_result_type (signature_argument_type)
            > unary_functor_expression_type;
    typedef typename unary_functor_expression_type::result_type result_type;
    typedef typename unary_functor_expression_type::expression_type expression_type;
};


namespace /*<unnamed>*/ {

/// Wrapper for ::std::isfinite.
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::disable_if<
            ::boost::is_complex<T>,
            int
>::type isfinite_impl(T x)
{
    return ::std::isfinite(x);
}


/// Auxiliary function used to replace ::std::isfinite when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::enable_if<
            ::boost::is_complex<T>,
            int
>::type isfinite_impl(T x)
{
    return ::std::isfinite(x.real()) && ::std::isfinite(x.imag());
}

} // Namespace <unnamed>

} // Namespace detail


/**
 * \brief Applies the \c std::isfinite function to a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::isfinite to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_isfinite_functor_traits<VectorExprT>::result_type isfinite(vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_isfinite_functor_traits<VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_isfinite_functor_traits<VectorExprT>::signature_argument_type signature_argument_type;

    return expression_type(ve(), detail::isfinite_impl<signature_argument_type>);
}


/**
 * \brief Applies the \c std::isfinite function to a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::isfinite to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_isfinite_functor_traits<MatrixExprT>::result_type isfinite(matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_isfinite_functor_traits<MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_isfinite_functor_traits<MatrixExprT>::signature_argument_type signature_argument_type;

    return expression_type(me(), detail::isfinite_impl<signature_argument_type>);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ISFINITE_HPP
