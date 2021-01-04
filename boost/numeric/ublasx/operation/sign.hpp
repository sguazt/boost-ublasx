/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/sign.hpp
 *
 * \brief Compute the sign function for each element of a vector or matrix
 *  expression.
 *
 * The sign function for real numbers as defined as follows:
 * \f[
 * \operatorname{sign}(x):=\begin{cases}
 *                          -1 & \text{if } x<0,\\
 *                           0 & \text{if } x=0,\\
 *                           1 & \text{if } x>0.
 *                         \end{cases}
 * \f]
 * In case of complex numbers, the sign function is defined as follows:
 * \f[
 * \operatorname{sign}(z)=\begin{cases}
 *                          \frac{z}{|z|} & \text{if } z \ne 0,\\
 *                          0 & \text{if } z = 0+0i.
 *                        \end{cases}
 * \f]
 *
 * \sa The sign function at Wikipedia: https://en.wikipedia.org/wiki/Sign_function
 *
 * \author comcon1 (original version)
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_SIGN_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_SIGN_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
#include <boost/type_traits/is_complex.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <complex>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT>
struct vector_sign_functor_traits
{
    typedef VectorExprT input_expression_type;
    typedef typename vector_traits<input_expression_type>::value_type signature_argument_type;
    typedef typename type_traits<signature_argument_type>::value_type signature_result_type;
    typedef vector_unary_functor_traits<
                input_expression_type,
                signature_result_type (signature_argument_type)
            > unary_functor_expression_type;
    typedef typename unary_functor_expression_type::result_type result_type;
    typedef typename unary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT>
struct matrix_sign_functor_traits
{
    typedef MatrixExprT input_expression_type;
    typedef typename matrix_traits<input_expression_type>::value_type signature_argument_type;
    typedef typename type_traits<signature_argument_type>::value_type signature_result_type;
    typedef matrix_unary_functor_traits<
                input_expression_type,
                signature_result_type (signature_argument_type)
            > unary_functor_expression_type;
    typedef typename unary_functor_expression_type::result_type result_type;
    typedef typename unary_functor_expression_type::expression_type expression_type;
};

/// Auxiliary function for real types: sign(x) = 1 if x > 0, 0 if x == 0, -1 otherwise.
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::disable_if<
            ::boost::is_complex<T>,
            T
>::type sign_impl(T x)
{
    if (::std::isnan(x))
    {
        return ::std::numeric_limits<T>::quiet_NaN();
    }
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

/// Auxiliary function for complex types: sign(x) = x ./ abs(x)
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::enable_if<
            ::boost::is_complex<T>,
            T
>::type sign_impl(T x)
{
    typename T::value_type a = ::std::abs(x);
    return (a == 0) ? T(0,0) : (x / a);
}

//template <typename RealType> 
//BOOST_UBLAS_INLINE 
//RealType sign(RealType v)
//{
//    return ( -(RealType)( ::std::signbit(v) ) + 0.5 ) * 2.0;
//}


} // Namespace detail


/**
 * \brief Applies the \c std::sign function to a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::sign to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_sign_functor_traits<VectorExprT>::result_type sign(vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_sign_functor_traits<VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_sign_functor_traits<VectorExprT>::signature_result_type signature_result_type;

    return expression_type(ve(), detail::sign_impl<signature_result_type>);
}


/**
 * \brief Applies the \c std::sign function to a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::sign to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_sign_functor_traits<MatrixExprT>::result_type sign(matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_sign_functor_traits<MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_sign_functor_traits<MatrixExprT>::signature_result_type signature_result_type;

    return expression_type(me(), detail::sign_impl<signature_result_type>);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_SIGN_HPP
