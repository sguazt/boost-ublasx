/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/pow.hpp
 *
 * \brief Apply the \c std::pow function to each element of a vector or a matrix
 *  expression.
 *
 * <hr/>
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * Copyright (c) 2015, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_binary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_binary_functor.hpp>
#include <cmath>
#include <complex>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename VectorExprT, typename Arg2T>
struct vector_pow_functor1_traits
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


template <typename Arg1T, typename VectorExprT>
struct vector_pow_functor2_traits
{
    typedef VectorExprT input_expression_type;
    typedef Arg1T signature_argument1_type;
    typedef typename vector_traits<input_expression_type>::value_type signature_argument2_type;
    //typedef signature_argument_type signature_result_type;
    typedef typename promote_traits<
                signature_argument1_type,
                signature_argument2_type
            >::promote_type signature_result_type;
    typedef vector_binary_functor2_traits<
                Arg1T,
                input_expression_type,
                signature_result_type (signature_argument1_type, signature_argument2_type)
            > binary_functor_expression_type;
    typedef typename binary_functor_expression_type::result_type result_type;
    typedef typename binary_functor_expression_type::expression_type expression_type;
};


template <typename MatrixExprT, typename Arg2T>
struct matrix_pow_functor1_traits
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


template <typename Arg1T, typename MatrixExprT>
struct matrix_pow_functor2_traits
{
    typedef MatrixExprT input_expression_type;
    typedef Arg1T signature_argument1_type;
    typedef typename matrix_traits<input_expression_type>::value_type signature_argument2_type;
    //typedef signature_argument_type signature_result_type;
    typedef typename promote_traits<
                signature_argument1_type,
                signature_argument2_type
            >::promote_type signature_result_type;
    typedef matrix_binary_functor2_traits<
                Arg1T,
                input_expression_type,
                signature_result_type (signature_argument1_type, signature_argument2_type)
            > binary_functor_expression_type;
    typedef typename binary_functor_expression_type::result_type result_type;
    typedef typename binary_functor_expression_type::expression_type expression_type;
};


// Wrappers to the std::pow function to avoid compiler errors

template <typename T1, typename T2>
BOOST_UBLAS_INLINE
typename promote_traits<T1,T2>::promote_type pow(T1 x, T2 y)
{
    return ::std::pow(x, y);
}

template <typename T1, typename T2>
BOOST_UBLAS_INLINE
std::complex<T1> pow(std::complex<T1> const& x, T2 y)
{
    return ::std::pow(x, y);
}

template <typename T1, typename T2>
BOOST_UBLAS_INLINE
std::complex<T1> pow(T1 x, std::complex<T2> const& y)
{
    // Remember: if z=(a + ib) is a complex number and c is a scalar => c^z = e^{ln(c)*z}
    return ::std::exp(::std::log(x)*y);
}

} // Namespace detail


/**
 * \brief Applies the \c std::pow function to a given vector expression,
 *  where each element of the vector is treated as the base of the
 *  exponentiation.
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
typename detail::vector_pow_functor1_traits<VectorExprT,T>::result_type pow(vector_expression<VectorExprT> const& ve, T p)
{
    typedef typename detail::vector_pow_functor1_traits<VectorExprT,T>::expression_type expression_type;
    typedef typename detail::vector_pow_functor1_traits<VectorExprT,T>::signature_argument1_type signature_argument1_type;
    typedef typename detail::vector_pow_functor1_traits<VectorExprT,T>::signature_argument2_type signature_argument2_type;
    typedef typename detail::vector_pow_functor1_traits<VectorExprT,T>::signature_result_type signature_result_type;

//  return expression_type(ve(), detail::pow<signature_result_type>);
//  signature_result_type (*)(ptr_pow_fun)(signature_argument_type)(BOOST_NUMERIC_UBLASX_OPERATION_POW_NS_::pow); 
    typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
    fun_ptr_type ptr_pow_fun(&detail::pow); 
    return expression_type(ve(), p, ptr_pow_fun);
}


/**
 * \brief Applies the \c std::pow function to a given vector expression,
 *  where each element of the vector is treated as the power of the
 *  exponentiation.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param b The base.
 * \param ve The input vector expression.
 * \return A vector expression representing the application of \c std::pow to
 *  each element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename T, typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_pow_functor2_traits<T,VectorExprT>::result_type pow(T b, vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_pow_functor2_traits<T,VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_pow_functor2_traits<T,VectorExprT>::signature_argument1_type signature_argument1_type;
    typedef typename detail::vector_pow_functor2_traits<T,VectorExprT>::signature_argument2_type signature_argument2_type;
    typedef typename detail::vector_pow_functor2_traits<T,VectorExprT>::signature_result_type signature_result_type;

//  return expression_type(ve(), detail::pow<signature_result_type>);
//  signature_result_type (*)(ptr_pow_fun)(signature_argument_type)(BOOST_NUMERIC_UBLASX_OPERATION_POW_NS_::pow); 
    typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
    fun_ptr_type ptr_pow_fun(&detail::pow); 
    return expression_type(b, ve(), ptr_pow_fun);
}


/**
 * \brief Applies the \c std::pow function to a given matrix expression,
 *  where each element of the matrix is treated as the base of the
 *  exponentiation.
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
typename detail::matrix_pow_functor1_traits<MatrixExprT,T>::result_type pow(matrix_expression<MatrixExprT> const& me, T p)
{
    typedef typename detail::matrix_pow_functor1_traits<MatrixExprT,T>::expression_type expression_type;
    typedef typename detail::matrix_pow_functor1_traits<MatrixExprT,T>::signature_argument1_type signature_argument1_type;
    typedef typename detail::matrix_pow_functor1_traits<MatrixExprT,T>::signature_argument2_type signature_argument2_type;
    typedef typename detail::matrix_pow_functor1_traits<MatrixExprT,T>::signature_result_type signature_result_type;

//  return expression_type(me(), detail::pow<signature_result_type>(signature_argument_type));
    typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
    fun_ptr_type ptr_pow_fun(&detail::pow); 
    return expression_type(me(), p, ptr_pow_fun);
}

/**
 * \brief Applies the \c std::pow function to a given matrix expression,
 *  where each element of the matrix is treated as the power of the
 *  exponentiation.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param b The base.
 * \param me The input matrix expression.
 * \return A matrix expression representing the application of \c std::pow to
 *  each element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename T, typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_pow_functor2_traits<T,MatrixExprT>::result_type pow(T b, matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_pow_functor2_traits<T,MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_pow_functor2_traits<T,MatrixExprT>::signature_argument1_type signature_argument1_type;
    typedef typename detail::matrix_pow_functor2_traits<T,MatrixExprT>::signature_argument2_type signature_argument2_type;
    typedef typename detail::matrix_pow_functor2_traits<T,MatrixExprT>::signature_result_type signature_result_type;

//  return expression_type(me(), detail::pow<signature_result_type>(signature_argument_type));
    typedef signature_result_type(*fun_ptr_type)(signature_argument1_type, signature_argument2_type);
    fun_ptr_type ptr_pow_fun(&detail::pow); 
    return expression_type(b, me(), ptr_pow_fun);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP
