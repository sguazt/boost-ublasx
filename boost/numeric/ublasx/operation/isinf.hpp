/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/isinf.hpp
 *
 * \brief Returns a logical vector/matrix expression telling whether each
 *  element of a given vector/matrix expression is infinite.
 *
 * The logical vector/matrix expression returned by this function is an
 * integral vector/matrix expression containing only `1` (i.e., `true`) or `0`
 * (i.e., `false`) values.
 *
 * Specifically, given a vector/matrix expression `A`, a call to `isinf(A)`
 * returns an integral vector/matrix expression of the same dimension as `A` and
 * containing `1`s (i.e., `true`) or `0`s (i.e., `false`) depending on whether
 * the corresponding elements of `A` are infinite or not, respecitively.
 * When the type of the elements of the input vector/matrix expression is a C++
 * built-in arithmetic type, this function calls the `std::isinf` function for
 * every element of the vector/matrix expression, and converts the returned
 * value to `1` or `0` depending on whether such value is `true` or `false`,
 * respectively.
 * If the vector/matrix expression contains complex numbers, the value returned
 * by this function contains `1` when the corresponding element in the input
 * expression has infinite real or imaginary part, and `0` otherwise.
 * This behavior is similar to the one used by the MATLAB's \c isinf function.
 *
 * \note The definition of infinity for complex numbers used by this function
 * is different from the _complex infinity_ concept (e.g., used by Wolfram
 * Mathematica), which represents a quantity with infinite magnitude, but
 * undetermined complex phase, as the former considers as infinite also complex
 * numbers that do not represent a complex infinity (e.g., \f$3+1i/0\f$,
 * which is not a complex infinity, is considered an infinite complex number).
 *
 * \sa C++'s `std::inf` function: https://en.cppreference.com/w/cpp/numeric/math/isinf
 * \sa Complex infinity: https://mathworld.wolfram.com/ComplexInfinity.html
 * \sa MATLAB's `isinf` function: https://www.mathworks.com/help/matlab/ref/isinf.html
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone (marco.guazzone@gmail.com)
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ISINF_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ISINF_HPP


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

/// Helper type traits used by the `isinf` function when it takes a matrix expresion as input paramter.
template <typename VectorExprT>
struct vector_isinf_functor_traits
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


/// Helper type traits used by the `isinf` function when it takes a matrix expresion as input paramter.
template <typename MatrixExprT>
struct matrix_isinf_functor_traits
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


/// Wrapper for ::std::isinf.
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::disable_if<
            ::boost::is_complex<T>,
            int
>::type isinf_impl(T x)
{
    return ::std::isinf(x);
}


/// Auxiliary function used to replace ::std::isinf when that is not available.
template <typename T>
BOOST_UBLAS_INLINE
typename ::boost::enable_if<
            ::boost::is_complex<T>,
            int
>::type isinf_impl(T x)
{
    // See the MATLAB's isinf function
    // (https://www.mathworks.com/help/matlab/ref/isinf.html)

    return ::std::isinf(x.real()) || ::std::isinf(x.imag());
}

} // Namespace detail


/**
 * \brief Returns a logical vector expression telling whether each element
 *  of a given vector expression is infinity.
 *
 * \tparam VectorExprT The type of the input vector expression.
 *
 * \param ve The input vector expression.
 * \return A vector expression representing a logical (integral) vector
 *  containing `1` (i.e., `true`) where the elements of \a ve are either
 *  positive or negative infinity, and `0` (i.e., `false`) where they are not.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename detail::vector_isinf_functor_traits<VectorExprT>::result_type isinf(vector_expression<VectorExprT> const& ve)
{
    typedef typename detail::vector_isinf_functor_traits<VectorExprT>::expression_type expression_type;
    typedef typename detail::vector_isinf_functor_traits<VectorExprT>::signature_argument_type signature_argument_type;

    return expression_type(ve(), detail::isinf_impl<signature_argument_type>);
}


/**
 * \brief Returns a logical matrix expression telling whether each element
 *  of a given matrix expression is infinity.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \return A matrix expression representing a logical (integral) matrix
 *  containing `1` (i.e., `true`) where the elements of \a me are either
 *  positive or negative infinity, and `0` (i.e., `false`) where they are not.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename detail::matrix_isinf_functor_traits<MatrixExprT>::result_type isinf(matrix_expression<MatrixExprT> const& me)
{
    typedef typename detail::matrix_isinf_functor_traits<MatrixExprT>::expression_type expression_type;
    typedef typename detail::matrix_isinf_functor_traits<MatrixExprT>::signature_argument_type signature_argument_type;

    return expression_type(me(), detail::isinf_impl<signature_argument_type>);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ISINF_HPP
