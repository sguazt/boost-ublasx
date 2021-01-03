/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/arithmetic_ops.hpp
 *
 * \brief Collection of arithmetic operators for matrix and vector expressions.
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ARITHMETIC_OPS_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ARITHMETIC_OPS_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>


/// (t / v)[i] = t / v[i]
template <typename T1, typename E2>
BOOST_UBLAS_INLINE
typename ::boost::enable_if<
    ::boost::is_convertible<T1, typename E2::value_type>,
    typename ::boost::numeric::ublas::vector_binary_scalar1_traits<
                const T1,
                E2,
                ::boost::numeric::ublas::scalar_divides<T1, typename E2::value_type>
    >::result_type
>::type operator/(T1 const& e1, ::boost::numeric::ublas::vector_expression<E2> const& e2)
{
    typedef typename ::boost::numeric::ublas::vector_binary_scalar1_traits<
                const T1,
                E2,
                ::boost::numeric::ublas::scalar_divides<T1, typename E2::value_type>
        >::expression_type expression_type;

    return expression_type(e1, e2());
}


/// (t / A)(i,j) = t / A(i,j)
template <typename T1, typename E2>
BOOST_UBLAS_INLINE
typename ::boost::enable_if<
    ::boost::is_convertible<T1, typename E2::value_type>,
    typename ::boost::numeric::ublas::matrix_binary_scalar1_traits<
                const T1,
                E2,
                ::boost::numeric::ublas::scalar_divides<T1, typename E2::value_type>
    >::result_type
>::type operator/(T1 const& e1, ::boost::numeric::ublas::matrix_expression<E2> const& e2)
{
    typedef typename ::boost::numeric::ublas::matrix_binary_scalar1_traits<
                const T1,
                E2,
                ::boost::numeric::ublas::scalar_divides<T1, typename E2::value_type>
        >::expression_type expression_type;

    return expression_type(e1, e2());
}


#endif //  BOOST_NUMERIC_UBLASX_OPERATION_ARITHMETIC_OPS_HPP
