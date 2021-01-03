/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/num_columns.hpp
 *
 * \brief The \c num_columns operation.
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_NUM_COLUMNS_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_NUM_COLUMNS_HPP


/*
#include <boost/version.hpp>


#if BOOST_VERSION > 105100L


#include <boost/numeric/ublas/operation/num_columns.hpp>


namespace boost { namespace numeric { namespace ublasx {

using ::boost::numeric::ublas::num_columns;

}}} // Namespace boost::numeric::ublasx


#else // BOOST_VERSION


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/traits.hpp>


namespace boost { namespace numeric { namespace ublasx {

template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::size_type num_columns(matrix_expression<MatrixExprT> const& me)
{
    return me().size2();
}


#endif // BOOST_VERSION
*/
#include <boost/numeric/ublas/operation/num_columns.hpp>


namespace boost { namespace numeric { namespace ublasx {

using ::boost::numeric::ublas::num_columns;

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_NUM_COLUMNS_HPP
