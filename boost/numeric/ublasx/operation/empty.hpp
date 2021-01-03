/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/empty.hpp
 *
 * \brief Check for emptiness a ginve vector/matrix expression.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLAS_OPERATION_EMPTY_HPP
#define BOOST_NUMERIC_UBLAS_OPERATION_EMPTY_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief Check for emptiness the given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 * \param ve The input vector expression.
 * \return \c true if \a ve is a zero-sized vector; \c false otherwise.
 *
 * A zero-sized vector is a vector of zero length.
 */
template <typename VectorExprT>
bool empty(vector_expression<VectorExprT> const& ve)
{
    return size(ve) == 0;
}


/**
 * \brief Check for emptiness the given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \param me The input matrix expression.
 * \return \c true if \a me is a zero-sized matrix; \c false otherwise.
 *
 * A zero-sized matrix is a vector with either zero rows or zero columns or
 * both.
 */
template <typename MatrixExprT>
bool empty(matrix_expression<MatrixExprT> const& me)
{
    return num_rows(me) == 0 || num_columns(me) == 0;
}


}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLAS_OPERATION_EMPTY_HPP
