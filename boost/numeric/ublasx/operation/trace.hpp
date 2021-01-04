/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/trace.hpp
 *
 * \brief Compute the trace of a matrix.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_TRACE_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_TRACE_HPP


#include <algorithm>
#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>


namespace boost { namespace numeric { namespace ublasx {

namespace ublas = ::boost::numeric::ublas;


/**
 * \brief Compute the trace of a given matrix.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param A The input matrix expression.
 * \return The trace of \a A.
 *
 * The trace of a m-by-n matrix \f$A\f$ is defined as
 * \f[
 *   \operatorname{tr}(A) = \sum_{i=1}^k a_{ii}
 * \f]
 * where \f$k=\min(m,n)\f$.
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename ublas::matrix_traits<MatrixExprT>::value_type trace(ublas::matrix_expression<MatrixExprT> const& A)
{
    typedef typename ublas::matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename ublas::matrix_traits<MatrixExprT>::size_type size_type;

    size_type n = ::std::min(num_rows(A), num_columns(A));
    value_type res = 0;

    for (size_type i = 0; i < n; ++i)
    {
        res += A()(i,i);
    }

    return res;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_TRACE_HPP
