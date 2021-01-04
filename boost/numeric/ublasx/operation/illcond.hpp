/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/illcond.hpp
 *
 * \brief Check if a matrix is ill-conditioned.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ILLCOND_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ILLCOND_HPP


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublasx/operation/rcond.hpp>
#include <cmath>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

template <typename MatrixExprT>
BOOST_UBLAS_INLINE
bool illcond(matrix_expression<MatrixExprT> const& A)
{
    double r = rcond(A);
    volatile double rp1 = r + 1.0;

    return (rp1 == 1.0) || ::std::isnan(r);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ILLCOND_HPP
