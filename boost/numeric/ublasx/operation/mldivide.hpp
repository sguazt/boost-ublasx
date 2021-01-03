/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/mldivide.hpp
 *
 * \brief Matrix left division.
 *
 * Inspired by the \c mldivide MATLAB function.
 *
 * <hr/>
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#ifndef BOOST_NUMERIC_UBLASX_MLDIVIDE_HPP
#define BOOST_NUMERIC_UBLASX_MLDIVIDE_HPP


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/lu.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

template<typename AMatrixT,
         typename BVectorT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixT>::size_type mldivide_inplace(matrix_expression<AMatrixT> const& A,
                                                             vector_container<BVectorT>& b)
{
    return lu_solve_inplace(A, b);
}

template<typename AMatrixT,
         typename BVectorT,
         typename XVectorT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixT>::size_type mldivide(matrix_expression<AMatrixT> const& A,
                                                     vector_expression<BVectorT> const& b,
                                                     vector_container<XVectorT>& x)
{
    return lu_solve(A, b, x());
}

template<typename AMatrixT,
         typename BMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixT>::size_type mldivide_inplace(matrix_expression<AMatrixT> const& A,
                                                             matrix_container<BMatrixT>& B)
{
    return lu_solve_inplace(A, B);
}

template<typename AMatrixT,
         typename BMatrixT,
         typename XMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixT>::size_type mldivide(matrix_expression<AMatrixT> const& A,
                                                     matrix_expression<BMatrixT> const& B,
                                                     matrix_container<XMatrixT>& X)
{
    return lu_solve(A, B, X());
}

}}} // Namespace boost::numeric::ublasx

#endif // BOOST_NUMERIC_UBLASX_MLDIVIDE_HPP
