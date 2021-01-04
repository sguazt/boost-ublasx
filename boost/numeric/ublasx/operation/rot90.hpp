/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file path/to/include/file.hpp
 *
 * \brief Rotate matrix 90 degrees.
 *
 * Inspired by the \c rot90 MATLAB function.
 * See http://www.mathworks.com/help/techdoc/ref/rot90.html.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_ROT90_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_ROT90_HPP


//#include <boost/numeric/ublasx/detail/temporary.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cstddef>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail {

template <typename MatrixT>
struct rot90_matrix_traits
{
//Not all matrix types have this type (e.g., scalar_matrix<>)
//  typedef typename detail::matrix_temporary_traits<MatrixT>::type result_type;
    typedef matrix<typename matrix_traits<MatrixT>::value_type> result_type;
};

template <typename VectorT>
struct rot90_vector_traits
{
//Not all vector types have this type (e.g., scalar_vector<>)
//  typedef typename detail::vector_temporary_traits<VectorT>::type result_type;
    typedef vector<typename vector_traits<VectorT>::value_type> result_type;
};

} // Namespace detail


/// Rotate the given vector counterclockwise by (a multiple of) 90 degrees.
template <typename VectorT>
typename detail::rot90_vector_traits<VectorT>::result_type rot90(vector_expression<VectorT> const& v, int k=1)
{
    typedef typename detail::rot90_vector_traits<VectorT>::result_type result_type;
    typedef typename vector_traits<VectorT>::size_type size_type;

    size_type n(ublasx::size(v));

    result_type x;

    // Make sure k \in {0, 1, 2, 3}
    k %= 4;
    if (k < 0)
    {
        k += 4;
    }

    // NOTE: uBLAS makes no distinction between row and column vector.
    //       So rotations for k=0 and k=1 are considered identical.
    //       The same applies for rotations for k=2 and k=3.

    if (k == 2 || k == 3)
    {
        x.resize(n, false);

        for (size_type i = 0; i < n; ++i)
        {
            x(n-i-1) = v()(i);
        }
    }
    else
    {
        x = v;
    }

    return x;
}


/// Rotate the given vector counterclockwise by (a multiple of) 90 degrees.
template <typename VectorT>
BOOST_UBLAS_INLINE
void rot90_inplace(vector_container<VectorT>& v, int k=1)
{
    v() = rot90(v, k);
}


/// Rotate the given matrix counterclockwise by (a multiple of) 90 degrees
template <typename MatrixT>
typename detail::rot90_matrix_traits<MatrixT>::result_type rot90(matrix_expression<MatrixT> const& A, int k=1)
{
    typedef typename detail::rot90_matrix_traits<MatrixT>::result_type result_type;
    typedef typename matrix_traits<MatrixT>::size_type size_type;

    size_type nr(ublasx::num_rows(A));
    size_type nc(ublasx::num_columns(A));

    result_type X;

    // Make sure k \in {0, 1, 2, 3}
    k %= 4;
    if (k < 0)
    {
        k += 4;
    }

    if (k == 1)
    {
        X.resize(nc, nr, false);

        for (size_type c = 0; c < nc; ++c)
        {
            row(X, nc-c-1) = column(A(), c);
        }
    }
    else if (k == 2)
    {
        X.resize(nr, nc, false);

        for (size_type r = 0; r < nr; ++r)
        {
            for (size_type c = 0; c < nc; ++c)
            {
                X(nr-r-1,nc-c-1) = A()(r,c);
            }
        }
    }
    else if (k == 3)
    {
        X.resize(nc, nr, false);

        for (size_type r = 0; r < nr; ++r)
        {
            column(X, nr-r-1) = row(A(), r);
        }
    }
    else
    {
        X = A;
    }

    return X;
}

/// Rotate the given matrix counterclockwise by (a multiple of) 90 degrees
template <typename MatrixT>
BOOST_UBLAS_INLINE
void rot90_inplace(matrix_container<MatrixT>& A, int k=1)
{
    A() = rot90(A, k);
}


}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_ROT90_HPP
