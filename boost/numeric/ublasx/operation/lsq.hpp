/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/lsq.hpp
 *
 * \brief Least Square problem solvers.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_LSQ_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_LSQ_HPP


#include <algorithm>
#include <boost/numeric/bindings/lapack/driver/gels.hpp>
#include <boost/numeric/bindings/lapack/driver/gelss.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/rcond.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail { namespace /*<unnamed>*/ {


template <typename MatrixT, typename VectorT>
void llsq_qr_impl(MatrixT& A, VectorT& b, column_major_tag)
{
    typedef typename promote_traits<
                typename matrix_traits<MatrixT>::size_type,
                typename vector_traits<VectorT>::size_type
        >::promote_type size_type;

//  size_type m = num_rows(A);
    size_type n = num_columns(A);

    ::boost::numeric::bindings::lapack::gels(A, b);

    b.resize(n, true);
}


template <typename MatrixT, typename VectorT>
void llsq_qr_impl(MatrixT& A, VectorT& b, row_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_qr_impl(tmp_A, b, column_major_tag());
}


template <typename MatrixT, typename VectorT>
void llsq_qr_impl(matrix_expression<MatrixT> const& A, VectorT& b, column_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_qr_impl(tmp_A, b, column_major_tag());
}


template <typename MatrixT, typename VectorT>
void llsq_qr_impl(matrix_expression<MatrixT> const& A, VectorT& b, row_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_qr_impl(tmp_A, b, column_major_tag());
}


template <typename MatrixT, typename VectorT>
void llsq_svd_impl(MatrixT& A, VectorT& b, column_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef typename type_traits<value_type>::real_type real_type;
    typedef typename promote_traits<
                typename matrix_traits<MatrixT>::size_type,
                typename vector_traits<VectorT>::size_type
        >::promote_type size_type;
    typedef vector<real_type> work_vector_type;

    size_type m = num_rows(A);
    size_type n = num_columns(A);
    size_type k = ::std::min(m, n);
    real_type rc = rcond(A);
    ::fortran_int_t r;
    //work_vector_type tmp_b(b);//TODO: should we do this in order to avoid problem with other types of vector (like sparse vector)
    work_vector_type dummy_s(k);

    ::boost::numeric::bindings::lapack::gelss(A, b, dummy_s, rc, r);

    b.resize(n, true);
}


template <typename MatrixT, typename VectorT>
void llsq_svd_impl(MatrixT& A, VectorT& b, row_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_svd_impl(tmp_A, b, column_major_tag());
}


template <typename MatrixT, typename VectorT>
void llsq_svd_impl(matrix_expression<MatrixT> const& A, VectorT& b, column_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_svd_impl(tmp_A, b, column_major_tag());
}


template <typename MatrixT, typename VectorT>
void llsq_svd_impl(matrix_expression<MatrixT> const& A, VectorT& b, row_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);

    llsq_svd_impl(tmp_A, b, column_major_tag());
}

}} // Namespace detail::<unnamed>


/**
 * \brief Solve the linear (ordinary) least square problem by using the QR
 *  decomposition.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b On entry, the input vector (i.e., the observations vector); on exit,
 *  the least square solution.
 *
 * Orthogonal decomposition methods of solving the least squares problem are
 * slower than directly solving the normal equations but are more numerically
 * stable.
 */
template <typename MatrixExprT, typename VectorT>
BOOST_UBLAS_INLINE
void llsq_qr_inplace(matrix_expression<MatrixExprT> const& A, VectorT& b)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;

    detail::llsq_qr_impl(A, b, orientation_category());
}


/**
 * \brief Solve the linear (ordinary) least square problem by using the QR
 *  decomposition.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b The input vector (i.e., the observations vector).
 * \return The least square solution.
 *
 * Orthogonal decomposition methods of solving the least squares problem are
 * slower than directly solving the normal equations but are more numerically
 * stable.
 */
template <typename MatrixExprT, typename VectorExprT>
BOOST_UBLAS_INLINE
typename vector_temporary_traits<VectorExprT>::type llsq_qr(matrix_expression<MatrixExprT> const& A, vector_expression<VectorExprT> const& b)
{
    typedef typename vector_temporary_traits<VectorExprT>::type out_vector_type;

    out_vector_type x(b);

    llsq_qr_inplace(A, x);

    return x;
}


/**
 * \brief Solve the linear (ordinary) least square problem by using the Singular
 * Value  Decomposition (SVD) method.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b On entry, the input vector (i.e., the observations vector); on exit,
 *  the least square solution.
 *
 * This method is the most computationally intensive, but is particularly useful
 * if the normal equations matrix is very ill-conditioned (i.e. if its condition
 * number multiplied by the machine's relative round-off error is appreciably
 * large).
 */
template <typename MatrixExprT, typename VectorT>
BOOST_UBLAS_INLINE
void llsq_svd_inplace(matrix_expression<MatrixExprT> const& A, VectorT& b)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;

    detail::llsq_svd_impl(A, b, orientation_category());
}


/**
 * \brief Solve the linear (ordinary) least square problem by using the Singular
 *  Value Decomposition (SVD) method.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b The input vector (i.e., the observations vector).
 * \return The least square solution.
 *
 * This method is the most computationally intensive, but is particularly useful
 * if the normal equations matrix is very ill-conditioned (i.e. if its condition
 * number multiplied by the machine's relative round-off error is appreciably
 * large).
 */
template <typename MatrixExprT, typename VectorExprT>
BOOST_UBLAS_INLINE
typename vector_temporary_traits<VectorExprT>::type llsq_svd(matrix_expression<MatrixExprT> const& A, vector_expression<VectorExprT> const& b)
{
    typedef typename vector_temporary_traits<VectorExprT>::type out_vector_type;

    out_vector_type x(b);

    llsq_svd_inplace(A, x);

    return x;
}


/**
 * \brief Solve the linear (ordinary) least square problem.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b On entry, the input vector (i.e., the observations vector); on exit,
 *  the least square solution.
 */
template <typename MatrixExprT, typename VectorT>
BOOST_UBLAS_INLINE
void llsq_inplace(matrix_expression<MatrixExprT> const& A, VectorT& b)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;

    detail::llsq_svd_impl(A, b, orientation_category());
}


/**
 * \brief Solve the linear (ordinary) least square problem.
 * \tparam MatrixExprT Type of the input matrix expression.
 * \tparam VectorExprT Type of the input/output vector.
 * \param A The input matrix expression (i.e., the design matrix).
 * \param b The input vector (i.e., the observations vector).
 * \return The least square solution.
 */
template <typename MatrixExprT, typename VectorExprT>
BOOST_UBLAS_INLINE
typename vector_temporary_traits<VectorExprT>::type llsq(matrix_expression<MatrixExprT> const& A, vector_expression<VectorExprT> const& b)
{
    typedef typename vector_temporary_traits<VectorExprT>::type out_vector_type;

    out_vector_type x(b);

    llsq_inplace(A, x);

    return x;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_LSQ_HPP
