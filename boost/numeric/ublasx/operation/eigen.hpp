/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/eigen.hpp
 *
 * \brief Compute the eigenvalues and eigenvectors of a single matrix or the
 *  generalized eigenvalues and eigeinvector of a pair of matrices.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010-2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_EIGEN_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_EIGEN_HPP


#include <boost/mpl/and.hpp>
#include <boost/numeric/bindings/lapack/driver/geev.hpp>
#include <boost/numeric/bindings/lapack/driver/ggev.hpp>
#include <boost/numeric/bindings/lapack/driver/heev.hpp>
#include <boost/numeric/bindings/lapack/driver/hegv.hpp>
#include <boost/numeric/bindings/lapack/driver/syev.hpp>
#include <boost/numeric/bindings/lapack/driver/sygv.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/detail/lapack.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_complex.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <complex>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail {

/// Side of eigenvectors
enum eigenvectors_side
{
    none_eigenvectors, ///< None eigenvectors are to be selected.
    left_eigenvectors, ///< Only the left eigenvectors are to be selected.
    right_eigenvectors, ///< Only the right eigenvectors are to be selected.
    both_eigenvectors ///< Both left and right eigenvectors are to be selected.
};


//@{ Eigenvalues problem


/// Eigenvalues of a general real matrix (column-major case).
//FIXME: It seems that LAPACK (v. 3.2.1) wants that VR and VL have right dimensions even if they should be not referenced (e.g., jobvl='N' or jobvr='N').
template <
    typename MatrixExprT,    // must be of real type
    typename OutRealVectorT, // must be of real type
    typename OutImagVectorT, // must be of real type
    typename OutLeftMatrixT, // must be of complex type
    typename OutRightMatrixT // must be of complex type
>
void eigen_impl(matrix_expression<MatrixExprT> const& A, eigenvectors_side side, OutRealVectorT& rw, OutImagVectorT& iw, OutLeftMatrixT& LV, OutRightMatrixT& RV, column_major_tag)
{
    // precondition: A must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<MatrixExprT>::value_type>::value
    ));
    // precondition: rw must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<OutRealVectorT>::value_type>::value
    ));
    // precondition: iw must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<OutImagVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutLeftMatrixT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutRightMatrixT>::value_type>::value
    ));


    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef matrix<value_type, column_major> work_matrix_type;

    size_type n = num_rows(A);

    char jobvl;
    char jobvr;
    size_type work_n_LV;
    size_type out_n_LV;
    size_type work_n_RV;
    size_type out_n_RV;

    // Copy the original A matrix since LAPACK GEEV overwrites it.
    work_matrix_type tmp_A(A);

    switch (side)
    {
        case both_eigenvectors:
            jobvl = jobvr = 'V';
            work_n_LV = out_n_LV = work_n_RV = out_n_RV = n;
            break;
        case left_eigenvectors:
            jobvl = 'V';
            jobvr = 'N';
            work_n_LV = out_n_LV = n;
            work_n_RV = detail::lapack::min_array_size;
            out_n_RV = 0;
            break;
        case right_eigenvectors:
            jobvl = 'N';
            jobvr = 'V';
            work_n_LV = detail::lapack::min_array_size;
            out_n_LV = 0;
            work_n_RV = out_n_RV = n;
            break;
        case none_eigenvectors:
        default:
            jobvl = jobvr = 'N';
            work_n_LV = work_n_RV = detail::lapack::min_array_size;
            out_n_LV = out_n_RV = 0;
    }

    // Create temporary real matrices for eigenvectors
    // NOTE: LAPACK wants the arrays be correctly sized even if they are not
    // requested as output (i.e., jobvl == 'N' and/or jobvr == 'N').
    work_matrix_type tmp_LV(work_n_LV, work_n_LV);
    work_matrix_type tmp_RV(work_n_RV, work_n_RV);

    if (size(rw) != n)
    {
        rw.resize(n, false);
    }
    if (size(iw) != n)
    {
        iw.resize(n, false);
    }

    ::boost::numeric::bindings::lapack::geev(
        jobvl,
        jobvr,
        tmp_A,
        rw,
        iw,
        tmp_LV,
        tmp_RV
    );

    // Resize output complex eigenvectors matrices ...

    if (num_rows(LV) != out_n_LV || num_columns(LV) != out_n_LV)
    {
        LV.resize(out_n_LV, out_n_LV, false);
    }
    if (num_rows(RV) != out_n_RV || num_columns(RV) != out_n_RV)
    {
        RV.resize(out_n_RV, out_n_RV, false);
    }

    // ... and possibly fill eigenvectors matrices LV and RV

    if (out_n_LV > 0 || out_n_RV > 0)
    {
        // The algorithm for computing left and right eigenvectors is the
        // following:
        // - The left eigenvectors u(j) are stored one
        //   after another in the columns of LV, in the same order as
        //   their eigenvalues. If the j-th eigenvalue is real, then
        //   u(j) = LV(:,j), the j-th column of LV. If the j-th and
        //   (j+1)-th eigenvalues form a complex conjugate pair, then
        //   u(j) = LV(:,j)+i*LV(:,j+1) and u(j+1) = LV(:,j)-i*LV(:,j+1).
        // - The right eigenvectors v(j) are stored one
        //   after another in the columns of RV, in the same order as
        //   their eigenvalues. If the j-th eigenvalue is real, then
        //   v(j) = RV(:,j), the j-th column of RV. If the j-th and
        //   (j+1)-th eigenvalues form a complex conjugate pair, then
        //   v(j) = RV(:,j)+i*RV(:,j+1) and v(j+1) = RV(:,j)-i*RV(:,j+1).
        // .
        // Each eigenvector is scaled so the largest component has
        // abs(real part)+abs(imag. part)=1.
        //

        for (size_type i = 0; i < n; ++i)
        {
            for (size_type j = 0; j < n; ++j)
            {
                if (iw(j) != 0.0)
                {
                    if (out_n_LV > 0)
                    {
                        value_type lv1 = tmp_LV(i,j);
                        value_type lv2 = tmp_LV(i,j+1);

                        LV(i,j) = ::std::complex<value_type>(lv1, lv2);
                        LV(i,j+1) = ::std::complex<value_type>(lv1, -lv2);
                    }

                    if (out_n_RV > 0)
                    {
                        value_type rv1 = tmp_RV(i,j);
                        value_type rv2 = tmp_RV(i,j+1);

                        RV(i,j) = ::std::complex<value_type>(rv1, rv2);
                        RV(i,j+1) = ::std::complex<value_type>(rv1, -rv2);
                    }

                    ++j;
                }
                else
                {
                    if (out_n_LV > 0)
                    {
                        LV(i,j) = tmp_LV(i,j);
                    }
                    if (out_n_RV > 0)
                    {
                        RV(i,j) = tmp_RV(i,j);
                    }
                }
            }
        }
    }
}


/// Eigenvalues of a general real matrix (row-major case).
template <
    typename MatrixExprT,
    typename OutRealVectorT,
    typename OutImagVectorT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
void eigen_impl(matrix_expression<MatrixExprT> const& A, eigenvectors_side side, OutRealVectorT& rw, OutImagVectorT& iw, OutLeftMatrixT& LV, OutRightMatrixT& RV, row_major_tag)
{
    // Note: LAPACK works with column-major matrices

    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);
    colmaj_matrix_type tmp_LV;
    colmaj_matrix_type tmp_RV;

    eigen_impl(tmp_A, side, rw, iw, tmp_LV, tmp_RV, column_major_tag());

    LV = tmp_LV;
    RV = tmp_RV;
}


/// Eigenvalues of a general complex matrix (column-major case).
//FIXME: It seems that LAPACK (v. 3.2.1) wants that VR and VL have right dimensions even if they should be not referenced (e.g., jobvl='N' or jobvr='N').
template <
    typename MatrixExprT,    // must be of complex type
    typename OutVectorT,     // must be of complex type
    typename OutLeftMatrixT, // must be of complex type
    typename OutRightMatrixT // must be of complex type
>
typename ::boost::enable_if<
    ::boost::is_complex<typename matrix_traits<MatrixExprT>::value_type>,
    void
>::type eigen_impl(matrix_expression<MatrixExprT> const& A, eigenvectors_side side, OutVectorT& w, OutLeftMatrixT& LV, OutRightMatrixT& RV, column_major_tag)
{
    // precondition: A must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<MatrixExprT>::value_type>::value
    ));
    // precondition: w must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutLeftMatrixT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutRightMatrixT>::value_type>::value
    ));


    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;

    size_type n = num_rows(A);
    size_type n_lv;
    size_type n_rv;

    matrix<value_type, column_major> tmp_A(A); // LAPACK GEEV overwrites the original input matrix A

    char jobvl;
    char jobvr;

    switch (side)
    {
        case both_eigenvectors:
            jobvl = jobvr = 'V';
            n_lv = n_rv = n;
            break;
        case left_eigenvectors:
            jobvl = 'V';
            jobvr = 'N';
            n_lv = n;
            n_rv = 0;
            break;
        case right_eigenvectors:
            jobvl = 'N';
            jobvr = 'V';
            n_lv = 0;
            n_rv = n;
            break;
        case none_eigenvectors:
        default:
            jobvl = jobvr = 'N';
            n_lv = n_rv = 0;
    }

    if (size(w) != n)
    {
        w.resize(n, false);
    }
    if (num_rows(LV) != n || num_columns(LV) != n)
    {
        LV.resize(n, n, false);
    }
    if (num_rows(RV) != n || num_columns(RV) != n)
    {
        RV.resize(n, n, false);
    }

    ::boost::numeric::bindings::lapack::geev(jobvl, jobvr, tmp_A, w, LV, RV);

    if (num_rows(LV) != n_lv)
    {
        LV.resize(n_lv, n_lv, true);
    }
    if (num_rows(RV) != n_rv)
    {
        RV.resize(n_rv, n_rv, true);
    }
}


/// Eigenvalues of a general real matrix (column-major case).
template <
    typename MatrixExprT,
    typename OutVectorT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
typename ::boost::disable_if<
    ::boost::is_complex<typename matrix_traits<MatrixExprT>::value_type>,
    void
>::type eigen_impl(matrix_expression<MatrixExprT> const& A, eigenvectors_side side, OutVectorT& w, OutLeftMatrixT& LV, OutRightMatrixT& RV, column_major_tag)
{
    // precondition: A must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<MatrixExprT>::value_type>::value
    ));
    // precondition: w must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutLeftMatrixT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutRightMatrixT>::value_type>::value
    ));


    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename vector_traits<OutVectorT>::size_type size_type;
    typedef typename vector_traits<OutVectorT>::value_type out_value_type;

    size_type n_w = size(w);
    size_type n = num_rows(A);

    if (n_w != n)
    {
        w.resize(n, false);
        n_w = n;
    }

    vector<value_type> rw(n_w);
    vector<value_type> iw(n_w);

    eigen_impl(A, side, rw, iw, LV, RV, column_major_tag());

    for (size_type i = 0; i < n_w; ++i)
    {
        // Assume that out_value_type is a complex-like type
        //w(i) = ::std::complex<value_type>(rw(i), iw(i));
        w(i) = out_value_type(rw(i), iw(i));
    }
}


/// Eigenvalues of a general real/complex matrix (row-major case).
template <
    typename MatrixExprT,
    typename OutVectorT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
void eigen_impl(matrix_expression<MatrixExprT> const& A, eigenvectors_side side, OutVectorT& w, OutLeftMatrixT& LV, OutRightMatrixT& RV, row_major_tag)
{
    typedef typename promote_traits<
                        typename matrix_traits<MatrixExprT>::value_type,
                        typename promote_traits<
                            typename matrix_traits<OutRightMatrixT>::value_type,
                            typename matrix_traits<OutLeftMatrixT>::value_type
                        >::promote_type
            >::promote_type value_type;

    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);
    colmaj_matrix_type tmp_LV;
    colmaj_matrix_type tmp_RV;

    eigen_impl(tmp_A, side, w, tmp_LV, tmp_RV, column_major_tag());

    LV = tmp_LV;
    RV = tmp_RV;
}


/// Eigenvalues of a hermitian matrix (column-major case).
template <
    typename ValueT,
    typename TriangularT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen_impl(hermitian_matrix<ValueT,TriangularT,column_major> const& A, eigenvectors_side side, OutVectorT& w, OutMatrixT& V)
{
    // precondition: A must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<ValueT>::value
    ));
    // precondition: w must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: V must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<OutMatrixT>::value_type>::value
    ));


    typedef hermitian_matrix<ValueT, TriangularT, column_major> matrix_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    typedef OutMatrixT out_matrix_type;
    typedef hermitian_adaptor<out_matrix_type, TriangularT> work_matrix_type;

    size_type n = num_rows(A);
    size_type n_v;

    char jobvz;

    switch (side)
    {
        case both_eigenvectors:
            jobvz = 'V';
            n_v = n;
            break;
        case none_eigenvectors:
        default:
            jobvz = 'N';
            n_v = 0;
    }

    if (size(w) != n)
    {
        w.resize(n, false);
    }
    if (num_rows(V) != n_v)
    {
        V.resize(n_v, n_v, false);
    }

    out_matrix_type aux_A(A);
    work_matrix_type tmp_A(aux_A);

    ::boost::numeric::bindings::lapack::heev(jobvz, tmp_A, w);

    if (n_v > 0)
    {
        V = aux_A;
    }
}


/// Eigenvalues of a hermitian matrix (row-major case).
template <
    typename ValueT,
    typename TriangularT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen_impl(hermitian_matrix<ValueT,TriangularT,row_major> const& A, eigenvectors_side side, OutVectorT& w, OutMatrixT& V)
{
    typedef hermitian_matrix<ValueT, TriangularT, column_major> colmaj_in_matrix_type;
    typedef matrix<typename matrix_traits<OutMatrixT>::value_type, column_major> colmaj_out_matrix_type;

    colmaj_in_matrix_type tmp_A(A);
    colmaj_out_matrix_type tmp_V;

    eigen_impl(tmp_A, side, w, tmp_V);

    V = tmp_V;
}


/// Eigenvalues of a symmetric matrix (column-major case).
template <
    typename ValueT,
    typename TriangularT,
    typename OutVectorT,
    typename OutMatrixT
>
void  eigen_impl(symmetric_matrix<ValueT,TriangularT,column_major> const& A, eigenvectors_side side, OutVectorT& w, OutMatrixT& V)
{
    // NOTE: a symmetric matrix is a real hermitian matrix

    // precondition: A must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<ValueT>::value
    ));
    // precondition: w must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: V must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<OutMatrixT>::value_type>::value
    ));


    typedef symmetric_matrix<ValueT, TriangularT, column_major> matrix_type;
    //typedef symmetric_matrix<double, lower, column_major> matrix_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    //typedef matrix<ValueT, column_major> aux_matrix_type;
    typedef OutMatrixT out_matrix_type;
    typedef symmetric_adaptor<out_matrix_type, TriangularT> work_matrix_type;

    size_type n = num_rows(A);
    size_type n_v;

    char jobvz;

    switch (side)
    {
        case both_eigenvectors:
            jobvz = 'V';
            n_v = n;
            break;
        case none_eigenvectors:
        default:
            jobvz = 'N';
            n_v = 0;
    }

    if (size(w) != n)
    {
        w.resize(n, false);
    }
    if (num_rows(V) != n_v)
    {
        V.resize(n_v, n_v, false);
    }

    out_matrix_type aux_A(A);
    work_matrix_type tmp_A(aux_A);

    ::boost::numeric::bindings::lapack::syev(jobvz, tmp_A, w);

    if (n_v > 0)
    {
        V = aux_A;
    }
}


/// Eigenvalues of a symmetric matrix (row-major case).
template <
    typename ValueT,
    typename TriangularT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen_impl(symmetric_matrix<ValueT,TriangularT,row_major> const& A, eigenvectors_side side, OutVectorT& w, OutMatrixT& V)
{
    typedef symmetric_matrix<ValueT, TriangularT, column_major> colmaj_in_matrix_type;
    typedef matrix<typename matrix_traits<OutMatrixT>::value_type, column_major> colmaj_out_matrix_type;

    colmaj_in_matrix_type tmp_A(A);
    colmaj_out_matrix_type tmp_V;

    eigen_impl(tmp_A, side, w, tmp_V);

    V = tmp_V;
}


//@} Eigenvalues problem


//@{ Generalized Eigenvalues problem


/// Generalized eigenvectors for real matrix pair (A,B) (column-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename AlphaRVectorT,
    typename AlphaIVectorT,
    typename BetaVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
void geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, AlphaRVectorT& alphar, AlphaIVectorT& alphai, BetaVectorT& beta, LVMatrixT& LV, RVMatrixT& RV, column_major_tag)
{
    // precondition: A must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<AMatrixExprT>::value_type>::value
    ));
    // precondition: B must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<BMatrixExprT>::value_type>::value
    ));
    // precondition: alphar must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<AlphaRVectorT>::value_type>::value
    ));
    // precondition: alphai must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<AlphaIVectorT>::value_type>::value
    ));
    // precondition: beta must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<BetaVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<LVMatrixT>::value_type>::value
    ));
    // precondition: RV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<RVMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::value_type,
                typename matrix_traits<BMatrixExprT>::value_type
            >::promote_type value_type;
    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::size_type,
                typename matrix_traits<BMatrixExprT>::size_type
            >::promote_type size_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;
    //typedef typename type_traits<typename matrix_traits<LVMatrixT>::value_type>::real_type left_real_type;
    //typedef typename type_traits<typename matrix_traits<RVMatrixT>::value_type>::real_type right_real_type;


    size_type n = num_rows(A);

    // LAPACK GEEV overwrites the original input matrix A
    colmaj_matrix_type tmp_A(A);
    // LAPACK GEEV overwrites the original input matrix B
    colmaj_matrix_type tmp_B(B);

    char jobvl;
    char jobvr;
    size_type work_n_LV;
    size_type out_n_LV;
    size_type work_n_RV;
    size_type out_n_RV;

    switch (side)
    {
        case both_eigenvectors:
            jobvl = jobvr = 'V';
            work_n_LV = out_n_LV = work_n_RV = out_n_RV = n;
            break;
        case left_eigenvectors:
            jobvl = 'V';
            jobvr = 'N';
            work_n_LV = out_n_LV = n;
            work_n_RV = detail::lapack::min_array_size;
            out_n_RV = 0;
            break;
        case right_eigenvectors:
            jobvl = 'N';
            jobvr = 'V';
            work_n_LV = detail::lapack::min_array_size;
            out_n_LV = 0;
            work_n_RV = out_n_RV = n;
            break;
        case none_eigenvectors:
        default:
            jobvl = jobvr = 'N';
            work_n_LV = work_n_RV = detail::lapack::min_array_size;
            out_n_LV = out_n_RV = 0;
    }

    // Size arrays with the proper size
    // NOTE: LAPACK always computes the eigenvalues so we need to size them with
    // the corrected size even if we do not want them (i.e.,
    // when want_eigvals == false).

    if (size(alphar) != n)
    {
        alphar.resize(n, false);
    }
    if (size(alphai) != n)
    {
        alphai.resize(n, false);
    }
    if (size(beta) != n)
    {
        beta.resize(n, false);
    }

    // Create temporary real matrices for eigenvectors
    // NOTE: LAPACK wants the arrays be correctly sized even if they are not
    // requested as output (i.e., jobvl == 'N' and/or jobvr == 'N').
    colmaj_matrix_type tmp_LV(work_n_LV, work_n_LV);
    colmaj_matrix_type tmp_RV(work_n_RV, work_n_RV);

    ::boost::numeric::bindings::lapack::ggev(
        jobvl,
        jobvr,
        tmp_A,
        tmp_B,
        alphar,
        alphai,
        beta,
        tmp_LV,
        tmp_RV
    );

    // Resize output complex eigenvectors matrices ...

    if (num_rows(LV) != out_n_LV || num_columns(LV) != out_n_LV)
    {
        LV.resize(out_n_LV, out_n_LV, false);
    }
    if (num_rows(RV) != out_n_RV || num_columns(RV) != out_n_RV)
    {
        RV.resize(out_n_RV, out_n_RV, false);
    }

    // ... and possibly fill eigenvectors matrices LV and RV

    if (out_n_LV > 0 || out_n_RV > 0)
    {
        // The algorithm for computing left and right eigenvectors is the
        // following:
        // - The left eigenvectors u(j) are stored one
        //   after another in the columns of LV, in the same order as
        //   their eigenvalues. If the j-th eigenvalue is real, then
        //   u(j) = LV(:,j), the j-th column of LV. If the j-th and
        //   (j+1)-th eigenvalues form a complex conjugate pair, then
        //   u(j) = LV(:,j)+i*LV(:,j+1) and u(j+1) = LV(:,j)-i*LV(:,j+1).
        // - The right eigenvectors v(j) are stored one
        //   after another in the columns of RV, in the same order as
        //   their eigenvalues. If the j-th eigenvalue is real, then
        //   v(j) = RV(:,j), the j-th column of RV. If the j-th and
        //   (j+1)-th eigenvalues form a complex conjugate pair, then
        //   v(j) = RV(:,j)+i*RV(:,j+1) and v(j+1) = RV(:,j)-i*RV(:,j+1).
        // .
        // Each eigenvector is scaled so the largest component has
        // abs(real part)+abs(imag. part)=1.
        //

        for (size_type i = 0; i < n; ++i)
        {
            for (size_type j = 0; j < n; ++j)
            {
                if (alphai(j) != 0.0)
                {
                    if (out_n_LV > 0)
                    {
                        value_type lv1 = tmp_LV(i,j);
                        value_type lv2 = tmp_LV(i,j+1);

                        LV(i,j) = ::std::complex<value_type>(lv1, lv2);
                        LV(i,j+1) = ::std::complex<value_type>(lv1, -lv2);
                    }

                    if (out_n_RV > 0)
                    {
                        value_type rv1 = tmp_RV(i,j);
                        value_type rv2 = tmp_RV(i,j+1);

                        RV(i,j) = ::std::complex<value_type>(rv1, rv2);
                        RV(i,j+1) = ::std::complex<value_type>(rv1, -rv2);
                    }

                    ++j;
                }
                else
                {
                    if (out_n_LV > 0)
                    {
                        LV(i,j) = tmp_LV(i,j);
                    }
                    if (out_n_RV > 0)
                    {
                        RV(i,j) = tmp_RV(i,j);
                    }
                }
            }
        }
    }

    // When possible, resize to save memory

//  if (num_rows(LV) != out_n_LV)
//  {
//      LV.resize(out_n_LV, out_n_LV, false);
//  }
//  if (num_rows(RV) != out_n_RV)
//  {
//      RV.resize(out_n_RV, out_n_RV, false);
//  }
    if (!want_eigvals)
    {
        alphar.resize(0, false);
        alphai.resize(0, false);
        beta.resize(0, false);
    }
}


/// Generalized eigenvectors for real matrix pair (A,B) (row-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename AlphaRVectorT,
    typename AlphaIVectorT,
    typename BetaVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
void geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, AlphaRVectorT& alphar, AlphaIVectorT& alphai, BetaVectorT& beta, LVMatrixT& LV, RVMatrixT& RV, row_major_tag)
{
    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::value_type,
                typename matrix_traits<BMatrixExprT>::value_type
            >::promote_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;


    colmaj_matrix_type tmp_A(A);
    colmaj_matrix_type tmp_B(B);
    colmaj_matrix_type tmp_LV;
    colmaj_matrix_type tmp_RV;

    geigen_impl(A, B, side, want_eigvals, alphar, alphai, beta, tmp_LV, tmp_RV, column_major_tag());

    LV = tmp_LV;
    RV = tmp_RV;
}


/// Generalized eigenvectors for complex matrix pair (A,B) (column-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename AlphaVectorT,
    typename BetaVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
void geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, AlphaVectorT& alpha, BetaVectorT& beta, LVMatrixT& LV, RVMatrixT& RV, column_major_tag)
{
    // precondition: A must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<AMatrixExprT>::value_type>::value
    ));
    // precondition: B must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<BMatrixExprT>::value_type>::value
    ));
    // precondition: alpha must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<AlphaVectorT>::value_type>::value
    ));
    // precondition: beta must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<BetaVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<LVMatrixT>::value_type>::value
    ));
    // precondition: RV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<RVMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::value_type,
                typename matrix_traits<BMatrixExprT>::value_type
            >::promote_type value_type;
    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::size_type,
                typename matrix_traits<BMatrixExprT>::size_type
            >::promote_type size_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;


    size_type n = num_rows(A);

    colmaj_matrix_type tmp_A(A); // LAPACK GEEV overwrites the original input matrix A
    colmaj_matrix_type tmp_B(B); // LAPACK GEEV overwrites the original input matrix B

    char jobvl;
    char jobvr;
    size_type n_LV;
    size_type n_RV;

    switch (side)
    {
        case both_eigenvectors:
            jobvl = jobvr = 'V';
            n_LV = n_RV = n;
            break;
        case left_eigenvectors:
            jobvl = 'V';
            jobvr = 'N';
            n_LV = n;
            n_RV = 0;
            break;
        case right_eigenvectors:
            jobvl = 'N';
            jobvr = 'V';
            n_LV = 0;
            n_RV = n;
            break;
        case none_eigenvectors:
        default:
            jobvl = jobvr = 'N';
            n_LV = n_RV = 0;
    }

    // Size matrices and arrays with the proper size
    //
    // NOTE: LAPACK wants the matrices/arrays be correctly sized even if they
    // are not requested as output (i.e., want_eigvals == false or jobvl='N' or jobvr='N').

    if (size(alpha) != n)
    {
        alpha.resize(n, false);
    }
    if (size(beta) != n)
    {
        beta.resize(n, false);
    }
    if (num_rows(LV) != n || num_columns(LV) != n)
    {
        LV.resize(n, n, false);
    }
    if (num_rows(RV) != n || num_columns(RV) != n)
    {
        RV.resize(n, n, false);
    }

    ::boost::numeric::bindings::lapack::ggev(
        jobvl,
        jobvr,
        tmp_A,
        tmp_B,
        alpha,
        beta,
        LV,
        RV
    );

    // When possible, resize to save memory

    if (num_rows(LV) != n_LV)
    {
        LV.resize(n_LV, n_LV, false);
    }
    if (num_rows(RV) != n_RV)
    {
        RV.resize(n_RV, n_RV, false);
    }
    if (!want_eigvals)
    {
        alpha.resize(0, false);
        beta.resize(0, false);
    }
}


/// Generalized eigenvectors for complex matrix pair (A,B) (row-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename AlphaVectorT,
    typename BetaVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
void geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, AlphaVectorT& alpha, BetaVectorT& beta, LVMatrixT& LV, RVMatrixT& RV, row_major_tag)
{
    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::value_type,
                typename matrix_traits<BMatrixExprT>::value_type
            >::promote_type value_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;


    colmaj_matrix_type tmp_A(A);
    colmaj_matrix_type tmp_B(B);
    colmaj_matrix_type tmp_LV;
    colmaj_matrix_type tmp_RV;

    geigen_impl(tmp_A, tmp_B, side, want_eigvals, alpha, beta, tmp_LV, tmp_RV, column_major_tag());

    LV = tmp_LV;
    RV = tmp_RV;
}


/// Generalized eigenvectors for either real or complex matrix pair (A,B) (column-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
typename ::boost::disable_if<
    typename ::boost::mpl::and_<
        ::boost::is_complex<typename matrix_traits<AMatrixExprT>::value_type>,
        ::boost::is_complex<typename matrix_traits<BMatrixExprT>::value_type>
    >::type,
    void
>::type geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, OutVectorT& w, LVMatrixT& LV, RVMatrixT& RV, column_major_tag)
{
    // precondition: A can be either a real or a complex matrix -> no check
    // precondition: B can be either a real or a complex matrix -> no check
    // precondition: w must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<LVMatrixT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<RVMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::size_type,
                typename matrix_traits<BMatrixExprT>::size_type
            >::promote_type size_type;
    typedef typename vector_traits<OutVectorT>::value_type eigval_value_type;
    typedef typename type_traits<eigval_value_type>::real_type eigval_real_type;

    size_type n = num_rows(A);

    vector<eigval_real_type> tmp_alphar; // (n);
    vector<eigval_real_type> tmp_alphai; // (n);
    vector<eigval_real_type> tmp_beta; // (n);

    geigen_impl(A, B, side, want_eigvals, tmp_alphar, tmp_alphai, tmp_beta, LV, RV, column_major_tag());

    if (want_eigvals)
    {
        if (size(w) != n)
        {
            w.resize(n, false);
        }
#ifdef BOOST_UBLASX_DEBUG
        // Underflow threshold (used in the 'safety check' inside the 'for' loop)
        eigval_real_type rmin(::std::numeric_limits<eigval_real_type>::min());
#endif // BOOST_UBLASX_DEBUG
        for (size_type i = 0; i < n; ++i)
        {
#ifdef BOOST_UBLASX_DEBUG
            // Safety check: when beta_i is near to zero the corresponding
            //               eigenvalue is infinite.
            //               This test was inspired by the 'f08wafe.f'
            //               subrouting found on NAG libraries.
            if ((::std::abs(tmp_alphar(i))+::std::abs(tmp_alphai(i)))*rmin >= ::std::abs(tmp_beta(i)))
            {
                BOOST_UBLASX_DEBUG_TRACE("[Warning] Eigenvalue(" << i << ") is numerically infinite or undetermined: alpha_r(" << i << ") = " << tmp_alphar(i) << ", alpha_i(" << i << ") = " << tmp_alphai(i) << ", beta(" << i << ") = " << tmp_beta(i));
            }
#endif // BOOST_UBLASX_DEBUG

            w(i) = eigval_value_type(tmp_alphar(i), tmp_alphai(i))/tmp_beta(i);
        }
    }
    else
    {
        w.resize(0, false);
    }
}


/// Generalized eigenvectors for either real or complex matrix pair (A,B) (column-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
typename ::boost::enable_if<
    typename ::boost::mpl::and_<
        ::boost::is_complex<typename matrix_traits<AMatrixExprT>::value_type>,
        ::boost::is_complex<typename matrix_traits<BMatrixExprT>::value_type>
    >::type,
    void
>::type geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, OutVectorT& w, LVMatrixT& LV, RVMatrixT& RV, column_major_tag)
{
    // precondition: A can be either a real or a complex matrix -> no check
    // precondition: B can be either a real or a complex matrix -> no check
    // precondition: w must be a complex vector
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename vector_traits<OutVectorT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<LVMatrixT>::value_type>::value
    ));
    // precondition: LV must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<RVMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<
                typename matrix_traits<AMatrixExprT>::size_type,
                typename matrix_traits<BMatrixExprT>::size_type
            >::promote_type size_type;
    typedef typename vector_traits<OutVectorT>::value_type eigval_value_type;

    size_type n = num_rows(A);

    vector<eigval_value_type> tmp_alpha; // (n);
    vector<eigval_value_type> tmp_beta; // (n);

    geigen_impl(A, B, side, want_eigvals, tmp_alpha, tmp_beta, LV, RV, column_major_tag());

    if (want_eigvals)
    {
        if (size(w) != n)
        {
            w.resize(n, false);
        }
#ifdef BOOST_UBLASX_DEBUG
        typedef typename type_traits<eigval_value_type>::real_type eigval_real_type;

        // Underflow threshold (used in the 'safety check' inside the 'for' loop)
        eigval_real_type rmin(::std::numeric_limits<eigval_real_type>::min());
#endif // BOOST_UBLASX_DEBUG
        for (size_type i = 0; i < n; ++i)
        {
#ifdef BOOST_UBLASX_DEBUG
            // Safety check: when beta_i is near to zero the corresponding
            //               eigenvalue is infinite.
            //               This test was inspired by the 'f08wnfe.f'
            //               subrouting found on NAG libraries.
            if (::std::abs(tmp_alpha(i))*rmin >= ::std::abs(tmp_beta(i)))
            {
                BOOST_UBLASX_DEBUG_TRACE("[Warning] Eigenvalue(" << i << ") is numerically infinite or undetermined: alpha(" << i << ") = " << tmp_alpha(i) << ", beta(" << i << ") = " << tmp_beta(i));
            }
#endif // BOOST_UBLASX_DEBUG

            w(i) = tmp_alpha(i)/tmp_beta(i);
        }
    }
    else
    {
        w.resize(0, false);
    }
}


/// Generalized eigenvectors for real/complex matrix pair (A,B) (row-major case).
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename LVMatrixT,
    typename RVMatrixT
>
//typename ::boost::enable_if<
//  typename ::boost::mpl::and_<
//      ::boost::is_complex<typename matrix_traits<AMatrixExprT>::value_type>,
//      ::boost::is_complex<typename matrix_traits<BMatrixExprT>::value_type>
//  >::type,
//  void
//>::type
void geigen_impl(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, eigenvectors_side side, bool want_eigvals, OutVectorT& w, LVMatrixT& LV, RVMatrixT& RV, row_major_tag)
{
    typedef typename promote_traits<
                        typename matrix_traits<AMatrixExprT>::value_type,
                        typename promote_traits<
                            typename matrix_traits<BMatrixExprT>::value_type,
                            typename promote_traits<
                                typename matrix_traits<LVMatrixT>::value_type,
                                typename matrix_traits<RVMatrixT>::value_type
                            >::promote_type
                        >::promote_type
            >::promote_type value_type;

    typedef matrix<value_type, column_major> colmaj_matrix_type;

    colmaj_matrix_type tmp_A(A);
    colmaj_matrix_type tmp_B(B);
    colmaj_matrix_type tmp_LV;
    colmaj_matrix_type tmp_RV;

    geigen_impl(tmp_A, tmp_B, side, want_eigvals, w, tmp_LV, tmp_RV, column_major_tag());

    LV = tmp_LV;
    RV = tmp_RV;
}


/// Generalized eigenvectors for real symmetric matrix pair (A,B).
template <
    typename AValueT,
    typename TriangularT,
    typename BValueT,
    typename WVectorT,
    typename VMatrixT
>
void geigen_impl(symmetric_matrix<AValueT,TriangularT,column_major> const& A, symmetric_matrix<BValueT,TriangularT,column_major> const& B, eigenvectors_side side, bool want_eigvals, WVectorT& w, VMatrixT& V)
{
    // precondition: A must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<AValueT>::value
    ));
    // precondition: B must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<BValueT>::value
    ));
    // precondition: w must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<WVectorT>::value_type>::value
    ));
    // precondition: V must be a real matrix
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename matrix_traits<VMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<AValueT,BValueT>::promote_type value_type;
    typedef symmetric_matrix<AValueT,TriangularT,column_major> A_matrix_type;
    typedef symmetric_matrix<BValueT,TriangularT,column_major> B_matrix_type;
    typedef typename promote_traits<
                typename matrix_traits<A_matrix_type>::size_type,
                typename matrix_traits<B_matrix_type>::size_type
            >::promote_type size_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;
    typedef symmetric_adaptor<colmaj_matrix_type, TriangularT> work_matrix_type;


    size_type n = num_rows(A);

    ::fortran_int_t itype = 1; // We always solve A*v=w*B*v
    char jobz;
    size_type n_V;

    switch (side)
    {
        case both_eigenvectors:
            jobz = 'V';
            n_V = n;
            break;
        case none_eigenvectors:
        default:
            jobz = 'N';
            n_V = 0;
    }

    // Size arrays with the proper size
    // NOTE: LAPACK wants the arrays be correctly sized even if they are not
    // requested as output (i.e., want_eigvals == false).

    if (size(w) != n)
    {
        w.resize(n, false);
    }

    //TODO: 
    // OPTIMIZATION: maybe using SPGV in place of SYGV may save us from
    // copying A and B into a dense temporary matrix.

    // LAPACK GEEV overwrites the original input matrix A
    colmaj_matrix_type aux_A(A);
    work_matrix_type tmp_A(aux_A);
    // LAPACK GEEV overwrites the original input matrix B
    colmaj_matrix_type aux_B(B);
    work_matrix_type tmp_B(aux_B);

    ::boost::numeric::bindings::lapack::sygv(
        itype,
        jobz,
        tmp_A,
        tmp_B,
        w
    );

    // When possible, resize to save memory

    if (!want_eigvals)
    {
        w.resize(0, false);
    }
    if (n_V == 0)
    {
        V.resize(0, 0, false);
    }
    else
    {
        V = aux_A;
    }
}


/// Generalized eigenvectors for real symmetric matrix pair (A,B).
/// (row-major case)
template <
    typename AValueT,
    typename TriangularT,
    typename BValueT,
    typename WVectorT,
    typename VMatrixT
>
void geigen_impl(symmetric_matrix<AValueT,TriangularT,row_major> const& A, symmetric_matrix<BValueT,TriangularT,row_major> const& B, eigenvectors_side side, bool want_eigvals, WVectorT& w, VMatrixT& V)
{
    typedef symmetric_matrix<AValueT, TriangularT, column_major> colmaj_A_matrix_type;
    typedef symmetric_matrix<BValueT, TriangularT, column_major> colmaj_B_matrix_type;
    typedef matrix<typename matrix_traits<VMatrixT>::value_type, column_major> colmaj_V_matrix_type;

    colmaj_A_matrix_type tmp_A(A);
    colmaj_B_matrix_type tmp_B(B);
    colmaj_V_matrix_type tmp_V;

    geigen_impl(tmp_A, tmp_B, side, want_eigvals, w, tmp_V);

    V = tmp_V;
}


/// Generalized eigenvectors for complex hermitian matrix pair (A,B)
/// (column-major case).
template <
    typename AValueT,
    typename TriangularT,
    typename BValueT,
    typename WVectorT,
    typename VMatrixT
>
void geigen_impl(hermitian_matrix<AValueT,TriangularT,column_major> const& A, hermitian_matrix<BValueT,TriangularT,column_major> const& B, eigenvectors_side side, bool want_eigvals, WVectorT& w, VMatrixT& V)
{
    // precondition: A must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<AValueT>::value
    ));
    // precondition: B must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<BValueT>::value
    ));
    // precondition: w must be a real vector
    BOOST_STATIC_ASSERT((
        !::boost::is_complex<typename vector_traits<WVectorT>::value_type>::value
    ));
    // precondition: V must be a complex matrix
    BOOST_STATIC_ASSERT((
        ::boost::is_complex<typename matrix_traits<VMatrixT>::value_type>::value
    ));


    typedef typename promote_traits<AValueT,BValueT>::promote_type value_type;
    typedef symmetric_matrix<AValueT,TriangularT,column_major> A_matrix_type;
    typedef symmetric_matrix<BValueT,TriangularT,column_major> B_matrix_type;
    typedef typename promote_traits<
                typename matrix_traits<A_matrix_type>::size_type,
                typename matrix_traits<B_matrix_type>::size_type
            >::promote_type size_type;
    typedef matrix<value_type, column_major> colmaj_matrix_type;
    typedef hermitian_adaptor<colmaj_matrix_type, TriangularT> work_matrix_type;


    size_type n = num_rows(A);

    ::fortran_int_t itype = 1; // We always solve A*v=w*B*v
    char jobz;
    size_type n_V;

    switch (side)
    {
        case both_eigenvectors:
            jobz = 'V';
            n_V = n;
            break;
        case none_eigenvectors:
        default:
            jobz = 'N';
            n_V = 0;
    }

    // Size arrays with the proper size
    // NOTE: LAPACK wants the arrays be correctly sized even if they are not
    // requested as output (i.e., want_eigvals == false).

    if (size(w) != n)
    {
        w.resize(n, false);
    }

    //TODO: 
    // OPTIMIZATION: maybe using HPGV in place of HEGV may save us from
    // copying A and B into a dense temporary matrix.

    // LAPACK GEEV overwrites the original input matrix A
    colmaj_matrix_type aux_A(A);
    work_matrix_type tmp_A(aux_A);
    // LAPACK GEEV overwrites the original input matrix B
    colmaj_matrix_type aux_B(B);
    work_matrix_type tmp_B(aux_B);

    ::boost::numeric::bindings::lapack::hegv(
        itype,
        jobz,
        tmp_A,
        tmp_B,
        w
    );

    // When possible, resize to save memory

    if (!want_eigvals)
    {
        w.resize(0, false);
    }
    if (n_V == 0)
    {
        V.resize(0, 0, false);
    }
    else
    {
        V = aux_A;
    }
}


/// Generalized eigenvectors for complex hermitian matrix pair (A,B).
/// (row-major case)
template <
    typename AValueT,
    typename TriangularT,
    typename BValueT,
    typename WVectorT,
    typename VMatrixT
>
void geigen_impl(hermitian_matrix<AValueT,TriangularT,row_major> const& A, hermitian_matrix<BValueT,TriangularT,row_major> const& B, eigenvectors_side side, bool want_eigvals, WVectorT& w, VMatrixT& V)
{
    typedef hermitian_matrix<AValueT, TriangularT, column_major> colmaj_A_matrix_type;
    typedef hermitian_matrix<BValueT, TriangularT, column_major> colmaj_B_matrix_type;
    typedef matrix<typename matrix_traits<VMatrixT>::value_type, column_major> colmaj_V_matrix_type;

    colmaj_A_matrix_type tmp_A(A);
    colmaj_B_matrix_type tmp_B(B);
    colmaj_V_matrix_type tmp_V;

    geigen_impl(tmp_A, tmp_B, side, want_eigvals, w, tmp_V);

    V = tmp_V;
}


//@} Generalized Eigenvalues problem

} // Namespace detail


/**
 * \brief Compute the eigenvalues and the left and right eigenvectors
 *  of the given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutLeftMatrixT The type of the left eigenvectors matrix.
 * \tparam OutRightMatrixT The type of the right eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param LV The output left eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 * \param RV The output right eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, \a LV, and \a RV contain the eigenvalues, the
 *  left eigenvectors, and the right eigenvector of \a A, respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename MatrixExprT,
    typename OutVectorT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
void eigen(matrix_expression<MatrixExprT> const& A, vector_container<OutVectorT>& v, matrix_container<OutLeftMatrixT>& LV, matrix_container<OutRightMatrixT>& RV)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutLeftMatrixT>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutRightMatrixT>::orientation_category orientation_category3;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    //matrix<value_type, typename layout_type<MatrixExprT>::type> tmp_A(A);

    detail::eigen_impl(A, detail::both_eigenvectors, v(), LV(), RV(), orientation_category1());
}


/**
 * \brief Compute the eigenvalues and the eigenvectors of the given symmetric
 *  matrix.
 *
 * \tparam ValueT The type of the input matrix elements.
 * \tparam TriangularT The type of the triangular shape.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  eigenvectors of \a A, respectively.
 *
 * \note
 * Since \a A is a symmetric matrix, the left and right eigenvectors are simply
 * each other's transpose.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen(symmetric_matrix<ValueT,TriangularT,LayoutT> const& A, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef symmetric_matrix<ValueT, TriangularT, LayoutT> matrix_type;
    typedef typename matrix_traits<matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    detail::eigen_impl(A, detail::both_eigenvectors, v(), V());

    // If eigen_impl assume that the input matrix can be changed, then use
    // these instructions below instead of the above one.
    //matrix_type tmp_A(A);
    //detail::eigen_impl(tmp_A, detail::both_eigenvectors, v, V);
}


/**
 * \brief Compute the eigenvalues and the eigenvectors of the given hermitian
 *  matrix.
 *
 * \tparam ValueT The type of the input matrix elements.
 * \tparam TriangularT The type of the triangular shape.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  eigenvectors of \a A, respectively.
 *
 * \note
 * Since \a A is a hermitian matrix, the left and right eigenvectors are simply
 * each other's conjugate transpose.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen(hermitian_matrix<ValueT,TriangularT,LayoutT> const& A, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef hermitian_matrix<ValueT, TriangularT, LayoutT> matrix_type;
    typedef typename matrix_traits<matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    detail::eigen_impl(A, detail::both_eigenvectors, v(), V());

    // If eigen_impl assume that the input matrix can be changed, then use
    // these instructions below instead of the above one.
    //matrix_type tmp_A(A);
    //detail::eigen_impl(tmp_A, detail::both_eigenvectors, v, V);
}


/**
 * \brief Compute the eigenvalues and the left and right eigenvectors
 *  of the given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutLeftMatrixT The type of the left eigenvectors matrix.
 * \tparam OutRightMatrixT The type of the right eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param LV The output left eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 * \param RV The output right eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, \a LV, and \a RV contain the eigenvalues, the
 *  left eigenvectors, and the right eigenvector of \a A, respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
void eigen(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, vector_container<OutVectorT>& v, matrix_container<OutLeftMatrixT>& LV, matrix_container<OutRightMatrixT>& RV)
{
    typedef typename matrix_traits<AMatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<BMatrixExprT>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutLeftMatrixT>::orientation_category orientation_category3;
    typedef typename matrix_traits<OutRightMatrixT>::orientation_category orientation_category4;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::mpl::and_<
                ::boost::is_same<orientation_category1,orientation_category3>,
                ::boost::is_same<orientation_category1,orientation_category4>
            >
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );
    // precondition: B is square
    BOOST_UBLAS_CHECK(
        (num_rows(B) == num_columns(B)),
        bad_argument()
    );

    detail::geigen_impl(A, B, detail::both_eigenvectors, true, v(), LV(), RV(), orientation_category1());
}


/**
 * \brief Compute the eigenvalues and the eigenvectors of the given symmatrix
 *  matrix expression.
 *
 * \tparam AValueT The type of the elements of the first input matrix
 *  expression.
 * \tparam ATriangularT The type of the triangular shape of the first input
 *  matrix expression.
 * \tparam BValueT The type of the elements of the second input matrix
 *  expression.
 * \tparam BTriangularT The type of the triangular shape of the second input
 *  matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, \a LV, and \a RV contain the eigenvalues, the
 *  left eigenvectors, and the right eigenvector of \a A, respectively.
 *
 * \note
 *  The second input matrix \a B is assumed to be positive definite.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename AValueT,
    typename ATriangularT,
    typename ALayoutT,
    typename BValueT,
    typename BTriangularT,
    typename BLayoutT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen(symmetric_matrix<AValueT,ATriangularT,ALayoutT> const& A, symmetric_matrix<BValueT,BTriangularT,BLayoutT> const& B, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef symmetric_matrix<AValueT,ATriangularT,ALayoutT> A_matrix_type;
    typedef symmetric_matrix<BValueT,BTriangularT,BLayoutT> B_matrix_type;
    typedef typename matrix_traits<A_matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<B_matrix_type>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category3;

    // precondition: same triangular shape
    BOOST_STATIC_ASSERT((
        ::boost::is_same<ATriangularT,BTriangularT>::value
    ));
    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );
    // precondition: B is square
    BOOST_UBLAS_CHECK(
        (num_rows(B) == num_columns(B)),
        bad_argument()
    );

    detail::geigen_impl(A, B, detail::both_eigenvectors, true, v(), V());
}


/**
 * \brief Compute the eigenvalues and the eigenvectors of the given symmatrix
 *  matrix expression.
 *
 * \tparam AValueT The type of the elements of the first input matrix
 *  expression.
 * \tparam ATriangularT The type of the triangular shape of the first input
 *  matrix expression.
 * \tparam BValueT The type of the elements of the second input matrix
 *  expression.
 * \tparam BTriangularT The type of the triangular shape of the second input
 *  matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, \a LV, and \a RV contain the eigenvalues, the
 *  left eigenvectors, and the right eigenvector of \a A, respectively.
 *
 * \note
 *  The second input matrix \a B is assumed to be positive definite.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename AValueT,
    typename ATriangularT,
    typename ALayoutT,
    typename BValueT,
    typename BTriangularT,
    typename BLayoutT,
    typename OutVectorT,
    typename OutMatrixT
>
void eigen(hermitian_matrix<AValueT,ATriangularT,ALayoutT> const& A, hermitian_matrix<BValueT,BTriangularT,BLayoutT> const& B, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef symmetric_matrix<AValueT,ATriangularT,ALayoutT> A_matrix_type;
    typedef symmetric_matrix<BValueT,BTriangularT,BLayoutT> B_matrix_type;
    typedef typename matrix_traits<A_matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<B_matrix_type>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category3;

    // precondition: same triangular shape
    BOOST_STATIC_ASSERT((
        ::boost::is_same<ATriangularT,BTriangularT>::value
    ));
    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );
    // precondition: B is square
    BOOST_UBLAS_CHECK(
        (num_rows(B) == num_columns(B)),
        bad_argument()
    );

    detail::geigen_impl(A, B, detail::both_eigenvectors, true, v(), V());
}


/**
 * \brief Compute the eigenvalues and the left eigenvectors of the given matrix
 *  expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the left eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output left eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  left eigenvectors of \a A, respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename MatrixExprT,
    typename OutVectorT,
    typename OutMatrixT
>
void left_eigen(matrix_expression<MatrixExprT> const& A, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;
    typedef typename layout_type<OutMatrixT>::type out_layout_type;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    //OutMatrixT dummy_RV;
    matrix<typename matrix_traits<OutMatrixT>::value_type, out_layout_type> dummy_RV;

    detail::eigen_impl(A, detail::left_eigenvectors, v(), V(), dummy_RV, orientation_category1());
}


/**
 * \brief Compute the generazlied eigenvalues and the left generazlied
 *  eigenvectors of the given matrix pair expression.
 *
 * \tparam AMatrixExprT The type of the first input matrix expression.
 * \tparam BMatrixExprT The type of the second input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the left eigenvectors matrix.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output left eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  left eigenvectors of \a (A,B), respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename OutMatrixT
>
void left_eigen(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef typename matrix_traits<AMatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<BMatrixExprT>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category3;
    typedef typename layout_type<OutMatrixT>::type out_layout_type;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );
    // precondition: B is square
    BOOST_UBLAS_CHECK(
        (num_rows(B) == num_columns(B)),
        bad_argument()
    );

    //OutMatrixT dummy_RV;
    matrix<typename matrix_traits<OutMatrixT>::value_type, out_layout_type> dummy_RV;

    detail::geigen_impl(A, B, detail::left_eigenvectors, true, v(), V(), dummy_RV, orientation_category1());
}


/**
 * \brief Compute the eigenvalues and the right eigenvectors of the given matrix
 *  expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the right eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output right eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  right eigenvectors of \a A, respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename MatrixExprT,
    typename OutVectorT,
    typename OutMatrixT
>
void right_eigen(matrix_expression<MatrixExprT> const& A, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;
    typedef typename layout_type<OutMatrixT>::type out_layout_type;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    //OutMatrixT dummy_LV;
    matrix<typename matrix_traits<OutMatrixT>::value_type, out_layout_type> dummy_LV;

    detail::eigen_impl(A, detail::right_eigenvectors, v(), dummy_LV, V(), orientation_category1());
}


/**
 * \brief Compute the generazlied eigenvalues and the right generazlied
 *  eigenvectors of the given matrix pair expression.
 *
 * \tparam AMatrixExprT The type of the first input matrix expression.
 * \tparam BMatrixExprT The type of the second input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 * \tparam OutMatrixT The type of the right eigenvectors matrix.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \param v The output eigenvalues vector.
 * \param V The output right eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing, but \a v, and \a V contain the eigenvalues, and the
 *  right eigenvectors of \a (A,B), respectively.
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename AMatrixExprT,
    typename BMatrixExprT,
    typename OutVectorT,
    typename OutMatrixT
>
void right_eigen(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, vector_container<OutVectorT>& v, matrix_container<OutMatrixT>& V)
{
    typedef typename matrix_traits<AMatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<BMatrixExprT>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category3;
    typedef typename layout_type<OutMatrixT>::type out_layout_type;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );
    // precondition: B is square
    BOOST_UBLAS_CHECK(
        (num_rows(B) == num_columns(B)),
        bad_argument()
    );

    //OutMatrixT dummy_LV;
    matrix<typename matrix_traits<OutMatrixT>::value_type, out_layout_type> dummy_LV;

    detail::geigen_impl(A, B, detail::right_eigenvectors, true, v(), dummy_LV, V(), orientation_category1());
}


/**
 * \brief Compute the eigenvalues of the given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutVectorT The type of the eigenvalues vector.
 *
 * \param A The input matrix expression.
 * \param v The output eigenvalues vector.
 * \return Nothing; however the parameter \a v will store on exit the
 *  eigenvalues vector of \a A.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename MatrixExprT,
    typename OutVectorT
>
void eigenvalues(matrix_expression<MatrixExprT> const& A, vector_container<OutVectorT>& v)
{
    typedef typename vector_traits<OutVectorT>::value_type out_value_type;
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;

    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    matrix<out_value_type, typename layout_type<MatrixExprT>::type> tmp_LV;
    matrix<out_value_type, typename layout_type<MatrixExprT>::type> tmp_RV;

    detail::eigen_impl(A, detail::none_eigenvectors, v(), tmp_LV, tmp_RV, orientation_category());
}


/**
 * \brief Compute the eigenvalues of the given symmetrix matrix.
 *
 * \tparam ValueT The type of the elements of the input symmetric matrix.
 * \tparam Triangular The triangular type of the input symmetric matrix.
 * \tparam LayoutT The storage layout type of the input symmetric matrix.
 * \tparam OutVectorT The type of the eigenvalues vector.
 *
 * \param A The input symmetric matrix.
 * \param v The output eigenvalues vector.
 * \return Nothing; however the parameter \a v will store on exit the
 *  eigenvalues vector of \a A.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutVectorT
>
void eigenvalues(symmetric_matrix<ValueT,TriangularT,LayoutT> const& A, vector_container<OutVectorT>& v)
{
    typedef typename vector_traits<OutVectorT>::value_type out_value_type;

    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    matrix<out_value_type, LayoutT> tmp_V;

    detail::eigen_impl(A, detail::none_eigenvectors, v(), tmp_V);
}


/**
 * \brief Compute the eigenvalues of the given hermitian matrix.
 *
 * \tparam ValueT The type of the elements of the input hermitian matrix.
 * \tparam Triangular The triangular type of the input hermitian matrix.
 * \tparam LayoutT The storage layout type of the input hermitian matrix.
 * \tparam OutVectorT The type of the eigenvalues vector.
 *
 * \param A The input hermitian matrix.
 * \param v The output eigenvalues vector.
 * \return Nothing; however the parameter \a v will store on exit the
 *  eigenvalues vector of \a A.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutVectorT
>
void eigenvalues(hermitian_matrix<ValueT,TriangularT,LayoutT> const& A, vector_container<OutVectorT>& v)
{
    typedef ValueT in_value_type;

    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    matrix<in_value_type, LayoutT> tmp_V;

    detail::eigen_impl(A, detail::none_eigenvectors, v(), tmp_V);
}


/**
 * \brief Compute the left and right eigenvectors of the given matrix
 *  expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam OutLeftMatrixT The type of the left eigenvectors matrix.
 * \tparam OutRightMatrixT The type of the right eigenvectors matrix.
 *
 * \param A The input matrix expression.
 * \param LV The output left eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 * \param RV The output right eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing; however parameter \a LV and \a RV will store on exit the
 *  left and right eigenvectors matrix of \a A, respectively.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename MatrixExprT,
    typename OutLeftMatrixT,
    typename OutRightMatrixT
>
void eigenvectors(matrix_expression<MatrixExprT> const& A, matrix_container<OutLeftMatrixT>& LV, matrix_container<OutRightMatrixT>& RV)
{
    typedef typename promote_traits<
                typename matrix_traits<OutLeftMatrixT>::value_type,
                typename matrix_traits<OutRightMatrixT>::value_type
            >::promote_type out_value_type;
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutLeftMatrixT>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutRightMatrixT>::orientation_category orientation_category3;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    vector<out_value_type> tmp_v;

    detail::eigen_impl(A, detail::both_eigenvectors, tmp_v, LV(), RV(), orientation_category1());
}


/**
 * \brief Compute the left and right eigenvectors of the given symmetric matrix.
 *
 * \tparam ValueT The type of the elements of the input symmetric matrix.
 * \tparam TriangularT The triangular type of the input symmetric matrix.
 * \tparam LayoutT The storage layout type of the input symmetric matrix.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The input symmetric matrix.
 * \param V The output (right) eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing; however parameter \a V will store on exit the right
 *  eigenvectors matrix of \a A.
 *
 * \note
 * Since \a A is a symmetric matrix, the left and right eigenvectors are simply
 * each other's transpose.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutMatrixT
>
void eigenvectors(symmetric_matrix<ValueT, TriangularT, LayoutT> const& A, matrix_container<OutMatrixT>& V)
{
    typedef symmetric_matrix<ValueT, TriangularT, LayoutT> in_matrix_type;
    typedef typename matrix_traits<OutMatrixT>::value_type out_value_type;
    typedef typename matrix_traits<in_matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    vector<out_value_type> tmp_v;

    detail::eigen_impl(A, detail::both_eigenvectors, tmp_v, V());
}


/**
 * \brief Compute the left and right eigenvectors of the given hermitian matrix.
 *
 * \tparam ValueT The type of the elements of the input hermitian matrix.
 * \tparam TriangularT The triangular type of the input hermitian matrix.
 * \tparam LayoutT The storage layout type of the input hermitian matrix.
 * \tparam OutMatrixT The type of the eigenvectors matrix.
 *
 * \param A The input hermitian matrix.
 * \param V The output (right) eigenvectors matrix (each eigenvector is stored
 *  column-wise).
 *
 * \return Nothing; however parameter \a V will store on exit the right
 *  eigenvectors matrix of \a A.
 *
 * \note
 * Since \a A is a hermitian matrix, the left and right eigenvectors are simply
 * each other's conjugate transpose.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename TriangularT,
    typename LayoutT,
    typename OutMatrixT
>
void eigenvectors(hermitian_matrix<ValueT, TriangularT, LayoutT> const& A, matrix_container<OutMatrixT>& V)
{
    typedef hermitian_matrix<ValueT, TriangularT, LayoutT> in_matrix_type;
    typedef typename matrix_traits<OutMatrixT>::value_type out_value_type;
    typedef typename matrix_traits<in_matrix_type>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrixT>::orientation_category orientation_category2;

    // precondition: same orientation category
    BOOST_STATIC_ASSERT((
        ::boost::is_same<orientation_category1,orientation_category2>::value
    ));
    // precondition: A is square
    BOOST_UBLAS_CHECK(
        (num_rows(A) == num_columns(A)),
        bad_argument()
    );

    vector<out_value_type> tmp_v;

    detail::eigen_impl(A, detail::both_eigenvectors, tmp_v, V());
}


//TODO
//template <
//  typename MatrixExprT,
//  typename OutMatrixT
//>
//void left_eigenvectors(matrix_expression<MatrixExprT> const& A, OutMatrixT& V)
//{
//}


//TODO
//template <
//  typename MatrixExprT,
//  typename OutMatrixT
//>
//void right_eigenvectors(matrix_expression<MatrixExprT> const& A, OutMatrixT& V)
//{
//}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_EIGEN_HPP
