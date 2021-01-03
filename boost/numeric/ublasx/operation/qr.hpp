/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/qr.hpp
 *
 * \brief The QR matrix decomposition.
 *
 * Given a matrix \f$A\f$, its QR-decomposition is a matrix decomposition of the
 * form:
 * \f[
 *   A=QR
 * \f]
 * where \f$R\f$ is an m-by-n upper trapezoidal (or, when \f$m \ge n\f$,
 * triangular) matrix and \f$Q\f$ is an m-by-m orthogonal (or unitary) matrix,
 * that is one satisfying:
 * \f[
 *   Q^{T}Q=I,
 * \f]
 * where \f$Q^{T}\f$ is the transpose of \f$Q\f$ and \f$I\f$ is the identity
 * matrix.
 *
 * For the special case of \f$m \ge n\f$, the factorization can be rewritten as:
 * \f[
 *  A=\begin{pmatrix}
 *     Q_1 & Q_2
 *     \end{pmatrix}
 *     \begin{pmatrix}
 *     R_1 \\
 *     R_2 \\
 *     \end{pmatrix}
 *   =\begin{pmatrix}
 *     Q_1 & Q_2
 *     \end{pmatrix}
 *     \begin{pmatrix}
 *     R_1 \\
 *     0 \\
 *     \end{pmatrix}
 *   = Q_1 R_1
 * \f] 
 * where \f$Q_1\f$ is an m-by-n matrix, \f$Q_2\f$ is an m-by-(m-n) matrix,
 * \f$R_1\f$ is an n-by-n lower triangular matrix, and \f$R_2\f$ is an
 * (m-n)-by-n zero matrix.
 *
 * This matrix decomposition can be used to solve linear systems of equations,
 * especially the ones involved in the linear least squares problem.
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_QR_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_QR_HPP


#include <algorithm>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/numeric/bindings/lapack/computational/geqrf.hpp>
#include <boost/numeric/bindings/lapack/computational/orgqr.hpp>
#include <boost/numeric/bindings/lapack/computational/ormqr.hpp>
#include <boost/numeric/bindings/lapack/computational/ungqr.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/bindings/tag.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/type_traits/is_same.hpp>
#include <complex>
#include <cstddef>
#include <stdint.h>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {

/**
 * \brief Common operations for QR decomposition.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
struct qr_decomposition_impl_common;

/**
 * \brief Type-oriented operations for QR decomposition.
 *
 * \tparam IsComplex Logical parameter telling if the we are doing either a real
 *  or a complex QR decomposition.
 *
 * This class makes distinction between the real and the complex case.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <bool IsComplex>
struct qr_decomposition_impl;


struct qr_decomposition_impl_common
{
    /// Performan QR decomposition of the given input matrix \a A
    /// (row-major case).
    template <typename AMatrixT, typename TauVectorT>
        static void decompose(AMatrixT& A, TauVectorT& tau, row_major_tag)
    {
        matrix<typename matrix_traits<AMatrixT>::value_type, column_major> tmp_A(A);

        decompose(tmp_A, tau, column_major_tag());

        A = tmp_A;
    }


    /// Performan QR decomposition of the given input matrix \a A
    /// (column-major case).
    template <typename AMatrixT, typename TauVectorT>
        static void decompose(AMatrixT& A, TauVectorT& tau, column_major_tag)
    {
        typedef typename matrix_traits<AMatrixT>::size_type size_type;

        size_type m = num_rows(A);
        size_type n = num_columns(A);
        size_type k = ::std::min(m,n);

        if (size(tau) != k)
        {
            tau.resize(k, false);
        }

        ::boost::numeric::bindings::lapack::geqrf(A, tau);
    }


    /**
     * \brief Extract the R matrix from a previously computing QR decomposition
     * (row-major case).
     *
     * Let QR be an m-by-n matrix, then the R matrix is built by taking the
     * min(m,n)-by-n upper trapezoidal (triangular, if m >= n) elements of QR.
     */
    template <typename QRMatrixT, typename RMatrixT>
        static void extract_R(QRMatrixT const& QR, RMatrixT& R, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QRMatrixT>::value_type, column_major> tmp_QR(QR);
        matrix<typename matrix_traits<RMatrixT>::value_type, column_major> tmp_R(R);

        extract_R(tmp_QR, tmp_R, full, column_major_tag());

        R = tmp_R;
    }


    /**
     * \brief Extract the R matrix from a previously computing QR decomposition
     * (row-major case).
     *
     * Let QR be an m-by-n matrix, then the R matrix is built by taking the
     * min(m,n)-by-n upper trapezoidal (triangular, if m >= n) elements of QR.
     */
    template <typename QRMatrixT, typename RMatrixT>
        static void extract_R(QRMatrixT const& QR, RMatrixT& R, bool full, column_major_tag)
    {
        typedef typename matrix_traits<RMatrixT>::size_type size_type;
        typedef typename matrix_traits<RMatrixT>::value_type value_type;

        size_type m = num_rows(QR);
        size_type n = num_columns(QR);
        size_type nr = full ? m : ::std::min(m,n);

        if (num_rows(R) != nr && num_columns(R) != n)
        {
            R.resize(nr, n, false);
        }

        if (m >= n)
        {
            // The upper triangle of the submatrix QR(1:n,1:n) contains the
            // min(m,n)-by-n upper triangular matrix R.
            for (size_type row = 0; row < n; ++row)
            {
                for (size_type col = 0; col < n; ++col)
                {
                    if (col >= row)
                    {
                        R(row,col) = QR(row,col);
                    }
                    else
                    {
                        R(row,col) = value_type/*zero*/();
                    }
                }
            }

            // Set to zero the last m-n rows
            if (full)
            {
                subrange(R, n, m, 0, n) = scalar_matrix<value_type>(m-n, n, value_type/*zero*/());
            }
        }
        else
        {
            // The elements on and upper the n-th subdiagonal contain the
            // m-by-n upper trapezoidal matrix R.
            for (size_type row = 0; row < m; ++row)
            {
                for (size_type col = 0; col < n; ++col)
                {
                    if (col >= row)
                    {
                        R(row,col) = QR(row,col);
                    }
                    else
                    {
                        R(row,col) = value_type/*zero*/();
                    }
                }
            }
        }
    }


    /**
     * \brief Multiply the given \a C matrix by the \c Q matrix obtained from
     *  the QR decomposition.
     *
     * \tparam QRMatrixT The type of the \a QR matrix.
     * \tparam TAUMatrixT The type of the \a tau vector.
     * \tparam CMatrixT The type of the \a C matrix.
     *
     * \param QR The matrix obtained by the QR decomposition such that the i-th
     *  column contains the vector which defines the elementary reflector
     *  \f$H(i)\f$, for \f$i = 1,2,\ldots,k\f$.
     * \param tau The vector obtained by the QR decomposition containing the
     *  scalar factors of the elementary reflectors \f$H(i)\f$, for
     *  \f$i=1,2,\ldots,k\f$.
     * \param left_Q A boolean value indicating which side of the product the
     *  matrix \c Q will occupy. A \c true value indicates that \c Q is the left
     *  operand, while a \c false value indicates that \c Q is the right
     *  operand.
     * \param trans_Q A boolean value indicating if the matrix \c Q is to be
     *  transposed. A \c true value indicates that \c Q is to be transposed,
     *  while a \c false value indicates that \c Q is to be taken as-is.
     * \param orientation The matrix orientation fixed to row-major.
     *
     * Let \c Q be the matrix obtained from the QR decomposition represented
     * by the \a QR matrix and the \a tau vector parameters. 
     * Then this function computes the following matrix product:
     * \f{equation*}{
     *   \begin{cases}
     *   Q C, & \text{\texttt{left\_Q} = \emph{true} and \texttt{trans\_Q} = \emph{false}}, \\
     *   Q^T C, & \text{\texttt{left\_Q} = \emph{true} and \texttt{trans\_Q} = \emph{true}}, \\
     *   C Q, & \text{\texttt{left\_Q} = \emph{false} and \texttt{trans\_Q} = \emph{false}}, \\
     *   C Q^T, & \text{\texttt{left\_Q} = \emph{false} and \texttt{trans\_Q} = \emph{true}}.
     *  \end{cases}
     * \f}
     */
    template <typename QRMatrixT, typename TAUVectorT, typename CMatrixT>
        static void prod(QRMatrixT& QR, TAUVectorT const& tau, CMatrixT& C, bool left_Q, bool trans_Q, row_major_tag)
    {
        //NOTE: QR cannot be const since LAPACK::ORMQR modified it, restoring
        //      it at the end of the function.

        matrix<typename matrix_traits<QRMatrixT>::value_type, column_major> tmp_QR(QR);
        matrix<typename matrix_traits<CMatrixT>::value_type, column_major> tmp_C(C);

        prod(tmp_QR, tau, tmp_C, left_Q, trans_Q, column_major_tag());

        C = tmp_C;
    }


    /**
     * \brief Multiply the given \a C matrix by the \c Q matrix obtained from
     *  the QR decomposition.
     *
     * \tparam QRMatrixT The type of the \a QR matrix.
     * \tparam TAUMatrixT The type of the \a tau vector.
     * \tparam CMatrixT The type of the \a C matrix.
     *
     * \param QR The matrix obtained by the QR decomposition such that the i-th
     *  column contains the vector which defines the elementary reflector
     *  \f$H(i)\f$, for \f$i = 1,2,\ldots,k\f$.
     * \param tau The vector obtained by the QR decomposition containing the
     *  scalar factors of the elementary reflectors \f$H(i)\f$, for
     *  \f$i=1,2,\ldots,k\f$.
     * \param left_Q A boolean value indicating which side of the product the
     *  matrix \c Q will occupy. A \c true value indicates that \c Q is the left
     *  operand, while a \c false value indicates that \c Q is the right
     *  operand.
     * \param trans_Q A boolean value indicating if the matrix \c Q is to be
     *  transposed. A \c true value indicates that \c Q is to be transposed,
     *  while a \c false value indicates that \c Q is to be taken as-is.
     * \param orientation The matrix orientation fixed to column-major.
     *
     * Let \c Q be the matrix obtained from the QR decomposition represented
     * by the \a QR matrix and the \a tau vector parameters. 
     * Then this function computes the following matrix product:
     * \f{equation*}{
     *   \begin{cases}
     *   Q C, & \text{\texttt{left\_Q} = \emph{true} and \texttt{trans\_Q} = \emph{false}}, \\
     *   Q^T C, & \text{\texttt{left\_Q} = \emph{true} and \texttt{trans\_Q} = \emph{true}}, \\
     *   C Q, & \text{\texttt{left\_Q} = \emph{false} and \texttt{trans\_Q} = \emph{false}}, \\
     *   C Q^T, & \text{\texttt{left\_Q} = \emph{false} and \texttt{trans\_Q} = \emph{true}}.
     *  \end{cases}
     * \f}
     */
    template <typename QRMatrixT, typename TAUVectorT, typename CMatrixT>
        static void prod(QRMatrixT& QR, TAUVectorT const& tau, CMatrixT& C, bool left_Q, bool trans_Q, column_major_tag /*orientation*/)
    {
        //NOTE: QR cannot be const since LAPACK::ORMQR modified it, restoring
        //      it at the end of the function.

//      typedef typename matrix_traits<QRMatrixT>::value_type value_type;
//      typedef typename matrix_traits<QRMatrixT>::size_type size_type;
//      typedef typename type_traits<value_type>::real_type real_type;
//
//      const ::fortran_int_t m = num_rows(C);
//      const ::fortran_int_t n = num_columns(C);
//      const ::fortran_int_t k = size(tau);
//      const ::fortran_int_t lda = num_rows(QR);
//      const ::fortran_int_t ldc = m;
//      real_type* work;
//      real_type opt_work_size;
//      ::fortran_int_t lwork;
//      ::std::ptrdiff_t info;

        if (left_Q)
        {
            if (trans_Q)
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormqr has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormqr(
                    ::boost::numeric::bindings::tag::left(),
                    ::boost::numeric::bindings::trans(QR),
                    tau,
                    C
                );
            }
            else
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormqr has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormqr(
                    ::boost::numeric::bindings::tag::left(),
                    QR,
                    tau,
                    C
                );
            }
        }
        else
        {
            if (trans_Q)
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormqr has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormqr(
                    ::boost::numeric::bindings::tag::right(),
                    ::boost::numeric::bindings::trans(QR),
                    tau,
                    C
                );
            }
            else
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormqr has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormqr(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QR.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormqr(
                    ::boost::numeric::bindings::tag::right(),
                    QR,
                    tau,
                    C
                );
            }
        }
    }
};


template <>
struct qr_decomposition_impl<false>: public qr_decomposition_impl_common
{
    /// Extract the Q matrix from a previously computing QR decomposition
    /// (row-major case).
    template <typename QRMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QRMatrixT const& QR, TauVectorT const& tau, QMatrixT& Q, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QRMatrixT>::value_type, column_major> tmp_QR(QR);
        matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);

        extract_Q(tmp_QR, tau, tmp_Q, full, column_major_tag());

        Q = tmp_Q;
    }


    /// Extract the Q matrix from a previously computing QR decomposition
    /// (column-major case).
    template <typename QRMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QRMatrixT const& QR, TauVectorT& tau, QMatrixT& Q, bool full, column_major_tag)
    {
        typedef typename matrix_traits<QMatrixT>::size_type size_type;
        typedef typename matrix_traits<QMatrixT>::value_type value_type;

        size_type m = num_rows(QR);
        size_type n = num_columns(QR);
        size_type nc = full ? m : ::std::min(m,n);

        if (num_rows(Q) != m || num_columns(Q) != nc)
        {
            Q.resize(m, nc, false);
        }

        if (m > n)
        {
            if (full)
            {
                subrange(Q, 0, m, 0, n) = QR;
                subrange(Q, 0, m, n, m) = scalar_matrix<value_type>(m, m-n, value_type/*zero*/());
            }
            else
            {
                Q = QR;
            }
        }
        else if (m < n)
        {
            Q = subrange(QR, 0, m, 0, nc);
        }
        else
        {
            Q = QR;
        }

       ::boost::numeric::bindings::lapack::orgqr(Q, tau);
    }
};


template <>
struct qr_decomposition_impl<true>: public qr_decomposition_impl_common
{
    /// Extract the Q matrix from a previously computing QR decomposition
    /// (row-major case).
    template <typename QRMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QRMatrixT const& QR, TauVectorT const& tau, QMatrixT& Q, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QRMatrixT>::value_type, column_major> tmp_QR(QR);
        matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);

        extract_Q(tmp_QR, tau, tmp_Q, full, column_major_tag());

        Q = tmp_Q;
    }


    /// Extract the Q matrix from a previously computing QR decomposition
    /// (column-major case).
    template <typename QRMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QRMatrixT const& QR, TauVectorT const& tau, QMatrixT& Q, bool full, column_major_tag)
    {
        typedef typename matrix_traits<QMatrixT>::size_type size_type;
        typedef typename matrix_traits<QMatrixT>::value_type value_type;

        size_type m = num_rows(QR);
        size_type n = num_columns(QR);
        size_type nc = full ? m : ::std::min(m,n);

        if (num_rows(Q) != m || num_columns(Q) != nc)
        {
            Q.resize(m, nc, false);
        }

        if (m > n)
        {
            if (full)
            {
                subrange(Q, 0, m, 0, n) = QR;
                subrange(Q, 0, m, n, m) = scalar_matrix<value_type>(m, m-n, value_type/*zero*/());
            }
            else
            {
                Q = QR;
            }
        }
        else if (m < n)
        {
            Q = subrange(QR, 0, m, 0, nc);
        }
        else
        {
            Q = QR;
        }

        ::boost::numeric::bindings::lapack::ungqr(Q, tau);
    }
};


/// Free function performing the QR decomposition of the given matrix expression \a A.
template<typename MatrixExprT, typename QMatrixT, typename RMatrixT, typename OrientationT>
void qr_decompose_impl(matrix_expression<MatrixExprT> const& A, QMatrixT& Q, RMatrixT& R, bool full, OrientationT orientation)
{
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    matrix<value_type, typename layout_type<MatrixExprT>::type> tmp_QR(A);
    vector<value_type> tmp_tau;

    qr_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template decompose(tmp_QR, tmp_tau, orientation);


    qr_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template extract_Q(tmp_QR, tmp_tau, Q, full, orientation);


    qr_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template extract_R(tmp_QR, R, full, orientation);
}

}} // Namespace detail::<unnamed>


/**
 * \brief QR decomposition.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ValueT>
class qr_decomposition
{
    public: typedef ValueT value_type;
    private: typedef matrix<value_type, column_major> work_matrix_type;
    public: typedef work_matrix_type QR_matrix_type;
    public: typedef work_matrix_type Q_matrix_type;
    public: typedef work_matrix_type R_matrix_type;
    private: typedef vector<value_type> tau_vector_type;


    public: qr_decomposition()
    {
        // empty
    }


    public: template <typename MatrixExprT>
        qr_decomposition(matrix_expression<MatrixExprT> const& A)
        : QR_(A)
    {
        decompose();
    }


    public: template <typename MatrixExprT>
        void decompose(matrix_expression<MatrixExprT> const& A)
    {
        QR_ = A;

        decompose();
    }


    public: Q_matrix_type Q(bool full = true) const
    {
        Q_matrix_type tmp_Q;

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template extract_Q(QR_, tau_, tmp_Q, full, column_major_tag());

        return tmp_Q;
    }


    public: R_matrix_type R(bool full = true) const
    {
        R_matrix_type tmp_R;

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template extract_R(QR_, tmp_R, full, column_major_tag());

        return tmp_R;
    }


    /// Perform the product \f$Q C\f$ and store the result in \a C.
    public: template <typename CMatrixT>
        void lprod_inplace(CMatrixT& C) const
    {
        typedef typename matrix_traits<CMatrixT>::orientation_category orientation_category;

        lprod_inplace(C, orientation_category());
    }


    /// Perform the product \f$C Q\f$ and store the result in \a C.
    public: template <typename CMatrixT>
        void rprod_inplace(CMatrixT& C) const
    {
        typedef typename matrix_traits<CMatrixT>::orientation_category orientation_category;

        rprod_inplace(C, orientation_category());
    }


    /// Perform the product \f$Q^T C\f$ and store the result in \a C.
    public: template <typename CMatrixT>
        void tlprod_inplace(CMatrixT& C) const
    {
        typedef typename matrix_traits<CMatrixT>::orientation_category orientation_category;

        tlprod_inplace(C, orientation_category());
    }


    /// Perform the product \f$C Q^T\f$ and store the result in \a C.
    public: template <typename CMatrixT>
        void trprod_inplace(CMatrixT& C) const
    {
        typedef typename matrix_traits<CMatrixT>::orientation_category orientation_category;

        trprod_inplace(C, orientation_category());
    }


    /// Perform the product \f$Q C\f$ and return the result.
    public: template <typename CMatrixExprT>
        typename matrix_temporary_traits<CMatrixExprT>::type lprod(matrix_expression<CMatrixExprT> const& C) const
    {
        typename matrix_temporary_traits<CMatrixExprT>::type tmp_C(C);

        lprod_inplace(tmp_C);

        return tmp_C;
    }


    /// Perform the product \f$C Q\f$ and return the result.
    public: template <typename CMatrixExprT>
        typename matrix_temporary_traits<CMatrixExprT>::type rprod(matrix_expression<CMatrixExprT> const& C) const
    {
        typename matrix_temporary_traits<CMatrixExprT>::type tmp_C(C);

        rprod_inplace(tmp_C);

        return tmp_C;
    }


    /// Perform the product \f$Q^T C\f$ and return the result.
    public: template <typename CMatrixExprT>
        typename matrix_temporary_traits<CMatrixExprT>::type tlprod(matrix_expression<CMatrixExprT> const& C) const
    {
        typename matrix_temporary_traits<CMatrixExprT>::type tmp_C(C);

        tlprod_inplace(tmp_C);

        return tmp_C;
    }


    /// Perform the product \f$C Q^T\f$ and return the result.
    public: template <typename CMatrixExprT>
        typename matrix_temporary_traits<CMatrixExprT>::type trprod(matrix_expression<CMatrixExprT> const& C) const
    {
        typename matrix_temporary_traits<CMatrixExprT>::type tmp_C(C);

        trprod_inplace(tmp_C);

        return tmp_C;
    }


    private: void decompose()
    {
        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template decompose(QR_, tau_, column_major_tag());
    }


    /// Perform the product \f$Q C\f$ and store the result in \a C (column-major
    /// case).
    private: template <typename CMatrixT>
        void lprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, C, true, false, column_major_tag());
    }


    /// Perform the product \f$Q C\f$ and store the result in \a C (row-major
    /// case).
    private: template <typename CMatrixT>
        void lprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, tmp_C, true, false, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$C Q\f$ and store the result in \a C (column-major
    /// case).
    private: template <typename CMatrixT>
        void rprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::qr_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template prod(QR_, tau_, C, false, false, column_major_tag());
    }


    /// Perform the product \f$C Q\f$ and store the result in \a C (row-major
    /// case).
    private: template <typename CMatrixT>
        void rprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, tmp_C, false, false, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$Q^T C\f$ and store the result in \a C
    /// (column-major case).
    private: template <typename CMatrixT>
        void tlprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::qr_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template prod(QR_, tau_, C, true, true, column_major_tag());
    }


    /// Perform the product \f$Q^T C\f$ and store the result in \a C
    /// (row-major case).
    private: template <typename CMatrixT>
        void tlprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, tmp_C, true, true, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$C Q^T\f$ and store the result in \a C
    /// (column-major case).
    private: template <typename CMatrixT>
        void trprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, C, false, true, column_major_tag());
    }


    /// Perform the product \f$C Q^T\f$ and store the result in \a C
    /// (row-major case).
    private: template <typename CMatrixT>
        void trprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::qr_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QR_, tau_, tmp_C, false, true, column_major_tag());

        C = tmp_C;
    }


    // NOTE: the 'mutable' keyword is needed in order to make 'const' the
    //       '?prod' methods ('lprod', 'tlprod', 'rprod', 'trprod').
    //       Indeed, these methods call the respective '?prod_inplace' methods
    //       which, in turns, call the LAPACK::ORMQR function which temporarily
    //       changes the QR matrix (and restores it before returning).
    private: mutable QR_matrix_type QR_;
    private: tau_vector_type tau_;
};


/// Free function performing the QR decomposition of the given matrix expression \a A.
template<typename MatrixExprT, typename OutMatrix1T, typename OutMatrix2T>
BOOST_UBLAS_INLINE
void qr_decompose(matrix_expression<MatrixExprT> const& A, OutMatrix1T& Q, OutMatrix2T& R, bool full = true)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category1;
    typedef typename matrix_traits<OutMatrix1T>::orientation_category orientation_category2;
    typedef typename matrix_traits<OutMatrix2T>::orientation_category orientation_category3;

    // precondition: same orientation category
    BOOST_MPL_ASSERT(
        (::boost::mpl::and_<
            ::boost::is_same<orientation_category1,orientation_category2>,
            ::boost::is_same<orientation_category1,orientation_category3>
        >)
    );

    detail::qr_decompose_impl(A, Q, R, full, orientation_category1());
}


/// Free function performing the QR decomposition of the given matrix expression \a A.
template<typename MatrixExprT>
BOOST_UBLAS_INLINE
qr_decomposition<typename matrix_traits<MatrixExprT>::value_type> qr_decompose(matrix_expression<MatrixExprT> const& A)
{
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    return qr_decomposition<value_type>(A);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_QR_HPP
