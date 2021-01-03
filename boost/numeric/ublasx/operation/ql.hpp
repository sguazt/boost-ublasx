/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/ql.hpp
 *
 * \brief The QL matrix decomposition.
 *
 * Given an \f$m\f$-by-\f$n\f$ matrix \f$A\f$, its QL-decomposition is a matrix
 * decomposition of the form:
 * \f[
 *   A=QL
 * \f]
 * where \f$L\f$ is an m-by-n lower trapezoidal (or, when \f$m \ge n\f$,
 * triangular) matrix and \f$Q\f$ is an m-by-m orthogonal (or unitary) matrix,
 * that is one satisfying:
 * \f[
 *  Q^{T}Q=I
 * \f]
  where \f$Q^{T}\f$ is the transpose of \f$Q\f$ and \f$I\f$ is the identity
 * matrix.
 *
 * For the special case of \f$m \ge n\f$, the factorization can be rewritten as:
 * \f[
 *  A=\begin{pmatrix}
 *     Q_1 & Q_2
 *     \end{pmatrix}
 *     \begin{pmatrix}
 *     L_1 \\
 *     L_2 \\
 *     \end{pmatrix}
 *   =\begin{pmatrix}
 *     Q_1 & Q_2
 *     \end{pmatrix}
 *     \begin{pmatrix}
 *     0 \\
 *     L_2 \\
 *     \end{pmatrix}
 *   = Q_2 L_2
 * \f] 
 * where \f$Q_1\f$ is an m-by-(m-n) matrix, \f$Q_2\f$ is an m-by-n matrix,
 * \f$L_1\f$ is an (m-n)-by-n zero matrix, and \f$L_2\f$ is an n-by-n lower
 * triangular matrix.
 *
 * The QL factorization is particular useful for computing minimum-phase
 * filters.
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_QL_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_QL_HPP


#include <algorithm>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/numeric/bindings/lapack/computational/geqlf.hpp>
#include <boost/numeric/bindings/lapack/computational/orgql.hpp>
#include <boost/numeric/bindings/lapack/computational/ormql.hpp>
#include <boost/numeric/bindings/lapack/computational/ungql.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/bindings/tag.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
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

struct ql_decomposition_impl_common;

/**
 * \brief Type-oriented operations for QL decomposition.
 *
 * \tparam IsComplex Logical parameter telling if the we are doing either a real
 *  or a complex QL decomposition.
 *
 * This class makes distinction between the real and the complex case.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <bool IsComplex>
struct ql_decomposition_impl;


/**
 * \brief Common operations for QL decomposition.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
struct ql_decomposition_impl_common
{
    /// Performan QL decomposition of the given input matrix \a A
    /// (row-major case).
    template <typename AMatrixT, typename TauVectorT>
        static void decompose(AMatrixT& A, TauVectorT& tau, row_major_tag)
    {
        matrix<typename matrix_traits<AMatrixT>::value_type, column_major> tmp_A(A);

        decompose(tmp_A, tau, column_major_tag());

        A = tmp_A;
    }


    /// Performan QL decomposition of the given input matrix \a A
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

        ::boost::numeric::bindings::lapack::geqlf(A, tau);
    }


    /// Extract the L matrix from a previously computing QL decomposition
    /// (row-major case).
    template <typename QLMatrixT, typename LMatrixT>
        static void extract_L(QLMatrixT const& QL, LMatrixT& L, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QLMatrixT>::value_type, column_major> tmp_QL(QL);
        matrix<typename matrix_traits<LMatrixT>::value_type, column_major> tmp_L(L);

        extract_L(tmp_QL, tmp_L, full, column_major_tag());

        L = tmp_L;
    }


    /**
     * \brief Extract the L matrix from a previously computing QL decomposition
     * (column-major case).
     *
     * Let QL be an m-by-n matrix, then the L matrix is built as:
     * - If m >= n, the lower triangle of the submatrix QL(m-n+1:m,1:n)
     *   contains the n-by-n lower triangular matrix L;
     * - if m <= n, the elements on and below the (n-m)-th
     *   superdiagonal contain the m-by-n lower trapezoidal matrix L.
     * .
     */
    template <typename QLMatrixT, typename LMatrixT>
        static void extract_L(QLMatrixT const& QL, LMatrixT& L, bool full, column_major_tag)
    {
        typedef typename matrix_traits<LMatrixT>::size_type size_type;
        typedef typename matrix_traits<LMatrixT>::value_type value_type;

        size_type m = num_rows(QL);
        size_type n = num_columns(QL);
        size_type nr = full ? m : ::std::min(m,n);

        if (num_rows(L) != nr && num_columns(L) != n)
        {
            L.resize(nr, n, false);
        }

        //::std::fill(L.data().begin(), L.data().end(), value_type/*zero*/());
        if (m >= n)
        {
            size_type k = m-n;
            size_type kr = k;

            // Set to zero the first m-n rows
            if (full)
            {
                subrange(L, 0, k, 0, n) = scalar_matrix<value_type>(k, n, value_type/*zero*/());
//              for (size_type row = 0; row < k; ++row)
//              {
//                  for(size_type col = 0; col < n; ++col)
//                  {
//                      L(row,col) = value_type/*zero*/();
//                  }
//              }
                kr = 0;
            }
            // the lower triangle of the submatrix QL(m-n+1:m,1:n) contains the
            // n-by-n lower triangular matrix L
            for (size_type row = k; row < m; ++row)
            {
                for (size_type col = 0; col < n; ++col)
                {
                    if (col <= (row-k))
                    {
                        L(row-kr,col) = QL(row,col);
                    }
                    else
                    {
                        L(row-kr,col) = value_type/*zero*/();
                    }
                }
            }
        }
        else
        {
             // the elements on and below the (n-m)-th
             // superdiagonal contain the m-by-n lower trapezoidal matrix L.
            size_type k = n-m;
            for (size_type row = 0; row < m; ++row)
            {
                for(size_type col = 0; col < n; ++col)
                {
                    if (col <= (row+k))
                    {
                        L(row,col) = QL(row,col);
                    }
                    else
                    {
                        L(row,col) = value_type/*zero*/();
                    }
                }
            }
        }
    }


    /**
     * \brief Multiply the given \a C matrix by the \c Q matrix obtained from
     *  the QL decomposition.
     *
     * \tparam QLMatrixT The type of the \a QL matrix.
     * \tparam TAUMatrixT The type of the \a tau vector.
     * \tparam CMatrixT The type of the \a C matrix.
     *
     * \param QL The matrix obtained by the QL decomposition such that the i-th
     *  column contains the vector which defines the elementary reflector
     *  \f$H(i)\f$, for \f$i = 1,2,\ldots,k\f$.
     * \param tau The vector obtained by the QL decomposition containing the
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
     * Let \c Q be the matrix obtained from the QL decomposition represented
     * by the \a QL matrix and the \a tau vector parameters. 
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
    template <typename QLMatrixT, typename TAUVectorT, typename CMatrixT>
        static void prod(QLMatrixT& QL, TAUVectorT const& tau, CMatrixT& C, bool left_Q, bool trans_Q, row_major_tag)
    {
        //NOTE: QL cannot be const since LAPACK::ORMQL modified it, restoring
        //      it at the end of the function.

        matrix<typename matrix_traits<QLMatrixT>::value_type, column_major> tmp_QL(QL);
        matrix<typename matrix_traits<CMatrixT>::value_type, column_major> tmp_C(C);

        prod(tmp_QL, tau, tmp_C, left_Q, trans_Q, column_major_tag());

        C = tmp_C;
    }


    /**
     * \brief Multiply the given \a C matrix by the \c Q matrix obtained from
     *  the QL decomposition.
     *
     * \tparam QLMatrixT The type of the \a QL matrix.
     * \tparam TAUMatrixT The type of the \a tau vector.
     * \tparam CMatrixT The type of the \a C matrix.
     *
     * \param QL The matrix obtained by the QL decomposition such that the i-th
     *  column contains the vector which defines the elementary reflector
     *  \f$H(i)\f$, for \f$i = 1,2,\ldots,k\f$.
     * \param tau The vector obtained by the QL decomposition containing the
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
     * Let \c Q be the matrix obtained from the QL decomposition represented
     * by the \a QL matrix and the \a tau vector parameters. 
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
    template <typename QLMatrixT, typename TAUVectorT, typename CMatrixT>
        static void prod(QLMatrixT& QL, TAUVectorT const& tau, CMatrixT& C, bool left_Q, bool trans_Q, column_major_tag /*orientation*/)
    {
        //NOTE: QL cannot be const since LAPACK::ORMQL modified it, restoring
        //      it at the end of the function.

//      typedef typename matrix_traits<QLMatrixT>::value_type value_type;
//      typedef typename matrix_traits<QLMatrixT>::size_type size_type;
//      typedef typename type_traits<value_type>::real_type real_type;
//
//      const ::fortran_int_t m = num_rows(C);
//      const ::fortran_int_t n = num_columns(C);
//      const ::fortran_int_t k = size(tau);
//      const ::fortran_int_t lda = num_rows(QL);
//      const ::fortran_int_t ldc = m;
//      real_type* work;
//      real_type opt_work_size;
//      ::fortran_int_t lwork;
//      ::std::ptrdiff_t info;

        if (left_Q)
        {
            if (trans_Q)
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormql has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormql(
                    ::boost::numeric::bindings::tag::left(),
                    ::boost::numeric::bindings::trans(QL),
                    tau,
                    C
                );
            }
            else
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormql has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::left(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormql(
                    ::boost::numeric::bindings::tag::left(),
                    QL,
                    tau,
                    C
                );
            }
        }
        else
        {
            if (trans_Q)
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormql has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormql(
                    ::boost::numeric::bindings::tag::right(),
                    ::boost::numeric::bindings::trans(QL),
                    tau,
                    C
                );
            }
            else
            {
//              //FIXME: actually (2010-08-13) bindinds::lapack::ormql has problems
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  &opt_work_size,
//                  -1
//              );
//              lwork = static_cast< ::fortran_int_t >(opt_work_size);
//              work = new real_type[lwork];
//              info = ::boost::numeric::bindings::lapack::detail::ormql(
//                  ::boost::numeric::bindings::tag::right(),
//                  ::boost::numeric::bindings::tag::no_transpose(),
//                  m,
//                  n,
//                  k,
//                  QL.data().begin(),
//                  lda,
//                  tau.data().begin(),
//                  C.data().begin(),
//                  ldc,
//                  work,
//                  lwork
//              );
//              delete[] work;
                ::boost::numeric::bindings::lapack::ormql(
                    ::boost::numeric::bindings::tag::right(),
                    QL,
                    tau,
                    C
                );
            }
        }
    }
};


/**
 * \brief QL decomposition operations for non-complex types.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <>
struct ql_decomposition_impl<false>: public ql_decomposition_impl_common
{
    /// Extract the Q matrix from a previously computing QL decomposition
    /// (row-major case).
    template <typename QLMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QLMatrixT const& QL, TauVectorT const& tau, QMatrixT& Q, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QLMatrixT>::value_type, column_major> tmp_QL(QL);
        matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);

        extract_Q(tmp_QL, tau, tmp_Q, full, column_major_tag());

        Q = tmp_Q;
    }


    /// Extract the Q matrix from a previously computing QL decomposition
    /// (column-major case).
    template <typename QLMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QLMatrixT const& QL, TauVectorT& tau, QMatrixT& Q, bool full, column_major_tag)
    {
        typedef typename matrix_traits<QMatrixT>::size_type size_type;
        typedef typename matrix_traits<QMatrixT>::value_type value_type;

        size_type m = num_rows(QL);
        size_type n = num_columns(QL);
        size_type nc = full ? m : ::std::min(m,n);

        if (num_rows(Q) != m || num_columns(Q) != nc)
        {
            Q.resize(m, nc, false);
        }

        if (m > n)
        {
            if (full)
            {
                subrange(Q, 0, m, 0, m-n) = scalar_matrix<value_type>(m, m-n, value_type/*zero*/());
                subrange(Q, 0, m, m-n, m) = QL;
            }
            else
            {
                Q = QL;
            }
        }
        else if (m < n)
        {
            subrange(Q, 0, m-1, 0, 1) = scalar_matrix<value_type>(m-1, 1, value_type/*zero*/());
            subrange(Q, m-1, m, 0, m) = scalar_matrix<value_type>(1, m, value_type/*zero*/());
            subrange(Q, 0, m-1, 1, m) = subrange(QL, 0, m-1, n-m+1, n);
        }
        else
        {
            Q = QL;
        }

        ::boost::numeric::bindings::lapack::orgql(Q, tau);
/*
        // Compute Q without LAPACK
        //
        // The matrix Q is represented as a product of elementary reflectors
        //   Q = H(k) . . . H(2) H(1), where k = min(m,n).
        // Each H(i) has the form
        //   H(i) = I - tau * v * v'
        // where tau is a real scalar, and v is a real vector with
        // v(m-k+i+1:m) = 0 and v(m-k+i) = 1; v(1:m-k+i-1) is stored on exit in
        // A(1:m-k+i-1,n-k+i), and tau in TAU(i).

        size_type k = std::min(m, n);

        if (num_rows(Q) != m || num_columns(Q) != m)
        {
            Q.resize(m, m, false);
        }

        identity_matrix<value_type> I(m);

        Q = I;
        for (size_type i = k-1; (i+1) > 0; --i)
        {
            // Build v = [ A(1:m-k+i-1,n-k+i) 1 0 ... 0 ]

            vector<value_type> v(m, value_type());
            for (size_type j = 0; j < m-k+i; ++j)
            {
                v(j) = QL(j,i);
            }
            v(m-k+i) = value_type(1);

            matrix<value_type, column_major> H = I - tau(i)*outer_prod(v, v);
            Q = prod(Q, H);
        }
*/
    }
};


/**
 * \brief QL decomposition operations for complex types.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <>
struct ql_decomposition_impl<true>: public ql_decomposition_impl_common
{
    /// Extract the Q matrix from a previously computing QL decomposition
    /// (row-major case).
    template <typename QLMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QLMatrixT const& QL, TauVectorT const& tau, QMatrixT& Q, bool full, row_major_tag)
    {
        matrix<typename matrix_traits<QLMatrixT>::value_type, column_major> tmp_QL(QL);
        matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);

        extract_Q(tmp_QL, tau, tmp_Q, full, column_major_tag());

        Q = tmp_Q;
    }


    /// Extract the Q matrix from a previously computing QL decomposition
    /// (column-major case).
    template <typename QLMatrixT, typename TauVectorT, typename QMatrixT>
        static void extract_Q(QLMatrixT const& QL, TauVectorT& tau, QMatrixT& Q, bool full, column_major_tag)
    {
        typedef typename matrix_traits<QMatrixT>::size_type size_type;
        typedef typename matrix_traits<QMatrixT>::value_type value_type;

        size_type m = num_rows(QL);
        size_type n = num_columns(QL);
        size_type nc = full ? m : std::min(m,n);

        if (num_rows(Q) != m || num_columns(Q) != nc)
        {
            Q.resize(m, nc, false);
        }

        if (m > n)
        {
            if (full)
            {
                subrange(Q, 0, m, 0, m-n) = scalar_matrix<value_type>(m, m-n, 0);
                subrange(Q, 0, m, m-n, m) = QL;
            }
            else
            {
                Q = QL;
            }
        }
        else if (m < n)
        {
            subrange(Q, 0, m-1, 0, 1) = scalar_matrix<value_type>(m-1, 1, 0);
            subrange(Q, m-1, m, 0, m) = scalar_matrix<value_type>(1, m, 0);
            subrange(Q, 0, m-1, 1, m) = subrange(QL, 0, m-1, n-m+1, n);
        }
        else
        {
            Q = QL;
        }

        ::boost::numeric::bindings::lapack::ungql(Q, tau);
/*
        // Compute Q without LAPACK
        //
        // The matrix Q is represented as a product of elementary reflectors
        //   Q = H(k) . . . H(2) H(1), where k = min(m,n).
        // Each H(i) has the form
        //   H(i) = I - tau * v * v'
        // where tau is a real scalar, and v is a real vector with
        // v(m-k+i+1:m) = 0 and v(m-k+i) = 1; v(1:m-k+i-1) is stored on exit in
        // A(1:m-k+i-1,n-k+i), and tau in TAU(i).

        size_type k = std::min(m, n);

        if (num_rows(Q) != m || num_columns(Q) != m)
        {
            Q.resize(m, m, false);
        }

        identity_matrix<value_type> I(m);

        Q = I;
        for (size_type i = k-1; (i+1) > 0; --i)
        {
            // Build v = [ A(1:m-k+i-1,n-k+i) 1 0 ... 0 ]

            vector<value_type> v(m, value_type());
            for (size_type j = 0; j < m-k+i; ++j)
            {
                v(j) = QL(j,i);
            }
            v(m-k+i) = value_type(1);

            matrix<value_type, column_major> H = I - tau(i)*outer_prod(v, v);
            Q = prod(Q, H);
        }
*/
    }
};


/// Free function performing the QL decomposition of the given matrix expression \a A.
template<typename MatrixExprT, typename QMatrixT, typename LMatrixT, typename OrientationT>
void ql_decompose_impl(matrix_expression<MatrixExprT> const& A, QMatrixT& Q, LMatrixT& L, bool full, OrientationT orientation)
{
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    matrix<value_type, typename layout_type<MatrixExprT>::type> tmp_QL(A);
    vector<value_type> tmp_tau;

    ql_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template decompose(tmp_QL, tmp_tau, orientation);


    ql_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template extract_Q(tmp_QL, tmp_tau, Q, full, orientation);


    ql_decomposition_impl<
            ::boost::is_complex<value_type>::value
        >::template extract_L(tmp_QL, L, full, orientation);
}

}} // Namespace detail::<unnamed>


/**
 * \brief QL decomposition.
 *
 * \tparam ValueT The type of the elements stored in the input matrices.
 *
 * \todo Currently, the type of the L matrix is a dense matrix.
 *  Can we use a better matrix structure?
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ValueT>
class ql_decomposition
{
    public: typedef ValueT value_type;
    private: typedef matrix<value_type, column_major> work_matrix_type;
    private: typedef vector<value_type> tau_vector_type;
    public: typedef work_matrix_type QL_matrix_type;
    public: typedef work_matrix_type Q_matrix_type;
    public: typedef work_matrix_type L_matrix_type;//TODO: Can I use a special matrix


    /// Default constructor.
    public: ql_decomposition()
    {
        // empty
    }


    /// Decompose the given matrix expression \a A.
    public: template <typename MatrixExprT>
        ql_decomposition(matrix_expression<MatrixExprT> const& A)
        : QL_(A)
    {
        decompose();
    }


    /// Decompose the given matrix expression \a A.
    public: template <typename MatrixExprT>
        void decompose(matrix_expression<MatrixExprT> const& A)
    {
        QL_ = A;

        decompose();
    }


    /**
     * \brief Extract the \c Q matrix.
     * \param full If \c false enables the economy-size mode whereby a
     *  reduced (rectangular) Q matrix is returned instead of full (square) one.
     * \return The \c Q matrix.
     *
     * The <em>economy-size</em> mode is useful when \f$m > n\f$ (where \f$m\f$
     * and \f$n\f$ are the number of rows and columns of the decomposed matrix
     * \f$A\f$).
     * As a matter of fact, in this case, the QL factorization can be viewed as:
     * \f[
     *   A = QL = \begin{pmatrix} Q_1 & Q_2 \end{pmatrix} \begin{pmatrix} 0 \\ L \end{pmatrix} = Q_2 L
     * \f]
     * where \f$Q_2\f$ is an m-by-n matrix containing the n trailing columns of
     * \f$Q\f$.
     */
    public: Q_matrix_type Q(bool full = true) const
    {
        Q_matrix_type tmp_Q;

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template extract_Q(QL_, tau_, tmp_Q, full, column_major_tag());

        return tmp_Q;
    }


    /**
     * \brief Extract the \c L matrix.
     * \param full If \c false enables the economy-size mode whereby a
     *  reduced \f$\min(m,n)\f$-by\f$n\f$ \c L matrix is returned instead of the
     *  full \f$m\f$-by-\f$n\f$ one.
     * \return The \c L matrix.
     *
     * The <em>economy-size</em> mode is useful when \f$m > n\f$ (where \f$m\f$
     * and \f$n\f$ are the number of rows and columns of the decomposed matrix
     * \f$A\f$).
     * As a matter of fact, in this case, the QL factorization can be viewed as:
     * \f[
     *   A = QL = \begin{pmatrix} Q_1 & Q_2 \end{pmatrix} \begin{pmatrix} 0 \\ L \end{pmatrix} = Q_2 L
     * \f]
     * where \f$Q_2\f$ is an m-by-n matrix containing the n trailing columns of
     * \f$Q\f$.
     */
    public: L_matrix_type L(bool full = true) const
    {
        L_matrix_type tmp_L;

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template extract_L(QL_, tmp_L, full, column_major_tag());

        return tmp_L;
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
        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template decompose(QL_, tau_, column_major_tag());
    }


    /// Perform the product \f$Q C\f$ and store the result in \a C (column-major
    /// case).
    private: template <typename CMatrixT>
        void lprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, C, true, false, column_major_tag());
    }


    /// Perform the product \f$Q C\f$ and store the result in \a C (row-major
    /// case).
    private: template <typename CMatrixT>
        void lprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, tmp_C, true, false, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$C Q\f$ and store the result in \a C (column-major
    /// case).
    private: template <typename CMatrixT>
        void rprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, C, false, false, column_major_tag());
    }


    /// Perform the product \f$C Q\f$ and store the result in \a C (row-major
    /// case).
    private: template <typename CMatrixT>
        void rprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, tmp_C, false, false, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$Q^T C\f$ and store the result in \a C
    /// (column-major case).
    private: template <typename CMatrixT>
        void tlprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, C, true, true, column_major_tag());
    }


    /// Perform the product \f$Q^T C\f$ and store the result in \a C
    /// (row-major case).
    private: template <typename CMatrixT>
        void tlprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, tmp_C, true, true, column_major_tag());

        C = tmp_C;
    }


    /// Perform the product \f$C Q^T\f$ and store the result in \a C
    /// (column-major case).
    private: template <typename CMatrixT>
        void trprod_inplace(CMatrixT& C, column_major_tag) const
    {
        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, C, false, true, column_major_tag());
    }


    /// Perform the product \f$C Q^T\f$ and store the result in \a C
    /// (row-major case).
    private: template <typename CMatrixT>
        void trprod_inplace(CMatrixT& C, row_major_tag) const
    {
        work_matrix_type tmp_C(C);

        detail::ql_decomposition_impl<
                ::boost::is_complex<value_type>::value
            >::template prod(QL_, tau_, tmp_C, false, true, column_major_tag());

        C = tmp_C;
    }


    // NOTE: the 'mutable' keyword is needed in order to make 'const' the
    //       '?prod' methods ('lprod', 'tlprod', 'rprod', 'trprod').
    //       Indeed, these methods call the respective '?prod_inplace' methods
    //       which, in turns, call the LAPACK::ORMQL function which temporarily
    //       changes the QL matrix (and restores it before returning).
    private: mutable QL_matrix_type QL_;
    private: tau_vector_type tau_;
};


/// Free function performing the QL decomposition of the given matrix expression \a A.
template<typename MatrixExprT, typename OutMatrix1T, typename OutMatrix2T>
BOOST_UBLAS_INLINE
void ql_decompose(matrix_expression<MatrixExprT> const& A, OutMatrix1T& Q, OutMatrix2T& L, bool full = true)
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

    detail::ql_decompose_impl(A, Q, L, full, orientation_category1());
}


/// Free function performing the QL decomposition of the given matrix expression \a A.
template<typename MatrixExprT>
BOOST_UBLAS_INLINE
ql_decomposition<typename matrix_traits<MatrixExprT>::value_type> ql_decompose(matrix_expression<MatrixExprT> const& A)
{
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    return ql_decomposition<value_type>(A);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_QL_HPP
