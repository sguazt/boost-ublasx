/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/rcond.hpp
 *
 * \brief Matrix reciprocal condition number estimate.
 *
 * The condition number of a regular (square) matrix is the product
 * of the \e norm of the matrix and the norm of its inverse (or
 * pseudo-inverse), and hence depends on the kind of matrix-norm.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_RCOND_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_RCOND_HPP


#include <algorithm>
#include <boost/numeric/bindings/lapack/auxiliary/lange.hpp>
#include <boost/numeric/bindings/lapack/computational/gbcon.hpp>
#include <boost/numeric/bindings/lapack/computational/gbtrf.hpp>
#include <boost/numeric/bindings/lapack/computational/gecon.hpp>
#include <boost/numeric/bindings/lapack/computational/getrf.hpp>
#include <boost/numeric/bindings/lapack/computational/hecon.hpp>
#include <boost/numeric/bindings/lapack/computational/hetrf.hpp>
#include <boost/numeric/bindings/lapack/computational/sycon.hpp>
#include <boost/numeric/bindings/lapack/computational/sytrf.hpp>
#include <boost/numeric/bindings/lapack/computational/trcon.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/qr.hpp>
#include <stdexcept>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail {

enum matrix_norm_category
{
    matrix_norm_1,
    matrix_norm_2,
    matrix_norm_frobenius,
    matrix_norm_inf
};


//template <
//  typename ValueT,
//  typename LayoutT,
//  typename StorageT
//>
//ValueT* band_mat_to_lapack_vec(banded_matrix<ValueT,LayoutT,StorageT> const& A, ::fortran_int_t& n)
//{
//  typedef banded_matrix<ValueT,LayoutT,StorageT> matrix_type;
//  typedef typename matrix_traits<matrix_type>::size_type size_type;
//
//  const size_type kl = A.lower();
//  const size_type ku = A.upper();
//  const size_type nr = num_rows(A);
//  const size_type nc = num_columns(A);
//
//  n = (2*kl+ku+1)*nc;
//
//  ValueT* v = new ValueT[n];
//
//  for (size_type r = 0; r < nr; ++r)
//  {
//      for (size_type c = 0; c < nc; ++c)
//      {
//          size_type rr = kl+ku+r-c;
//          size_type k = (2*kl+ku+1)*c+rr;
//
//          if ((r == c) || (c <= (ku+r) && r <= (c+kl)))
//          {
//              // In band
//              v[k] = A(r,c);
//          }
//          else
//          {
//              // Out of band
//              v[k] = ValueT();
//          }
//      }
//  }
//
//  return v;
//}

//template <typename T>
//void print_vector(std::string const& desc, T const* a, std::size_t n)
//{
//  std::cout << std::endl << desc << std::endl << "  [";
//  for(std::size_t i = 0; i < n; ++i)
//  {
//      std::cout << " " << std::fixed << a[i];
//  }
//  std::cout << "]" << std::endl;
//}



template <typename MatrixT>
typename type_traits<
    typename matrix_traits<MatrixT>::value_type
>::real_type rcond_impl(MatrixT const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef typename matrix_traits<MatrixT>::value_type value_type;
    typedef typename type_traits<value_type>::real_type result_type;
    typedef typename matrix_traits<MatrixT>::size_type size_type;
    typedef matrix<value_type, column_major> work_matrix_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);
    size_type k = ::std::min(nr,nc);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(false), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(false), norm_category, column_major_tag());
        }
    }

    char what_norm;
    result_type norm;
    result_type res;

    //FIXME: actually, in bindings this function is broken
//  switch (norm_category)
//  {
//      case matrix_norm_1:
//          what_norm = 'O';
//          break;
//      case matrix_norm_inf:
//          what_norm = 'I';
//          break;
////        case matrix_norm_frobenius:
////            what_norm = 'F';
////            break;
//      default:
//          throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
//  }

    // Compute the norm of A
    //FIXME: actually, in bindings this function is broken
//  ::boost::numeric::bindings::lapack::lange(
//      what_norm,
//      A
//  );
    switch (norm_category)
    {
        case matrix_norm_1:
            what_norm = 'O';
            norm = norm_1(A);
            break;
        case matrix_norm_inf:
            what_norm = 'I';
            norm = norm_inf(A);
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the LUP factorization of A
    work_matrix_type tmp_LU(A);
    vector< ::fortran_int_t > dummy_ipiv(k);
    ::boost::numeric::bindings::lapack::getrf(
        tmp_LU,
        dummy_ipiv
    );
    dummy_ipiv.resize(0, false); // free memory

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::gecon(
        what_norm,
        tmp_LU,
        norm,
        res
    );

    return res;
}


template <typename MatrixT>
typename type_traits<
    typename matrix_traits<MatrixT>::value_type
>::real_type rcond_impl(MatrixT const& A, matrix_norm_category norm_category, row_major_tag)
{
    typedef matrix<
                typename matrix_traits<MatrixT>::value_type,
                column_major
            > work_matrix_type;

    work_matrix_type tmp_A(A);

    return rcond_impl(tmp_A, norm_category, column_major_tag());
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(triangular_matrix<ValueT,TriangularT,column_major,StorageT> const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef triangular_matrix<ValueT,TriangularT,column_major,StorageT> matrix_type;
    typedef typename matrix_traits<matrix_type>::value_type value_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    typedef typename type_traits<value_type>::real_type result_type;
    typedef matrix<value_type,column_major> auxiliary_matrix_type;
    typedef triangular_adaptor<auxiliary_matrix_type, TriangularT> work_matrix_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(), norm_category, column_major_tag());
        }
    }


    char what_norm;
    result_type res;

    switch (norm_category)
    {
        case matrix_norm_1:
            what_norm = 'O';
            break;
        case matrix_norm_inf:
            what_norm = 'I';
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    auxiliary_matrix_type aux_A(A);
    work_matrix_type tmp_A(aux_A);

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::trcon(
        what_norm,
        tmp_A,
        res
    );

    return res;
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(triangular_matrix<ValueT,TriangularT,row_major,StorageT> const& A, matrix_norm_category norm_category, row_major_tag)
{
    typedef triangular_matrix<ValueT,TriangularT,column_major,StorageT> work_matrix_type;

    work_matrix_type tmp_A(A);

    return rcond_impl(tmp_A, norm_category, column_major_tag());
}


/*
template <
    typename ValueT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(banded_matrix<ValueT,column_major,StorageT> const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef banded_matrix<ValueT,column_major,StorageT> matrix_type;
    typedef typename matrix_traits<matrix_type>::value_type value_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    typedef typename type_traits<value_type>::real_type result_type;
    //typedef matrix<value_type,LayoutT> auxiliary_matrix_type;
    //typedef banded_adaptor<auxiliary_matrix_type> work_matrix_type;
    typedef banded_matrix<ValueT,row_major,StorageT> work_matrix_type;
    typedef vector< ::fortran_int_t > vector_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(), norm_category, column_major_tag());
        }
    }

    char what_norm;
    result_type norm;
    result_type res;
    size_type k = ::std::min(nr,nc);
    size_type kl = A.lower();
    size_type ku = A.upper();
//  size_type ldab = 2*kl+ku+1;

    switch (norm_category)
    {
        case matrix_norm_1:
            what_norm = 'O';
            break;
        case matrix_norm_inf:
            what_norm = 'I';
            break;
//      case matrix_norm_frobenius:
//          what_norm = 'F';
//          break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the norm of A
    //FIXME: actually, in bindings this function is broken
//  ::boost::numeric::bindings::lapack::lange(
//      what_norm,
//      A
//  );
    switch (norm_category)
    {
        case matrix_norm_1:
            norm = norm_1(A);
            break;
        case matrix_norm_inf:
            norm = norm_inf(A);
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the LUP factorization of A
//  ::fortran_int_t* aux_ab = band_mat_to_lapack_vec(A, n);
//  vector< ::fortran_int_t> tmp_ipiv(k);
//  ::std::ptrdiff_t info;
//  info = ::boost::numeric::bindings::lapack::detail::gbtrf(
//      nr,
//      nc,
//      kl,
//      ku,
//      aux_ab,
//      ldab,
//      tmp_ipiv.data().begin()
//  );
    work_matrix_type AB(A, kl, kl+ku); //NOTE: "kl+ku" is not a typo
    vector_type ipiv(k);
    ::boost::numeric::bindings::lapack::gbtrf(AB, ipiv);

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::gbcon(
        what_norm,
        AB,
        ipiv,
        norm,
        res
    );

    return res;
}
*/


template <
    typename ValueT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(banded_matrix<ValueT,column_major,StorageT> const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef banded_matrix<ValueT,row_major,StorageT> work_matrix_type;

    work_matrix_type tmp_A(A, A.lower(), A.upper());

    return rcond_impl(tmp_A, norm_category, row_major_tag());
}


template <
    typename ValueT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(banded_matrix<ValueT,row_major,StorageT> const& A, matrix_norm_category norm_category, row_major_tag)
{
    typedef banded_matrix<ValueT,row_major,StorageT> matrix_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(), norm_category, column_major_tag());
        }
    }

    typedef typename matrix_traits<matrix_type>::value_type value_type;

//[FIXME] (2020-12-30)
//  The function 'boost::numeric::bindings::lapack::gbtrf(AB, ipiv)' does not
//  work as I expected.
//  For such reason, the following has been temporarily commented in favor of
//  the one below where the banded matrix is first converted to a dense matrix,
//  which is then passed as parameter to the function that computer the rcond
//  on dense matrices.
//
#if 0
    typedef typename type_traits<value_type>::real_type result_type;
    typedef banded_matrix<ValueT,row_major,StorageT> work_matrix_type;
    typedef vector< ::fortran_int_t > vector_type;

    char what_norm;
    result_type norm;
    result_type res;
    size_type k = ::std::min(nr,nc);
    size_type kl = A.lower();
    size_type ku = A.upper();
//  size_type ldab = 2*kl+ku+1;

    switch (norm_category)
    {
        case matrix_norm_1:
            what_norm = 'O';
            break;
        case matrix_norm_inf:
            what_norm = 'I';
            break;
//      case matrix_norm_frobenius:
//          what_norm = 'F';
//          break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the norm of A
    //FIXME: actually, in bindings this function is broken
//  ::boost::numeric::bindings::lapack::lange(
//      what_norm,
//      A
//  );
    switch (norm_category)
    {
        case matrix_norm_1:
            norm = norm_1(A);
            break;
        case matrix_norm_inf:
            norm = norm_inf(A);
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the LUP factorization of A
//  ::fortran_int_t* aux_ab = band_mat_to_lapack_vec(A, n);
//  vector< ::fortran_int_t> tmp_ipiv(k);
//  ::std::ptrdiff_t info;
//  info = ::boost::numeric::bindings::lapack::detail::gbtrf(
//      nr,
//      nc,
//      kl,
//      ku,
//      aux_ab,
//      ldab,
//      tmp_ipiv.data().begin()
//  );
    work_matrix_type AB(A, kl, kl+ku); //NOTE: "kl+ku" is not a typo
    vector_type ipiv(k);
    ::boost::numeric::bindings::lapack::gbtrf(AB, ipiv);

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::gbcon(
        what_norm,
        AB,
        ipiv,
        norm,
        res
    );

    return res;
#else
    ublas::matrix<value_type> tmp_A(A);

    return rcond_impl(ublas::matrix<value_type>(A), matrix_norm_1, row_major_tag());
#endif
//[/FIXME]
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(symmetric_matrix<ValueT,TriangularT,column_major,StorageT> const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef symmetric_matrix<ValueT,TriangularT,column_major,StorageT> matrix_type;
    typedef typename matrix_traits<matrix_type>::value_type value_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    typedef typename type_traits<value_type>::real_type result_type;
    typedef matrix<ValueT,column_major> auxiliary_matrix_type;
    typedef symmetric_adaptor<auxiliary_matrix_type,TriangularT> work_matrix_type;
    typedef vector< ::fortran_int_t > vector_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);
    size_type n = ::std::min(nr, nc);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(), norm_category, column_major_tag());
        }
    }

    result_type norm;
    result_type res;

    // Compute the norm of A
    //FIXME: actually, in bindings this function is broken
//  ::boost::numeric::bindings::lapack::lange(
//      what_norm,
//      A
//  );
    switch (norm_category)
    {
        case matrix_norm_1:
            norm = norm_1(A);
            break;
        case matrix_norm_inf:
            norm = norm_inf(A);
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the LUP factorization of A
    //work_matrix_type AB(A);
    auxiliary_matrix_type aux_A(A);
    work_matrix_type AB(aux_A);
    vector_type ipiv(n);
    ::boost::numeric::bindings::lapack::sytrf(AB, ipiv);

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::sycon(
        AB,
        ipiv,
        norm,
        res
    );

    return res;
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(symmetric_matrix<ValueT,TriangularT,row_major,StorageT> const& A, matrix_norm_category norm_category, row_major_tag)
{
    typedef symmetric_matrix<ValueT,TriangularT,column_major,StorageT> work_matrix_type;

    work_matrix_type tmp_A(A);

    return rcond_impl(tmp_A, norm_category, column_major_tag());
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(hermitian_matrix<ValueT,TriangularT,column_major,StorageT> const& A, matrix_norm_category norm_category, column_major_tag)
{
    typedef hermitian_matrix<ValueT,TriangularT,column_major,StorageT> matrix_type;
    typedef typename matrix_traits<matrix_type>::value_type value_type;
    typedef typename matrix_traits<matrix_type>::size_type size_type;
    typedef typename type_traits<value_type>::real_type result_type;
    typedef matrix<ValueT,column_major> auxiliary_matrix_type;
    typedef hermitian_adaptor<auxiliary_matrix_type,TriangularT> work_matrix_type;
    typedef vector< ::fortran_int_t > vector_type;

    size_type nr = num_rows(A);
    size_type nc = num_columns(A);
    size_type n = ::std::min(nr, nc);

    // Check if A is a square matrix
    if (nr != nc)
    {
        // Non-square matrix -> Use QR decomposition
        if (nr < nc)
        {
            return rcond_impl(qr_decompose(trans(A)).R(), norm_category, column_major_tag());
        }
        else
        {
            return rcond_impl(qr_decompose(A).R(), norm_category, column_major_tag());
        }
    }

    result_type norm;
    result_type res;

    // Compute the norm of A
    //FIXME: actually, in bindings this function is broken
//  ::boost::numeric::bindings::lapack::lange(
//      what_norm,
//      A
//  );
    switch (norm_category)
    {
        case matrix_norm_1:
            norm = norm_1(A);
            break;
        case matrix_norm_inf:
            norm = norm_inf(A);
            break;
        default:
            throw std::runtime_error("[rcond::detail::rcond_impl] Unsupported norm category.");
    }

    // Compute the LUP factorization of A
    auxiliary_matrix_type aux_A(A);
    work_matrix_type AB(aux_A);
    vector_type ipiv(n);
    ::boost::numeric::bindings::lapack::hetrf(AB, ipiv);

    // Finally, compute the reciprocal condition number
    ::boost::numeric::bindings::lapack::hecon(
        AB,
        ipiv,
        norm,
        res
    );

    return res;
}


template <
    typename ValueT,
    typename TriangularT,
    typename StorageT
>
typename type_traits<ValueT>::real_type rcond_impl(hermitian_matrix<ValueT,TriangularT,row_major,StorageT> const& A, matrix_norm_category norm_category, row_major_tag)
{
    typedef hermitian_matrix<ValueT,TriangularT,column_major,StorageT> work_matrix_type;

    work_matrix_type tmp_A(A);

    return rcond_impl(tmp_A, norm_category, column_major_tag());
}

} // Namespace detail


/**
 * \brief Matrix reciprocal condition number estimate based on 1-norm.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param A The input \e square matrix expression.
 * \return The estimate of the reciprocal condition number of \a A.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
typename type_traits<
    typename matrix_traits<MatrixExprT>::value_type
>::real_type rcond(matrix_expression<MatrixExprT> const& A)
{
    typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;

    return detail::rcond_impl(A(), detail::matrix_norm_1, orientation_category());
}


//FIXME: Does we also need this?
///**
// * \brief Matrix reciprocal condition number estimate based on the matrix norm
// *  defined by parameter \a what_norm.
// *
// * \tparam MatrixExprT The type of the input matrix expression.
// * \tparam NormTagT The type of the matrix norm category.
// *
// * \param A The input \e square matrix expression.
// * \param what_norm The matrix norm category.
// * \return The estimate of the reciprocal condition number of \a A.
// *
// * \author Marco Guazzone, marco.guazzone@gmail.com
// */
//template <typename MatrixExprT, typename NormTagT>
//typename type_traits<
//  typename matrix_traits<MatrixExprT>::value_type
//>::real_type rcond(matrix_expression<MatrixExprT> const& A, NormTagT what_norm)
//{
//  typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;
//
//  return detail::rcond_impl(A(), detail::matrix_norm_1, orientation_category());
//}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_RCOND_HPP
