/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/lu.hpp
 *
 * \brief LU decomposition and solver.
 *
 * Computes an LU factorization of a general m-by-n matrix \f$A\f$ optionally
 * using partial pivoting with row interchanges.
 * The factorization has the form
 * \f[
 *  A = L U
 * \f]
 * or, if partial pivoting is used:
 * \f[
 *  A = P L U
 * \f]
 * where \f$P\f$ is a permutation matrix, \f$L\f$ is lower triangular with unit
 * diagonal elements (lower trapezoidal if \f$m > n\f$), and \f$U\f$ is upper
 * triangular (upper trapezoidal if \f$m < n\f$).
 * If matrix \f$A\f$ is rectangular \f$L and \f$P are square matrices each
 * having the same number of rows as \f$A\f$, while \f$U\f$ is exactly the same
 * shape as \f$A\f$.
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_LU_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_LU_HPP

//TODO: Use LAPACK functions in order to handle different types of matrices.
//TODO: How about full pivoting?
//TODO: Create a \c lu_decomposition class (e.g., \sa qr.hpp).


#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief LU decomposition without pivoting of the given matrix \a A.
 * \param A The matrix to be decomposed.
 * \return Zero if decomposition succeed; non-zero if decompositon fails (the
 *  value is 1 + the numer of the failing row). The input matrix \a A is
 *  modified in order to contain the L*U matrix.
 *
 * Perform LU decomposition of matrix \a A and replaces the strict lower
 * triangular part of \a m with the computed matrix L and the upper triangular
 * part of \a m is replaced by the computed matrix U.
 * For obtaining the single matrices L and U proceed as follows:
 * - for L: extract the strict lower-triangular part (i.e., without the main
 *   diagonal) from the computed matrix \a m and add to it the identity matrix
 *   of the same order of \a m;
 * - for U: extract the upper-triangular part (with the main diagonal) from the
 *   computer matrix \a m.
 */
template <typename MatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixT>::size_type lu_decompose_inplace(matrix_container<MatrixT>& A)
{
    return lu_factorize(A());
}


/**
 * \brief LU decomposition with partial pivoting of the matrix \a A.
 *
 * \param A The matrix to be decomposed.
 * \param P The permutation matrix reporting permutated rows of \a A after
 *  the decomposition.
 * \return Zero if decomposition succeed; non-zero if decompositon fails (the
 *  value is 1 + the numer of the failing row). The input matrix \a A is
 *  modified in order to contain the L*U matrix.
 *
 * Perform LUP decomposition of matrix \a A and replaces the strict lower
 * triangular part of \a A with the computed matrix L and the upper triangular
 * part of \a A is replaced by the computed matrix U.
 * For obtaining the single matrices L and U proceed as follows:
 * - for L: extract the strict lower-triangular part (i.e., without the main
 *   diagonal) from the computed matrix \a m and add to it the identity matrix
 *   of the same order of \a A;
 * - for U: extract the upper-triangular part (with the main diagonal) from the
 *   computer matrix \a A.
 * .
 */
template <typename MatrixT, typename PermutationMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixT>::size_type lu_decompose_inplace(matrix_container<MatrixT>& A, PermutationMatrixT& P)
{
    typedef typename matrix_traits<MatrixT>::size_type size_type;

    // Safety check: P is squared && size(P) == num_rows(A)
    //BOOST_UBLAS_CHECK( size(P) == num_columns(P) && size(P) == num_rows(A), bad_size() );
    size_type nr_A = num_rows(A);
    if (size(P) != nr_A)
    {
        P.resize(nr_A, false);
    }

    return lu_factorize(A(), P);
//
//  // postcondition: P is squared && size(P) == num_rows(A)
//  BOOST_UBLAS_CHECK( size(P) == num_columns(P) && size(P) == num_rows(A), bad_size() );
}


/**
 * \brief LU decomposition without pivoting of the matrix \a A.
 *
 * \tparam MatrixT The type of the matrix to be decomposed.
 * \param A The matrix to be decomposed.
 * \return Zero if decomposition succeed; non-zero if decompositon fails (the
 *  value is 1 + the numer of the failing row). The input matrix \a A is
 *  modified in order to contain the L*U matrix.
 *
 * Perform LU decomposition of matrix \a A and replaces the strict lower
 * triangular part of \a A with the computed matrix L and the upper triangular
 * part of \a A is replaced by the computed matrix U.
 * For obtaining the single matrices L and U proceed as follows:
 * - for L: extract the strict lower-triangular part (i.e., without the main
 *   diagonal) from the computed matrix \a A and add to it the identity matrix
 *   of the same order of \a A;
 * - for U: extract the upper-triangular part (with the main diagonal) from the
 *   computer matrix \a A.
 * .
 */
template <typename AMatrixExprT, typename LUMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixExprT>::size_type lu_decompose(matrix_expression<AMatrixExprT> const& A, matrix_container<LUMatrixT>& LU)
{
    LU = A;

    return lu_decompose_inplace(LU);
}


/**
 * \brief LU decomposition with partial pivoting of the matrix \a A.
 *
 * \pre Matrix \a A must be squared.
 * \tparam MatrixT The type of the matrix to be decomposed.
 * \param A The matrix to be decomposed.
 * \param P The permutation matrix reporting permutated rows of \a A after
 *  the decomposition.
 * \return The LU matrix.
 *
 * Perform LUP decomposition of matrix \a A and return the LU matrix.
 * For obtaining the single matrices L and U proceed as follows:
 * - for L: extract the strict lower-triangular part (i.e., without the main
 *   diagonal) from the computed matrix \c LU and add to it the identity matrix
 *   of the same order of \c LU;
 * - for U: extract the upper-triangular part (with the main diagonal) from the
 *   computer matrix \c LU.
 * .
 */
template <typename AMatrixExprT, typename PermutationMatrixT, typename LUMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixExprT>::size_type lu_decompose(matrix_expression<AMatrixExprT> const& A, PermutationMatrixT& P, matrix_container<LUMatrixT>& LU)
{
    typedef typename matrix_traits<AMatrixExprT>::size_type size_type;

    // Safety check: size(P) == num_rows(A)
    //BOOST_UBLAS_CHECK( size(P) == num_rows(A), bad_size() );
    size_type nr_A = num_rows(A);
    if (size(P) != nr_A)
    {
        P.resize(nr_A, false);
    }

    LU = A;

    return lu_decompose_inplace(LU, P);
}


/**
 * \brief Complete the LU forward/backward substitution for solving the system
 *  \f$LUx=b\f$.
 *
 * \pre The size of \a b must be the same as the number of rows of \a LU.
 * \tparam MatrixExprT The type of the matrix obtained by the LU decomposition.
 * \tparam VectorT The type of the constant terms vector.
 * \param LU A matrix representing an LU decomposition.
 * \param b The vector of coefficients.
 * \return Nothing. However, the vector \a b is replaced with the value of
 *  unknowns \f$x_i\f$ satisfying the system \f$LUx=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LU\f$, where \f$L\f$ is a lower-triangular matrix with ones on the
 * diagonal, and \f$U\f$ is an upper-triangular matrix). Solving the system
 * \f$Ax=b\f$ is then equivalent to solving two simpler systems \f$Ly=b\f$ and
 * \f$Ux=y\f$. Since \f$L\f$ is lower-triangular, the system \f$Ly=b\f$ can be
 * solved by forward substitution. Moreover, since \f$U\f$ is upper-triangular,
 * the system \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename MatrixExprT, typename VectorT>
BOOST_UBLAS_INLINE
void lu_apply_inplace(matrix_expression<MatrixExprT> const& LU, vector_container<VectorT>& b)
{
    // pre: size(b) == size(P)
    BOOST_UBLAS_CHECK( size(b) == num_rows(LU), bad_size() );

    lu_substitute(LU(), b());
}


/**
 * \brief Complete the LU forward/backward substitution for solving the system
 *  \f$LUX=B\f$.
 *
 * \pre The number of rows of \a LU and \a b must be the same.
 * \tparam LUMatrixExprT The type of the matrix obtained by the LU decomposition.
 * \tparam BMatrixT The type of the constant terms vector.
 * \param LU A matrix representing an LU decomposition.
 * \param B The matrix of coefficients.
 * \return Nothing. However, the matrix \a B is replaced with the value of
 *  unknowns \f$X_{ij}\f$ satisfying the system \f$LUX=B\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=B\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LU\f$, where \f$L\f$ is a lower-triangular matrix with ones on the
 * diagonal, and \f$U\f$ is an upper-triangular matrix). Solving the system
 * \f$AX=B\f$ is then equivalent to solving two simpler systems \f$LY=B\f$ and
 * \f$UX=Y\f$. Since \f$L\f$ is lower-triangular, the system \f$LY=B\f$ can be
 * solved by forward substitution. Moreover, since \f$U\f$ is upper-triangular,
 * the system \f$UX=Y\f$ can be solved by backward substitution.
 */
template <typename LUMatrixExprT, typename BMatrixT>
BOOST_UBLAS_INLINE
void lu_apply_inplace(matrix_expression<LUMatrixExprT> const& LU, matrix_container<BMatrixT>& B)
{
    // pre: num_rows(B) == num_rows(LU)
    BOOST_UBLAS_CHECK( num_rows(B) == num_rows(LU), bad_size() );

    lu_substitute(LU(), B());
}


/**
 * \brief Complete the LUP forward/backward substitution for solving the system
 *  \f$LU*X=P*b\f$.
 * 
 * \pre The size of \a P must be the same as the number of rows of \a LU.
 * \pre The size of \a b must be the same as the number of rows of \a LU.
 * \param LU A matrix representing an LU decomposition.
 * \param P A permutation matrix.
 * \param b The vector constant terms.
 * \return Nothing. However, the vector \a b is replaced with the value
 *  of unknowns \f$x_i\f$ satisfying the system \f$LU*x=P*b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of \f$A\f$ before so
 * that it can be decomposed). Solving the system \f$Ax=b\f$ is then equivalent
 * to solving two simpler systems \f$Ly=Pb\f$ and \f$Ux=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$Ly=Pb\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename LUMatrixExprT, typename PermutationMatrixT, typename BVectorT>
BOOST_UBLAS_INLINE
void lu_apply_inplace(matrix_expression<LUMatrixExprT> const& LU, PermutationMatrixT const& P, vector_container<BVectorT>& b)
{
    // precondition: size(P) == num_rows(LU)
    BOOST_UBLAS_CHECK( size(P) == num_rows(LU), bad_size() );
    // precondition: size(b) == num_rows(LU)
    BOOST_UBLAS_CHECK( size(b) == num_rows(LU), bad_size() );

    lu_substitute(LU(), P, b());
}


/**
 * \brief Complete the LUP forward/backward substitution for solving the system
 *  \f$LU*X=P*B\f$.
 * 
 * \pre The size of \a P must be the same as the number of rows of \a LU.
 * \pre The number of rows of \a LU and of \a B must be the same.
 * \param LU A matrix representing an LU decomposition.
 * \param P A permutation matrix.
 * \param B The matrix of constant terms.
 * \return Nothing. However, the matrix \a B is replaced with the value
 *  of unknowns \f$X_{ij}\f$ satisfying the system \f$LU*X=P*B\f$.
 *
 * An \f$n \times n\f$ linear system \f$AX=B\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of \f$A\f$ before so
 * that it can be decomposed). Solving the system \f$AX=B\f$ is then equivalent
 * to solving two simpler systems \f$LY=Pb\f$ and \f$UX=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$LY=PB\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$UX=Y\f$ can be solved by backward substitution.
 */
template <typename LUMatrixExprT, typename PermutationMatrixT, typename BMatrixT>
BOOST_UBLAS_INLINE
void lu_apply_inplace(matrix_expression<LUMatrixExprT> const& LU, PermutationMatrixT const& P, matrix_container<BMatrixT>& B)
{
    // pre: size(P) == num_rows(LU)
    BOOST_UBLAS_CHECK( size(P) == num_rows(LU), bad_size() );
    // pre: num_rows(b) == num_rows(LU)
    BOOST_UBLAS_CHECK( num_rows(B) == num_rows(LU), bad_size() );

    lu_substitute(LU(), P, B());
}


/**
 * \brief Complete the LU forward/backward substitution for solving the system
 *  \f$LUx=b\f$.
 *
 * \param LU A matrix representing an LU decomposition.
 * \param b The vector of coefficients.
 * \return The vector of unknowns \f$x\f$ satisfying the system \f$LUPx=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LU\f$, where \f$L\f$ is a lower-triangular matrix with ones on the
 * diagonal, and \f$U\f$ is an upper-triangular matrix). Solving the system
 * \f$Ax=b\f$ is then equivalent to solving two simpler systems \f$Ly=b\f$ and
 * \f$Ux=y\f$. Since \f$L\f$ is lower-triangular, the system \f$Ly=b\f$ can be
 * solved by forward substitution. Moreover, since \f$U\f$ is upper-triangular,
 * the system \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename LUMatrixExprT, typename BVectorExprT>
BOOST_UBLAS_INLINE
typename vector_temporary_traits<BVectorExprT>::type lu_apply(matrix_expression<LUMatrixExprT> const& LU, vector_expression<BVectorExprT> const& b)
{
    //  preconditions check delegated to lu_apply_inplace

    typedef typename vector_temporary_traits<BVectorExprT>::type out_vector_type;

    out_vector_type x(b);

    lu_apply_inplace(LU, x);

    return x;
}


/**
 * \brief Complete the LU forward/backward substitution for solving the system
 *  \f$LU*X=B\f$.
 *
 * \param LU A matrix representing an LU decomposition.
 * \param B The matrix of constant terms.
 * \return The matrix of unknowns \f$X\f$ satisfying the system \f$LU*X=B\f$.
 */
template <typename LUMatrixExprT, typename BMatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_temporary_traits<BMatrixExprT>::type lu_apply(matrix_expression<LUMatrixExprT> const& LU, matrix_expression<BMatrixExprT> const& B)
{
    //  preconditions check delegated to lu_apply_inplace

    typedef typename matrix_temporary_traits<BMatrixExprT>::type out_matrix_type;

    out_matrix_type X(B);

    lu_apply_inplace(LU, X);

    return X;
}


/**
 * \brief Complete the LUP forward/backward substitution for solving the system
 *  \f$LUx=b\f$.
 *
 * \param LU A matrix representing an LU decomposition.
 * \param P A permutation matrix.
 * \param b The vector of coefficients.
 * \return The vector of unknowns \f$x\f$ satisfying the system \f$LUPx=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of A before so that it
 * can be decomposed). Solving the system \f$Ax=b\f$ is then equivalent to
 * solving two simpler systems \f$Ly=Pb\f$ and \f$Ux=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$Ly=Pb\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename LUMatrixExprT, typename PermutationMatrixT, typename BVectorExprT>
BOOST_UBLAS_INLINE
typename vector_temporary_traits<BVectorExprT>::type lu_apply(matrix_expression<LUMatrixExprT> const& LU, PermutationMatrixT const& P, vector_expression<BVectorExprT> const& b)
{
    //  preconditions check delegated to lu_apply_inplace

    typedef typename vector_temporary_traits<BVectorExprT>::type out_vector_type;

    out_vector_type x(b);

    lu_apply_inplace(LU, P, x);

    return x;
}


/**
 * \brief Complete the LUP forward/backward substitution for solving the system
 *  \f$LUP*X=B\f$.
 *
 * \param LU A matrix representing an LU decomposition.
 * \param P A permutation matrix.
 * \param B The matrix of constant terms.
 * \return The matrix of unknowns \f$X\f$ satisfying the system \f$LUP*X=B\f$.
 */
template <typename LUMatrixExprT, typename PermutationMatrixT, typename BMatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_temporary_traits<BMatrixExprT>::type lu_apply(matrix_expression<LUMatrixExprT> const& LU, PermutationMatrixT const& P, matrix_expression<BMatrixExprT> const& B)
{
    //  preconditions check delegated to lu_apply_inplace

    typedef typename matrix_temporary_traits<BMatrixExprT>::type out_matrix_type;

    out_matrix_type X(B);

    lu_apply_inplace(LU, P, X);

    return X;
}


/**
 * \brief Solve the linear system \f$Ax=b\f$ by LUP decomposition.
 *
 * \pre The size of \a b must be the same as the number of rows of \a A.
 * \param A The input matrix representing the coefficients of the unknowns.
 * \param b The vector of coefficients.
 * \return A zero value if the system is solvable; a number greater than zero if
 *  it is not. Moreover, the vector \a b is replaced with the value of unknowns
 *  \f$x_i\f$ satisfying the system \f$Ax=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of \f$A\f$ before so
 * that it can be decomposed). Solving the system \f$Ax=b\f$ is then equivalent
 * to solving two simpler systems \f$Ly=Pb\f$ and \f$Ux=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$Ly=Pb\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename MatrixExprT, typename VectorT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::size_type lu_solve_inplace(matrix_expression<MatrixExprT> const& A, vector_container<VectorT>& b)
{
    // precondition: size(b) == num_rows(A)
    BOOST_UBLAS_CHECK( size(b) == num_rows(A), bad_size() );

    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename layout_type<MatrixExprT>::type layout_type;

    // Ax=b ==> LUx=b ==> Ly=b AND Ux=y

    matrix<value_type, layout_type> LU(A);
    permutation_matrix<size_type> P(num_rows(LU));

    size_type singular;
    singular = lu_decompose_inplace(LU, P);

    if (!singular)
    {
        lu_apply_inplace(LU, P, b());
    }

    return singular;
}


/**
 * \brief Solve the linear system \f$AX=B\f$ by LUP decomposition.
 *
 * \pre The number of rows of \a A and \a B must be the same.
 * \param A The input matrix representing the coefficients of the unknowns.
 * \param B The matrix of coefficients.
 * \return A zero value if the system is solvable; a number greater than zero if
 *  it is not. Moreover, the matrix \a B is replaced with the value of unknowns
 *  \f$X_{ij}\f$ satisfying the system \f$AX=B\f$.
 *
 * An \f$n \times n\f$ linear system \f$AX=B\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of \f$A\f$ before so
 * that it can be decomposed). Solving the system \f$AX=B\f$ is then equivalent
 * to solving two simpler systems \f$LY=PB\f$ and \f$UX=Y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$LY=PB\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$UX=Y\f$ can be solved by backward substitution.
 */
template <typename AMatrixExprT, typename BMatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixExprT>::size_type lu_solve_inplace(matrix_expression<AMatrixExprT> const& A, matrix_container<BMatrixExprT>& B)
{
    // pre: num_rows(B) == num_rows(A)
    BOOST_UBLAS_CHECK( num_rows(B) == num_rows(A), bad_size() );

    typedef typename matrix_traits<AMatrixExprT>::size_type size_type;
    typedef typename matrix_traits<AMatrixExprT>::value_type value_type;
    typedef typename layout_type<AMatrixExprT>::type layout_type;

    // Ax=B ==> LUx=B ==> Ly=B AND Ux=y

    matrix<value_type, layout_type> LU(A());
    permutation_matrix<size_type> P(num_rows(LU));

    size_type singular;
    singular = lu_decompose_inplace(LU, P);

    if (!singular)
    {
        lu_apply_inplace(LU, P, B());
    }

    return singular;
}


/**
 * \brief Solve the linear system \f$Ax=b\f$ by LUP decomposition.
 *
 * \pre The size of \a b must be the same as the number of rows of \a A.
 * \param A The input matrix representing the coefficients of the unknowns..
 * \param b The vector of coefficients.
 * \return The vector of unknowns \f$x\f$ satisfying the system \f$Ax=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of A before so that it
 * can be decomposed). Solving the system \f$Ax=b\f$ is then equivalent to
 * solving two simpler systems \f$Ly=Pb\f$ and \f$Ux=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$Ly=Pb\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename MatrixExprT, typename VectorExprT, typename OutVectorT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::size_type lu_solve(matrix_expression<MatrixExprT> const& A, vector_expression<VectorExprT> const& b, OutVectorT& x)
{
    // precondition: size(b) == num_rows(A)
    BOOST_UBLAS_CHECK( size(b) == num_rows(A), bad_size() );

    // Ax=b ==> LUx=b ==> Ly=b AND Ux=y

    typedef typename matrix_traits<MatrixExprT>::value_type size_type;
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename layout_type<MatrixExprT>::type layout_type;

    matrix<value_type,layout_type> LU(A);
    permutation_matrix<size_type> P(num_rows(LU));

    size_type singular;
    singular = lu_decompose_inplace(LU, P);

    if (!singular)
    {
        x = b;

        lu_apply_inplace(LU, P, x);
    }

    return singular;
}

/**
 * \brief Solve the linear system \f$Ax=b\f$ by LUP decomposition.
 *
 * \pre The size of \a b must be the same as the number of rows of \a A.
 * \param A The input matrix representing the coefficients of the unknowns..
 * \param b The vector of coefficients.
 * \return The vector of unknowns \f$x\f$ satisfying the system \f$Ax=b\f$.
 *
 * An \f$n \times n\f$ linear system \f$Ax=b\f$ can often be solved efficiently
 * by \f$LU\f$ decomposition (that is by decomposing matrix \f$A\f$ into a
 * product \f$LUP\f$, where \f$L\f$ is a lower-triangular matrix with ones on
 * the diagonal, \f$U\f$ is an upper-triangular matrix, and \f$P\f$ is a row
 * permutation matrix that is used to rearrange the rows of A before so that it
 * can be decomposed). Solving the system \f$Ax=b\f$ is then equivalent to
 * solving two simpler systems \f$Ly=Pb\f$ and \f$Ux=y\f$. Since \f$L\f$ is
 * lower-triangular, the system \f$Ly=Pb\f$ can be solved by forward
 * substitution. Moreover, since \f$U\f$ is upper-triangular, the system
 * \f$Ux=y\f$ can be solved by backward substitution.
 */
template <typename AMatrixExprT, typename BMatrixExprT, typename OutMatrixT>
BOOST_UBLAS_INLINE
typename matrix_traits<AMatrixExprT>::size_type lu_solve(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, OutMatrixT& X)
{
    // precondition: size(b) == num_rows(A)
    BOOST_UBLAS_CHECK( num_rows(B) == num_rows(A), bad_size() );

    // Ax=b ==> LUx=b ==> Ly=b AND Ux=y

    typedef typename matrix_traits<AMatrixExprT>::value_type size_type;
    typedef typename matrix_traits<AMatrixExprT>::value_type value_type;
    typedef typename layout_type<AMatrixExprT>::type layout_type;

    matrix<value_type,layout_type> LU(A);
    permutation_matrix<size_type> P(num_rows(LU));

    size_type singular;
    singular = lu_decompose_inplace(LU, P);

    if (!singular)
    {
        X = B;

        lu_apply_inplace(LU, P, X);
    }

    return singular;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_LU_HPP
