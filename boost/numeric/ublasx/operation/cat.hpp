/**
 * \file boost/numeric/ublasx/operation/cat.hpp
 *
 * \brief Concatenate arrays along specified dimension.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \todo cat<1>, cat<2>, cat<tag::major>, ...
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLAS_OPERATION_EX_CAT_HPP
#define BOOST_NUMERIC_UBLAS_OPERATION_EX_CAT_HPP


#include <algorithm>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>


//TODO: add overloaded function to cat vectors and vector-matrix pairs

namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <
	typename M1,
	typename M2
>
struct matrix_cat_traits
{
	typedef typename promote_traits<
			typename matrix_traits<M1>::value_type,
			typename matrix_traits<M2>::value_type
		>::promote_type value_type;
	// Currently, the result type is simply a dense matrix since it is difficult
	// to know in advance whether a particular matrix structure will be
	// preserved after the 'cat' operation.
	// TODO: we might use the matrix_temporary_traits. However, how do we choose
	// between matrix_temporary_traits<M1> and matrix_temporary_traits<M2>?
	typedef matrix<
			value_type,
			typename layout_type<M1>::type
		> result_type;
};


/**
 * \brief Concatenate arrays along columns.
 *
 * \tparam InMatrixExpr1T The type of the first input matrix expression.
 * \tparam InMatrixExpr2T The type of the second input matrix expression.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \return A new matrix with \c max(num_rows(A),num_rows(B)) rows and
 *  \c num_columns(A)+num_columns(B) columns.
 *
 * For two input matrices A and B, append each column of B to its respective
 * column of A. 
 * If \a A and \a B have a different number of columns, the number of colums of
 * the resulting matrix will be the maximum between the number of columns of the
 * two input matrices and the elements of the matrix with the smaller number of
 * columns will be replaced with a zero value.
 *
 * Examples:
 * <pre>
 * A = [1 2 3;
 *      4 5 6;
 *      7 8 9]
 * B = [10 11;
 *      12 13]
 * C = cat_columns(A,B);
 * C == [ 1  2 3;
 *        4  5 6;
 *        7  8 9;
 *       10 11 0;
 *       12 13 0]
 * </pre>
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
	typename InMatrixExpr1T,
	typename InMatrixExpr2T
>
typename matrix_cat_traits<InMatrixExpr1T,InMatrixExpr2T>::result_type cat_columns(matrix_expression<InMatrixExpr1T> const& A, matrix_expression<InMatrixExpr2T> const& B)
{
	typedef typename matrix_cat_traits<InMatrixExpr1T,InMatrixExpr2T>::result_type out_matrix_type;
	typedef typename matrix_traits<out_matrix_type>::value_type value_type;
	typedef typename matrix_traits<out_matrix_type>::size_type size_type;

//	// precondition: num_columns(A) == num_columns(B)
//	BOOST_UBLAS_CHECK(
//		num_columns(A) == num_columns(B),
//		bad_argument()
//	);

	size_type A_nr = num_rows(A);
	size_type A_nc = num_columns(A);
	size_type B_nr = num_rows(B);
	size_type B_nc = num_columns(B);
	size_type nc = ::std::max(A_nc, B_nc);

	out_matrix_type X(A_nr+B_nr, nc, value_type());

	for (size_type c = 0; c < nc; ++c)
	{
		if (c < A_nc)
		{
			for (size_type r = 0; r < A_nr; ++r)
			{
				X(r,c) = A()(r,c);
			}
		}
		if (c < B_nc)
		{
			for (size_type r = 0; r < B_nr; ++r)
			{
				X(r+A_nr,c) = B()(r,c);
			}
		}
	}

	return X;
}


/**
 * \brief Concatenate arrays along rows.
 *
 * \tparam InMatrixExpr1T The type of the first input matrix expression.
 * \tparam InMatrixExpr2T The type of the second input matrix expression.
 *
 * \param A The first input matrix expression.
 * \param B The second input matrix expression.
 * \return A new matrix with \c num_rows(A)+num_rows(B) rows and
 *  \c max(num_columns(A),num_columns(B)) columns.
 *
 * For two input matrices A and B, append each column of B to its respective
 * column of A.
 * If \a A and \a B have a different number of columns, the number of rows of
 * the resulting matrix will be the maximum between the number of rows of the
 * two input matrices and the elements of the matrix with the smaller number of
 * columns will be replaced with a zero value.
 *
 * Examples:
 * <pre>
 * A = [1 2 3;
 *      4 5 6;
 *      7 8 9]
 * B = [10 11;
 *      12 13]
 * C = cat_rows(A,B);
 * C == [1 2 3 10 11;
 *       4 5 6 12 13;
 *       7 8 9  0  0];
 * </pre>
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
	typename InMatrixExpr1T,
	typename InMatrixExpr2T
>
typename matrix_cat_traits<InMatrixExpr1T,InMatrixExpr2T>::result_type cat_rows(matrix_expression<InMatrixExpr1T> const& A, matrix_expression<InMatrixExpr2T> const& B)
{
	typedef typename matrix_cat_traits<InMatrixExpr1T,InMatrixExpr2T>::result_type out_matrix_type;
	typedef typename matrix_traits<out_matrix_type>::value_type value_type;
	typedef typename matrix_traits<out_matrix_type>::size_type size_type;

	size_type A_nr = num_rows(A);
	size_type A_nc = num_columns(A);
	size_type B_nr = num_rows(B);
	size_type B_nc = num_columns(B);
	size_type nr = ::std::max(A_nr, B_nr);

	out_matrix_type X(nr, A_nc+B_nc, value_type());

	for (size_type r = 0; r < nr; ++r)
	{
		if (r < A_nr)
		{
			for (size_type c = 0; c < A_nc; ++c)
			{
				X(r,c) = A()(r,c);
			}
		}
		if (r < B_nr)
		{
			for (size_type c = 0; c < B_nc; ++c)
			{
				X(r,c+A_nc) = B()(r,c);
			}
		}
	}

	return X;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_CAT_HPP
