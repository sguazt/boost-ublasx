/**
 * \file boost/numeric/ublasx/operation/inv.hpp
 *
 * \brief Matrix inverse.
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_INV_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_INV_HPP


#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/illcond.hpp>
#include <boost/numeric/ublasx/operation/lu.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace boost::numeric::ublas;

/**
 * \brief Matrix inversion of a square matrix.
 *
 * \return \c true if the given input matrix is invertible; \c false is the
 *  input matrix is (nearly) singular.
 */
template <typename MatrixT>
bool inv_inplace(MatrixT& A)
{
	typedef typename matrix_traits<MatrixT>::value_type value_type;
	typedef typename matrix_traits<MatrixT>::size_type size_type;

	//pre: A is square
	BOOST_UBLAS_CHECK(
		num_rows(A) == num_columns(A),
		bad_size()
	);

	// Compute the inverse X=A^{-1} as the solution of the linear system
	//  AX = I

	MatrixT X(identity_matrix<value_type>(num_rows(A)));

	size_type sing;
	sing = lu_solve_inplace(A, X);

	// Check if matrix is singular
	if (sing)
	{
		BOOST_UBLASX_DEBUG_TRACE("Warning: Matrix is (nearly) singular: cannot compute its inverse.");

		// Fill the matrix with Inf (like MATLAB does)
		A = scalar_matrix<value_type>(
				num_rows(A),
				num_columns(A),
				::std::numeric_limits<value_type>::infinity()
			);

		return false;
	}

	// Check if matrix is ill-conditioned
	if (illcond(A))
	{
		BOOST_UBLASX_DEBUG_TRACE("Warning: Matrix is close to singular or badly scaled.  Results may be inaccurate.");
		::std::clog << "[Warning] Matrix is close to singular or badly scaled.  Results may be inaccurate." << ::std::endl;
	}

	A = X;

	return true;
}

/**
 * \brief Matrix inversion of a square matrix.
 */
template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> inv(matrix_expression<MatrixExprT> const& A)
{
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	typedef matrix<value_type> out_matrix_type;

	out_matrix_type X(A);

	inv_inplace(X);

	return X;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_INV_HPP
