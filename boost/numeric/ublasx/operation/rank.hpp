/**
 * \file boost/numeric/ublasx/operation/rank.hpp
 *
 * \brief Rank of a matrix.
 *
 * The rank of a matrix is the number of linearly independent rows or columns.
 *
 * The \c rank function provides an estimate of the number of linearly
 * independent rows or columns of a matrix.
 * There are a number of ways to compute the rank of a matrix.
 * The currently adopted method is  based on the singular value decomposition
 * (SVD) which is the most time consuming, but also the most reliable.
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_RANK_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_RANK_HPP


#include <algorithm>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/eps.hpp>
#include <boost/numeric/ublasx/operation/max.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/operation/svd.hpp>
#include <boost/numeric/ublasx/operation/which.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief Estimate the rank as the number of singular values of \a A that are
 *  greater than a given tolerance.
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam RealT The floating-point type of the tolerance.
 * \param A The input matrix expression.
 * \param tol The tolerance.
 * \return The number of singular values of \a A that are greater than \a tol.
 */
template <typename MatrixExprT, typename RealT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::size_type rank(matrix_expression<MatrixExprT> const& A, RealT tol)
{
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	typedef typename type_traits<value_type>::real_type real_type;

	vector<real_type> s = svd_values(A);
	return size(which(s, ::std::bind2nd(::std::greater<real_type>(), tol)));
}


/**
 * \brief Estimate the rank as the number of singular values of \a A that are
 *  greater than the default tolerance.
 * \tparam MatrixExprT The type of the input matrix expression.
 * \param A The input matrix expression.
 * \return The number of singular values of \a A that are greater than \a tol.
 *
 * The default tolerance is
 * \f[
 * 	 \max(n,m) \|A\|_2 {\epsilon}_m
 * \f]
 * where \f$n\f$ is the number of rows of \f$A\f$, \f$m\f$ is the number of
 * columns of \f$A\f$, and \f${\epsilon}_m\f$ is the floating-point machine
 * precision.
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::size_type rank(matrix_expression<MatrixExprT> const& A)
{
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	typedef typename type_traits<value_type>::real_type real_type;

	vector<real_type> s = svd_values(A);
	real_type tol = ::std::max(num_rows(A), num_columns(A))*eps(max(s)); // note: max(s) == norm_2(A)
	return size(which(s, ::std::bind2nd(::std::greater<real_type>(), tol)));
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_RANK_HPP
