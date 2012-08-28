/**
 * \file boost/numeric/ublasx/operation/relational_ops.hpp
 *
 * \brief Relational operators defined over matrix and vector expressions.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_RELATIONAL_OPS_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_RELATIONAL_OPS_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>


/// Equality operator for vector expressions.
template <typename VectorExpr1T, typename VectorExpr2T>
BOOST_UBLAS_INLINE
bool operator==(::boost::numeric::ublas::vector_expression<VectorExpr1T> const& ve1,
				::boost::numeric::ublas::vector_expression<VectorExpr2T> const& ve2)
{
	typedef typename ::boost::numeric::ublas::promote_traits<
			typename ::boost::numeric::ublas::vector_traits<VectorExpr1T>::size_type,
			typename ::boost::numeric::ublas::vector_traits<VectorExpr2T>::size_type
		>::promote_type size_type;
	typedef typename ::boost::numeric::ublas::promote_traits<
			typename ::boost::numeric::ublas::vector_traits<VectorExpr1T>::value_type,
			typename ::boost::numeric::ublas::vector_traits<VectorExpr2T>::value_type
		>::promote_type value_type;

	size_type n1(::boost::numeric::ublasx::size(ve1));
	size_type n2(::boost::numeric::ublasx::size(ve2));

	if (n1 != n2)
	{
		return false;
	}


	for (size_type i = 0; i < n1; ++i)
	{
		if (static_cast<value_type>(ve1()(i)) != static_cast<value_type>(ve2()(i)))
		{
			return false;
		}
	}

	return true;
}


/// Inequality operator for vector expressions.
template <typename VectorExpr1T, typename VectorExpr2T>
BOOST_UBLAS_INLINE
bool operator!=(::boost::numeric::ublas::vector_expression<VectorExpr1T> const& ve1,
				::boost::numeric::ublas::vector_expression<VectorExpr2T> const& ve2)
{
	return !(ve1 == ve2);
}


/// Equality operator for matrix expressions.
template <typename MatrixExpr1T, typename MatrixExpr2T>
BOOST_UBLAS_INLINE
bool operator==(::boost::numeric::ublas::matrix_expression<MatrixExpr1T> const& me1,
				::boost::numeric::ublas::matrix_expression<MatrixExpr2T> const& me2)
{
	typedef typename ::boost::numeric::ublas::promote_traits<
			typename ::boost::numeric::ublas::matrix_traits<MatrixExpr1T>::size_type,
			typename ::boost::numeric::ublas::matrix_traits<MatrixExpr2T>::size_type
		>::promote_type size_type;
	typedef typename ::boost::numeric::ublas::promote_traits<
			typename ::boost::numeric::ublas::matrix_traits<MatrixExpr1T>::value_type,
			typename ::boost::numeric::ublas::matrix_traits<MatrixExpr2T>::value_type
		>::promote_type value_type;

	size_type nr1(::boost::numeric::ublasx::num_rows(me1));
	size_type nc1(::boost::numeric::ublasx::num_columns(me1));
	size_type nr2(::boost::numeric::ublasx::num_rows(me2));
	size_type nc2(::boost::numeric::ublasx::num_columns(me2));

	if (nr1 != nr2 || nc1 != nc2)
	{
		return false;
	}


	for (size_type r = 0; r < nr1; ++r)
	{
		for (size_type c = 0; c < nc1; ++c)
		{
			if (static_cast<value_type>(me1()(r,c)) != static_cast<value_type>(me2()(r,c)))
			{
				return false;
			}
		}
	}

	return true;
}


/// Inequality operator for matrix expressions.
template <typename MatrixExpr1T, typename MatrixExpr2T>
BOOST_UBLAS_INLINE
bool operator!=(::boost::numeric::ublas::matrix_expression<MatrixExpr1T> const& me1,
				::boost::numeric::ublas::matrix_expression<MatrixExpr2T> const& me2)
{
	return !(me1 == me2);
}


#endif // BOOST_NUMERIC_UBLASX_OPERATION_RELATIONAL_OPS_HPP
