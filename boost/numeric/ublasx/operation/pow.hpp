/**
 * \file boost/numeric/ublasx/operation/pow.hpp
 *
 * \brief Apply the \c std::pow function to a vector or matrix expression.
 *
 * Copyright (c) 2015, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP


#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/inv.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <cmath>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

namespace detail {

template <typename MatrixExprT>
struct matrix_pow_traits
{
	typedef typename MatrixExprT::matrix_temporary_type result_type;
};

} // Namespace detail


/**
 * \brief Computes \a me to the power of \a p (me^p).
 *
 * If \a me is a square matrix and \a p is a positive integer, me^p effectively
 * multiplies \a me by itself p-1 times.
 * If \a me is square and nonsingular, me^(-p) effectively multiplies the
 * inverse of \a me by itself p-1 times.
 *
 * \note Fractional exponents are not currently supported.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 *
 * \param me The input matrix expression.
 * \param p The exponent.
 * \return The result of \a me to the power of \a p.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT, typename T>
BOOST_UBLAS_INLINE
typename detail::matrix_pow_traits<MatrixExprT>::result_type pow(matrix_expression<MatrixExprT> const& me, T p)
{
	typedef typename detail::matrix_pow_traits<MatrixExprT>::result_type result_type;

	result_type res;

	if (p > 0)
	{
		res = me;

		--p;
		while (p >= 1)
		{
			res = prod(res, me);
			--p;
		}
	}
	else if (p < 0)
	{
		result_type inv_me = inv(me);
		res = inv_me;
		p = -p;

		--p;
		while (p >= 1)
		{
			res = prod(res, inv_me);
			--p;
		}
	}
	else // p == 0
	{
		res = identity_matrix<typename matrix_traits<MatrixExprT>::value_type>(num_rows(me));
	}

	return res;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_POW_HPP
