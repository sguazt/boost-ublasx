/**
 * \file boost/numeric/ublasx/operation/rep.hpp
 *
 * \brief Replicate and tile a matrix or a vector.
 *
 * This operation mimic the MATLAB \c repmat function.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_REP_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_REP_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cstddef>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> rep(matrix_expression<MatrixExprT> const& me, ::std::size_t nr, ::std::size_t nc)
{
	typedef matrix<typename matrix_traits<MatrixExprT>::value_type> result_matrix_type;
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;

	size_type nr_me(num_rows(me));
	size_type nc_me(num_columns(me));

	result_matrix_type res(nr_me*nr, nc_me*nc);

	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			subrange(res, r*nr_me, (r+1)*nr_me, c*nc_me, (c+1)*nc_me) = me;
		}
	}

	return res;
}


template <typename VectorExprT>
matrix<typename vector_traits<VectorExprT>::value_type> rep(vector_expression<VectorExprT> const& ve, ::std::size_t nr, ::std::size_t nc)
{
	typedef matrix<typename vector_traits<VectorExprT>::value_type> result_matrix_type;
	typedef typename vector_traits<VectorExprT>::size_type size_type;
	typedef matrix_vector_range<result_matrix_type> mat_vec_range_type;

	size_type n_ve(size(ve));

	result_matrix_type res(n_ve*nr, nc);

	for (size_type r = 0; r < nr; ++r)
	{
		for (size_type c = 0; c < nc; ++c)
		{
			for (size_type i = 0; i < n_ve; ++i)
			{
				res(r*n_ve+i, c) = ve()(i);
			}
		}
	}

	return res;
}


template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> rep(matrix_expression<MatrixExprT> const& me, ::std::size_t n)
{
	return rep(me, n, n);
}


template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> rep(vector_expression<MatrixExprT> const& ve, ::std::size_t n)
{
	return rep(ve, n, n);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_REP_HPP
