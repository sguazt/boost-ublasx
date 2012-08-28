/**
 * \file boost/numeric/ublasx/operation/triu.hpp
 *
 * \brief Upper triangular view of matrix.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail&gt;
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_TRIU_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_TRIU_HPP


#include <algorithm>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>


//TODO:
// - Actually the triu function does not create a view; instead it copies the
//   elements of the input matrix according to the requested triangular view.
//   We need to create a 'generalized_triangular_matrix' and
//   'generalized_triangular_adaptor' class containers.
//


namespace boost { namespace numeric { namespace ublasx {


using namespace ::boost::numeric::ublas;


template <typename MatrixT>
struct triu_traits
{
//	typedef generalized_triangular_adaptor<MatrixT,upper> triangular_matrix_type;
	typedef matrix<typename matrix_traits<MatrixT>::value_type,
				   typename layout_type<MatrixT>::type> triangular_matrix_type;
};


template <typename MatrixExprT>
typename triu_traits<MatrixExprT>::triangular_matrix_type triu(matrix_expression<MatrixExprT> const& A,
															   typename matrix_traits<MatrixExprT>::difference_type k=0)
{
	typedef typename triu_traits<MatrixExprT>::triangular_matrix_type triangular_matrix_type;
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	typedef typename matrix_traits<MatrixExprT>::size_type size_type;
	typedef typename matrix_traits<MatrixExprT>::difference_type difference_type;

//	return triangular_matrix_type(A());

	const difference_type zero(0);
	size_type nr(num_rows(A));
	size_type nc(num_columns(A));

	triangular_matrix_type X(nr, nc, value_type/*zero*/());

	size_type row_start(zero);
	size_type row_stop;

	if (k > zero)
	{
		row_stop = ::std::min(nr,nc-k);
	}
	else if (k < zero)
	{
		//row_stop = ::std::min(nr,nc);
		row_stop = nr;
	}
	else
	{
		row_stop = ::std::min(nr,nc);
	}

	for (size_type r = row_start; r < row_stop; ++r)
	{
		size_type col_start(zero);
		size_type col_stop(nc);

		if (k > zero)
		{
			col_start = r+::std::max(zero,k);
			//col_start = ::std::max(static_cast<difference_type>(r-k-1), zero);
			col_stop = nc;
		}
		else if (k < zero)
		{
			col_start = ::std::max(static_cast<difference_type>(r+k), zero);
			col_stop = nc;
		}
		else
		{
			col_start = r+k;
			col_stop = nc;
		}

		for (size_type c = col_start; c < col_stop; ++c)
		{
			X(r,c) = A()(r,c);
		}
	}

	return X;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_TRIU_HPP
