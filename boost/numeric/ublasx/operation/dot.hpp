/**
 * \file boost/numeric/ublasx/operation/dot.hpp
 *
 * \brief Vector dot product.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_DOT_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_DOT_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/operation/sum.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <typename VecExpr1T, typename VecExpr2T>
struct vdot_traits_type
{
	typedef typename vector_scalar_binary_traits<
					VecExpr1T,
					VecExpr2T,
					vector_inner_prod<
						VecExpr1T,
						VecExpr2T,
						typename promote_traits<
							typename vector_traits<VecExpr1T>::value_type,
							typename vector_traits<VecExpr2T>::value_type
						>::promote_type
					>
			>::result_type result_type;
};


template <typename MatExpr1T, typename MatExpr2T>
struct mdot_traits_type
{
	typedef vector<
				typename promote_traits<
					typename matrix_traits<MatExpr1T>::value_type,
					typename matrix_traits<MatExpr2T>::value_type
				>::promote_type
//				typename vector_traits<
//					typename matrix_binary_traits<
//							MatExpr1T,
//							MatExpr2T,
//							scalar_multiplies<
//									typename MatExpr1T::value_type,
//									typename MatExpr2T::value_type
//							>
//					>::result_type
//				>::value_type
			> result_type;
};


/**
 * \brief Scalar product of two vectors.
 *
 * \tparam VecExpr1T The type of the first vector.
 * \tparam VecExpr2T The type of the second vector.
 * \param v1 The first vector.
 * \param v2 The second vector.
 * \return The scalar product of vectors \a v1 and \a v2.
 *
 * The scalar product of two vectors \f$u\f$ and \f$v\f$ is defined as:
 * \f[
 *   \sum_{i} u_{i}v_{i}
 * \f]
 */
template <typename VecExpr1T, typename VecExpr2T>
BOOST_UBLAS_INLINE
typename vdot_traits_type<VecExpr1T,VecExpr2T>::result_type dot(vector_expression<VecExpr1T> const& v1,
																vector_expression<VecExpr2T> const& v2)
{
	return inner_prod(v1, v2);
}


/**
 * \brief Scalar product of two matrices along a given dimension.
 *
 * \tparam Dim The dimension used for computing the scalar product.
 * \tparam MatExpr1T The type of the first matrix.
 * \tparam MatExpr2T The type of the second matrix.
 * \param M1 The first matrix.
 * \param M2 The second matrix.
 * \return A vector representing the scalar product of vectors \a M1 and \a M2
 *  along dimension \a Dim.
 *
 * The scalar product of two matrices \f$A\f$ and \f$B\f$ along dimensino
 * \f$d\f$ is defined as:
 * - If \f$d=1\f$:
 *  \f[
 *    \sum_{j} A_{ij}B_{ij}, i=1,2,\ldots
 *  \f]
 * - If \f$d=2\f$:
 *  \f[
 *    \sum_{i} A_{ij}B_{ij}, j=1,2,\ldots
 *  \f]
 * .
 */
template <std::size_t Dim, typename MatExpr1T, typename MatExpr2T>
typename mdot_traits_type<MatExpr1T,MatExpr2T>::result_type dot(matrix_expression<MatExpr1T> const& M1,
																matrix_expression<MatExpr2T> const& M2)
{
	return sum<Dim>(element_prod(M1, M2));
}

}}} // Namespace boost::numeric::ublas


#endif // BOOST_NUMERIC_UBLASX_OPERATION_DOT_HPP
