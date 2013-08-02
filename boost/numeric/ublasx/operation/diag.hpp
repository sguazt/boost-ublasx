/**
 * \file boost/numeric/ublasx/operation/diag.hpp
 *
 * \brief The \c diag operation.
 *
 * The \c diag operation takes inspiration from the \e diag MATLAB's function
 * and the \e DiagonalMatrix Mathematica's function.
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLAS_DIAG_HPP
#define BOOST_NUMERIC_UBLAS_DIAG_HPP

#include <boost/numeric/ublasx/container/generalized_diagonal_matrix.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublasx/proxy/matrix_diagonal.hpp>
#include <boost/numeric/ublas/traits.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

/**
 * \brief A traits class for vector-to-diagonal-matrix transformation.
 * \tparam VectorT A model of VectorExpression.
 * \tparam Layout A matrix layout type.
 */
template <typename VectorT, typename LayoutT>
struct vector_matrix_diag_traits
{
	typedef typename vector_traits<VectorT>::value_type value_type;
	typedef typename vector_traits<VectorT>::difference_type difference_type;
	typedef typename vector_traits<VectorT>::size_type size_type;
	typedef LayoutT layout_type;
	typedef typename VectorT::array_type array_type; //FIXME: not in vector_traits
	typedef generalized_diagonal_matrix<value_type, layout_type, array_type> result_type;
};


/**
 * \brief A traits class for vector-to-diagonal-matrix transformation.
 * \tparam VectorExprT A model of VectorExpression.
 * \tparam Layout A matrix layout type.
 */
template <typename VectorExprT, typename LayoutT>
struct vector_matrix_diag_traits<vector_expression<VectorExprT>, LayoutT>
{
	typedef typename vector_traits<VectorExprT>::value_type value_type;
	typedef typename vector_traits<VectorExprT>::difference_type difference_type;
	typedef typename vector_traits<VectorExprT>::size_type size_type;
	typedef LayoutT layout_type;
	typedef typename VectorExprT::array_type array_type; //FIXME: not in vector_traits
	typedef generalized_diagonal_matrix<value_type, layout_type, array_type> result_type;
};


/**
 * \brief Create a square matrix of order \f$n+abs(k)\f$, with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \tparam LayoutT The layout type of the resulting matrix (e.g., row_major).
 * \param v A vector expression.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \param l The matrix layout.
 * \return A square diagonal matrix with the elements of \a v on the \a k-th
 *  diagonal.
 */
template <typename VectorT, typename LayoutT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT,LayoutT>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,LayoutT>::difference_type k=0, LayoutT /*l*/=LayoutT())
{
	typedef vector_matrix_diag_traits<VectorT,LayoutT> traits;
	typedef typename traits::size_type size_type;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	size_type d(k > 0 ? k : -k);

	return result_type(
			v().size() + d,
			k,
			array_type(v().data())
	);
}


/**
 * \brief Create a square matrix of order \f$n+abs(k)\f$, with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \tparam LayoutT The layout type of the resulting matrix (e.g., row_major).
 * \param v A vector expression.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \param l The matrix layout.
 * \return A square diagonal matrix with the elements of \a v on the \a k-th
 *  diagonal.
 *
 * Variant for const reference to vectors.
 */
template <typename VectorT, typename LayoutT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT const,LayoutT>::result_type diag(vector_expression<VectorT> const& v, typename vector_matrix_diag_traits<VectorT,LayoutT>::difference_type k=0, LayoutT /*l*/=LayoutT())
{
	typedef vector_matrix_diag_traits<VectorT const,LayoutT> traits;
	typedef typename traits::size_type size_type;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	size_type d(k > 0 ? k : -k);

	return result_type(
			v().size() + d,
			k,
			array_type(v().data())
	);
}


/**
 * \brief create a square matrix of order \f$n+abs(k)\f$, with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam vectort a model of vectorexpression.
 * \param v a vector expression.
 * \param k the offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return a square diagonal matrix with the elements of \a v on the \a k-th
 *  diagonal and with a row-major layout.
 */
template <typename VectorT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT,row_major>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,row_major>::difference_type k=0)
{
	typedef vector_matrix_diag_traits<VectorT,row_major> traits;
	typedef typename traits::size_type size_type;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	size_type d(k > 0 ? k : -k);

	return result_type(
			v().size() + d,
			k,
			array_type(v().data())
	);
}


/**
 * \brief create a square matrix of order \f$n+abs(k)\f$, with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam vectort a model of vectorexpression.
 * \param v a vector expression.
 * \param k the offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return a square diagonal matrix with the elements of \a v on the \a k-th
 *  diagonal and with a row-major layout.
 *
 * Variant for const reference to vectors.
 */
template <typename VectorT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT const,row_major>::result_type diag(vector_expression<VectorT> const& v, typename vector_matrix_diag_traits<VectorT,row_major>::difference_type k=0)
{
	typedef vector_matrix_diag_traits<VectorT const,row_major> traits;
	typedef typename traits::size_type size_type;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	size_type d(k > 0 ? k : -k);

	return result_type(
			v().size() + d,
			k,
			array_type(v().data())
	);
}


/**
 * \brief Create a \a size1 by \a size2 matrix with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \tparam LayoutT The layout type of the resulting matrix (e.g., row_major).
 * \param v A vector expression. If too long it will be truncated.
 * \param size1 The number of rows of the resulting matrix.
 * \param size2 The number of columns of the resulting matrix.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \param l The matrix layout.
 * \return A rectangular diagonal matrix with the elements of \a v on the
 *  \a k-th diagonal.
 */
template <typename VectorT, typename LayoutT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT,LayoutT>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,LayoutT>::size_type size1, typename vector_matrix_diag_traits<VectorT,LayoutT>::size_type size2, typename vector_matrix_diag_traits<VectorT,LayoutT>::difference_type k=0, LayoutT /*l*/=LayoutT())
{
	typedef vector_matrix_diag_traits<VectorT,row_major> traits;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	return result_type(
			size1,
			size2,
			k,
			array_type(v().data())
	);
}


/**
 * \brief Create a \a size1 by \a size2 matrix with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \tparam LayoutT The layout type of the resulting matrix (e.g., row_major).
 * \param v A vector expression. If too long it will be truncated.
 * \param size1 The number of rows of the resulting matrix.
 * \param size2 The number of columns of the resulting matrix.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \param l The matrix layout.
 * \return A rectangular diagonal matrix with the elements of \a v on the
 *  \a k-th diagonal.
 *
 * Variant for const reference to vectors.
 */
template <typename VectorT, typename LayoutT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT const,LayoutT>::result_type diag(vector_expression<VectorT> const& v, typename vector_matrix_diag_traits<VectorT,LayoutT>::size_type size1, typename vector_matrix_diag_traits<VectorT,LayoutT>::size_type size2, typename vector_matrix_diag_traits<VectorT,LayoutT>::difference_type k=0, LayoutT /*l*/=LayoutT())
{
	typedef vector_matrix_diag_traits<VectorT const,row_major> traits;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	return result_type(
			size1,
			size2,
			k,
			array_type(v().data())
	);
}


/**
 * \brief Create a \a size1 by \a size2 matrix with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \param v A vector expression. If too long it will be truncated.
 * \param size1 The number of rows of the resulting matrix.
 * \param size2 The number of columns of the resulting matrix.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return A rectangular diagonal matrix with the elements of \a v on the
 *  \a k-th diagonal and with a row-major layout.
 */
template <typename VectorT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT,row_major>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,row_major>::size_type size1, typename vector_matrix_diag_traits<VectorT,row_major>::size_type size2, typename vector_matrix_diag_traits<VectorT,row_major>::difference_type k=0)
{
	typedef vector_matrix_diag_traits<VectorT,row_major> traits;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	return result_type(
			size1,
			size2,
			k,
			array_type(v().data())
	);
}


/**
 * \brief Create a \a size1 by \a size2 matrix with the elements of
 *  \a v on the \a k-th diagonal.
 * \tparam VectorT A model of VectorExpression.
 * \param v A vector expression. If too long it will be truncated.
 * \param size1 The number of rows of the resulting matrix.
 * \param size2 The number of columns of the resulting matrix.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return A rectangular diagonal matrix with the elements of \a v on the
 *  \a k-th diagonal and with a row-major layout.
 *
 * Variant for const reference to vectors.
 */
template <typename VectorT>
BOOST_UBLAS_INLINE
typename vector_matrix_diag_traits<VectorT const,row_major>::result_type diag(vector_expression<VectorT> const& v, typename vector_matrix_diag_traits<VectorT,row_major>::size_type size1, typename vector_matrix_diag_traits<VectorT,row_major>::size_type size2, typename vector_matrix_diag_traits<VectorT,row_major>::difference_type k=0)
{
	typedef vector_matrix_diag_traits<VectorT const,row_major> traits;
	typedef typename traits::array_type array_type;
	typedef typename traits::result_type result_type;

	return result_type(
			size1,
			size2,
			k,
			array_type(v().data())
	);
}


//[FIXME]: The two functions below are commented for problems on overloading:
// the ambiguity is caused by difference_type, size_type and LayoutT.
///**
// * \brief Create a square matrix of order \f$n+abs(k)\f$, with the elements of
// *  \a v on the \a k-th diagonal.
// * \tparam VectorT A model of VectorExpression.
// * \tparam LayoutT The layout type of the resulting matrix (e.g., row_major).
// * \param v A vector expression. If longer than \a size it will be truncated.
// * \param size The size of the resulting matrix.
// * \param k The offset from the main diagonal:
// *  - \a k = 0 represents the main diagonal,
// *  - \a k > 0 is the offset above the main diagonal,
// *  - \a k < 0 is the offset below the main diagonal.
// * \param l The matrix layout.
// *  .
// * \return A square diagonal matrix with the elements of \a v on the \a k-th
// *  diagonal.
// */
//template <typename VectorT, typename LayoutT>
//BOOST_UBLAS_INLINE
//typename vector_matrix_diag_traits<VectorT,LayoutT>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,LayoutT>::size_type size, typename vector_matrix_diag_traits<VectorT,LayoutT>::difference_type k, LayoutT /*l*//*=LayoutT()*/)
//{
//	typedef vector_matrix_diag_traits<VectorT,LayoutT> traits;
//	typedef typename traits::size_type size_type;
//	typedef typename traits::array_type array_type;
//	typedef typename traits::result_type result_type;
//
//	return result_type(
//			size,
//			k,
//			array_type(v().data())
//	);
//}


///**
// * \brief Create a square matrix of order \f$n+abs(k)\f$, with the elements of
// *  \a v on the \a k-th diagonal.
// * \tparam VectorT A model of VectorExpression.
// * \param v A vector expression. If longer than \a size it will be truncated.
// * \param size The size of the resulting matrix.
// * \param k The offset from the main diagonal:
// *  - \a k = 0 represents the main diagonal,
// *  - \a k > 0 is the offset above the main diagonal,
// *  - \a k < 0 is the offset below the main diagonal.
// *  .
// * \return A square diagonal matrix with the elements of \a v on the \a k-th
// *  diagonal and with a row-major layout.
// */
//template <typename VectorT>
//BOOST_UBLAS_INLINE
//typename vector_matrix_diag_traits<VectorT,row_major>::result_type diag(vector_expression<VectorT>& v, typename vector_matrix_diag_traits<VectorT,row_major>::size_type size, typename vector_matrix_diag_traits<VectorT,row_major>::difference_type k)
//{
//	typedef vector_matrix_diag_traits<VectorT,row_major> traits;
//	typedef typename traits::size_type size_type;
//	typedef typename traits::array_type array_type;
//	typedef typename traits::result_type result_type;
//
//	return result_type(
//			size,
//			k,
//			array_type(v().data())
//	);
//}
//[/FIXME]


/**
 * \brief Create a view of the \a k-th diagonal of a matrix.
 * \tparam MatrixT A model of MatrixExpression.
 * \param me A matrix expression from which taking the \a k-th diagonal.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return A view of the \a k-th diagonal of matrix \a me.
 */
template<typename MatrixT>
BOOST_UBLAS_INLINE
matrix_diagonal<MatrixT> diag(matrix_expression<MatrixT>& me, typename MatrixT::difference_type k=0)
{
	return matrix_diagonal<MatrixT>(me(), k);
}


/**
 * \brief Create an unmutable view of the \a k-th diagonal of a matrix.
 * \tparam MatrixT A model of MatrixExpression.
 * \param me A matrix expression from which taking the \a k-th diagonal.
 * \param k The offset from the main diagonal:
 *  - \a k = 0 represents the main diagonal,
 *  - \a k > 0 is the offset above the main diagonal,
 *  - \a k < 0 is the offset below the main diagonal.
 *  .
 *  Default to zero.
 * \return An unmutable view of the \a k-th diagonal of matrix \a me.
 */
template<typename MatrixT>
BOOST_UBLAS_INLINE
matrix_diagonal<MatrixT const> const diag(matrix_expression<MatrixT> const& me, typename MatrixT::difference_type k=0)
{
	return matrix_diagonal<MatrixT const>(me(), k);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_DIAG_HPP
