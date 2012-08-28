/**
 * \file boost/numeric/ublasx/operation/balance.hpp
 *
 * \brief Balance a matrix or a pair of matrices to improve eigenvalue accuracy.
 *
 * <hr/>
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_BALANCE_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_BALANCE_HPP


#include <boost/mpl/assert.hpp>
#include <boost/numeric/bindings/lapack/computational/gebak.hpp>
#include <boost/numeric/bindings/lapack/computational/gebal.hpp>
#include <boost/numeric/bindings/lapack/computational/ggbak.hpp>
#include <boost/numeric/bindings/lapack/computational/ggbal.hpp>
#include <boost/numeric/bindings/tag.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/type_traits/is_same.hpp>


//TODO: implement overloaded functions for matrices with special structure (e.g., symmetric matrices).


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {

/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 * \tparam SVectorT The type of the scaling vector.
 * \tparam PVectorT The type of the permuting vector.
 * \tparam BVectorT The type of the balancing matrix.
 *
 * \param A The matrix to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \param want_scaling_vec Tells if the caller has requested to compute the
 *  scaling vector.
 * \param scaling_vec The computed scaling vector.
 * \param want_permuting_vec Tells if the caller has requested to compute the
 *  permutation vector.
 * \param permuting_vec The computed permuting vector.
 * \param want_balancing_mat Tells if the caller has requested to compute the
 *  balancing matrix.
 * \param balancing_mat The computed balancing matrix.
 * \return none, but \a A, \a scaling_vec, \a permuting_vec, and
 *  \a balancing_mat are changed (possibly, if requested).
 *
 * Version for matrices with column-major layout.
 */
template <
	typename MatrixT,
	typename SVectorT,
	typename PVectorT,
	typename BMatrixT
>
void balance_impl(MatrixT& A,
				  bool scale,
				  bool permute,
				  bool want_scaling_vec,
				  SVectorT& scaling_vec,
				  bool want_permuting_vec,
				  PVectorT& permuting_vec,
				  bool want_balancing_mat,
				  BMatrixT& balancing_mat,
				  column_major_tag)
{
	typedef typename matrix_traits<MatrixT>::value_type value_type;
	typedef typename matrix_traits<MatrixT>::size_type size_type;
	typedef typename type_traits<value_type>::real_type real_type;
	typedef vector<real_type> work_vector_type;

	// pre: A must be square
	BOOST_UBLAS_CHECK(
			ublasx::num_rows(A) == ublasx::num_columns(A),
			bad_size()
		);

	char job;

	if (scale && !permute)
	{
		// Only scale
		job = 'S';
	}
	else if (!scale && permute)
	{
		// Only permute
		job = 'P';
	}
	else if (!scale && !permute)
	{
		// Do nothing but simply set ILO = 1, IHI = N, SCALE(I) = 1.0 for i = 1,...,N;
		job = 'N';
	}
	else
	{
		// Both scale and permute
		job = 'B';
	}

	size_type n = num_rows(A);
	::fortran_int_t ilo;
	::fortran_int_t ihi;
	work_vector_type tmp_scale_vec(n);

	::boost::numeric::bindings::lapack::gebal(job,
											  A,
											  ilo,
											  ihi,
											  tmp_scale_vec);

	if (want_scaling_vec)
	{
		if (size(scaling_vec) != n)
		{
			scaling_vec.resize(n, false);
		}

		for (size_type i = 0; i < (static_cast<size_type>(ilo)-1); ++i)
		{
			scaling_vec(i) = real_type(1);
		}
		for (size_type i = ilo-1; i < static_cast<size_type>(ihi); ++i)
		{
			scaling_vec(i) = tmp_scale_vec(i);
		}
		for (size_type i = ihi; i < n; ++i)
		{
			scaling_vec(i) = real_type(1);
		}
	}
	if (want_permuting_vec)
	{
		if (size(permuting_vec) != n)
		{
			permuting_vec.resize(n, false);
		}

		for (size_type i = 0; i < n; ++i)
		{
			permuting_vec(i) = i;
		}
		for (size_type i = n-1; i >= static_cast<size_type>(ihi); --i)
		{
			size_type j(tmp_scale_vec(i)-1);
			::std::swap(permuting_vec(i), permuting_vec(j));
		}
		for (size_type i = 0; i < (static_cast<size_type>(ilo)-1); ++i)
		{
			size_type j(tmp_scale_vec(i)-1);
			::std::swap(permuting_vec(i), permuting_vec(j));
		}
	}
	if (want_balancing_mat)
	{
		balancing_mat = identity_matrix<real_type>(n,n);

		::boost::numeric::bindings::tag::right side;

		::boost::numeric::bindings::lapack::gebak(job,
												  side,
												  ilo,
												  ihi,
												  tmp_scale_vec,
												  balancing_mat);
	}
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 * \tparam SVectorT The type of the scaling vector.
 * \tparam PVectorT The type of the permuting vector.
 * \tparam BVectorT The type of the balancing matrix.
 *
 * \param A The matrix to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \param want_scaling_vec Tells if the caller has requested to compute the
 *  scaling vector.
 * \param scaling_vec The computed scaling vector.
 * \param want_permuting_vec Tells if the caller has requested to compute the
 *  permutation vector.
 * \param permuting_vec The computed permuting vector.
 * \param want_balancing_mat Tells if the caller has requested to compute the
 *  balancing matrix.
 * \param balancing_mat The computed balancing matrix.
 * \return none, but \a A, \a scaling_vec, \a permuting_vec, and
 *  \a balancing_mat are changed (possibly, if requested).
 *
 * Version for matrices with row-major layout.
 */
template <
	typename MatrixT,
	typename SVectorT,
	typename PVectorT,
	typename BMatrixT
>
void balance_impl(MatrixT& A,
				  bool scale,
				  bool permute,
				  bool want_scaling_vec,
				  SVectorT& scaling_vec,
				  bool want_permuting_vec,
				  PVectorT& permuting_vec,
				  bool want_balancing_mat,
				  BMatrixT& balancing_mat,
				  row_major_tag)
{
    // Note: LAPACK works with column-major matrices

    typedef typename matrix_traits<MatrixT>::value_type value_type;

    typedef matrix<value_type, column_major> colmaj_matrix_type;

	colmaj_matrix_type tmp_A(A);
	colmaj_matrix_type tmp_balancing_mat;

	balance_impl(tmp_A,
				 scale,
				 permute,
				 want_scaling_vec,
				 scaling_vec,
				 want_permuting_vec,
				 permuting_vec,
				 want_balancing_mat,
				 tmp_balancing_mat,
				 column_major_tag());

	A = tmp_A;

	if (want_balancing_mat)
	{
		balancing_mat = tmp_balancing_mat;
	}
}


/**
 * \brief Diagonal matrix balancing to improve generalized eigenvalue accuracy.
 *
 * \tparam Matrix1T The type of the first matrix in the input pencil to be
 *  balanced.
 * \tparam Matrix2T The type of the second matrix in the input pencil to be
 *  balanced.
 * \tparam SLVectorT The type of the scaling vector applied to the left side
 *  of the input pencil.
 * \tparam SRVectorT The type of the scaling vector applied to the right side
 *  of the input pencil.
 * \tparam PLVectorT The type of the permuting vector applied to the left side
 *  pf the input pencil.
 * \tparam PRVectorT The type of the permuting vector applied to the right side
 *  pf the input pencil.
 * \tparam BVectorT The type of the balancing matrix.
 *
 * \param A The first matrix in the input pencil (A,B) to be balanced.
 * \param B The second matrix in the input pencil (A,B) to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \param want_scaling_vec Tells if the caller has requested to compute the
 *  scaling vectors.
 * \param left_scaling_vec The computed scaling vector applied to the left side
 *  of the input pencil (A,B).
 * \param right_scaling_vec The computed scaling vector applied to the right
 *  side of the input pencil (A,B).
 * \param want_permuting_vec Tells if the caller has requested to compute the
 *  permutation vectors.
 * \param left_permuting_vec The computed permuting vector applied to the left
 *  size of the input pencil (A,B).
 * \param right_permuting_vec The computed permuting vector applied to the right
 *  size of the input pencil (A,B).
 * \param want_balancing_mat Tells if the caller has requested to compute the
 *  balancing matrix.
 * \param balancing_mat The computed balancing matrix.
 * \return none, but \a A, \a B, \a left_scaling_vec, \a right_scaling_vec,
 *  \a left_permuting_vec, \a right_permuting_vec, and \a balancing_mat are
 *  changed (possibly, if requested).
 *
 * Version for matrices with column-major layout.
 */
template <
	typename Matrix1T,
	typename Matrix2T,
	typename SLVectorT,
	typename PLVectorT,
	typename SRVectorT,
	typename PRVectorT,
	typename BMatrixT
>
void balance_impl(Matrix1T& A,
				  Matrix2T& B,
				  bool scale,
				  bool permute,
				  bool want_scaling_vec,
				  SLVectorT& left_scaling_vec,
				  SRVectorT& right_scaling_vec,
				  bool want_permuting_vec,
				  PLVectorT& left_permuting_vec,
				  PRVectorT& right_permuting_vec,
				  bool want_balancing_mat,
				  BMatrixT& balancing_mat,
				  column_major_tag)
{
	typedef typename promote_traits<
						typename matrix_traits<Matrix1T>::value_type,
						typename matrix_traits<Matrix2T>::value_type
				>::promote_type value_type;
	typedef typename promote_traits<
						typename matrix_traits<Matrix1T>::size_type,
						typename matrix_traits<Matrix2T>::size_type
				>::promote_type size_type;
	typedef typename type_traits<value_type>::real_type real_type;
	typedef vector<real_type> work_vector_type;

    // pre: same orientation category
	BOOST_MPL_ASSERT(
		(
			::boost::is_same<
					typename matrix_traits<Matrix1T>::orientation_category,
					typename matrix_traits<Matrix2T>::orientation_category
			>
		)
	);
	// pre: A must be square
	BOOST_UBLAS_CHECK(
			ublasx::num_rows(A) == ublasx::num_columns(A),
			bad_size()
		);
	// pre: B must be square
	BOOST_UBLAS_CHECK(
			ublasx::num_rows(B) == ublasx::num_columns(B),
			bad_size()
		);
	// pre: A and B must be of the same order
	BOOST_UBLAS_CHECK(
			ublasx::num_rows(A) == ublasx::num_rows(B),
			bad_size()
		);

	char job;

	if (scale && !permute)
	{
		// Only scale
		job = 'S';
	}
	else if (!scale && permute)
	{
		// Only permute
		job = 'P';
	}
	else if (!scale && !permute)
	{
		// Do nothing but simply set ILO = 1, IHI = N, LSCALE(I) = 1.0 and RSCALE(I) for i = 1,...,N;
		job = 'N';
	}
	else
	{
		// Both scale and permute
		job = 'B';
	}

	size_type n = num_rows(A);
	::fortran_int_t ilo;
	::fortran_int_t ihi;
	work_vector_type tmp_lscale_vec(n);
	work_vector_type tmp_rscale_vec(n);

	::boost::numeric::bindings::lapack::ggbal(job,
											  A,
											  ilo,
											  ihi,
											  tmp_lscale_vec,
											  tmp_rscale_vec);

	if (want_scaling_vec)
	{
		if (size(left_scaling_vec) != n)
		{
			left_scaling_vec.resize(n, false);
		}
		if (size(right_scaling_vec) != n)
		{
			right_scaling_vec.resize(n, false);
		}

		for (size_type i = 0; i < (static_cast<size_type>(ilo)-1); ++i)
		{
			left_scaling_vec(i) = right_scaling_vec(i)
								= real_type(1);
		}
		for (size_type i = ilo-1; i < static_cast<size_type>(ihi); ++i)
		{
			left_scaling_vec(i) = tmp_lscale_vec(i);
			right_scaling_vec(i) = tmp_rscale_vec(i);
		}
		for (size_type i = ihi; i < n; ++i)
		{
			left_scaling_vec(i) = right_scaling_vec(i)
								= real_type(1);
		}
	}
	if (want_permuting_vec)
	{
		if (size(left_permuting_vec) != n)
		{
			left_permuting_vec.resize(n, false);
		}
		if (size(right_permuting_vec) != n)
		{
			right_permuting_vec.resize(n, false);
		}

		for (size_type i = 0; i < n; ++i)
		{
			left_permuting_vec(i) = right_permuting_vec(i)
								  = i;
		}
		for (size_type i = n-1; i >= static_cast<size_type>(ihi); --i)
		{
			size_type j;
			j = tmp_lscale_vec(i)-1;
			::std::swap(left_permuting_vec(i), left_permuting_vec(j));
			j = tmp_rscale_vec(i)-1;
			::std::swap(right_permuting_vec(i), right_permuting_vec(j));
		}
		for (size_type i = 0; i < (static_cast<size_type>(ilo)-1); ++i)
		{
			size_type j;
			j = tmp_lscale_vec(i)-1;
			::std::swap(left_permuting_vec(i), left_permuting_vec(j));
			j = tmp_rscale_vec(i)-1;
			::std::swap(right_permuting_vec(i), right_permuting_vec(j));
		}
	}
	if (want_balancing_mat)
	{
		balancing_mat = identity_matrix<real_type>(n,n);

		::boost::numeric::bindings::tag::right side;

		::boost::numeric::bindings::lapack::ggbak(job,
												  side,
												  ilo,
												  ihi,
												  tmp_lscale_vec,
												  tmp_rscale_vec,
												  balancing_mat);
	}
}


/**
 * \brief Diagonal matrix balancing to improve generalized eigenvalue accuracy.
 *
 * \tparam Matrix1T The type of the first matrix in the input pencil to be
 *  balanced.
 * \tparam Matrix2T The type of the second matrix in the input pencil to be
 *  balanced.
 * \tparam SLVectorT The type of the scaling vector applied to the left side
 *  of the input pencil.
 * \tparam SRVectorT The type of the scaling vector applied to the right side
 *  of the input pencil.
 * \tparam PLVectorT The type of the permuting vector applied to the left side
 *  pf the input pencil.
 * \tparam PRVectorT The type of the permuting vector applied to the right side
 *  pf the input pencil.
 * \tparam BVectorT The type of the balancing matrix.
 *
 * \param A The first matrix in the input pencil (A,B) to be balanced.
 * \param B The second matrix in the input pencil (A,B) to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \param want_scaling_vec Tells if the caller has requested to compute the
 *  scaling vectors.
 * \param left_scaling_vec The computed scaling vector applied to the left side
 *  of the input pencil (A,B).
 * \param right_scaling_vec The computed scaling vector applied to the right
 *  side of the input pencil (A,B).
 * \param want_permuting_vec Tells if the caller has requested to compute the
 *  permutation vectors.
 * \param left_permuting_vec The computed permuting vector applied to the left
 *  size of the input pencil (A,B).
 * \param right_permuting_vec The computed permuting vector applied to the right
 *  size of the input pencil (A,B).
 * \param want_balancing_mat Tells if the caller has requested to compute the
 *  balancing matrix.
 * \param balancing_mat The computed balancing matrix.
 * \return none, but \a A, \a B, \a left_scaling_vec, \a right_scaling_vec,
 *  \a left_permuting_vec, \a right_permuting_vec, and \a balancing_mat are
 *  changed (possibly, if requested).
 *
 * Version for matrices with row-major layout.
 */
template <
	typename Matrix1T,
	typename Matrix2T,
	typename SLVectorT,
	typename PLVectorT,
	typename SRVectorT,
	typename PRVectorT,
	typename BMatrixT
>
void balance_impl(Matrix1T& A,
				  Matrix2T& B,
				  bool scale,
				  bool permute,
				  bool want_scaling_vec,
				  SLVectorT& left_scaling_vec,
				  SRVectorT& right_scaling_vec,
				  bool want_permuting_vec,
				  PLVectorT& left_permuting_vec,
				  PRVectorT& right_permuting_vec,
				  bool want_balancing_mat,
				  BMatrixT& balancing_mat,
				  row_major_tag)
{
    // Note: LAPACK works with column-major matrices

	typedef typename promote_traits<
						typename matrix_traits<Matrix1T>::value_type,
						typename matrix_traits<Matrix2T>::value_type
				>::promote_type value_type;

    typedef matrix<value_type, column_major> colmaj_matrix_type;

	colmaj_matrix_type tmp_A(A);
	colmaj_matrix_type tmp_B(B);
	colmaj_matrix_type tmp_balancing_mat;

	balance_impl(tmp_A,
				 tmp_B,
				 scale,
				 permute,
				 want_scaling_vec,
				 left_scaling_vec,
				 right_scaling_vec,
				 want_permuting_vec,
				 left_permuting_vec,
				 right_permuting_vec,
				 want_balancing_mat,
				 tmp_balancing_mat,
				 column_major_tag());

	A = tmp_A;
	B = tmp_B;

	if (want_balancing_mat)
	{
		balancing_mat = tmp_balancing_mat;
	}
}

}} // Namespace detail::<unnamed>


//@{ Balance of Single Matrix


/// Traits type class for the \c balance operation (single matrix version).
template <typename MatrixT>
struct balance_traits
{
	/// The type of the balanced matrix.
	typedef matrix<typename matrix_traits<MatrixT>::value_type,
				   typename layout_type<MatrixT>::type> balanced_matrix_type;
	/// The type of the balancing matrix.
	typedef matrix<typename matrix_traits<MatrixT>::value_type,
				   typename layout_type<MatrixT>::type> balancing_matrix_type;
	/// The type of the scaling vector.
	typedef vector<typename type_traits<
						typename matrix_traits<MatrixT>::value_type
					>::real_type> scaling_vector_type;
	/// The type of the permuting vector.
	typedef vector<typename matrix_traits<MatrixT>::size_type> permuting_vector_type;
};


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 *
 * \param A The matrix to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return none, but matrix \a A is overwritten by its balanced counterpart.
 */
template <typename MatrixT>
BOOST_UBLAS_INLINE
void balance_inplace(MatrixT& A, bool scale = true, bool permute = true)
{
	typedef typename matrix_traits<MatrixT>::orientation_category orientation_category;
//	typedef typename matrix_traits<MatrixT>::size_type size_type;
//	typedef typename matrix_traits<MatrixT>::value_type value_type;
//	typedef typename type_traits<value_type>::real_type real_type;

//	vector<real_type> dummy_scaling_vec;
//	vector<size_type> dummy_permuting_vec;
//	matrix<value_type,column_major> dummy_balancing_mat;
	typename balance_traits<MatrixT>::scaling_vector_type dummy_scaling_vec;
	typename balance_traits<MatrixT>::permuting_vector_type dummy_permuting_vec;
	typename balance_traits<MatrixT>::balancing_matrix_type dummy_balancing_mat;

	detail::balance_impl(A,
						 scale,
						 permute,
						 false,
						 dummy_scaling_vec,
						 false,
						 dummy_permuting_vec,
						 false,
						 dummy_balancing_mat,
						 orientation_category());
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 * \tparam MatrixExprT The type of the balancing matrix.
 *
 * \param A The matrix to be balanced.
 * \param balancing_mat The computed balancing matrix.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return none, but matrices \a A and \a balancing_mat are overwritten by the
 *  balanced counterpart of \a A and by the computed balanced matrix,
 *  respectively.
 */
template <typename MatrixT, typename MatrixExprT>
BOOST_UBLAS_INLINE
void balance_inplace(MatrixT& A,
					 matrix_container<MatrixExprT>& balancing_mat,
					 bool scale = true,
					 bool permute = true)
{
	typedef typename matrix_traits<MatrixT>::orientation_category orientation_category;
//	typedef typename matrix_traits<MatrixT>::size_type size_type;
//	typedef typename matrix_traits<MatrixT>::value_type value_type;
//	typedef typename type_traits<value_type>::real_type real_type;

//	vector<real_type> dummy_scaling_vec;
//	vector<size_type> dummy_permuting_vec;
	typename balance_traits<MatrixT>::scaling_vector_type dummy_scaling_vec;
	typename balance_traits<MatrixT>::permuting_vector_type dummy_permuting_vec;

	detail::balance_impl(A,
						 scale,
						 permute,
						 false,
						 dummy_scaling_vec,
						 false,
						 dummy_permuting_vec,
						 true,
						 balancing_mat(),
						 orientation_category());
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 * \tparam VectorExprT The type of the scaling vector.
 *
 * \param A The matrix to be balanced.
 * \param scaling_vec The computed scaling vector.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return none, but matrix \a A and vector \a scaling_vec are overwritten by
 *  the balanced counterpart of \a A and by the computed scaling vector,
 *  respectively.
 */
template <typename MatrixT, typename VectorExprT>
BOOST_UBLAS_INLINE
void balance_inplace(MatrixT& A,
					 vector_container<VectorExprT>& scaling_vec,
					 bool scale = true,
					 bool permute = true)
{
	typedef typename matrix_traits<MatrixT>::orientation_category orientation_category;
//	typedef typename matrix_traits<MatrixT>::size_type size_type;
//	typedef typename matrix_traits<MatrixT>::value_type value_type;

//	vector<size_type> dummy_permuting_vec;
//	matrix<value_type,column_major> dummy_balancing_mat;
	typename balance_traits<MatrixT>::permuting_vector_type dummy_permuting_vec;
	typename balance_traits<MatrixT>::balancing_matrix_type dummy_balancing_mat;

	detail::balance_impl(A,
						 scale,
						 permute,
						 true,
						 scaling_vec(),
						 false,
						 dummy_permuting_vec,
						 false,
						 dummy_balancing_mat,
						 orientation_category());
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 * \tparam SVectorExprT The type of the scaling vector.
 * \tparam PVectorExprT The type of the permuting vector.
 *
 * \param A The matrix to be balanced.
 * \param scaling_vec The computed scaling vector.
 * \param permuting_vec The computed permuting vector.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return none, but matrix \a A and vectors \a scaling_vec and \a permuting_vec
 *  are overwritten by the balanced counterpart of \a A, by the computed
 *  scaling vector, and by the computed permuting vector, respectively.
 */
template <typename MatrixT, typename SVectorExprT, typename PVectorExprT>
BOOST_UBLAS_INLINE
void balance_inplace(MatrixT& A,
					 vector_container<SVectorExprT>& scaling_vec,
					 vector_container<PVectorExprT>& permuting_vec,
					 bool scale = true,
					 bool permute = true)
{
	typedef typename matrix_traits<MatrixT>::orientation_category orientation_category;
//	typedef typename matrix_traits<MatrixT>::value_type value_type;

//	matrix<value_type,column_major> dummy_balancing_mat;
	typename balance_traits<MatrixT>::balancing_matrix_type dummy_balancing_mat;

	detail::balance_impl(A,
						 scale,
						 permute,
						 true,
						 scaling_vec(),
						 true,
						 permuting_vec(),
						 false,
						 dummy_balancing_mat,
						 orientation_category());
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 *
 * \param A The matrix to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return The balanced matrix.
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename balance_traits<MatrixExprT>::balanced_matrix_type balance(matrix_expression<MatrixExprT> const& A,
																   bool scale = true,
																   bool permute = true)
{
	//typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	//typedef typename layout_type<MatrixExprT>::type balanced_layout_type;
	//typedef matrix<value_type,balanced_layout_type> balanced_matrix_type;
	typedef typename balance_traits<MatrixExprT>::balanced_matrix_type balanced_matrix_type;

	balanced_matrix_type X(A);

	balance_inplace(X, scale, permute);

	return X;
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam AMatrixExprT The type of the matrix to be balanced.
 * \tparam BMatrixExprT The type of the balancing matrix.
 *
 * \param A The matrix to be balanced.
 * \param balancing_mat The balancing matrix.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return The balanced matrix; furthermore, the matrix \a balancing_mat is
 *  overwritten by the computed balancing matrix.
 */
template <typename AMatrixExprT, typename BMatrixExprT>
BOOST_UBLAS_INLINE
typename balance_traits<AMatrixExprT>::balanced_matrix_type balance(matrix_expression<AMatrixExprT> const& A,
																	matrix_container<BMatrixExprT>& balancing_mat,
																	bool scale = true,
																	bool permute = true)
{
	//typedef typename matrix_traits<AMatrixExprT>::value_type value_type;
	//typedef typename layout_type<AMatrixExprT>::type balanced_layout_type;
	//typedef matrix<value_type,balanced_layout_type> balanced_matrix_type;
	typedef typename balance_traits<AMatrixExprT>::balanced_matrix_type balanced_matrix_type;

	balanced_matrix_type X(A);

	balance_inplace(X, balancing_mat, scale, permute);

	return X;
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixExprT The type of the matrix to be balanced.
 * \tparam VectorExprT The type of the scaling vector.
 *
 * \param A The matrix to be balanced.
 * \param scaling_vec The scaling vector.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return The balanced matrix; furthermore, the vector \a scaling_vec is
 *  overwritten by the computed scaling vector.
 */
template <typename MatrixExprT, typename VectorExprT>
BOOST_UBLAS_INLINE
typename balance_traits<MatrixExprT>::balanced_matrix_type balance(matrix_expression<MatrixExprT> const& A,
																   vector_container<VectorExprT>& scaling_vec,
																   bool scale = true,
																   bool permute = true)
{
	//typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	//typedef typename layout_type<MatrixExprT>::type balanced_layout_type;
	//typedef matrix<value_type,balanced_layout_type> balanced_matrix_type;
	typedef typename balance_traits<MatrixExprT>::balanced_matrix_type balanced_matrix_type;

	balanced_matrix_type X(A);

	balance_inplace(X, scaling_vec, scale, permute);

	return X;
}


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixExprT The type of the matrix to be balanced.
 * \tparam SVectorExprT The type of the scaling vector.
 * \tparam PVectorExprT The type of the permuting vector.
 *
 * \param A The matrix to be balanced.
 * \param scaling_vec The scaling vector.
 * \param permuting_vec The permuting vector.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return The balanced matrix; furthermore, the vector \a scaling_vec is
 *  overwritten by the computed scaling vector.
 */
template <typename MatrixExprT, typename SVectorExprT, typename PVectorExprT>
BOOST_UBLAS_INLINE
typename balance_traits<MatrixExprT>::balanced_matrix_type balance(matrix_expression<MatrixExprT> const& A,
																   vector_container<SVectorExprT>& scaling_vec,
																   vector_container<PVectorExprT>& permuting_vec,
																   bool scale = true,
																   bool permute = true)
{
	typedef typename balance_traits<MatrixExprT>::balanced_matrix_type balanced_matrix_type;

	balanced_matrix_type X(A);

	balance_inplace(X, scaling_vec, permuting_vec, scale, permute);

	return X;
}


//@} Balance of Single Matrix


//@{ Balance of Matrix Pair


/// Traits type class for the \c balance operation (matrix pencil version).
template <typename Matrix1T, typename Matrix2T>
struct pair_balance_traits
{
	/// The type of the balanced matrix.
	typedef matrix<typename promote_traits<
							typename matrix_traits<Matrix1T>::value_type,
							typename matrix_traits<Matrix2T>::value_type
						>::promote_type,
				   typename layout_type<Matrix1T>::type
				> balanced_matrix_type;
	/// The type of the balancing matrix.
	typedef matrix<typename promote_traits<
							typename matrix_traits<Matrix1T>::value_type,
							typename matrix_traits<Matrix2T>::value_type
						>::promote_type,
				   typename layout_type<Matrix1T>::type
				> balancing_matrix_type;
	/// The type of the scaling vector.
	typedef vector<typename type_traits<
						typename promote_traits<
							typename matrix_traits<Matrix1T>::value_type,
							typename matrix_traits<Matrix2T>::value_type
						>::promote_type
					>::real_type> scaling_vector_type;
	/// The type of the permuting vector.
	typedef vector<typename promote_traits<
							typename matrix_traits<Matrix1T>::size_type,
							typename matrix_traits<Matrix2T>::size_type
						>::promote_type
				> permuting_vector_type;
};


/**
 * \brief Diagonal matrix balancing to improve eigenvalue accuracy.
 *
 * \tparam MatrixT The type of the matrix to be balanced.
 *
 * \param A The matrix to be balanced.
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return none, but matrix \a A is overwritten by its balanced counterpart.
 */
template <typename MatrixExpr1T, typename MatrixExpr2T>
BOOST_UBLAS_INLINE
void balance_inplace(matrix_container<MatrixExpr1T>& A, matrix_container<MatrixExpr2T>& B, bool scale = true, bool permute = true)
{
	typedef typename matrix_traits<MatrixExpr1T>::orientation_category orientation_category;
	typedef typename pair_balance_traits<MatrixExpr1T,MatrixExpr2T>::scaling_vector_type scaling_vector_type;
	typedef typename pair_balance_traits<MatrixExpr1T,MatrixExpr2T>::permuting_vector_type permuting_vector_type;
	typedef typename pair_balance_traits<MatrixExpr1T,MatrixExpr2T>::balancing_matrix_type balancing_matrix_type;

	scaling_vector_type dummy_left_scaling_vec;
	scaling_vector_type dummy_right_scaling_vec;
	permuting_vector_type dummy_left_permuting_vec;
	permuting_vector_type dummy_right_permuting_vec;
	balancing_matrix_type dummy_balancing_mat;

	detail::balance_impl(A,
						 B,
						 scale,
						 permute,
						 false,
						 dummy_left_scaling_vec,
						 dummy_right_scaling_vec,
						 false,
						 dummy_left_permuting_vec,
						 dummy_right_permuting_vec,
						 false,
						 dummy_balancing_mat,
						 orientation_category());
}


/**
 * \brief Diagonal matrix balancing to improve generalized eigenvalue accuracy.
 *
 * \tparam MatrixExpr1T The type of the first matrix of the input pencil to be
 *  balanced.
 * \tparam MatrixExpr2T The type of the second matrix of the input pencil to be
 *  balanced.
 * \tparam SLVectorExprT The type of the scaling vector applied to the left side
 *  of the input pencil.
 * \tparam SRVectorExprT The type of the scaling vector applied to the right
 *  side of the input pencil.
 * \tparam PLVectorExprT The type of the permuting vector applied to the left
 *  size of the input pencil.
 * \tparam PRVectorExprT The type of the permuting vector applied to the right
 *  size of the input pencil.
 *
 * \param A The first matrix of the input pencil (A,B) to be balanced.
 * \param B The second matrix of the input pencil (A,B) to be balanced.
 * \param left_scaling_vec The scaling vector applied to the left side of the
 *  input pencil (A,B).
 * \param right_scaling_vec The scaling vector applied to the right side of the
 *  input pencil (A,B).
 * \param left_permuting_vec The permuting vector applied to the left side of
 *  the input pencil (A,B)..
 * \param right_permuting_vec The permuting vector applied to the right side of
 *  the input pencil (A,B)..
 * \param scale Tells if scaling is to be applied.
 * \param permute Tells if permutation is to be applied.
 * \return The balanced matrix; furthermore, the vectors \a left_scaling_vec,
 *  \a right_scaling_vec, \a left_permuting_vec, and \a right_permuting_vec are
 *  overwritten by the computed scaling vectors and by the permuting scaling
 *  vectors, respectively.
 */
template <
	typename MatrixExpr1T,
	typename MatrixExpr2T,
	typename BMatrixExpr1T,
	typename BMatrixExpr2T,
	typename SLVectorExprT,
	typename SRVectorExprT,
	typename PLVectorExprT,
	typename PRVectorExprT
>
BOOST_UBLAS_INLINE
void balance(matrix_expression<MatrixExpr1T> const& A,
			 matrix_expression<MatrixExpr2T> const& B,
			 matrix_container<BMatrixExpr1T>& BA,
			 matrix_container<BMatrixExpr2T>& BB,
			 vector_container<SLVectorExprT>& left_scaling_vec,
			 vector_container<SRVectorExprT>& right_scaling_vec,
			 vector_container<PLVectorExprT>& left_permuting_vec,
			 vector_container<PRVectorExprT>& right_permuting_vec,
			 bool scale = true,
			 bool permute = true)
{
	typedef typename pair_balance_traits<MatrixExpr1T,MatrixExpr2T>::balanced_matrix_type balanced_matrix_type;

	balanced_matrix_type X(A);
	balanced_matrix_type Y(B);

	balance_inplace(X,
					Y,
					left_scaling_vec,
					right_scaling_vec,
					left_permuting_vec,
					right_permuting_vec,
					scale,
					permute);

	BA() = X;
	BB() = Y;
}


//@} Balance of Matrix Pair

}}} // Namespace boost:numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_BALANCE_HPP
