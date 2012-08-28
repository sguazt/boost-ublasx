/**
 * \file boost/numeric/ublasx/operation/svd.hpp
 *
 * \brief Singular Value Decomposition problem.
 *
 * The <em>singular value decomposition</em> (SVD) of a m-by-n real/complex
 * matrix \f$A\f$ is:
 * \f[
 *   A = U \Sigma V^{H}
 * \f]
 * where \f$\Sigma\f$ is an m-by-n matrix which is zero except for its
 * \f$\min(m,n)\f$ diagonal elements, \f$U\f$ is an m-by-m unitary matrix, and
 * \f$V\f$ is an n-by-n unitary matrix.
 * The diagonal elements of \f$\Sigma\f$ are the <em>singular values</em> of
 * \f$A\f$; they are real and non-negative, and are returned in descending order.
 * The first \f$\min(m,n)\f$ columns of \f$U\f$ and \f$V\f$ are the left and
 * right singular vectors of \f$A\f$.
 *
 * When an economy-size SVD is requested, if \f$k=\min(m.n)\f$, it results that
 * \f$\Sigma\f$ is a k-by-k diagonal matrix, \f$U\f$ is an m-by-k unitary matrix
 * and \f$V\f$ is an n-by-k unitary matrix.
 * In this case the original matrix \f$A\f$ cannot be reconstructed.
 *
 * \note
 *  For real m-by-n matrix \f$A\f$ the associated SVD is:
 * \f[
 *   A = U \Sigma V^{T}
 * \f]
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_SVD_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_SVD_HPP


#include <algorithm>
#include <boost/numeric/bindings/lapack/driver/gesvd.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/lapack.hpp>
#include <boost/numeric/ublasx/operation/diag.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/type_traits/is_complex.hpp>
#include <boost/utility/enable_if.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {

template <
	typename AMatrixT,
	typename SVectorT,
	typename UMatrixT,
	typename VTMatrixT
>
void svd_impl(AMatrixT const& A, SVectorT& s, bool want_U, bool full_U, UMatrixT& U, bool want_VT, bool full_VT, VTMatrixT& VT, column_major_tag)
{
	typedef typename matrix_traits<AMatrixT>::value_type value_type;
	typedef typename matrix_traits<AMatrixT>::size_type size_type;
	typedef matrix<value_type, column_major> work_matrix_type;

	char jobu = 'N';
	char jobvt = 'N';
	size_type m = num_rows(A);
	size_type n = num_columns(A);
	size_type k = ::std::min(m, n);
	size_type U_nr = detail::lapack::min_array_size;
	size_type U_nc = detail::lapack::min_array_size;
	size_type VT_nr = detail::lapack::min_array_size;
	size_type VT_nc = detail::lapack::min_array_size;

	if (want_U)
	{
		U_nr = m;
		if (full_U)
		{
			jobu = 'A';
			U_nc = m;
		}
		else
		{
			jobu = 'S';
			U_nc = k;
		}
	}
	if (want_VT)
	{
		VT_nc = n;
		if (full_VT)
		{
			jobvt = 'A';
			VT_nr = n;
		}
		else
		{
			jobvt = 'S';
			VT_nr = k;
		}
	}

	if (size(s) != k)
	{
		s.resize(k, false);
	}
	if (num_rows(U) != U_nr || num_columns(U) != U_nc)
	{
		U.resize(U_nr, U_nc, false);
	}
	if (num_rows(VT) != VT_nr || num_columns(VT) != VT_nc)
	{
		VT.resize(VT_nr, VT_nc, false);
	}

	work_matrix_type tmp_A(A);

	::boost::numeric::bindings::lapack::gesvd(
		jobu,
		jobvt,
		tmp_A,
		s,
		U,
		VT
	);
}


template <
	typename AMatrixT,
	typename SVectorT,
	typename UMatrixT,
	typename VTMatrixT
>
void svd_impl(AMatrixT const& A, SVectorT& s, bool want_U, bool full_U, UMatrixT& U, bool want_VT, bool full_VT, VTMatrixT& VT, row_major_tag)
{
	typedef typename matrix_traits<AMatrixT>::value_type value_type;
	typedef matrix<value_type, column_major> colmaj_matrix_type;

	colmaj_matrix_type tmp_A(A);
	colmaj_matrix_type tmp_U;
	colmaj_matrix_type tmp_VT;

	svd_impl(tmp_A, s, want_U, full_U, tmp_U, want_VT, full_VT, tmp_VT, column_major_tag());

	if (want_U)
	{
		U = tmp_U;
	}
	if (want_VT)
	{
		VT = tmp_VT;
	}
}


template <typename MatrixT>
typename ::boost::enable_if<
	::boost::is_complex<typename matrix_traits<MatrixT>::value_type>,
	MatrixT
>::type make_V(MatrixT const& VH)
{
	return herm(VH);
}


template <typename MatrixT>
typename ::boost::disable_if<
	::boost::is_complex<typename matrix_traits<MatrixT>::value_type>,
	MatrixT
>::type make_V(MatrixT const& VT)
{
	return trans(VT);
}

}} // Namespace detail::<unnamed>


/**
 * \brief Computes the singular value decomposition (SVD) of a matrix.
 *
 * Computes the <em>singular value decomposition</em> (SVD) of a m-by-n matrix,
 * optionally computing the left and/or right singular vectors.
 * The SVD is written as:
 * \f[
 *   A = U \Sigma V^{H}
 * \f]
 * where \f$\Sigma\f$ is an m-by-n matrix which is zero except for its
 * \f$\min(m,n)\f$ diagonal elements, \f$U\f$ is an m-by-m unitary matrix, and
 * \f$V\f$ is an n-by-n unitary matrix.
 * The diagonal elements of \f$\Sigma\f$ are the <em>singular values</em> of
 * \f$A\f$; they are real and non-negative, and are returned in descending order.
 * The first \f$\min(m,n)\f$ columns of \f$U\f$ and \f$V\f$ are the left and
 * right singular vectors of \f$A\f$.
 *
 * When <em>full mode</em> is disabled, an economy-size SVD is computed, such
 * that, if \f$k=\min(m.n)\f$, \f$\Sigma\f$ is a k-by-k diagonal matrix,
 * \f$U\f$ is an m-by-k unitary matrix and \f$V\f$ is an n-by-k unitary matrix.
 *
 * \author Marco Guazzone, marco.guazzoe@gmail.com
 */
template <typename ValueT>
class svd_decomposition
{
	public: typedef ValueT value_type;
	public: typedef typename type_traits<value_type>::real_type real_type;
	public: typedef vector<real_type> vector_type;
	public: typedef matrix<value_type, column_major> matrix_type;
	public: typedef matrix<real_type, column_major> real_matrix_type;
	private: typedef typename matrix_traits<matrix_type>::size_type size_type;


	/// Default constructor
	public: svd_decomposition()
		: full_(false),
		  m_(0),
		  n_(0),
		  k_(0)
	{
	}


	/// A constructor.
	public: template <typename MatrixExprT>
		svd_decomposition(matrix_expression<MatrixExprT> const& A, bool full = true)
	{
		decompose(A, full);
	}


	/// Compute the SVD \f$A=U \Sigma V^H\f$
	public: template <typename MatrixExprT>
		void decompose(matrix_expression<MatrixExprT> const& A, bool full = true)
	{
		// Cache some values (useful for later info retrieval)
		full_ = full;
		m_ = num_rows(A);
		n_ = num_columns(A);
		k_ = ::std::min(m_, n_);

		detail::svd_impl(A(), s_, true, full, U_, true, full, VH_, column_major_tag());
	}


	/// Return the U matrix of the SVD \f$U \Sigma V^H\f$.
	public: matrix_type const& U() const
	{
		return U_;
	}


	/// Return the \f$\operatorname{diag}(\Sigma)\f$ vector of the SVD \f$U \Sigma V^H\f$.
	public: vector_type const& s() const
	{
		return s_;
	}


	/// Return the \f$\Sigma\f$ matrix of the SVD \f$U \Sigma V^H\f$.
	public: real_matrix_type S() const
	{
		if (full_)
		{
			//return diag(s_, m_, n_);
			return diag<vector_type,column_major>(s_, m_, n_);
		}
		//return diag(s_, k_, k_);
		return diag<vector_type,column_major>(s_, k_, k_);
	}


	/// Return the \f$V^H\f$ matrix (\f$V^T\f$ for real types) of the SVD
	/// \f$U \Sigma V^H\f$.
	public: matrix_type const& VH() const
	{
		return VH_;
	}


	/// Return the \f$V\f$ matrix of the SVD \f$U \Sigma V^H\f$.
	public: matrix_type V() const
	{
		return detail::make_V(VH_);
	}


//	TODO: interesting but not urgent.
//	/**
//	 * \brief  Compute the approximate error bound for the computed singular
//	 *  values.
//	 * \note
//	 *  For the 2-norm, \f$\Sigma(1,1) = \operatorname{norm}(A)\f$.
//	 */
//	public: real_type s_error() const
//	{
//		return ::std::numeric_limits<real_type>::epsilon()*s_(0);
//	}


//	TODO: interesting but not urgent.
//	/**
//	 * \brief  Compute the approximate error bound for the computed singular
//	 *  values.
//	 * \note
//	 *  For the 2-norm, \f$\Sigma(1,1) = \operatorname{norm}(A)\f$.
//	 */
//	public: vector_type U_error() const
//	{
//		// Estimate reciprocal condition numbers for the singular vectors
//		vector_type rcond(n);
//		DDISNA('Left',M,N,s_,rcond)
//		// Compute the error estimates for the singular vectors
//		real_type s_err = s_error();
//		vector_type err(n);
//		for (size_type i = 0; i < n; ++i)
//		{
//			err(i) = s_err/rcond(i)
//		}
//
//		return err;
//	}


//	TODO: interesting but not urgent.
//	/**
//	 * \brief  Compute the approximate error bound for the computed singular
//	 *  values.
//	 * \note
//	 *  For the 2-norm, \f$\Sigma(1,1) = \operatorname{norm}(A)\f$.
//	 */
//	public: vector_type VH_error() const
//	{
//		// Estimate reciprocal condition numbers for the singular vectors
//		vector_type rcond(n);
//		DDISNA('Right',M,N,s_,rcond)
//		// Compute the error estimates for the singular vectors
//		real_type s_err = s_error();
//		vector_type err(n);
//		for (size_type i = 0; i < n; ++i)
//		{
//			err(i) = s_err/rcond(i)
//		}
//
//		return err;
//	}


	/// Tell if the current SVD is in full or economy mode.
	bool full_;
	/// The number of rows of the original decomposed matrix.
	size_type m_;
	/// The number of columns of the original decomposed matrix.
	size_type n_;
	/// The minimum between the number of rows and columns of the original
	/// decomposed matrix.
	size_type k_;
	/// The vector of singular values.
	private: vector_type s_;
	/// The matrix containing the left singular vectors.
	private: matrix_type U_;
	/// The matrix containing the right singular vectors.
	private: matrix_type VH_;
};


/// Compute the singular values of matrix \a A.
template <typename MatrixExprT>
vector<
	typename type_traits<
		typename matrix_traits<MatrixExprT>::value_type
	>::real_type
> svd_values(matrix_expression<MatrixExprT> const& A)
{
	typedef typename matrix_traits<MatrixExprT>::orientation_category orientation_category;
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;
	typedef typename type_traits<value_type>::real_type real_type;
	typedef vector<real_type> vector_type;
	typedef typename layout_type<MatrixExprT>::type layout_type;
	typedef matrix<value_type, layout_type> work_matrix_type;

	vector_type s;
	work_matrix_type dummy_U;
	work_matrix_type dummy_VT;

	detail::svd_impl(A(), s, false, false, dummy_U, false, false, dummy_VT, orientation_category());

	return s;
}


/// Compute the singular value decomposition of matrix \a A.
template <typename MatrixExprT>
svd_decomposition<typename matrix_traits<MatrixExprT>::value_type> svd_decompose(matrix_expression<MatrixExprT> const& A, bool full = true)
{
	typedef typename matrix_traits<MatrixExprT>::value_type value_type;

	return svd_decomposition<value_type>(A, full);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_SVD_HPP
