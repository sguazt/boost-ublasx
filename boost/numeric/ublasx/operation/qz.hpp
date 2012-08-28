/**
 * \file boost/numeric/ublasx/operation/qz.hpp
 *
 * \brief QZ factorization for generalized eigenvalues.
 *
 * Given two square matrices, \f$A\f$ and \f$B\f$, there exist two unitary
 * matrices, \f$Q\f$ and \f$Z\f$, such that \f$S=Q^{H}AZ\f$ and \f$T=Q^{H}BZ\f$
 * are both upper quasi-triangular matrices (they are triangular if \f$A\f$ and
 * \f$B\f$ are complex matrices).
 *
 * Equivalently, given two square matrices \f$A\f$ and \f$B\f$, there exist four
 * matrices, \f$S\f$, \f$T\f$, \f$Q\f$, and \f$Z\f$, such that \f$A=QSZ^{H}\f$
 * and \f$B=QTZ^{H}\f$.
 *
 * This decomposition is called the <em>generalized Schur decomposition</em> and
 * is also known as <em>QZ decomposition</em>.
 *
 * The pair of matrices \f$(A,B)\f$ is also referred to as <em>matrix
 * pencil</em> and the problem of finding the eigenvalues of a pencil is called
 * the <em>generalized eigenvalue problem</em>.
 * A pencil is called <em>regular</em> if there is at least one value of
 * \f$\lambda\f$ such that \f$\det(A-\lambda B) \ne 0\f$.
 * We call <em>eigenvalues</em> of a matrix pencil \f$(A,B)\f$ all complex
 * numbers \f$\lambda\f$ for which \f$\det(A-\lambda B) = 0\f$.
 * The set of the eigenvalues is called the <em>spectrum</em> of the pencil and
 * is written \f$\sigma(A,B)\f$; specifically:
 * \f[
 *  \sigma (A,B) = \left\{z \in \mathbb{C} : \det(A-zB)=0\right\}
 * \f]
 * If \f$\lambda \in \sigma (A,B)\f$ and
 * \f[
 *   Ax = \lambda Bx, \quad x \ne 0
 * \f]
 * then \f$x\f$ is referred to as an eigenvector of \f$A-\lambda B\f$.
 * Moreover, the pencil is said to have one or more eigenvalues at infinity if
 * \f$B\f$ has one or more \f$0\f$ eigenvalues.
 *
 * Although the decomposition is not unique, it yields the same generalized
 * eigenvalues that can be obtained by dividing the diagonal entries of \f$S\f$
 * by the corresponding diagonal entries of \f$T\f$.
 * Specifically, the generalized eigenvalues \f$\lambda\f$ that solve the
 * generalized eigenvalue problem  \f$A x = \lambda B x\f$ (where \f$x\f$ is an
 * unknown nonzero vector) can be calculated as the ratio of the diagonal
 * elements of \f$S\f$ to those of \f$T\f$. That is, using subscripts to denote
 * matrix elements, the \f$i\f$-th generalized eigenvalue \f$\lambda_i\f$
 * satisfies \f$\lambda_i = S_{ii}  / T_{ii}\f$.
 * The eigenvalues are finite when all the diagonal entries of \f$T\f$ are
 * nonzero.
 * By convention, the eigenvalues corresponding to zero diagonal entries of
 * \f$T\f$ are \f$\infty\f$.
 * If both \f$A\f$ and \f$B\f$ are real, the complex eigenvalues occur in
 * conjugate pairs.
 * In this case, \f$S\f$ is a real quasi-upper triangular matrix.
 * Each \f$2 \times 2\f$ block on the diagonal of \f$S\f$ corresponds to a
 * complex conjugate pair of eigenvalues, and the scalar diagonal entries
 * correspond to the real eigenvalues.
 * Such a decomposition is sometimes referred to as the <em>generalized real
 * Schur decomposition</em>.
 *
 * For more information see [1,2].
 *
 * References:
 * - [1] Anderson et al,
 *       <em>The LAPACK User Guide</em>,
 *       http://www.netlib.org/lapack/lug/node56.html
 * - [2] Golub et al,
 *       <em>Matrix Computations, 3rd ed.</em>,
 *       (Sec. 7.7), Johns Hopkins University Press, 1996
 * .
 *
 * \todo Optimize mem allocation similarly to eigen.hpp (where we used
 *  work_n_LV, out_n_LV, work_n_RV, and out_n_RV).
 *
 * <hr/>
 *
 * Copyright (c) 2010-2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_QZ_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_QZ_HPP


#include <boost/function.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/numeric/bindings/lapack/computational/tgsen.hpp>
#include <boost/numeric/bindings/lapack/computational/tgevc.hpp>
#include <boost/numeric/bindings/lapack/driver/gges.hpp>
#include <boost/numeric/bindings/tag.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublasx/detail/compiler.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
//#include <boost/numeric/ublasx/operation/balance.hpp>
//#include <boost/numeric/ublasx/operation/eigen.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/traits/layout_type.hpp>
#include <boost/type_traits/is_complex.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>


/// Cast the given function \a f to the QZ selector signature.
#define BOOST_UBLASX_DETAIL_QZ_SELECTOR_CAST(T,f) \
		( \
			::boost::is_complex<T>::value \
				? ( \
					::boost::is_same<float, typename type_traits<T>::real_type>::value \
						? reinterpret_cast< ::external_fp >(static_cast<detail::qz_complex_single_selector_type>(f)) \
						: reinterpret_cast< ::external_fp >(static_cast<detail::qz_complex_double_selector_type>(f)) \
				) \
				: ( \
					::boost::is_same<float, typename type_traits<T>::real_type>::value \
						? reinterpret_cast< ::external_fp >(static_cast<detail::qz_single_selector_type>(f)) \
						: reinterpret_cast< ::external_fp >(static_cast<detail::qz_double_selector_type>(f)) \
				) \
		)


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/// Eigenvalues selectors for QZ factorization.
enum qz_eigenvalues_selection
{
	all_qz_eigenvalues, ///< select all eigenvalues in the order of appearance (essentially, no ordering is performed).
	lhp_qz_eigenvalues, ///< Stable continuous-time space: select eigenvalues in the left-half plane (\f$\operatorname{real}(E) < 0\f$).
	rhp_qz_eigenvalues, ///< Unstable continuous-time space: select eigenvalues in the right-half plane (\f$\operatorname{real}(E) > 0\f$).
	udi_qz_eigenvalues, ///< Stable discrete-time space: select eigenvalues which are interior of unit disk (\f$\operatorname{abs}(E) < 1\f$).
	udo_qz_eigenvalues ///< Unstable discrete-time space: select eigenvalues which are exterior of unit disk (\f$\operatorname{abs}(E) > 1\f$).
};
	

namespace detail {

/**
 * \brief Select eigenvalues \f$\lambda\f$ in the left-half plane
 *  (\f$\operatorname{real}(\lambda) < 0\f$) [complex single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_lhp_eigenval_sel(::std::complex<float> a, ::std::complex<float> b)
{
	// Based on function ZB02OW.f from SLICOT 5.0
	return ::std::abs(b) != 0 && ::std::complex<float>(a/b).real() < 0;
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the left-half plane
 *  (\f$\operatorname{real}(\lambda) < 0\f$) [complex double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_lhp_eigenval_sel(::std::complex<double> a, ::std::complex<double> b)
{
	// Based on function ZB02OW.f from SLICOT 5.0
	return ::std::abs(b) != 0 && ::std::complex<double>(a/b).real() < 0;
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the left-half plane
 *  (\f$\operatorname{real}(\lambda) < 0\f$) [single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_lhp_eigenval_sel(float ar, float ai, float b)
{
	BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(ai);

	// Based on function SB02OW from SLICOT 5.0
	return ((ar > 0 && b < 0) || (ar < 0 && b > 0))
		   && ::std::abs(b) > (::std::abs(ar)*::std::numeric_limits<float>::epsilon());
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the left-half plane
 *  (\f$\operatorname{real}(\lambda) < 0\f$) [double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_lhp_eigenval_sel(double ar, double ai, double b)
{
	BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(ai);

	// Based on function SB02OW from SLICOT 5.0
	return ((ar > 0 && b < 0) || (ar < 0 && b > 0))
		   && ::std::abs(b) > (::std::abs(ar)*::std::numeric_limits<double>::epsilon());
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the right-half plane
 *  (\f$\operatorname{real}(\lambda) > 0\f$) [complex single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_rhp_eigenval_sel(::std::complex<float> a, ::std::complex<float> b)
{
	// Based on function ZB02OW.f from SLICOT 5.0
	return ::std::abs(b) != 0 && ::std::complex<float>(a/b).real() > 0;
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the right-half plane
 *  (\f$\operatorname{real}(\lambda) > 0\f$) [complex double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_rhp_eigenval_sel(::std::complex<double> a, ::std::complex<double> b)
{
	// Based on function ZB02OW.f from SLICOT 5.0
	return ::std::abs(b) != 0 && ::std::complex<double>(a/b).real() > 0;
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the right-half plane
 *  (\f$\operatorname{real}(\lambda) > 0\f$) [single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_rhp_eigenval_sel(float ar, float ai, float b)
{
	BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(ai);

	// Based on function SB02OW from SLICOT 5.0
	return ((ar > 0 && b > 0) || (ar < 0 && b < 0))
		   && ::std::abs(b) > (::std::abs(ar)*::std::numeric_limits<float>::epsilon());
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ in the right-half plane
 *  (\f$\operatorname{real}(\lambda) > 0\f$) [double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_rhp_eigenval_sel(double ar, double ai, double b)
{
	BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(ai);

	// Based on function SB02OW from SLICOT 5.0
	return ((ar > 0 && b > 0) || (ar < 0 && b < 0))
		   && ::std::abs(b) > (::std::abs(ar)*::std::numeric_limits<double>::epsilon());
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are interior of unit disk
 *  (\f$\operatorname{abs}(\lambda) < 1\f$) [complex single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udi_eigenval_sel(::std::complex<float> a, ::std::complex<float> b)
{
	// Based on function ZB02OX from SLICOT 5.0
	return ::std::abs(a) < ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are interior of unit disk
 *  (\f$\operatorname{abs}(\lambda) < 1\f$) [complex double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udi_eigenval_sel(::std::complex<double> a, ::std::complex<double> b)
{
	// Based on function ZB02OX from SLICOT 5.0
	return ::std::abs(a) < ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are interior of unit disk
 *  (\f$\operatorname{abs}(\lambda) < 1\f$) [single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udi_eigenval_sel(float ar, float ai, float b)
{
	// Based on function SB02OX from SLICOT 5.0
	return ::std::abs(::std::complex<float>(ar, ai)) < ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are interior of unit disk
 *  (\f$\operatorname{abs}(\lambda) < 1\f$) [double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udi_eigenval_sel(double ar, double ai, double b)
{
	// Based on function SB02OX from SLICOT 5.0
	return ::std::abs(::std::complex<double>(ar, ai)) < ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are exterior of unit disk
 *  (\f$\operatorname{abs}(\lambda) > 1\f$) [complex single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udo_eigenval_sel(::std::complex<float> a, ::std::complex<float> b)
{
	// Based on function ZB02OX from SLICOT 5.0
	// FIXME: Shall we use '>' instead of '>='?
	//        In MATLAB, the 'udo' function takes the 'exterior of unit disk (abs(E) > 1)'.
	//        In Octave, the 'big' function takes the 'leading block has all |lambda| >= 1'.
	return ::std::abs(a) >= ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are exterior of unit disk
 *  (\f$\operatorname{abs}(\lambda) > 1\f$) [complex double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udo_eigenval_sel(::std::complex<double> a, ::std::complex<double> b)
{
	// Based on function ZB02OX from SLICOT 5.0
	// FIXME: Shall we use '>' instead of '>='?
	//        In MATLAB, the 'udo' function takes the 'exterior of unit disk (abs(E) > 1)'.
	//        In Octave, the 'big' function takes the 'leading block has all |lambda| >= 1'.
	return ::std::abs(a) >= ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are exterior of unit disk
 *  (\f$\operatorname{abs}(\lambda) > 1\f$) [single precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udo_eigenval_sel(float ar, float ai, float b)
{
	// Based on function SB02OX from SLICOT 5.0
	// FIXME: Shall we use '>' instead of '>='?
	//        In MATLAB, the 'udo' function takes the 'exterior of unit disk (abs(E) > 1)'.
	//        In Octave, the 'big' function takes the 'leading block has all |lambda| >= 1'.
	return ::std::abs(::std::complex<float>(ar, ai)) >= ::std::abs(b);
}


/**
 * \brief Select eigenvalues \f$\lambda\f$ which are exterior of unit disk
 *  (\f$\operatorname{abs}(\lambda) > 1\f$) [double precision case].
 */
BOOST_UBLAS_INLINE
::fortran_bool_t qz_udo_eigenval_sel(double ar, double ai, double b)
{
	// Based on function SB02OX from SLICOT 5.0
	// FIXME: Shall we use '>' instead of '>='?
	//        In MATLAB, the 'udo' function takes the 'exterior of unit disk (abs(E) > 1)'.
	//        In Octave, the 'big' function takes the 'leading block has all |lambda| >= 1'.
	return ::std::abs(::std::complex<double>(ar, ai)) >= ::std::abs(b);
}


namespace /*<anoymous>*/ {

/// Type for the QZ eigenvalues selector (single precision case).
typedef ::fortran_bool_t (*qz_single_selector_type)(float, float, float);
/// Type for the QZ eigenvalues selector (double precision case).
typedef ::fortran_bool_t (*qz_double_selector_type)(double, double, double);
/// Type for the QZ eigenvalues selector (single precision complex case).
typedef ::fortran_bool_t (*qz_complex_single_selector_type)(::std::complex<float>, ::std::complex<float>);
/// Type for the QZ eigenvalues selector (single precision complex case).
typedef ::fortran_bool_t (*qz_complex_double_selector_type)(::std::complex<double>, ::std::complex<double>);


/// Options to select what Schur vectors compute in the QZ decomposition.
enum qz_schurvectors_side
{
	none_qz_schurvectors, ///< Do not compute vectors.
	left_qz_schurvectors, ///< Compute only left vectors.
	right_qz_schurvectors, ///< Compute only right vectors.
	both_qz_schurvectors ///< Compute both left and right vectors.
};


/// Options to select what generalized eigenvectors compute from the QZ decomposition.
enum qz_eigenvectors_side
{
//	none_qz_eigenvectors, ///< Do not compute generalized eigenvectors.
	left_qz_eigenvectors, ///< Compute only left generalized eigenvectors.
	right_qz_eigenvectors, ///< Compute only right generalized eigenvectors.
	both_qz_eigenvectors ///< Compute both left and right generalized eigenvectors.
};


/// Options for reordering QZ factorizations.
enum qz_order_option
{
	no_extra_qz_option, ///< Only reorder w.r.t. SELECT. No extras.
	projections_qz_option, ///<  Reciprocal of norms of \e projections onto left and righteigenspaces w.r.t. the selected cluster (PL and PR).
	upper_bounds_fnorm_qz_option, ///< Upper bounds on Difu and Difl. F-norm-based estimate (DIF(1:2)).
	upper_bounds_1norm_qz_option, ///< Estimate of Difu and Difl. 1-norm-based estimate (DIF(1:2)).
	projections_upper_bounds_fnorm_qz_option, ///< Compute PL, PR and DIF (F-norm based): Economic version to get it all.
	projections_upper_bounds_1norm_qz_option ///< Compute PL, PR and DIF (1-norm based).
};


/// Options for computing generalized Schur eigenvectors.
enum qz_eigenvectors_option
{
	all_qz_eigenvectors_option, ///< Compute all right and/or left eigenvectors.
	backtransform_qz_eigenvectors_option, ///< Compute all right and/or left eigenvectors, backtransformed by the matrices in VR and/or VL.
	select_qz_eigenvectors_option ///< Compute selected right and/or left eigenvectors.
};


/// Create a selector function according to the given \a selection method.
template <typename ValueT>
BOOST_UBLAS_INLINE
::external_fp create_qz_eigvals_selector(qz_eigenvalues_selection selection)
{
	::external_fp selctg = 0;

	switch (selection)
	{
		case lhp_qz_eigenvalues:
			selctg = BOOST_UBLASX_DETAIL_QZ_SELECTOR_CAST(ValueT, qz_lhp_eigenval_sel);
			break;
		case rhp_qz_eigenvalues:
			selctg = BOOST_UBLASX_DETAIL_QZ_SELECTOR_CAST(ValueT, qz_rhp_eigenval_sel);
			break;
		case udi_qz_eigenvalues:
			selctg = BOOST_UBLASX_DETAIL_QZ_SELECTOR_CAST(ValueT, qz_udi_eigenval_sel);
			break;
		case udo_qz_eigenvalues:
			selctg = BOOST_UBLASX_DETAIL_QZ_SELECTOR_CAST(ValueT, qz_udo_eigenval_sel);
			break;
		case all_qz_eigenvalues:
		default:
			selctg = 0;
			break;
	}

	return selctg;
}


/**
 * \brief Call the proper selector function according to the given \a selection
 *  method.
 *
 * \todo Replace this function with something at compile-time, like below:
 *  <pre>
 *   // Declare the selector with the proper type (this should emulate the
 *   // switch below, but at compile-time)
 *   BOOST_UBLAS_DETAIL_QZ_DECLARE_SELECTOR(selection, selector);
 *   ...
 *   // Invoke the selector
 *   selector(a, b);
 *  </pre>
 *  This way, we avoid to perform the switch at every invocation of the
 *  selector.
 */
template <typename AValueT, typename BValueT>
BOOST_UBLAS_INLINE
bool invoke_qz_eigvals_selector(qz_eigenvalues_selection selection, AValueT a, BValueT b)
{
	switch (selection)
	{
		case lhp_qz_eigenvalues:
			return qz_lhp_eigenval_sel(a, b);
		case rhp_qz_eigenvalues:
			return qz_rhp_eigenval_sel(a, b);
		case udi_qz_eigenvalues:
			return qz_udi_eigenval_sel(a, b);
			break;
		case udo_qz_eigenvalues:
			return qz_udo_eigenval_sel(a, b);
		case all_qz_eigenvalues:
		default:
			return true;
	}
}


/// Extract the generalized Schur eigenvectors from a QZ decomposition
/// (column-major case).
template <
	typename SMatrixT,
	typename TMatrixT,
	typename QMatrixT,
	typename ZMatrixT,
	typename LVMatrixT,
	typename RVMatrixT
>
void extract_eigenvectors(SMatrixT const& S, TMatrixT const& T, qz_eigenvectors_side eigvec_side, qz_eigenvectors_option eigvec_opt, vector< ::fortran_bool_t > eigvecs_sel, QMatrixT const& Q, ZMatrixT const& Z, LVMatrixT& LV, RVMatrixT& RV, column_major_tag)
{
	typedef typename matrix_traits<SMatrixT>::size_type size_type;

	size_type nr_LV = 0;
	size_type nc_LV = 0;
	size_type nr_RV = 0;
	size_type nc_RV = 0;
	bool resize_LV = true;
	bool resize_RV = true;
	char howmny;
	::fortran_int_t n = static_cast< ::fortran_int_t >(num_rows(S));
	::fortran_int_t mm = 0;
	::fortran_int_t m = 0;

	switch (eigvec_opt)
	{
		case backtransform_qz_eigenvectors_option:
			howmny = 'B';
			if (eigvec_side != right_qz_eigenvectors)
			{
				LV = Q;
			}
			if (eigvec_side != left_qz_eigenvectors)
			{
				RV = Z;
			}
			mm = n;
			break;
		case select_qz_eigenvectors_option:
			howmny = 'S';
			mm = static_cast< ::fortran_int_t >(size(eigvecs_sel));
			break;
		case all_qz_eigenvectors_option:
		default:
			howmny = 'A';
			mm = n;
	}

	switch (eigvec_side)
	{
		case left_qz_eigenvectors:
			nr_RV = nc_RV = 1;
			nr_LV = n;
			if (eigvec_opt == backtransform_qz_eigenvectors_option)
			{
				nc_LV = mm = ::std::max(n, mm);
				resize_LV = false;
			}
			else
			{
				nc_LV = mm;
			}
			break;
		case right_qz_eigenvectors:
			nr_LV = nc_LV = 1;
			nr_RV = n;
			if (eigvec_opt == backtransform_qz_eigenvectors_option)
			{
				nc_RV = mm = ::std::max(n, mm);
				resize_RV = false;
			}
			else
			{
				nc_RV = mm;
			}
			break;
		case both_qz_eigenvectors:
		default:
			nr_LV = nr_RV = n;
			if (eigvec_opt == backtransform_qz_eigenvectors_option)
			{
				nc_LV = nc_RV = mm = ::std::max(n, mm);
				resize_LV = resize_RV = false;
			}
			else
			{
				nc_LV = nc_RV = mm;
			}
	}

	if (resize_LV)
	{
		LV.resize(nr_LV, nc_LV, false);
	}
	if (resize_RV)
	{
		RV.resize(nr_RV, nc_RV, false);
	}

	switch (eigvec_side)
	{
		case left_qz_eigenvectors:
			::boost::numeric::bindings::lapack::tgevc(
				::boost::numeric::bindings::tag::left(),
				howmny,
				eigvecs_sel,
				S,
				T,
				LV,
				RV,
				mm,
				m
			);
			break;
		case right_qz_eigenvectors:
			::boost::numeric::bindings::lapack::tgevc(
				::boost::numeric::bindings::tag::right(),
				howmny,
				eigvecs_sel,
				S,
				T,
				LV,
				RV,
				mm,
				m
			);
			break;
		case both_qz_eigenvectors:
		default:
			::boost::numeric::bindings::lapack::tgevc(
				::boost::numeric::bindings::tag::both(),
				howmny,
				eigvecs_sel,
				S,
				T,
				LV,
				RV,
				mm,
				m
			);
	}
}


/**
 * \brief Generalized (real/complex) Schur decomposition.
 *
 * Given two square matrices \f$A\f$ and \f$B\f$, its Generalized Schur
 * decomposition is given by
 * \f{align}{
 *   A &= Q S Z^T,\\
 *   B &= Q T Z^T
 * \f}
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <bool IsComplex>
struct qz_decomposition_impl;


/**
 * \brief Generalized real Schur decomposition.
 *
 * Given two square real matrices \f$A\f$ and \f$B\f$, its Generalized Schur
 * decomposition is given by
 * \f{align}{
 *   A &= Q S Z^T,\\
 *   B &= Q T Z^T
 * \f}
 *
 * The diagonal elements of \f$S\f$ and \f$T\f$,
 * \f$\alpha = \operatorname{diag}{S}\f$ and
 * \f$\beta = \operatorname{diag}{S}\f$, are the generalized eigenvalues that
 * satisfy:
 * \f{align}{
 *   A V \beta &= B V \alpha,\\
 *   \beta W A &= \alpha W^T B.
 * \f}
 *
 * The ratios:
 * \f[
 *   \frac{\alpha_i}{\beta_i}
 * \f]
 * are the generalized eigenvalues.
 * \f$\alpha_i\f$ and \f$\beta_i\f$ are the diagonals of the complex Schur
 * form (S,T) that would result if the 2-by-2 diagonal blocks of
 * the real Schur form of \f$(A,B)\f$ were further reduced to
 * triangular form using 2-by-2 complex unitary transformations.
 * If \f$\operatorname{imag}(\alpha_i)\f$ is zero, then the \f$i\f$-th
 * eigenvalue is real; if positive, then the \f$j\f$-th and \f$(i+1)\f$-st
 * eigenvalues are a complex conjugate pair, with
 * \f$\operatorname{imag}(\alpha_{i+1})\f$ negative.
 * The quotients \f$\alpha_i/\beta_i\f$ may easily over- or underflow, and
 * \f$\beta_i\f$ may even be zero.
 * It should be avoided to naively computing the ratio \f$alpha_i/beta_i\f$.
 * However, \f$\alpha\f$ will be always less than and usually
 * comparable with \f$\operatorname{norm}(A)\f$ in magnitude, and \f\beta\f$
 * always less than and usually comparable with \f$\operatorname{norm}(B)\f$.
 *
 * \note
 * Matlab users: the Matlab \c qz function return the Hermitian of the \f$Q\f$
 * matrix computed by this decomposition.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <>
struct qz_decomposition_impl<false>
{
	/**
 	 * \brief Compute the generalized Schur (QZ) factorization (column-major
 	 *  case).
 	 *
	 * Given two square matrices \a A and \a B of order \f$n\f$, computes the
	 * generalized eigenvalues \a alpha\ and \a beta, the generalized real Schur
	 * form (\a S,\a T), and the left and right matrices \a Q and \a Z of Schur
	 * vectors such that the generalized Schur (QZ) factorization holds:
	 * \f[ 
	 *  (A,B) = ( Q S Z^T, Q T Z^T )
	 * \f]
	 */
	template <typename AMatrixT, typename BMatrixT, typename QMatrixT, typename ZMatrixT, typename AlphaVectorT, typename BetaVectorT>
		static void decompose(AMatrixT& A, BMatrixT& B, qz_schurvectors_side eigvecs_side, bool want_eigvals, bool reorder_eigvals, ::external_fp eigvals_selector, QMatrixT& Q, ZMatrixT& Z, AlphaVectorT& alpha, BetaVectorT& beta, column_major_tag)
	{
		typedef typename promote_traits<
					typename matrix_traits<AMatrixT>::value_type,
					typename matrix_traits<BMatrixT>::value_type
				>::promote_type value_type;
		typedef typename promote_traits<
					typename matrix_traits<AMatrixT>::size_type,
					typename matrix_traits<BMatrixT>::size_type
				>::promote_type size_type;
		typedef vector<value_type> work_vector_type;

		size_type n = num_rows(A);
		char jobvsl;
		char jobvsr;
		char sort;
		size_type n_q;
		size_type n_z;

		switch (eigvecs_side)
		{
			case both_qz_schurvectors:
				jobvsl = jobvsr = 'V';
				n_q = n_z = n;
				break;
			case left_qz_schurvectors:
				jobvsl = 'V';
				jobvsr = 'N';
				n_q = n;
				n_z = 0;
				break;
			case right_qz_schurvectors:
				jobvsl = 'N';
				jobvsr = 'V';
				n_q = 0;
				n_z = n;
				break;
			case none_qz_schurvectors:
			default:
				jobvsl = jobvsr = 'N';
				n_q = n_z = 0;
		}

		if (reorder_eigvals)
		{
			sort = 'S';
		}
		else
		{
			sort = 'N';
		}

		if (num_rows(Q) != n)
		{
			Q.resize(n, n, false);
		}
		if (num_rows(Z) != n)
		{
			Z.resize(n, n, false);
		}

		work_vector_type alpha_real(n);
		work_vector_type alpha_imag(n);
		if (size(beta) != n)
		{
			beta.resize(n, false);
		}

		::fortran_int_t sdim;

		::boost::numeric::bindings::lapack::gges(
			jobvsl,
			jobvsr,
			sort,
			eigvals_selector,
			A,
			B,
			sdim,
			alpha_real,
			alpha_imag,
			beta,
			Q,
			Z
		);

		// Create the alpha vector
		if (want_eigvals)
		{
			if (size(alpha) != n)
			{
				alpha.resize(n, false);
			}
			// From LAPACK ?GGES documentation
			// "If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
			// positive, then the j-th and (j+1)-st eigenvalues are a
			// complex conjugate pair, with ALPHAI(j+1) negative."
			//
#ifdef BOOST_UBLASX_DEBUG
			// Underflow threshold (used in the 'safety check' inside the 'for' loop)
			value_type rmin(::std::numeric_limits<value_type>::min());
#endif // BOOST_UBLASX_DEBUG
			for (size_type i = 0; i < n; ++i)
			{
#ifdef BOOST_UBLASX_DEBUG
				// Safety check: when beta_i is near to zero the corresponding
				//               eigenvalue is infinite.
				//               This test was inspired by the 'f08wafe.f'
				//               subrouting found on NAG libraries.
				if ((::std::abs(alpha_real(i))+::std::abs(alpha_imag(i)))*rmin >= ::std::abs(beta(i)))
				{
					BOOST_UBLASX_DEBUG_TRACE("[Warning] Eigenvalue(" << i << ") is numerically infinite or undetermined: alpha_r(" << i << ") = " << alpha_real(i) << ", alpha_i(" << i << ") = " << alpha_imag(i) << ", beta(" << i << ") = " << beta(i));
				}
#endif // BOOST_UBLASX_DEBUG

//				if (alpha_imag(i) == value_type/*zero*/())
//				{
//					alpha(i) = ::std::complex<value_type>(alpha_real(i), value_type/*zero*/());
//				}
//				else
//				{
//					alpha(i) = ::std::complex<value_type>(alpha_real(i), alpha_imag(i));
//					// safety check (even if it should not happen)
//					if ((i+1) < n)
//					{
//						alpha(i+1) = ::std::conj(alpha(i));
//					}
//					++i;
//				}
				if (alpha_imag(i) == value_type/*zero*/())
				{
					alpha(i) = ::std::complex<value_type>(alpha_real(i), value_type/*zero*/());
				}
				else
				{
					alpha(i) = ::std::complex<value_type>(alpha_real(i), alpha_imag(i));
				}
			}
		}
		else
		{
			alpha.resize(0, false);
			beta.resize(0, false);
		}

		if (num_rows(Q) != n_q)
		{
			Q.resize(n_q, n_q, n_q > 0 ? true : false);
		}
		if (num_rows(Z) != n_z)
		{
			Z.resize(n_z, n_z, n_z > 0 ? true : false);
		}
	}


	/**
 	 * \brief Compute the generalized Schur (QZ) factorization (row-major case).
 	 *
	 * Given two square matrices \a A and \a B of order \f$n\f$, computes the
	 * generalized eigenvalues \a alpha\ and \a beta, the generalized real Schur
	 * form (\a S,\a T), and the left and right matrices \a Q and \a Z of Schur
	 * vectors such that the generalized Schur (QZ) factorization holds:
	 * \f[ 
	 *  (A,B) = ( Q S Z^T, Q T Z^T )
	 * \f]
	 */
	template <typename AMatrixT, typename BMatrixT, typename QMatrixT, typename ZMatrixT, typename AlphaVectorT, typename BetaVectorT>
		static void decompose(AMatrixT& A, BMatrixT& B, qz_schurvectors_side eigvecs_side, bool want_eigvals, bool reorder_eigvals, ::external_fp eigvals_selector, QMatrixT& Q, ZMatrixT& Z, AlphaVectorT& alpha, BetaVectorT& beta, row_major_tag)
	{
		// LAPACK works with dense column-major matrices

//		matrix<typename matrix_traits<AMatrixT>::value_type, typename layout_type<AMatrixT>::type> tmp_A(A);
//		matrix<typename matrix_traits<BMatrixT>::value_type, typename layout_type<BMatrixT>::type> tmp_B(B);
//		matrix<typename matrix_traits<QMatrixT>::value_type, typename layout_type<QMatrixT>::type> tmp_Q(Q);
//		matrix<typename matrix_traits<ZMatrixT>::value_type, typename layout_type<ZMatrixT>::type> tmp_Z(Z);
		matrix<typename matrix_traits<AMatrixT>::value_type, column_major> tmp_A(A);
		matrix<typename matrix_traits<BMatrixT>::value_type, column_major> tmp_B(B);
		matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);
		matrix<typename matrix_traits<ZMatrixT>::value_type, column_major> tmp_Z(Z);

		decompose(tmp_A, tmp_B, eigvecs_side, want_eigvals, reorder_eigvals, eigvals_selector, tmp_Q, tmp_Z, alpha, beta, column_major_tag());

		A = tmp_A;
		B = tmp_B;
		Q = tmp_Q;
		Z = tmp_Z;

		if (!want_eigvals)
		{
			alpha.resize(0, false);
			beta.resize(0, false);
		}
	}


	/// Reorder the given QZ decomposition (column-major case).
	template <
		typename SMatrixT,
		typename TMatrixT,
		typename QMatrixT,
		typename ZMatrixT,
		typename AlphaVectorT,
		typename BetaVectorT
	>
	static void reorder(SMatrixT& S, TMatrixT& T, qz_order_option order_opt, vector< ::fortran_bool_t > eigvals_sel, AlphaVectorT& alpha, BetaVectorT& beta, bool update_Q, QMatrixT& Q, bool update_Z, ZMatrixT& Z, column_major_tag)
	{
		typedef typename promote_traits<
					typename matrix_traits<SMatrixT>::value_type,
					typename matrix_traits<TMatrixT>::value_type
				>::promote_type value_type;

		::fortran_int_t ijob; // type of job the LAPACK function has to perform [intput]
		::fortran_int_t m; // dimension of the specified pair of left and right eigenspaces [output]
		value_type projl; // lower bounds on the reciprocal of the norm of "projections" onto left eigenspace [output]
		value_type projr; // lower bounds on the reciprocal of the norm of "projections" onto right eigenspace [output]
		vector<value_type> dif; // store the estimates of Difu and Difl [output]

		// Decode order option
		switch (order_opt)
		{
			case projections_qz_option:
				ijob = 1;
				break;
			case upper_bounds_fnorm_qz_option:
				ijob = 2;
				break;
			case upper_bounds_1norm_qz_option:
				ijob = 3;
				break;
			case projections_upper_bounds_fnorm_qz_option:
				ijob = 4;
				break;
			case projections_upper_bounds_1norm_qz_option:
				ijob = 5;
				break;
			case no_extra_qz_option:
			default:
				ijob = 0;
		}

		if (ijob >= 2)
		{
			dif.resize(2, false);
		}

		vector<value_type> aux_alphar(real(alpha));
		vector<value_type> aux_alphai(imag(alpha));

		::boost::numeric::bindings::lapack::tgsen(
			ijob,
			static_cast< ::fortran_bool_t >(update_Q ? 1 : 0),
			static_cast< ::fortran_bool_t >(update_Z ? 1 : 0),
			eigvals_sel,
			S,
			T,
			aux_alphar,
			aux_alphai,
			beta,
			Q,
			Z,
			m,
			projl,
			projr,
			dif
		);

		// Update the alpha vector
		//
		// From LAPACK ?TGSEN documentation
		// "If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
		// positive, then the j-th and (j+1)-st eigenvalues are a
		// complex conjugate pair, with ALPHAI(j+1) negative."
		//
		typedef typename vector_traits<AlphaVectorT>::size_type alpha_size_type;
		alpha_size_type n = size(alpha);
		for (alpha_size_type i = 0; i < n; ++i)
		{
			if (aux_alphai(i) == value_type/*zero*/())
			{
				alpha(i) = ::std::complex<value_type>(aux_alphar(i), value_type/*zero*/());
			}
			else
			{
				alpha(i) = ::std::complex<value_type>(aux_alphar(i), aux_alphai(i));
				// safety check (even if it should not happen)
				if ((i+1) < n)
				{
					alpha(i+1) = ::std::conj(alpha(i));
				}
				++i;
			}
		}
	}


	/// Reorder the given QZ decomposition (row-major case).
	template <
		typename SMatrixT,
		typename TMatrixT,
		typename QMatrixT,
		typename ZMatrixT,
		typename AlphaVectorT,
		typename BetaVectorT
	>
	static void reorder(SMatrixT& S, TMatrixT& T, qz_order_option order_opt, vector< ::fortran_bool_t > eigvals_sel, AlphaVectorT& alpha, BetaVectorT& beta, bool update_Q, QMatrixT& Q, bool update_Z, ZMatrixT& Z, row_major_tag)
	{
		// LAPACK works with dense column-major matrices

		typedef typename promote_traits<
					typename matrix_traits<SMatrixT>::value_type,
					typename matrix_traits<TMatrixT>::value_type
				>::promote_type value_type;
		typedef matrix<value_type, column_major> colmaj_matrix_type;

		colmaj_matrix_type tmp_S(S);;
		colmaj_matrix_type tmp_T(T);;
		colmaj_matrix_type tmp_Q;//(Q); //FIXME: maybe unnecessary if update_Q == false
		colmaj_matrix_type tmp_Z;//(Z); //FIXME: maybe unnecessary if update_Z == false

		if (update_Q)
		{
			tmp_Q = Q;
		}
		if (update_Z)
		{
			tmp_Z = Z;
		}

		reorder(tmp_S, tmp_T, order_opt, eigvals_sel, alpha, beta, update_Q, tmp_Q, update_Z, tmp_Z, column_major_tag());

		S = tmp_S;
		T = tmp_T;
		if (update_Q)
		{
			Q = tmp_Q;
		}
		if (update_Z)
		{
			Z = tmp_Z;
		}
	}
}; // struct qz_decomposition_impl<false>


/**
 * \brief Generalized complex Schur decomposition.
 *
 * Given two square complex matrices \f$A\f$ and \f$B\f$, its Generalized Schur
 * decomposition is given by
 * \f{align}{
 *   A &= Q S Z^T,\\
 *   B &= Q T Z^T
 * \f}
 *
 * The diagonal elements of \f$S\f$ and \f$T\f$,
 * \f$\alpha = \operatorname{diag}{S}\f$ and
 * \f$\beta = \operatorname{diag}{S}\f$, are the generalized eigenvalues that
 * satisfy:
 * \f{align}{
 *   A V \beta &= B V \alpha,\\
 *   \beta W A &= \alpha W^T B.
 * \f}
 *
 * The ratios
 * \f[
 *   \lambda_i = \frac{\alpha_i}{\beta_i}
 * \f]
 * represents the generalized eigenvalues of the matrix pair \f$(A,B)\f$, such
 * that \f$A - \lambda_i*B\f$ is singular.
 * The quotients \f$\alpha_i/\beta_i\f$ may easily over- or underflow, and
 * \f$\beta_i\f$ may even be zero.
 * It should be avoided to naively computing the ratio \f$alpha_i/beta_i\f$.
 * However, \f$\alpha\f$ will be always less than and usually
 * comparable with \f$\operatorname{norm}(A)\f$ in magnitude, and \f\beta\f$
 * always less than and usually comparable with \f$\operatorname{norm}(B)\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <>
struct qz_decomposition_impl<true>
{
	/**
 	 * \brief Compute the generalized Schur (QZ) factorization (row-major case).
 	 *
	 * Given two square matrices \a A and \a B of order \f$n\f$, computes the
	 * generalized eigenvalues \a alpha\ and \a beta, the generalized real Schur
	 * form (\a S,\a T), and the left and right matrices \a Q and \a Z of Schur
	 * vectors such that the generalized Schur (QZ) factorization holds:
	 * \f[ 
	 *  (A,B) = ( Q S Z^T, Q T Z^T )
	 * \f]
	 */
	template <typename AMatrixT, typename BMatrixT, typename QMatrixT, typename ZMatrixT, typename AlphaVectorT, typename BetaVectorT>
		static void decompose(AMatrixT& A, BMatrixT& B, qz_schurvectors_side eigvecs_side, bool want_eigvals, bool reorder_eigvals, ::external_fp eigvals_selector, QMatrixT& Q, ZMatrixT& Z, AlphaVectorT& alpha, BetaVectorT& beta, row_major_tag)
	{
		// LAPACK works with dense column-major matrices

		matrix<typename matrix_traits<AMatrixT>::value_type, column_major> tmp_A(A);
		matrix<typename matrix_traits<BMatrixT>::value_type, column_major> tmp_B(B);
		matrix<typename matrix_traits<QMatrixT>::value_type, column_major> tmp_Q(Q);
		matrix<typename matrix_traits<ZMatrixT>::value_type, column_major> tmp_Z(Z);

		//decompose(tmp_A, tmp_B, tmp_Q, tmp_Z, alpha, beta, side, order, selctg, column_major_tag());
		decompose(tmp_A, tmp_B, tmp_Q, tmp_Z, alpha, beta, eigvecs_side, reorder_eigvals, eigvals_selector, column_major_tag());

		A = tmp_A;
		B = tmp_B;
		Q = tmp_Q;
		Z = tmp_Z;

		if (!want_eigvals)
		{
			alpha.resize(0, false);
			beta.resize(0, false);
		}
	}


	/**
 	 * \brief Compute the generalized Schur (QZ) factorization (column-major
 	 *  case).
 	 *
	 * Given two square matrices \a A and \a B of order \f$n\f$, computes the
	 * generalized eigenvalues \a alpha\ and \a beta, the generalized real Schur
	 * form (\a S,\a T), and the left and right matrices \a Q and \a Z of Schur
	 * vectors such that the generalized Schur (QZ) factorization holds:
	 * \f[ 
	 *  (A,B) = ( Q S Z^T, Q T Z^T )
	 * \f]
	 */
	template <typename AMatrixT, typename BMatrixT, typename QMatrixT, typename ZMatrixT, typename AlphaVectorT, typename BetaVectorT>
		static void decompose(AMatrixT& A, BMatrixT& B, qz_schurvectors_side eigvecs_side, bool want_eigvals, bool reorder_eigvals, ::external_fp eigvals_selector, QMatrixT& Q, ZMatrixT& Z, AlphaVectorT& alpha, BetaVectorT& beta, column_major_tag)
	{
		BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(want_eigvals);

		typedef typename promote_traits<
					typename matrix_traits<AMatrixT>::value_type,
					typename matrix_traits<BMatrixT>::value_type
				>::promote_type value_type;
		typedef typename type_traits<value_type>::real_type real_type;
		typedef typename promote_traits<
					typename matrix_traits<AMatrixT>::size_type,
					typename matrix_traits<BMatrixT>::size_type
				>::promote_type size_type;

		size_type n = num_rows(A);
		char jobvsl;
		char jobvsr;
		char sort;
		size_type n_q;
		size_type n_z;

		switch (eigvecs_side)
		{
			case both_qz_schurvectors:
				jobvsl = jobvsr = 'V';
				n_q = n_z = n;
				break;
			case left_qz_schurvectors:
				jobvsl = 'V';
				jobvsr = 'N';
				n_q = n;
				n_z = 0;
				break;
			case right_qz_schurvectors:
				jobvsl = 'N';
				jobvsr = 'V';
				n_q = 0;
				n_z = n;
				break;
			case none_qz_schurvectors:
			default:
				jobvsl = jobvsr = 'N';
				n_q = n_z = 0;
		}

		if (reorder_eigvals)
		{
			sort = 'S';
		}
		else
		{
			sort = 'N';
		}

		if (num_rows(Q) != n)
		{
			Q.resize(n, n, false);
		}
		if (num_rows(Z) != n)
		{
			Z.resize(n, n, false);
		}
		if (size(alpha) != n)
		{
			alpha.resize(n, false);
		}
		if (size(beta) != n)
		{
			beta.resize(n, false);
		}

		::fortran_int_t sdim;

		::boost::numeric::bindings::lapack::gges(
			jobvsl,
			jobvsr,
			sort,
			eigvals_selector,
			A,
			B,
			sdim,
			alpha,
			beta,
			Q,
			Z
		);

		if (want_eigvals)
		{
#ifdef BOOST_UBLASX_DEBUG
			// Safety check: when beta_i is near to zero the corresponding
			//               eigenvalue is infinite.
			//               This test was inspired by the 'f08wnfe.f'
			//               subrouting found on NAG libraries.
			real_type rmin(::std::numeric_limits<real_type>::min());
			for (size_type i = 0; i < n; ++i)
			{
				if (::std::abs(alpha(i))*rmin >= ::std::abs(beta(i)))
				{
					BOOST_UBLASX_DEBUG_TRACE("[Warning] Eigenvalue(" << i << ") is numerically infinite or undetermined: alpha(" << i << ") = " << alpha(i) << ", beta(" << i << ") = " << beta(i));
				}
			}
#endif // BOOST_UBLASX_DEBUG
		}
		else
		{
			alpha.resize(0, false);
			beta.resize(0, false);
		}

		if (num_rows(Q) != n_q)
		{
			Q.resize(n_q, n_q, n_q > 0 ? true : false);
		}
		if (num_rows(Z) != n_z)
		{
			Z.resize(n_z, n_z, n_z > 0 ? true : false);
		}
	}


	/// Reorder the given QZ decomposition (column-major case).
	template <
		typename SMatrixT,
		typename TMatrixT,
		typename QMatrixT,
		typename ZMatrixT,
		typename AlphaVectorT,
		typename BetaVectorT
	>
	static void reorder(SMatrixT& S, TMatrixT& T, qz_order_option order_opt, vector<fortran_bool_t> eigvals_sel, AlphaVectorT& alpha, BetaVectorT& beta, bool update_Q, QMatrixT& Q, bool update_Z, ZMatrixT& Z, column_major_tag)
	{
		typedef typename promote_traits<
					typename matrix_traits<SMatrixT>::value_type,
					typename matrix_traits<TMatrixT>::value_type
				>::promote_type value_type;
		typedef typename type_traits<value_type>::real_type real_type;

		::fortran_int_t ijob; // type of job the LAPACK function has to perform [intput]
		::fortran_int_t m; // dimension of the specified pair of left and right eigenspaces [output]
		real_type projl; // lower bounds on the reciprocal of the norm of "projections" onto left eigenspace [output]
		real_type projr; // lower bounds on the reciprocal of the norm of "projections" onto right eigenspace [output]
		vector<real_type> dif; // store the estimates of Difu and Difl [output]

		// Decode order option
		switch (order_opt)
		{
			case projections_qz_option:
				ijob = 1;
				break;
			case upper_bounds_fnorm_qz_option:
				ijob = 2;
				break;
			case upper_bounds_1norm_qz_option:
				ijob = 3;
				break;
			case projections_upper_bounds_fnorm_qz_option:
				ijob = 4;
				break;
			case projections_upper_bounds_1norm_qz_option:
				ijob = 5;
				break;
			case no_extra_qz_option:
			default:
				ijob = 0;
		}

		if (ijob >= 2)
		{
			dif.resize(2, false);
		}

		::boost::numeric::bindings::lapack::tgsen(
			ijob,
			static_cast< ::fortran_bool_t >(update_Q ? 1 : 0),
			static_cast< ::fortran_bool_t >(update_Z ? 1 : 0),
			eigvals_sel,
			S,
			T,
			alpha,
			beta,
			Q,
			Z,
			m,
			projl,
			projr,
			dif
		);
	}


	/// Reorder the given QZ decomposition (row-major case).
	template <
		typename SMatrixT,
		typename TMatrixT,
		typename QMatrixT,
		typename ZMatrixT,
		typename AlphaVectorT,
		typename BetaVectorT
	>
	static void reorder(SMatrixT& S, TMatrixT& T, qz_order_option order_opt, vector<bool> eigvals_sel, AlphaVectorT& alpha, BetaVectorT& beta, bool update_Q, QMatrixT& Q, bool update_Z, ZMatrixT& Z, row_major_tag)
	{
		// LAPACK works with dense column-major matrices

		typedef typename promote_traits<
					typename matrix_traits<SMatrixT>::value_type,
					typename matrix_traits<TMatrixT>::value_type
				>::promote_type value_type;
		typedef matrix<value_type, column_major> colmaj_matrix_type;

		colmaj_matrix_type tmp_S(S);;
		colmaj_matrix_type tmp_T(T);;
		colmaj_matrix_type tmp_Q; //(Q); //FIXME: maybe unnecessary if update_Q == false
		colmaj_matrix_type tmp_Z; //(Z); //FIXME: maybe unnecessary if update_Z == false

		if (update_Q)
		{
			tmp_Q = Q;
		}
		if (update_Z)
		{
			tmp_Z = Z;
		}

		reorder(tmp_S, tmp_T, order_opt, eigvals_sel, alpha, beta, update_Q, tmp_Q, update_Z, tmp_Z, column_major_tag());

		S = tmp_S;
		T = tmp_T;
		if (update_Q)
		{
			Q = tmp_Q;
		}
		if (update_Z)
		{
			Z = tmp_Z;
		}
	}
}; // struct qz_decomposition_impl<true>

}} // Namespace detail::<anonymous>


/**
 * \brief Generalized Schur (QZ) decomposition.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ValueT>
class qz_decomposition
{
	public: typedef ValueT value_type;
	//private: typedef typename type_traits<value_type>::real_type real_type;
	private: typedef matrix<value_type, column_major> work_matrix_type;
	public: typedef work_matrix_type S_matrix_type;
	public: typedef work_matrix_type T_matrix_type;
	public: typedef work_matrix_type Q_matrix_type;
	public: typedef work_matrix_type Z_matrix_type;
	public: typedef work_matrix_type eigvecs_matrix_type;
	public: typedef vector<
						typename ::boost::mpl::if_<
								::boost::is_complex<value_type>,
								value_type,
								::std::complex<value_type>
							>::type
				> alpha_vector_type; // NOTE: alpha is a complex vector both for real and complex case. 
	public: typedef vector<value_type> beta_vector_type; // NOTE: beta is a complex vector only for complex case.
	public: typedef alpha_vector_type eigvals_vector_type;
	private: typedef typename matrix_traits<work_matrix_type>::size_type size_type;


	/// Default constructor.
	public: qz_decomposition()
	{
		// empty
	}


	/**
	 * \brief A constructor: QZ decomposition of \a A and \a B with optional
	 *  reordering.
	 *
	 * \param A The first input matrix.
	 * \param B The second input matrix.
	 * \param selection The type of eigevalues selection to use for reordering.
	 */
	public: template <typename MatrixExprT1, typename MatrixExprT2>
		qz_decomposition(matrix_expression<MatrixExprT1> const& A, matrix_expression<MatrixExprT2> const& B, qz_eigenvalues_selection selection = all_qz_eigenvalues)
		: S_(A),
		  T_(B)
	{
		// NOTE: preconditions check moved inside decompose.

		decompose(selection);
	}


	/**
	 * \brief QZ decomposition of \a A and \a B with optional reordering.
	 *
	 * \param A The first input matrix.
	 * \param B The second input matrix.
	 * \param selection The type of eigevalues selection to use for reordering.
	 */
	public: template <typename MatrixExprT1, typename MatrixExprT2>
		void decompose(matrix_expression<MatrixExprT1> const& A, matrix_expression<MatrixExprT2> const& B, qz_eigenvalues_selection selection = all_qz_eigenvalues)
	{
		// precondition: A and B have same orientation category
		BOOST_MPL_ASSERT(
			(
				::boost::is_same<
						typename matrix_traits<MatrixExprT1>::orientation_category,
						typename matrix_traits<MatrixExprT2>::orientation_category
				>
			)
		);
		// precondition: A is square
		BOOST_UBLAS_CHECK( num_rows(A) == num_columns(A), bad_size() );
		// precondition: B is square
		BOOST_UBLAS_CHECK( num_rows(B) == num_columns(B), bad_size() );

		S_ = A;
		T_ = B;

		decompose(selection);
	}


	/**
	 * \brief Get the Schur form of the first input matrix \f$A\f$ in the
	 *  QZ decomposition of \f$(A,B)\f$.
	 *
	 * \return The Schur form of the input matrix \f$A\f$.
	 */
	public: S_matrix_type const& S() const
	{
		return S_;
	}


	/**
	 * \brief Get the Schur form of the second input matrix \f$B\f$ in the
	 *  QZ decomposition of \f$(A,B)\f$.
	 *
	 * \return The Schur form of the input matrix \f$B\f$.
	 */
	public: T_matrix_type const& T() const
	{
		return T_;
	}


	/**
	 * \brief Get the orthogonal (or unitary) \f$Q\f$ matrix in the
	 *  QZ decomposition of \f$(A,B)\f$.
	 *
	 * \return The orthogonal (or unitary) \f$Q\f$ matrix.
	 */
	public: Q_matrix_type const& Q() const
	{
		return Q_;
	}


	/**
	 * \brief Get the orthogonal (or unitary) \f$Z\f$ matrix in the
	 *  QZ decomposition of \f$(A,B)\f$.
	 *
	 * \return The orthogonal (or unitary) \f$Z\f$ matrix.
	 */
	public: Z_matrix_type const& Z() const
	{
		return Z_;
	}


	/**
	 * \brief Get the numerators of the generalized eigenvalues.
	 *
	 * \return The numerators of the generalized eigenvalues.
	 */
	public: alpha_vector_type const& alpha() const
	{
		return alpha_;
	}


	/**
	 * \brief Get the denominators of the generalized eigenvalues.
	 *
	 * \return The denominators of the generalized eigenvalues.
	 */
	public: beta_vector_type const& beta() const
	{
		return beta_;
	}


	/**
	 * \brief Compute the generalized eigenvalues.
	 *
	 * Compute the generalized eigenvalues as the ratio:
	 * \f[
	 *   \frac{\alpha_i}{\beta_i}
	 * \f]
	 * where \f$\alpha_i\f$ and \f$\beta_i\f$ are the are the diagonals of the
	 * Schur form of the original input matrices \f$(A,B)\f$.
	 *
	 * \return The generalized eigenvalues.
	 *
	 * \note
	 *  It is generally not safe to directly compute this ratio since it may
	 *  easily over- or underflow, and \f$\beta_i\f$ may even be zero.
	 *  However, \f$\alpha\f$ will be always less than and usually comparable
	 *  with \f$\operatorname{norm}(A)\f$ in magnitude, and \f$\beta\f$ always
	 *  less than and usually comparable with  \f$\operatorname{norm}(B)\f$.
	 */
	public: eigvals_vector_type eigenvalues() const
	{
		return element_div(alpha_, beta_);

//Another (more costly) alternative to get eigenvalues
//		eigvals_vector_type l;
//		matrix<typename vector_traits<eigvals_vector_type>::value_type,column_major> L;
//		matrix<typename vector_traits<eigvals_vector_type>::value_type,column_major> V;
//		eigen(S_, T_, l, L, V);

//Try to balance before computing eigenvalues (really needed?)
//		S_matrix_type SS(S_);
//		T_matrix_type TT(T_);
//
//		balance_inplace(SS, TT, false, true);
//		eigen(SS, TT, l, L, V);
//
//		return l;
	}


	/**
	 * \brief Compute the generalized left eigenvectors.
	 *
	 * \param backtransform Compute the generalized left eigenvectors of matrix
	 *  pair \f$(A,B)\f$ instead of \f$(S,T)\f$.
	 * \return A matrix whose columns are the generalized left eigenvectors.
	 *
	 * The right eigenvector \f$x\f$ and the left eigenvector \f$y\f$ of
	 * \f$(S,T)\f$, corresponding to an eigenvalue \f$w\f$ are defined by:
	 * \f{align}{ 
	 *  S x &= w T x,\\
	 *  y^{H} S = w y^{H} T
	 * \f}
	 * where \f$y^{H}\f$ denotes the conjugate tranpose of \f$y\f$, and \f$S\f$
	 * and \f$T\f$ are the Schur form of the input matrix pair \f$(A,B)\f$, as
	 * computed by the QZ decomposition.
	 * The eigenvalues are not input parameters, but are computed
	 * directly from the diagonal blocks of \f$S\f$ and \f$T\f$.
	 *  
	 * This function returns the matrix \f$Y\f$ of left eigenvectors of
	 * \f$(S,T)\f$, or the product \f$Q*Y\f$, where \f$Q\f$ and \f$Z\f$ are the
	 * Schur vectors computed by the QZ decomposition, representing the left
	 * eigenvectors of \f$(A,B)\f$.
	 */
	public: eigvecs_matrix_type left_eigenvectors(bool backtransform=true) const
	{
		vector< ::fortran_bool_t > dummy_eigvecs_sel;
		eigvecs_matrix_type LV;
		eigvecs_matrix_type dummy_RV;

		extract_eigenvectors(
			S_,
			T_,
			detail::left_qz_eigenvectors,
			(backtransform
				? detail::backtransform_qz_eigenvectors_option
				: detail::all_qz_eigenvectors_option),
			dummy_eigvecs_sel,
			Q_,
			Z_,
			LV,
			dummy_RV,
			column_major_tag()
		);

		return LV;
	}


	/**
	 * \brief Compute the generalized right eigenvectors.
	 *
	 * \param backtransform Compute the generalized left eigenvectors of matrix
	 *  pair \f$(A,B)\f$ instead of \f$(S,T)\f$.
	 * \return A matrix whose columns are the generalized right eigenvectors.
	 *
	 * The right eigenvector \f$x\f$ and the left eigenvector \f$y\f$ of
	 * \f$(S,T)\f$, corresponding to an eigenvalue \f$w\f$ are defined by:
	 * \f{align}{ 
	 *  S x &= w T x,\\
	 *  y^{H} S = w y^{H} T
	 * \f}
	 * where \f$y^{H}\f$ denotes the conjugate tranpose of \f$y\f$, and \f$S\f$
	 * and \f$T\f$ are the Schur form of the input matrix pair \f$(A,B)\f$, as
	 * computed by the QZ decomposition.
	 * The eigenvalues are not input parameters, but are computed
	 * directly from the diagonal blocks of \f$S\f$ and \f$T\f$.
	 *  
	 * This function returns the matrix \f$X\f$ of right eigenvectors of
	 * \f$(S,T)\f$, or the product \f$Z*X\f$, where \f$Q\f$ and \f$Z\f$ are the
	 * Schur vectors computed by the QZ decomposition, representing the right
	 * eigenvectors of \f$(A,B)\f$.
	 */
	public: eigvecs_matrix_type right_eigenvectors(bool backtransform=true) const
	{
		vector< ::fortran_bool_t > dummy_eigvecs_sel;
		eigvecs_matrix_type dummy_Y;
		eigvecs_matrix_type X;

		extract_eigenvectors(
			S_,
			T_,
			detail::right_qz_eigenvectors,
			(backtransform
				? detail::backtransform_qz_eigenvectors_option
				: detail::all_qz_eigenvectors_option),
			dummy_eigvecs_sel,
			Q_,
			Z_,
			dummy_Y,
			X,
			column_major_tag()
		);

		return X;
	}


	/**
	 * \brief Compute the generalized eigenvectors.
	 *
	 * \param LV A matrix whose columns are the generalized left eigenvectors.
	 * \param RV A matrix whose columns are the generalized right eigenvectors.
	 * \param backtransform Compute the generalized left eigenvectors of matrix
	 *  pair \f$(A,B)\f$ instead of \f$(S,T)\f$.
	 * \return None, but output parameters \a LV and \a RV stores, on exit, the
	 *  left and right eigenvectors, respectively.
	 *
	 * The right eigenvector \f$x\f$ and the left eigenvector \f$y\f$ of
	 * \f$(S,T)\f$, corresponding to an eigenvalue \f$w\f$ are defined by:
	 * \f{align}{ 
	 *  S x &= w T x,\\
	 *  y^{H} S = w y^{H} T
	 * \f}
	 * where \f$y^{H}\f$ denotes the conjugate tranpose of \f$y\f$, and \f$S\f$
	 * and \f$T\f$ are the Schur form of the input matrix pair \f$(A,B)\f$, as
	 * computed by the QZ decomposition.
	 * The eigenvalues are not input parameters, but are computed
	 * directly from the diagonal blocks of \f$S\f$ and \f$T\f$.
	 *  
	 * This function returns the matrices \f$X\f$ and \f$Y\f$ of right and
	 * left eigenvectors of \f$(S,T)\f$, or the products \f$Z*X\f$ and
	 * \f$Q*Y\f$, where \f$Q\f$ and \f$Z\f$ are the Schur vectors computed by
	 * the QZ decomposition, representing the right and left eigenvectors of
	 * \f$(A,B)\f$.
	 */
	public: void eigenvectors(eigvecs_matrix_type& X, eigvecs_matrix_type& Y, bool backtransform=true) const
	{
		vector< ::fortran_bool_t > dummy_eigvecs_sel;

		extract_eigenvectors(
			S_,
			T_,
			detail::both_qz_eigenvectors,
			(backtransform
				? detail::backtransform_qz_eigenvectors_option
				: detail::all_qz_eigenvectors_option),
			dummy_eigvecs_sel,
			Q_,
			Z_,
			Y,
			X,
			column_major_tag()
		);
	}


	/**
	 * \brief Reorder the QZ decomposition.
	 *
	 * Reorders the generalized real Schur decomposition so that a
	 * selected cluster of eigenvalues appears in the leading diagonal blocks
	 * of the upper quasi-triangular matrix \c S and the upper triangular
	 * \c T.
	 * The leading columns of \c Q and \c Z form orthonormal bases of the
	 * corresponding left and right eigenspaces (deflating subspaces).
	 *
	 * \note
	 *  The \c i-th element of the input vector \a selection must evaluate
	 *  to \c true if the \c i-th eigenvalue is to be selected.
	 * \note
	 *  As a side effect this function changes the original QZ decomposition
	 *  (i.e., matrices \c S, \c T, \c Q, and \c Z, and vector \c alpha and
	 *  \c beta).
	 */
	public: void reorder(qz_eigenvalues_selection selection)
	{
		//::external_fp selctg;
		size_type n = size(alpha_);
		vector< ::fortran_bool_t > eigvals_sel(n);

		for (size_type i = 0; i < n; ++i)
		{
			if (detail::invoke_qz_eigvals_selector(selection, alpha_(i), beta_(i)))
			{
				eigvals_sel(i) = 1;
			}
			else
			{
				eigvals_sel(i) = 0;
			}
		}
			
		detail::qz_decomposition_impl<
				::boost::is_complex<
					value_type
				>::value
			>::template reorder(S_, T_, detail::no_extra_qz_option, eigvals_sel, alpha_, beta_, true, Q_, true, Z_, column_major_tag());
	}


	/**
	 * \brief Reorder the QZ decomposition.
	 *
	 * \tparam VectorExprT The type of the input selection vector.
	 * \param selection Logical vector whose i-th element specifies whether the
	 * 	i-th eigenvalue is to be selected.
	 *
	 * Reorders the generalized real Schur decomposition so that a
	 * selected cluster of eigenvalues appears in the leading diagonal blocks
	 * of the upper quasi-triangular matrix \c S and the upper triangular
	 * \c T.
	 * The leading columns of \c Q and \c Z form orthonormal bases of the
	 * corresponding left and right eigenspaces (deflating subspaces).
	 *
	 * \note
	 *  The \c i-th element of the input vector \a selection must evaluate
	 *  to \c true if the \c i-th eigenvalue is to be selected.
	 * \note
	 *  As a side effect this function changes the original QZ decomposition
	 *  (i.e., matrices \c S, \c T, \c Q, and \c Z, and vector \c alpha and
	 *  \c beta).
	 */
	public: template <typename VectorExprT>
		void reorder(vector_expression<VectorExprT> const& selection)
	{
		// precondition: size(selection) == size(alpha_) [ == size(beta_) ]
		BOOST_UBLAS_CHECK( size(selection) == size(alpha_), bad_size() );

		size_type n = size(selection);
		vector< ::fortran_bool_t > eigvals_sel(n);

		for (size_type i = 0; i < n; ++i)
		{
			if (static_cast<bool>(selection()(i)))
			{
				eigvals_sel(i) = 1;
			}
			else
			{
				eigvals_sel(i) = 0;
			}
		}
			
		detail::qz_decomposition_impl<
				::boost::is_complex<
					value_type
				>::value
			>::template reorder(S_, T_, detail::no_extra_qz_option, eigvals_sel, alpha_, beta_, true, Q_, true, Z_, column_major_tag());
	}


	/**
	 * \brief QZ decomposition with optional reordering.
	 *
	 * \param selection The type of eigevalues selection to use for reordering.
	 */
	private: void decompose(qz_eigenvalues_selection selection)
	{
		//typedef typename type_traits<value_type>::real_type real_type;

		bool sort = false;
		::external_fp selctg = 0;

		selctg = detail::create_qz_eigvals_selector<value_type>(selection);
		if (selctg != 0)
		{
			sort = true;
		}

		detail::qz_decomposition_impl<
				::boost::is_complex<value_type>::value
			>::template decompose(S_, T_, detail::both_qz_schurvectors, true, sort, selctg, Q_, Z_, alpha_, beta_, column_major_tag());
	}


	/// The Schur form of the input matrix \f$A\f$.
	private: S_matrix_type S_;
	/// The Schur form of the input matrix \f$B\f$.
	private: T_matrix_type T_;
	/// The orthogonal/unitary matrix such that \f$QAZ=S\f$ and \f$QBZ=T\f$.
	private: Q_matrix_type Q_;
	/// The orthogonal/unitary matrix such that \f$QAZ=S\f$ and \f$QBZ=T\f$.
	private: Z_matrix_type Z_;
	/// The numerator of the generalized Schur eigenvalues.
	private: alpha_vector_type alpha_; // == diag(S_)
	/// The denominator of the generalized Schur eigenvalues.
	private: beta_vector_type beta_;
};


/**
 * \brief QZ decomposition of a matrix pair \f$(A,B)\f$.
 *
 * \tparam AMatrixExprT The type of the first matrix expression.
 * \tparam BMatrixExprT The type of the second matrix expression.
 *
 * \param A The first matrix expression.
 * \param B The second matrix expression.
 * \return An object containing information on the QZ decomposition of
 *  \f$(A,B)\f$ (\see qz_decomposition).
 *
 * For square matrices A and B, produces upper quasi-triangular matrices \a S
 * and \a T, and unitary matrices \a Q and \a Z such that
 * \f{align}{
 *   QSZ' &= A,\\
 *   QTZ' &= B.
 * \f}
 * For complex matrices, S and T are triangular.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename AMatrixExprT, typename BMatrixExprT>
BOOST_UBLAS_INLINE
qz_decomposition<
	typename promote_traits<
		typename matrix_traits<AMatrixExprT>::value_type,
		typename matrix_traits<BMatrixExprT>::value_type
	>::promote_type
> qz_decompose(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, qz_eigenvalues_selection selection = all_qz_eigenvalues)
{
	typedef typename promote_traits<
			typename matrix_traits<AMatrixExprT>::value_type,
			typename matrix_traits<BMatrixExprT>::value_type
		>::promote_type value_type;

	return qz_decomposition<value_type>(A, B, selection);
}


/**
 * \brief QZ decomposition of a matrix pair \f$(A,B)\f$.
 *
 * \tparam AMatrixT The type of the first input matrix.
 * \tparam BMatrixT The type of the second input matrix.
 *
 * \param A On entry, the first input matrix.
 *  On exit, the generalized Schur form of matrix \a A.
 * \param B On entry, the second input matrix.
 *  On exit, the generalized Schur form of matrix \a B.
 * \param Q An orthogonal (or unitary) matrix such that \f$QAZ=S\f$ and
 *  \f$QBZ=T\f$, where \f$S\f$ and \f$T\f$ denote the generalized Schur form
 *  of matrices \a A and \a B, respectively.
 * \param Z An orthogonal (or unitary) matrix such that \f$QAZ=S\f$ and
 *  \f$QBZ=T\f$, where \f$S\f$ and \f$T\f$ denote the generalized Schur form
 *  of matrices \a A and \a B, respectively.
 * \return No return value, but the function returns the computed QZ
 *  decomposition in the arguments \a A, \a B, \a Q, and \a Z.
 *
 * For square matrices A and B, produces upper quasi-triangular matrices \f$S\f$
 * and \f$T\f$, and unitary matrices \a Q and \a Z such that
 * \f{align}{
 *   QAZ &= S,\\
 *   QBZ &= T.
 * \f}
 * For complex matrices, S and T are triangular.
 * Matrix \f$S\f$ and \f$T\f$ are stored, at the exit of the function call, in
 * the arguments \a A and \a B, respectively.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
	typename AMatrixT,
	typename BMatrixT,
	typename QMatrixT,
	typename ZMatrixT
>
BOOST_UBLAS_INLINE
void qz_decompose_inplace(AMatrixT& A, BMatrixT& B, QMatrixT& Q, ZMatrixT& Z, qz_eigenvalues_selection selection = all_qz_eigenvalues)
{
	typedef typename promote_traits<
			typename matrix_traits<AMatrixT>::value_type,
			typename matrix_traits<BMatrixT>::value_type
		>::promote_type value_type;

	qz_decomposition<value_type> qz(A, B, selection);

	A = qz.S();
	B = qz.T();
	Q = qz.Q();
	Z = qz.Z();
}


/**
 * \brief QZ decomposition of a matrix pair \f$(A,B)\f$.
 *
 * \tparam AMatrixExprT The type of the first matrix expression.
 * \tparam BMatrixExprT The type of the second matrix expression.
 *
 * \param A The first matrix expression.
 * \param B The second matrix expression.
 * \param S The generalized Schur form of matrix \a A.
 * \param T The generalized Schur form of matrix \a B.
 * \param Q An orthogonal (or unitary) matrix such that \f$QAZ=S\f$ and
 *  \f$QBZ=T\f$.
 * \param Z An orthogonal (or unitary) matrix such that \f$QAZ=S\f$ and
 *  \f$QBZ=T\f$.
 * \return No return value, but the function returns the computed QZ
 *  decomposition in the arguments \a S, \a T, \a Q, and \a Z.
 *
 * For square matrices A and B, produces upper quasi-triangular matrices \a S
 * and \a T, and unitary matrices \a Q and \a Z such that
 * \f{align}{
 *   QAZ &= S,\\
 *   QBZ &= T.
 * \f}
 * For complex matrices, S and T are triangular.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
	typename AMatrixExprT,
	typename BMatrixExprT,
	typename SMatrixT,
	typename TMatrixT,
	typename QMatrixT,
	typename ZMatrixT
>
BOOST_UBLAS_INLINE
void qz_decompose(matrix_expression<AMatrixExprT> const& A, matrix_expression<BMatrixExprT> const& B, SMatrixT& S, TMatrixT& T, QMatrixT& Q, ZMatrixT& Z, qz_eigenvalues_selection selection = all_qz_eigenvalues)
{
	typedef typename promote_traits<
			typename matrix_traits<AMatrixExprT>::value_type,
			typename matrix_traits<BMatrixExprT>::value_type
		>::promote_type value_type;

	qz_decomposition<value_type> qz(A, B, selection);

	S = qz.S();
	T = qz.T();
	Q = qz.Q();
	Z = qz.Z();
}


/**
 * \brief Reorder the QZ decomposition.
 *
 * \tparam SMatrixT The type of the \a S matrix.
 * \tparam TMatrixT The type of the \a T matrix.
 * \tparam QMatrixT The type of the \a Q matrix.
 * \tparam ZMatrixT The type of the \a Z matrix.
 * \tparam SelVectorExprT The type of the \a selection vector.
 * \param S The Schur form of the matrix \f$A\f$ in QZ decomposition of the
 *  matrix pair \f$(A,B)\f$.
 * \param T The Schur form of the matrix \f$B\f$ in the QZ decomposition of the
 *  matrix pair \f$(A,B)\f$.
 * \param Q The orthogonal (or unitary) matrix obtained by the QZ decomposition
 *  of the matrix pair \f$(A,B)\f$.
 * \param Z The orthogonal (or unitary) matrix obtained by the QZ decomposition
 *  of the matrix pair \f$(A,B)\f$.
 * \param selection A vector where the i-th element specifies whether or not the
 *  i-th eigenvalue should be selected in order to appear in the leading (upper
 *  left) diagonal blocks of the quasi-triangular pair \f$(SS,TS)\f$,
 * \return None, but the result of the reordering is stored in the parameters
 * \a S, \a T, \a Q, and \a Z.
 *
 * Reorders the generalized Schur decomposition
 * \f{align}
 *  Q*A*Z &= S,
 *  Q*B*Z &= T.
 * \f}
 * for a matrix pair \f$(A,B)\f$ so that a selected cluster of eigenvalues
 * appears in the leading diagonal blocks of the upper quasi-triangular matrix
 * \a SS and the upper triangular \a TS.
 * The leading columns of cumulative orthogonal transformation \a QS and \a ZS
 * form orthonormal bases of the corresponding left and right eigenspaces
 * (deflating subspaces).
 * After reordering, the following relations are still valid:
 * \f{align}
 *  Q*A*Z &= S,
 *  Q*B*Z &= T.
 * \f}
 */
template <
	typename SMatrixT,
	typename TMatrixT,
	typename QMatrixT,
	typename ZMatrixT,
	typename SelVectorT
>
BOOST_UBLAS_INLINE
void qz_reorder_inplace(SMatrixT& S, TMatrixT& T, QMatrixT& Q, ZMatrixT& Z, vector_expression<SelVectorT> const& selection)
{
	typedef typename matrix_traits<SMatrixT>::orientation_category orientation_category;

	// precondition: check that orientation category is the same for all the matrices
	BOOST_MPL_ASSERT(
		(::boost::mpl::and_<
			::boost::is_same<
				orientation_category,
				typename matrix_traits<TMatrixT>::orientation_category
			>,
			::boost::mpl::and_<
				::boost::is_same<
					orientation_category,
					typename matrix_traits<QMatrixT>::orientation_category
				>,
				::boost::is_same<
					orientation_category,
					typename matrix_traits<ZMatrixT>::orientation_category
				>
			>
		>)
	);

	typedef typename promote_traits<
				typename matrix_traits<SMatrixT>::value_type,
				typename matrix_traits<TMatrixT>::value_type
			>::promote_type value_type;
	typedef typename type_traits<value_type>::real_type real_type;

	// NOTE: alpha is always a complex vector while beta is complex only for the complex case
	vector< ::std::complex<real_type> > dummy_alpha(num_columns(S));
	vector<value_type> dummy_beta(num_columns(S));

	detail::qz_decomposition_impl<
			::boost::is_complex<
				typename promote_traits<
					typename matrix_traits<SMatrixT>::value_type,
					typename matrix_traits<TMatrixT>::value_type
				>::promote_type
			>::value
		>::template reorder(S, T, detail::no_extra_qz_option, selection, dummy_alpha, dummy_beta, true, Q, true, Z, orientation_category());
}


/**
 * \brief Reorder the QZ decomposition.
 *
 * \tparam SMatrixT The type of the \a S matrix.
 * \tparam TMatrixT The type of the \a T matrix.
 * \tparam QMatrixT The type of the \a Q matrix.
 * \tparam ZMatrixT The type of the \a Z matrix.
 * \tparam SelVectorExprT The type of the \a selection vector.
 * \param S The Schur form of the matrix \f$A\f$ in QZ decomposition of the
 *  matrix pair \f$(A,B)\f$.
 * \param T The Schur form of the matrix \f$B\f$ in the QZ decomposition of the
 *  matrix pair \f$(A,B)\f$.
 * \param Q The orthogonal (or unitary) matrix obtained by the QZ decomposition
 *  of the matrix pair \f$(A,B)\f$.
 * \param Z The orthogonal (or unitary) matrix obtained by the QZ decomposition
 *  of the matrix pair \f$(A,B)\f$.
 * \param selection A vector where the i-th element specifies whether or not the
 *  i-th eigenvalue should be selected in order to appear in the leading (upper
 *  left) diagonal blocks of the quasi-triangular pair \f$(SS,TS)\f$,
 * \param SS The new matrix \a S obtained after the reordering.
 * \param TS The new matrix \a T obtained after the reordering.
 * \param QS The new matrix \a Q obtained after the reordering.
 * \param ZS The new matrix \a Z obtained after the reordering.
 * \return None, but the result of the reordering is stored in the output
 *  parameters \a SS, \a TS, \a QS, and \a ZS.
 *
 * Reorders the generalized Schur decomposition
 * \f{align}
 *  Q*A*Z &= S,
 *  Q*B*Z &= T.
 * \f}
 * for a matrix pair \f$(A,B)\f$ so that a selected cluster of eigenvalues
 * appears in the leading diagonal blocks of the upper quasi-triangular matrix
 * \a SS and the upper triangular \a TS.
 * The leading columns of cumulative orthogonal transformation \a QS and \a ZS
 * form orthonormal bases of the corresponding left and right eigenspaces
 * (deflating subspaces).
 * After reordering, the following relations are still valid:
 * \f{align}
 *  QS*A*ZS &= SS,
 *  QS*B*ZS &= TS.
 * \f}
 */
template <
	typename SMatrixT,
	typename TMatrixT,
	typename QMatrixT,
	typename ZMatrixT,
	typename SelVectorExprT
>
BOOST_UBLAS_INLINE
void qz_reorder(SMatrixT const& S, TMatrixT const& T, QMatrixT const& Q, ZMatrixT const& Z, vector_expression<SelVectorExprT> const& selection, SMatrixT& SS, TMatrixT& TS, QMatrixT& QS, ZMatrixT& ZS)
{
	SS = S;
	TS = T;
	QS = Q;
	ZS = Z;

	qz_reorder_inplace(SS, TS, QS, ZS, selection);
}

}}} // Namespace boost::numeric::ublas


#endif // BOOST_NUMERIC_UBLASX_OPERATION_QZ_HPP
