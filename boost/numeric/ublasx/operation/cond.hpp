/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/cond.hpp
 *
 * \brief Matrix condition number with respect to inversion.
 *
 * The condition number of a function, with respect to an argument, measures
 * the asymptotically worst case of how much the function can change in
 * proportion to small changes in the argument.
 * The "function" is the solution of a problem and the "arguments" are the data
 * in the problem.
 * A problem with a low condition number is said to be
 * <em>well-conditioned</em>, while a problem with a high condition number is
 * said to be <em>ill-conditioned</em>.
 * As a general rule of thumb, if the condition number \f$\kappa(A) = 10k\f$,
 * then you lose \f$k\f$ digits of accuracy on top of what would be lost to the
 * numerical method due to loss of precision from arithmetic methods.
 *
 * The condition number of a matrix measures the sensitivity of the solution of
 * a system of linear equations to errors in the data.
 * It gives an indication of the accuracy of the results from matrix inversion
 * and the linear equation solution.
 * Condition numbers near 1 indicate a well-conditioned matrix.
 * Mathematically, the condition number of a matrix \f$A\f$, with respect to a
 * given matrix p-norm, is defined as:
 * \f{equation}
 *  \kappa(A) = \begin{cases}
 *               \|A\|_p \|A^{-1}\|_p & A \text{ is not singular},\\
 *               +\infty, & A \text{ is singular}
 *              \end{cases}
 * \f}
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_COND_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_COND_HPP


#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/operation/any.hpp>
#include <boost/numeric/ublasx/operation/inv.hpp>
#include <boost/numeric/ublasx/operation/max.hpp>
#include <boost/numeric/ublasx/operation/min.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/svd.hpp>
#include <functional>
#include <limits>
#include <stdexcept>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {

enum norm_categories
{
    norm_inf_category = -1,
    norm_frobenius_category = 0,
    norm_1_category = 1,
    norm_2_category = 2
};


template <int Norm, typename MatrixExprT>
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond_impl(matrix_expression<MatrixExprT> const& A)
{
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;
    typedef typename type_traits<value_type>::real_type real_type;

    // pre: A is square OR (A is rectangular AND norm is 2)
    if (num_rows(A) != num_columns(A) && Norm != norm_2_category)
    {
        throw ::std::invalid_argument("[boost::numeric::ublasx::detail::cond_impl] For rectangular matrices use the 2 norm.");
    }

    real_type c;

    switch (Norm)
    {
        case norm_frobenius_category: // Frobenius norm
            try
            {
                c = norm_frobenius(A)*norm_frobenius(inv(A));
            }
            catch (...)
            {
                c = ::std::numeric_limits<real_type>::infinity();
            }
            break;
        case norm_inf_category: // Infinity norm
            try
            {
                c = norm_inf(A)*norm_inf(inv(A));
            }
            catch (...)
            {
                c = ::std::numeric_limits<real_type>::infinity();
            }
            break;
        case norm_1_category: // 1-norm
            try
            {
                c = norm_1(A)*norm_1(inv(A));
            }
            catch (...)
            {
                c = ::std::numeric_limits<real_type>::infinity();
            }
            break;
        case norm_2_category: // 2-norm
            {
                vector<real_type> s = svd_values(A);
                if (any(s, ::std::bind2nd(::std::equal_to<real_type>(), 0)))
                {
                    // Singular matrix
                    c = ::std::numeric_limits<real_type>::infinity();
                }
                else
                {
                    c = max(s)/min(s);
                }
            }
            break;
    }

    return c;
}

}} // Namespace detail::<unnamed>


/**
 * \brief The 1-norm matrix condition number with respect to inversion.
 *
 * \tparam MatrixExprT The matrix expression type.
 * \param A The input matrix expression.
 * \return The 1-norm condition number if \a A is not singular; otherwise,
 *  \f$+\infty\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond_1(matrix_expression<MatrixExprT> const& A)
{
    return detail::cond_impl<detail::norm_1_category>(A);
}


/**
 * \brief The 2-norm matrix condition number with respect to inversion.
 *
 * \tparam MatrixExprT The matrix expression type.
 * \param A The input matrix expression.
 * \return The 2-norm condition number if \a A is not singular; otherwise,
 *  \f$+\infty\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond_2(matrix_expression<MatrixExprT> const& A)
{
    return detail::cond_impl<detail::norm_2_category>(A);
}


/**
 * \brief The infinity norm matrix condition number with respect to inversion.
 *
 * \tparam MatrixExprT The matrix expression type.
 * \param A The input matrix expression.
 * \return The infinity norm condition number if \a A is not singular;
 *  otherwise, \f$+\infty\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond_inf(matrix_expression<MatrixExprT> const& A)
{
    return detail::cond_impl<detail::norm_inf_category>(A);
}


/**
 * \brief The Frobenius norm matrix condition number with respect to inversion.
 *
 * \tparam MatrixExprT The matrix expression type.
 * \param A The input matrix expression.
 * \return The Frobenius norm condition number if \a A is not singular;
 *  otherwise, \f$+\infty\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond_frobenius(matrix_expression<MatrixExprT> const& A)
{
    return detail::cond_impl<detail::norm_frobenius_category>(A);
}


/**
 * \brief The 2-norm matrix condition number with respect to inversion.
 *
 * \tparam MatrixExprT The matrix expression type.
 * \param A The input matrix expression.
 * \return The 2-norm condition number if \a A is not singular; otherwise,
 *  \f$+\infty\f$.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename type_traits<typename matrix_traits<MatrixExprT>::value_type>::real_type cond(matrix_expression<MatrixExprT> const& A)
{
    return detail::cond_impl<detail::norm_2_category>(A);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_COND_HPP
