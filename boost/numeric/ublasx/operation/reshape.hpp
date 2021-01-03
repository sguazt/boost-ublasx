/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/reshape.hpp
 *
 * \brief Reshape a matrix.
 *
 * Copyright (c) 2011, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_RESHAPE_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_RESHAPE_HPP


//#include <boost/mpl/has_xxx.hpp>
//#include <boost/mpl/if.hpp>
#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/tags.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/compiler.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
//#include <boost/utility/enable_if.hpp>
#include <cstddef>



namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


template <typename T>
struct reshape_traits;


namespace detail { namespace /*<unnamed>*/ {

// /// Define a \c has_matrix_temp_type trait class.
//BOOST_MPL_HAS_XXX_TRAIT_DEF(matrix_temp_type)


///**
// * \brief Wrapper type-traits used in \c boost::lazy_enabled_if for getting the
// *  matrix temporary type (see below).
// * \tparam MatrixT A matrix type.
// */
//template <typename MatrixT>
//struct matrix_temp_type
//{
//  /// The matrix temporary type.
//  typedef typename matrix_temporary_traits<MatrixT>::type type;
//};
 

/**
 * \brief The implementation of the \em reshape operation.
 * \tparam VectorExprT The type of the input vector expression.
 * \param ve The input vector expression.
 * \param nr The number of rows of the reshaped matrix.
 * \param nc The number of columns of the reshaped matrix.
 * \return The reshaped matrix.
 */
template <typename VectorExprT>
typename reshape_traits<VectorExprT>::result_type reshape_impl(vector_expression<VectorExprT> const& ve,
                                                               ::std::size_t nr,
                                                               ::std::size_t nc)
{
    //FIXME: this function behaves differently from the MATLAB/Octave countepart as it works row-wise instead of column-wise

    typedef typename reshape_traits<VectorExprT>::result_type result_matrix_type;
    typedef typename vector_traits<VectorExprT>::size_type size_type;

    size_type n(size(ve));

    // pre: to reshape, the number of elements must not change.
    // NOTE: assertion used to mimic the MATLAB 'reshape' function.
    BOOST_UBLAS_CHECK(
            nr*nc == n,
            bad_size()
        );

    // NOTE: this is needed to avoid this kind of warning when one compiles
    //       without debug information.
    BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING( n );

    result_matrix_type res(nr, nc);

    size_type k(0);

    for (size_type i = 0; i < nr; ++i)
    {
        for (size_type j = 0; j < nc; ++j)
        {
            res(i,j) = ve()(k);

            ++k;
        }
    }

    return res;
}


/**
 * \brief The implementation of the \em reshape operation.
 * \tparam MatrixExprT The type of the input matrix expression.
 * \param A The input matrix expression.
 * \param nr The number of rows of the reshaped matrix.
 * \param nr The number of columns of the reshaped matrix.
 * \param colw Tells if elements are taken from the input matrix either in a
 *  column-wise way (\c true value) or in a row-wise way (\c false value).
 * \return The reshaped matrix.
 */
template <typename MatrixExprT>
typename reshape_traits<MatrixExprT>::result_type reshape_impl(matrix_expression<MatrixExprT> const& me,
                                                               ::std::size_t nr,
                                                               ::std::size_t nc,
                                                               bool colw)
{
    // pre: to reshape, the number of elements must not change.
    // NOTE: assertion used to mimic the MATLAB 'reshape' function.
    BOOST_UBLAS_CHECK(
            nr*nc == num_rows(me)*num_columns(me),
            bad_size()
        );

//  typedef typename matrix_temporary_traits<MatrixExprT>::type result_matrix_type;
    typedef typename reshape_traits<MatrixExprT>::result_type result_matrix_type;
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;

    result_matrix_type res(nr, nc);

    size_type r(0);
    size_type c(0);

    if (colw)
    {
        // Reshape by taking the elements in a column-wise way.
        size_type nr_me(num_rows(me));
        for (size_type j = 0; j < nc; ++j)
        {
            for (size_type i = 0; i < nr; ++i)
            {
                if (r == nr_me)
                {
                    r = size_type/*zero*/();
                    ++c;
                }

                res(i,j) = me()(r,c);

                ++r;
            }
        }
    }
    else
    {
        // Reshape by taking the elements in a row-wise way.
        size_type nc_me(num_columns(me));
        for (size_type j = 0; j < nc; ++j)
        {
            for (size_type i = 0; i < nr; ++i)
            {
                if (c == nc_me)
                {
                    c = size_type/*zero*/();
                    ++r;
                }

                res(i,j) = me()(r,c);

                ++c;
            }
        }
    }

    return res;
}


/// Auxiliary class for the implementation of the by-dim \c reshape operation.
template <std::size_t Dim>
struct reshape_by_dim_impl;

template <>
struct reshape_by_dim_impl<1>
{
    /**
     * \brief Reshape the given matrix taking its elements in a column-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, true);
    }
};


template <>
struct reshape_by_dim_impl<2>
{
    /**
     * \brief Reshape the given matrix taking its elements in a row-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, false);
    }
};


/// Auxiliary class for the implementation of the by-tag \c reshape operation.
template <typename TagT, typename OrientationT>
struct reshape_by_tag_impl;

template <>
struct reshape_by_tag_impl<tag::major, row_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a column-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, true);
    }
};

template <>
struct reshape_by_tag_impl<tag::major, column_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a row-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, false);
    }
};

template <>
struct reshape_by_tag_impl<tag::minor, row_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a row-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, false);
    }
};

template <>
struct reshape_by_tag_impl<tag::minor, column_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a column-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, true);
    }
};


template <>
struct reshape_by_tag_impl<tag::leading, row_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a row-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, false);
    }
};


template <>
struct reshape_by_tag_impl<tag::leading, column_major_tag>
{
    /**
     * \brief Reshape the given matrix taken its elements in a column-wise way.
     * \tparam ExprT A matrix expression type.
     * \pre ExprT must be a model of MatrixExpression.
     */
    template <typename ExprT>
    BOOST_UBLAS_INLINE
    static typename reshape_traits<ExprT>::result_type apply(matrix_expression<ExprT> const& me, ::std::size_t nr, ::std::size_t nc)
    {
        return reshape_impl(me, nr, nc, true);
    }
};

}} // Namespace detail::<unnamed>


template <typename T>
struct reshape_traits
{
    typedef matrix<typename T::value_type> result_type;
};

template <typename VectorExprT>
struct reshape_traits< vector_expression<VectorExprT> >
{
    typedef matrix<typename vector_traits<VectorExprT>::value_type> result_type;
};

template <typename MatrixExprT>
struct reshape_traits< matrix_expression<MatrixExprT> >
{
//  typedef typename matrix_temporary_traits<MatrixExprT>::type result_type;
    typedef matrix<typename matrix_traits<MatrixExprT>::value_type> result_type;
};


template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename reshape_traits<VectorExprT>::result_type reshape(vector_expression<VectorExprT> const& ve,
                                                          ::std::size_t nr,
                                                          ::std::size_t nc)
{
    return detail::reshape_impl(ve, nr, nc);
}


template <std::size_t Dim, typename MatrixExprT>
BOOST_UBLAS_INLINE
typename reshape_traits<MatrixExprT>::result_type reshape(matrix_expression<MatrixExprT> const& me, ::std::size_t nr, ::std::size_t nc)
{
    return detail::reshape_by_dim_impl<Dim>::template apply(me, nr, nc);
}


/*
 * [BEGIN] Does Not Work!
 *
template <typename TagT, typename MatrixExprT>
BOOST_UBLAS_INLINE
typename ::boost::lazy_enable_if_c<
    detail::has_matrix_temp_type<MatrixExprT>::value,
    detail::matrix_temp_type<MatrixExprT>
>::type reshape(matrix_expression<MatrixExprT> const& me,
                ::std::size_t nr,
                ::std::size_t nc)
{
    return detail::reshape_by_tag_impl<TagT, typename matrix_traits<MatrixExprT>::orientation_category>::template apply(me);
}
 *
 * [END] Does Not Work!
 */


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename reshape_traits<MatrixExprT>::result_type reshape(matrix_expression<MatrixExprT> const& me,
                                                            ::std::size_t nr,
                                                            ::std::size_t nc)
{
    return detail::reshape_by_dim_impl<1>::template apply(me, nr, nc);
}


template <std::size_t Dim, typename MatrixExprT>
BOOST_UBLAS_INLINE
void reshape_inplace(matrix_container<MatrixExprT>& mc, ::std::size_t nr, ::std::size_t nc)
{
    typedef typename reshape_traits<MatrixExprT>::result_type work_matrix_type;

    work_matrix_type res;
    res = reshape<Dim>(mc, nr, nc);
    mc().resize(nr, nc, false);
    mc() = res;
}


/*
 * [BEGIN] Does Not Work!
 *
template <typename TagT, typename MatrixExprT>
BOOST_UBLAS_INLINE
void reshape_inplace(matrix_container<MatrixExprT>& mc,
                     ::std::size_t nr,
                     ::std::size_t nc)
{
    typedef typename reshape_traits<MatrixExprT>::result_type work_matrix_type;

    work_matrix_type res;
    res = reshape<TagT>(mc, nr, nc);
    mc().resize(nr, nc, false);
    mc() = res;
}
 *
 * [END] Does Not Work!
 */


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
void reshape_inplace(matrix_container<MatrixExprT>& mc,
                     ::std::size_t nr,
                     ::std::size_t nc)
{
    typedef typename reshape_traits<MatrixExprT>::result_type work_matrix_type;

    work_matrix_type res;
    res = reshape(mc, nr, nc);
    mc().resize(nr, nc, false);
    mc() = res;
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_RESHAPE_HPP
