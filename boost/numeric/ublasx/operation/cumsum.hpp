/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/cumsum.hpp
 *
 * \brief Compute the cumulative sum of an array.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_CUMSUM_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_CUMSUM_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <cstddef>


//FIXME: could not user boost::[matrix|vector]_temporary_traits since some
//container type (e.g., zero_matrix) does not define such type.


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

/**
 * \brief Compute the cumulative sum of the elements of the given vector
 *  expression.
 * \tparam VectorExprT The type of the vector expression.
 * \param ve The vector expression whose elements are cumsummed up.
 * \return The cumulative sum of the elements of the vector expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT>
vector<typename vector_traits<VectorExprT>::value_type> cumsum(vector_expression<VectorExprT> const& ve);


/**
 * \brief Compute the cumulative sum of the elements over each column of the
 *  given matrix expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param ve The matrix expression whose elements are cumulative summed up over
 *  each column.
 * \return The cumulative sum of the elements over each column of the give
 *  matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Compute the cumulative sum of the elements over each column of the
 *  given matrix expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param ve The matrix expression whose elements are cumulative summed up over
 *  each column.
 * \return The cumulative sum of the elements over each column of the given
 *  matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_rows(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Compute the cumulative sum of the elements over each row of the
 *  given matrix  expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are cumulative summed up over
 *  each row.
 * \return A vector containing the cumsum of the elements over each row of
 *  the given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_columns(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Compute the cumulative sum of the elements of the given matrix
 *  expression along the given dimension tag.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression whose elements are cumulative summed up along
 *  the given dimension.
 * \return A vector containing the cumulative sum of the elements along the
 *  given dimension of the given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename TagT, typename MatrixExprT>
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_by_tag(matrix_expression<MatrixExprT> const& me);

//@} Declarations


namespace detail {

//@{ Declarations

/**
 * \brief Auxiliary class for computing the cumsum of the elements over the given
 *  dimension for a container of the given category.
 * \tparam Dim The dimension number (starting from 1).
 * \tparam CategoryT The category type (e.g., vector_tag).
 */
template <std::size_t Dim, typename CategoryT>
struct cumsum_by_dim_impl;

/**
 * \brief Auxiliary class for computing the cumsum of the elements over the given
 *  dimension tag for a container of the given category.
 * \tparam TagT The dimension tag type (e.g., tag::major).
 * \tparam CategoryT The category type (e.g., vector_tag).
 * \tparam OrientationT The orientation category type (e.g., row_major_tag).
 */
template <typename TagT, typename CategoryT, typename OrientationT>
struct cumsum_by_tag_impl;

//@} Declarations


//@{ Definitions

template <>
struct cumsum_by_dim_impl<1, vector_tag>
{
    template <typename VectorExprT>
    BOOST_UBLAS_INLINE
    static vector<typename vector_traits<VectorExprT>::value_type> apply(vector_expression<VectorExprT> const& ve)
    {
        return cumsum(ve);
    }
};


template <>
struct cumsum_by_dim_impl<1, matrix_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_rows(me);
    }
};


template <>
struct cumsum_by_dim_impl<2, matrix_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_columns(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::major, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_rows(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::minor, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_columns(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::leading, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_columns(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::major, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_columns(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::minor, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_rows(me);
    }
};


template <>
struct cumsum_by_tag_impl<tag::leading, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static matrix<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return cumsum_rows(me);
    }
};


template <typename TagT>
struct cumsum_by_tag_impl<TagT, matrix_tag, unknown_orientation_tag>: cumsum_by_tag_impl<TagT, matrix_tag, row_major_tag>
{
    // Empty
};

//@} Definitions

} // Namespace detail


//@{ Definitions

template <typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> cumsum(vector_expression<VectorExprT> const& ve)
{
    typedef typename vector_traits<VectorExprT>::size_type size_type;
    typedef vector<typename vector_traits<VectorExprT>::value_type> result_type;

    size_type n = size(ve);
    result_type s(ve);

    for (size_type i = 1; i < n; ++i)
    {
        s(i) += s(i-1);
    }

    return s;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum(matrix_expression<MatrixExprT> const& me)
{
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef matrix<typename matrix_traits<MatrixExprT>::value_type> result_type;

    size_type nr = num_rows(me);
    size_type nc = num_columns(me);

    result_type s(me);

    for (size_type c = 0; c < nc; ++c)
    {
        for (size_type r = 1; r < nr; ++r)
        {
            s(r,c) += s(r-1,c);
        }
    }

    return s;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_rows(matrix_expression<MatrixExprT> const& me)
{
    return cumsum(me);
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_columns(matrix_expression<MatrixExprT> const& me)
{
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef matrix<typename matrix_traits<MatrixExprT>::value_type> result_type;

    size_type nr = num_rows(me);
    size_type nc = num_columns(me);

    result_type s(me);

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 1; c < nc; ++c)
        {
            s(r,c) += s(r,c-1);
        }
    }

    return s;
}


template <size_t Dim, typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> cumsum(vector_expression<VectorExprT> const& ve)
{
    return detail::cumsum_by_dim_impl<Dim, vector_tag>::template apply(ve);
}


template <size_t Dim, typename MatrixExprT>
BOOST_UBLAS_INLINE
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum(matrix_expression<MatrixExprT> const& me)
{
    return detail::cumsum_by_dim_impl<Dim, matrix_tag>::template apply(me);
}


template <typename TagT, typename MatrixExprT>
//template <typename MatrixExprT, typename TagT>
BOOST_UBLAS_INLINE
matrix<typename matrix_traits<MatrixExprT>::value_type> cumsum_by_tag(matrix_expression<MatrixExprT> const& me)
{
    return detail::cumsum_by_tag_impl<TagT, matrix_tag, typename matrix_traits<MatrixExprT>::orientation_category>::template apply(me);
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_CUMSUM_HPP
