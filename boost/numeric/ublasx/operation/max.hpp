/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/max.hpp
 *
 * \brief The \c max operation.
 *
 * \todo When possible, use vector/matrix_temporary_traits instead of
 *  expliciting the type of the container.
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

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_MAX_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_MAX_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//@{ Declarations

/**
 * \brief Find the maximum element of the given vector expression.
 * \tparam VectorExprT The type of the vector expression.
 * \param ve The vector expression over which to iterate for finding the maximum
 *  element.
 * \return The maximum element in the vector expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename VectorExprT>
typename vector_traits<VectorExprT>::value_type max(vector_expression<VectorExprT> const& ve);

/**
 * \brief Find the maximum element of the given matrix expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which to iterate for finding the maximum
 *  element.
 * \return The maximum element in the matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
typename matrix_traits<MatrixExprT>::value_type max(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Find the maximum element of each row in the given matrix expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which to iterate for finding the maximum
 *  element of each row.
 * \return A vector containing the maximum element for each row in the given
 *  matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> max_rows(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Find the maximum element of each column in the given matrix
 *  expression.
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which to iterate for finding the maximum
 *  element of each column.
 * \return A vector containing the maximum element for each column in the given
 *  matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> max_columns(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Find the maximum element of the given vector expression.
 * \tparam Dim The dimension number (for vector, only dimension 1 is valid).
 * \tparam MatrixExprT The type of the vectovector expression.
 * \param ve The vector expression over which to iterate for finding the maximum
 *  element over the given dimension.
 * \return A vector of size 1 containing the maximum element of the given vector
 *  expression.
 *
 * This function is provided for the sake of usability, in order to make to make
 * the call to \c size<1>(vec) a valid call.
 * For the same reason, the return type is a vector instead of a simple scalar.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <size_t Dim, typename VectorExprT>
vector<typename vector_traits<VectorExprT>::value_type> max(vector_expression<VectorExprT> const& ve);

/**
 * \brief Find the maximum elements over the given dimension of the given matrix
 *  expression.
 * \tparam Dim The dimension number (starting from 1).
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which to iterate for finding the maximum
 *  element over the given dimension.
 * \return A vector containing the maximum elements over the given dimension in
 *  the given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <size_t Dim, typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> max(matrix_expression<MatrixExprT> const& me);

/**
 * \brief Find the maximum elements over the given dimension tag of the given
 *  matrix expression.
 * \tparam TagT The dimension tag type (e.g., tag::major).
 * \tparam MatrixExprT The type of the matrix expression.
 * \param me The matrix expression over which to iterate for finding the maximum
 *  element over the given dimension.
 * \return A vector containing the maximum elements over the given dimension in
 *  the given matrix expression.
 *
 * \author Marco Guazzone, &lt;marco.guazzone@gmail.com&gt;
 */
template <typename TagT, typename MatrixExprT>
vector<typename matrix_traits<MatrixExprT>::value_type> max_by_tag(matrix_expression<MatrixExprT> const& me);

//@} Declarations


namespace detail { namespace /*<unnamed>*/ {

//@{ Declarations

/// Helper function for implementing the 'greater-than' relational operator over
/// generic types.
template <typename T>
bool greater_than_impl(T a, T b);

/// Helper function for implementing the 'greater-than' relational operator over
/// complex types.
template <typename T>
bool greater_than_impl(::std::complex<T> const& a, ::std::complex<T> const& b);


/// Helper class for real/complex minus infinity
template <typename T>
struct minus_infinity;

/**
 * \brief Auxiliary class for computing the maximum elements over the given
 *  dimension for a container of the given category.
 * \tparam Dim The dimension number (starting from 1).
 * \tparam CategoryT The category type (e.g., vector_tag).
 */
template <std::size_t Dim, typename CategoryT>
struct max_by_dim_impl;

/**
 * \brief Auxiliary class for computing the maximum elements over the given
 *  dimension tag for a container of the given category.
 * \tparam TagT The dimension tag type (e.g., tag::major).
 * \tparam CategoryT The category type (e.g., vector_tag).
 * \tparam OrientationT The orientation category type (e.g., row_major_tag).
 */
template <typename TagT, typename CategoryT, typename OrientationT>
struct max_by_tag_impl;

//@} Declarations


//@{ Definitions

template <typename T>
BOOST_UBLAS_INLINE
bool greater_than_impl(T a, T b)
{
    return a > b;
}


template <typename T>
BOOST_UBLAS_INLINE
bool greater_than_impl(::std::complex<T> const& a, ::std::complex<T> const& b)
{
    // For complex numbers compare modulus and phase angle
    // Use the same logic used by MATLAB for the 'max' function:
    // "For complex input A, max returns the complex number with the largest
    //  complex modulus (magnitude), computed with max(abs(A)). Then
    //  computes the largest phase angle with max(angle(x)), if necessary"

    const T ax(::std::abs(a));
    const T bx(::std::abs(b));
    return ax > bx 
           || (ax == bx && ::std::arg(a) > ::std::arg(b));
}


template <typename T>
struct minus_infinity
{
    static const T value;
};
template <typename T>
const T minus_infinity<T>::value = ::std::numeric_limits<T>::has_infinity ? -::std::numeric_limits<T>::infinity() : ::std::numeric_limits<T>::min();


template <typename T>
struct minus_infinity< ::std::complex<T> >
{
    static const ::std::complex<T> value;
};
template <typename T>
//const ::std::complex<T> minus_infinity< ::std::complex<T> >::value = ::std::complex<T>(minus_infinity<T>::value,::std::numeric_limits<T>::quiet_NaN());
const ::std::complex<T> minus_infinity< ::std::complex<T> >::value = ::std::complex<T>(T(0),T(0));


template <>
struct max_by_dim_impl<1, vector_tag>
{
    template <typename VectorExprT>
    BOOST_UBLAS_INLINE
    static vector<typename vector_traits<VectorExprT>::value_type> apply(vector_expression<VectorExprT> const& ve)
    {
        typedef typename vector_traits<VectorExprT>::value_type value_type;

        vector<value_type> res(1);

        res(0) = max(ve);

        return res;
    }
};


template <>
struct max_by_dim_impl<1, matrix_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_rows(me);
    }
};


template <>
struct max_by_dim_impl<2, matrix_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_columns(me);
    }
};


template <>
struct max_by_tag_impl<tag::major, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_rows(me);
    }
};


template <>
struct max_by_tag_impl<tag::minor, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_columns(me);
    }
};


template <>
struct max_by_tag_impl<tag::leading, matrix_tag, row_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_columns(me);
    }
};


template <>
struct max_by_tag_impl<tag::major, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_columns(me);
    }
};


template <>
struct max_by_tag_impl<tag::minor, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_rows(me);
    }
};


template <>
struct max_by_tag_impl<tag::leading, matrix_tag, column_major_tag>
{
    template <typename MatrixExprT>
    BOOST_UBLAS_INLINE
    static vector<typename matrix_traits<MatrixExprT>::value_type> apply(matrix_expression<MatrixExprT> const& me)
    {
        return max_rows(me);
    }
};


template <typename TagT>
struct max_by_tag_impl<TagT, matrix_tag, unknown_orientation_tag>: max_by_tag_impl<TagT, matrix_tag, row_major_tag>
{
    // Empty
};

//@} Definitions

}} // Namespace detail::<unnamed>


//@{ Definitions

template <typename VectorExprT>
BOOST_UBLAS_INLINE
typename vector_traits<VectorExprT>::value_type max(vector_expression<VectorExprT> const& ve)
{
    typedef typename vector_traits<VectorExprT>::size_type size_type;
    typedef typename vector_traits<VectorExprT>::value_type value_type;

    size_type n = size(ve);
//  value_type m = ::std::numeric_limits<value_type>::has_infinity
//                 ? - ::std::numeric_limits<value_type>::infinity()
//                 : ::std::numeric_limits<value_type>::min();
    value_type m = detail::minus_infinity<value_type>::value;

    for (size_type i = 0; i < n; ++i)
    {
//      if (ve()(i) > m)
        if (detail::greater_than_impl(ve()(i), m))
        {
            m = ve()(i);
        }
    }

    return m;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
typename matrix_traits<MatrixExprT>::value_type max(matrix_expression<MatrixExprT> const& me)
{
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    size_type nr = num_rows(me);
    size_type nc = num_columns(me);
//  value_type m = ::std::numeric_limits<value_type>::has_infinity
//                 ? - ::std::numeric_limits<value_type>::infinity()
//                 : ::std::numeric_limits<value_type>::min();
    value_type m = detail::minus_infinity<value_type>::value;

    for (size_type r = 0; r < nr; ++r)
    {
        for (size_type c = 0; c < nc; ++c)
        {
//          if (me()(r,c) > m)
            if (detail::greater_than_impl(me()(r,c), m))
            {
                m = me()(r,c);
            }
        }
    }

    return m;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> max_rows(matrix_expression<MatrixExprT> const& me)
{
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    size_type nr = num_rows(me);
    size_type nc = num_columns(me);

    vector<value_type> res(nr);
    size_type j = 0;
    for (size_type r = 0; r < nr; ++r)
    {
//      value_type m = ::std::numeric_limits<value_type>::has_infinity
//                     ? - ::std::numeric_limits<value_type>::infinity()
//                     : ::std::numeric_limits<value_type>::min();
        value_type m = detail::minus_infinity<value_type>::value;

        for (size_type c = 0; c < nc; ++c)
        {
//          if (me()(r,c) > m)
            if (detail::greater_than_impl(me()(r,c), m))
            {
                m = me()(r,c);
            }
        }

        res(j++) = m;
    }

    return res;
}


template <typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> max_columns(matrix_expression<MatrixExprT> const& me)
{
    typedef typename matrix_traits<MatrixExprT>::size_type size_type;
    typedef typename matrix_traits<MatrixExprT>::value_type value_type;

    size_type nr = num_rows(me);
    size_type nc = num_columns(me);

    vector<value_type> res(nc);
    size_type j = 0;
    for (size_type c = 0; c < nc; ++c)
    {
//      value_type m = ::std::numeric_limits<value_type>::has_infinity
//                     ? - ::std::numeric_limits<value_type>::infinity()
//                     : ::std::numeric_limits<value_type>::min();
        value_type m = detail::minus_infinity<value_type>::value;

        for (size_type r = 0; r < nr; ++r)
        {
//          if (me()(r,c) > m)
            if (detail::greater_than_impl(me()(r,c), m))
            {
                m = me()(r,c);
            }
        }

        res(j++) = m;
    }

    return res;
}


template <size_t Dim, typename VectorExprT>
BOOST_UBLAS_INLINE
vector<typename vector_traits<VectorExprT>::value_type> max(vector_expression<VectorExprT> const& ve)
{
    return detail::max_by_dim_impl<Dim, vector_tag>::template apply(ve);
}


template <size_t Dim, typename MatrixExprT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> max(matrix_expression<MatrixExprT> const& me)
{
    return detail::max_by_dim_impl<Dim, matrix_tag>::template apply(me);
}


template <typename TagT, typename MatrixExprT>
//template <typename MatrixExprT, typename TagT>
BOOST_UBLAS_INLINE
vector<typename matrix_traits<MatrixExprT>::value_type> max_by_tag(matrix_expression<MatrixExprT> const& me)
{
    return detail::max_by_tag_impl<TagT, matrix_tag, typename matrix_traits<MatrixExprT>::orientation_category>::template apply(me);
}

//@} Definitions

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_MAX_HPP
