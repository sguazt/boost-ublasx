/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/eye.hpp
 *
 * \brief The \c eye operation.
 *
 * The \c eye operation creates an identity matrix.
 * This operation takes inspiration from the MATLAB's \e eye function
 * and the Mathematica's \e IdentityMatrix function.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2021, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLAS_EYE_HPP
#define BOOST_NUMERIC_UBLAS_EYE_HPP

//#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <cstddef>
#include <memory>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

#if __cplusplus > 199711L
// C++0x allows default template parameters for functions

/**
 * \brief Create an unmutable identity square matrix.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...).
 *  By default, an integral type is used.
 * \tparam AllocT An allocator for storing the zeros and one elements.
 *  By default, a standard allocator is used.
 *
 * \param n The number of rows and columns of the resulting identity matrix.
 * \return An unmutable n-by-n identity matrix.
 */
template<typename ValueT = int, // The same default value used by the `identity_matrix` type.
         typename AllocT = std::allocator<ValueT> > // The same default allocator used by the `identity_matrix` type.
BOOST_UBLAS_INLINE
identity_matrix<ValueT,AllocT> eye(std::size_t n)
{
    return identity_matrix<ValueT,AllocT>(n);
}

/**
 * \brief Create an unmutable identity rectangular matrix.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...)
 *  By default, an integral type is used.
 * \tparam AllocT An allocator for storing the zeros and one elements.
 *  By default, a standard allocator is used.
 *
 * \param nr The number of rows of the resulting identity matrix.
 * \param nc The number of columns of the resulting identity matrix.
 * \return An unmutable nr-by-nc identity matrix.
 */
template<typename ValueT = int, // The same default value used by the `identity_matrix` type.
         typename AllocT = std::allocator<ValueT> > // The same default allocator used by the `identity_matrix` type.
BOOST_UBLAS_INLINE
identity_matrix<ValueT,AllocT> eye(std::size_t nr, std::size_t nc)
{
    return identity_matrix<ValueT,AllocT>(nr, nc);
}

#else // __cplusplus

// C++98 does not allow default template parameters for functions

/**
 * \brief Create an unmutable identity square matrix.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...).
 * \tparam AllocT An allocator for storing the zeros and one elements.
 *
 * \param n The number of rows and columns of the resulting identity matrix.
 * \return An unmutable n-by-n identity matrix.
 */
template<typename ValueT,
         typename AllocT>
BOOST_UBLAS_INLINE
identity_matrix<ValueT,AllocT> eye(std::size_t n)
{
    return identity_matrix<ValueT,AllocT>(n);
}

/**
 * \brief Create an unmutable identity square matrix with a default allocator.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...).
 *
 * \param n The number of rows and columns of the resulting identity matrix.
 * \return An unmutable n-by-n identity matrix.
 */
template<typename ValueT>
BOOST_UBLAS_INLINE
identity_matrix<ValueT> eye(std::size_t n)
{
    return identity_matrix<ValueT>(n);
}

/**
 * \brief Create an unmutable identity rectangular matrix.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...)
 *  By default, an integral type is used.
 * \tparam AllocT An allocator for storing the zeros and one elements.
 *  By default, a standard allocator is used.
 *
 * \param nr The number of rows of the resulting identity matrix.
 * \param nc The number of columns of the resulting identity matrix.
 * \return An unmutable nr-by-nc identity matrix.
 */
template<typename ValueT,
         typename AllocT>
BOOST_UBLAS_INLINE
identity_matrix<ValueT,AllocT> eye(std::size_t nr, std::size_t nc)
{
    return identity_matrix<ValueT,AllocT>(nr, nc);
}

/**
 * \brief Create an unmutable identity rectangular matrix with a default allocator.
 *
 * \tparam ValueT The type of object stored in the matrix (like double, float, complex, etc...)
 *  By default, an integral type is used.
 *
 * \param nr The number of rows of the resulting identity matrix.
 * \param nc The number of columns of the resulting identity matrix.
 * \return An unmutable nr-by-nc identity matrix.
 */
template<typename ValueT>
BOOST_UBLAS_INLINE
identity_matrix<ValueT> eye(std::size_t nr, std::size_t nc)
{
    return identity_matrix<ValueT>(nr, nc);
}

#endif // __cplusplus

/**
 * \brief Create an unmutable identity square matrix with default value type and allocator.
 *
 * \param n The number of rows and columns of the resulting identity matrix.
 * \return An unmutable n-by-n identity matrix.
 */
BOOST_UBLAS_INLINE
identity_matrix<> eye(std::size_t n)
{
    return identity_matrix<>(n);
}

/**
 * \brief Create an unmutable identity rectangular matrix with default value type and allocator.
 *
 * \param nr The number of rows of the resulting identity matrix.
 * \param nc The number of columns of the resulting identity matrix.
 * \return An unmutable nr-by-nc identity matrix.
 */
BOOST_UBLAS_INLINE
identity_matrix<> eye(std::size_t nr, std::size_t nc)
{
    return identity_matrix<>(nr, nc);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_EYE_HPP
