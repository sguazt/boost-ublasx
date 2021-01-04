/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/hild.hpp
 *
 * \brief Hilbert matrix.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2012, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_HILB_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_HILB_HPP


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/matrix.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

template <typename T>
BOOST_UBLAS_INLINE
matrix<T> hilb(::std::size_t n)
{
    matrix<T> H(n,n);

    for (::std::size_t r = 0; r < n; ++r)
    {
        for (::std::size_t c = 0; c < n; ++c)
        {
            //H(r,c) = T(1.0)/((r+1)+(c+1)-T(1.0));
            H(r,c) = T(1.0)/(r+c+T(1.0));
        }
    }

    return H;
}


BOOST_UBLAS_INLINE
matrix<double> hilb(::std::size_t n, int=0)
{
    return hilb<double>(n);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_HILB_HPP
