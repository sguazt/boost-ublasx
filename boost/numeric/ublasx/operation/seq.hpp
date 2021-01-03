/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/seq.hpp
 *
 * \brief Create a vector sequence.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_SEQ_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_SEQ_HPP


#include <boost/numeric/ublasx/container/sequence_vector.hpp>
#include <boost/numeric/ublas/storage.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace boost::numeric::ublas;


template <typename ValueT, typename SizeT>
sequence_vector<ValueT> seq(ValueT from, SizeT size)
{
    return sequence_vector<ValueT>(from, size);
}


template <typename ValueT, typename StrideT, typename SizeT>
sequence_vector<ValueT, StrideT> seq(ValueT from, StrideT stride, SizeT size)
{
    return sequence_vector<ValueT, StrideT>(from, stride, size);
}


sequence_vector<> seq(sequence_vector<>::value_type from, sequence_vector<>::size_type size)
{
    return sequence_vector<>(from, size);
}


sequence_vector<> seq(sequence_vector<>::value_type from, sequence_vector<>::stride_type stride, sequence_vector<>::size_type size)
{
    return sequence_vector<>(from, stride, size);
}


template <typename SizeT, typename DifferenceT>
sequence_vector<SizeT,DifferenceT> seq(basic_range<SizeT,DifferenceT> const& r)
{
    return sequence_vector<SizeT,DifferenceT>(r);
}


template <typename SizeT, typename DifferenceT>
sequence_vector<SizeT,DifferenceT> seq(basic_slice<SizeT,DifferenceT> const& s)
{
    return sequence_vector<SizeT,DifferenceT>(s);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_SEQ_HPP
