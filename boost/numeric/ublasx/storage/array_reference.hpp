/**
 * \file boost/numeric/ublasx/storage/array_reference.hpp
 *
 * \brief Storage class representing a reference to an array.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_STORAGE_ARRAY_REFERENCE_HPP
#define BOOST_NUMERIC_UBLASX_STORAGE_ARRAY_REFERENCE_HPP


#include <boost/numeric/ublas/storage.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief Storage class representing a reference to an array.
 *
 * \tparam ArrayT The type of the array to be referenced.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ArrayT>
class array_reference: public storage_array< array_reference<ArrayT> >
{
	private: typedef ArrayT array_type;
	public: typedef typename array_type::value_type value_type;
	public: typedef typename array_type::size_type size_type;
	public: typedef typename array_type::difference_type difference_type;
	public: typedef typename array_type::reference reference;
	public: typedef typename array_type::const_reference const_reference;
	public: typedef typename array_type::iterator iterator;
	public: typedef typename array_type::const_iterator const_iterator;
	public: typedef typename array_type::reverse_iterator reverse_iterator;
	public: typedef typename array_type::const_reverse_iterator const_reverse_iterator;


	/// A constructor
	public: explicit array_reference(array_type& data)
		: data_(data)
	{
	}


	/// Return the size of the referenced array.
	public: size_type size() const
	{
		return data_.size();
	}


	/// Element access: a(i)
	public: const_reference operator[](size_type i) const
	{
		return data_[i];
	}


	/// Element access: a[i]
	public: reference operator[](size_type i)
	{
		return data_[i];
	}


	/// Iterator to the first element of the referenced array.   
	public: const_iterator begin() const
	{
		return data_.begin();
	}


	/// Iterator after the last element of the referenced array.   
	public: const_iterator end() const
	{
		return data_.end();
	}


	public: iterator begin()
	{
		return data_.begin();
	}


	public: iterator end()
	{
		return data_.end();
	}


	public: const_reverse_iterator rbegin() const
	{
		return data_.rbegin();
	}


	public: const_reverse_iterator rend() const
	{
		return data_.rend();
	}


	public: reverse_iterator rbegin()
	{
		return data_.rbegin();
	}


	public: reverse_iterator rend()
	{
		return data_.rend();
	}


	private: array_type &data_;
};

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLAS_STORAGE_ARRAY_REFERENCE_HPP
