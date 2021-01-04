/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/container/sequence_vector.hpp
 *
 * \brief Sequence-based vector.
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


#ifndef BOOST_NUMERIC_UBLASX_CONTAINER_SEQUENCE_VECTOR_HPP
#define BOOST_NUMERIC_UBLASX_CONTAINER_SEQUENCE_VECTOR_HPP


#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
#include <boost/numeric/ublas/detail/iterator.hpp>
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>
#include <cstddef>
#include <iterator>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief A vector defined by a sequence.
 *
 * \tparam SizeT The sequence size type.
 * \tparam DifferenceT The sequence difference type.
 * \tparam AllocT The allocator type.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT = long,
    typename StrideT = ::std::ptrdiff_t,
    typename AllocT = ::std::allocator<ValueT>
>
class sequence_vector: public vector_container< sequence_vector<ValueT,StrideT,AllocT> >
{
    private: typedef sequence_vector<ValueT,StrideT,AllocT> self_type;
    private: typedef vector_container<self_type> base_type;
    private: typedef ValueT const* const_pointer;
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
    public: vector_container<self_type>::operator();
#endif
    public: typedef typename AllocT::size_type size_type;
    public: typedef typename AllocT::difference_type difference_type;
    public: typedef ValueT value_type;
    public: typedef StrideT stride_type;
    //public: typedef SizeT const& const_reference;
    public: typedef value_type const_reference;
    //public: typedef SizeT& reference;
    public: typedef const_reference reference;
    public: typedef const vector_reference<const self_type> const_closure_type;
    public: typedef vector_reference<self_type> closure_type;
    public: typedef sparse_tag storage_category;
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: typedef indexed_const_iterator<self_type, ::std::random_access_iterator_tag> const_iterator;
#else
    public: class const_iterator;
#endif
    public: typedef ::std::reverse_iterator<const_iterator> const_reverse_iterator;


    public: BOOST_UBLAS_INLINE sequence_vector()
        : base_type()
    {
    }


    public: BOOST_UBLAS_INLINE sequence_vector(value_type start, stride_type stride, size_type size)
        : base_type(),
          start_(start),
          stride_(stride),
          size_(size)
    {
    }


    public: BOOST_UBLAS_INLINE sequence_vector(value_type start, size_type size)
        : base_type(),
          start_(start),
          stride_(1),
          size_(size)
    {
    }


    public: template <typename S, typename D>
        BOOST_UBLAS_INLINE sequence_vector(basic_range<S,D> const& r)
        : base_type(),
          start_(r.start()),
          stride_(1),
          size_(r.size())
    {
    }


    public: template <typename S, typename D>
        BOOST_UBLAS_INLINE sequence_vector(basic_slice<S,D> const& s)
        : base_type(),
          start_(s.start()),
          stride_(s.stride()),
          size_(s.size())
    {
    }


    public: BOOST_UBLAS_INLINE sequence_vector(sequence_vector const& v)
        : base_type(),
          start_(v.start_),
          stride_(v.stride_),
          size_(v.size_)
    {
    }


    public: BOOST_UBLAS_INLINE size_type size() const
    {
        return size_;
    }


    public: BOOST_UBLAS_INLINE void resize(size_type size, bool /*preserve*/ = true)
    {
        size_ = size;
    }


//FIXME: how to avoid of returning a temp reference
//  public: BOOST_UBLAS_INLINE const_pointer find_element(size_type i) const
//  {
//      return &(sequence_[i]);
//  }


    public: BOOST_UBLAS_INLINE const_reference operator()(size_type i) const
    {
        // precondition: i < size_
        BOOST_UBLAS_CHECK(i < size_, bad_index());

        return start_+stride_*i;
    }


    public: BOOST_UBLAS_INLINE const_reference operator[](size_type i) const
    {
        return operator()(i);
    }


    public: BOOST_UBLAS_INLINE void clear()
    {
        start_ = 0;
        stride_ = 0;
        size_ = 0;
    }


#ifdef BOOST_UBLAS_MOVE_SEMANTICS
    public: BOOST_UBLAS_INLINE sequence_vector& operator=(sequence_vector v)
    {
        assign_temporary(v);
        return *this;
    }
#else
    public: BOOST_UBLAS_INLINE sequence_vector& operator=(sequence_vector const& v)
    {
        if (this != &v)
        {
            start_ = v.start_;
            stride_ = v.stride_;
            size_ = v.size_;
        }
        return *this;
    }
#endif // BOOST_UBLAS_MOVE_SEMANTICS


    public: BOOST_UBLAS_INLINE sequence_vector& assign_temporary(sequence_vector& v)
    {
        swap(v);
        return *this;
    }


    public: BOOST_UBLAS_INLINE void swap(sequence_vector& v)
    {
        if (this != &v)
        {
            ::std::swap(start_, v.start_);
            ::std::swap(stride_, v.stride_);
            ::std::swap(size_, v.size_);
        }
    }


    public: BOOST_UBLAS_INLINE friend void swap(sequence_vector& v1, sequence_vector& v2)
    {
        v1.swap(v2);
    }


    public: BOOST_UBLAS_INLINE const_iterator find(size_type i) const
    {
        //TODO
        return begin() + i;
    }


    public: BOOST_UBLAS_INLINE const_iterator begin() const
    {
        return const_iterator(*this, 0);
    }


    public: BOOST_UBLAS_INLINE const_iterator end() const
    {
        return const_iterator(*this, size_);
    }


    public: BOOST_UBLAS_INLINE const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(*this, 0);
    }


    public: BOOST_UBLAS_INLINE const_reverse_iterator rend() const
    {
        return const_reverse_iterator(*this, size_);
    }


//TODO
//  public: template <typename ArchiveT>
//      void serialize(ArchiveT& ar, const unsigned int /*file_version*/)
//  {
//      serialization::collection_size_type s(sequence_);
//      ar & serialization::make_nvp("sequence",s);
//      if (ArchiveT::is_loading::value)
//      {
//          sequence_ = s;
//      }
//  }


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
    private: typedef size_type const_subiterator_type;


    public: class const_iterator:   public container_const_reference<sequence_vector>,
                                    public random_access_iterator_base<
                                        ::std::random_access_iterator_tag,
                                        const_iterator,
                                        value_type
                                    >
    {
        public: typedef typename sequence_vector::value_type value_type;
        public: typedef typename sequence_vector::difference_type difference_type;
        public: typedef typename sequence_vector::const_reference reference;
        public: typedef typename sequence_vector::const_pointer pointer;

        public: BOOST_UBLAS_INLINE const_iterator()
            : container_const_reference<sequence_vector>(),
              it_()
        {
        }


        public: BOOST_UBLAS_INLINE const_iterator(sequence_vector const& s, const_subiterator_type const& it)
            : container_const_reference<sequence_vector>(s),
              it_(it)
        {
        }


        //@{ Arithmetic

        public: BOOST_UBLAS_INLINE const_iterator& operator++()
        {
            ++it_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator--()
        {
            BOOST_UBLAS_CHECK (it_ > 0, bad_index ());
            --it_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator+=(difference_type n)
        {
            BOOST_UBLAS_CHECK(n >= 0 || it_ >= size_type(-n), bad_index());
            it_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator-=(difference_type n)
        {
            BOOST_UBLAS_CHECK(n <= 0 || it_ >= size_type(n), bad_index());
            it_ -= n;
            return *this;
        }

        public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator const& it) const
        {
            return it_ - it.it_;
        }

        //@} Arithmetic


        //@{ Dereference

        public: BOOST_UBLAS_INLINE const_reference operator*() const
        {
            BOOST_UBLAS_CHECK(it_ < (*this)().size (), bad_index());
            return (*this)().start() + it_*(*this)().stride();
        }

        public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
        {
            return *(*this + n);
        }

        //@} Dereference


        //@{ Index

        public: BOOST_UBLAS_INLINE size_type index() const
        {
            BOOST_UBLAS_CHECK(it_ < (*this)().size(), bad_index());
            return it_;
        }

        //@} Index


        //@{ Assignement

        public: BOOST_UBLAS_INLINE const_iterator& operator=(const_iterator const& it)
        {
            // Comeau recommends...
            this->assign(&it());
            it_ = it.it_;
            return *this;
        }

        //@} Assignement


        //@{ Comparison

        public: BOOST_UBLAS_INLINE bool operator==(const_iterator const& it) const
        {
            BOOST_UBLAS_CHECK((*this)() == it(), external_logic());
            return it_ == it.it_;
        }


        public: BOOST_UBLAS_INLINE bool operator<(const_iterator const& it) const
        {
            BOOST_UBLAS_CHECK((*this)() == it(), external_logic());
            return it_ < it.it_;
        }

        //@} Comparison

        private: const_subiterator_type it_;
    };
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR

    private: value_type start_;
    private: stride_type stride_;
    private: size_type size_;
};

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_CONTAINER_SEQUENCE_VECTOR_HPP
