/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/expression/vector_unary_functor.hpp
 *
 * \brief Vector expression for applying unary functors to a vector expression.
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

#ifndef BOOST_NUMERIC_UBLASX_EXPRESSION_VECTOR_UNARY_FUNCTOR_HPP
#define BOOST_NUMERIC_UBLASX_EXPRESSION_VECTOR_UNARY_FUNCTOR_HPP


#include <boost/function.hpp>
#include <boost/mpl/if.hpp>
#include <boost/numeric/ublas/detail/iterator.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_same.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief Vector expression for applying unary functors to a vector expression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ExprT, typename SignatureT>
class vector_unary_functor: public vector_expression<vector_unary_functor<ExprT, SignatureT> >
{
#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: class const_iterator;
    public: friend class const_iterator;
#endif

    private: typedef vector_unary_functor<ExprT, SignatureT> self_type;
    public: typedef ::boost::function<SignatureT> functor_type;
    //public: typedef ExprT expression_type;
    public: typedef typename ::boost::mpl::if_<
                                ::boost::is_same<
                                    functor_type,
                                    scalar_identity<typename ExprT::value_type>
                                >,
                                ExprT,
                                const ExprT
                >::type expression_type;
    //public: typedef typename ExprT::const_closure_type expression_closure_type;
    public: typedef typename ::boost::mpl::if_<
                                ::boost::is_const<expression_type>,
                                typename ExprT::const_closure_type,
                                typename ExprT::closure_type
                >::type expression_closure_type;
    public: typedef typename ExprT::size_type size_type;
    public: typedef typename ExprT::difference_type difference_type;
    public: typedef typename functor_type::result_type value_type;
    public: typedef value_type const_reference;
    //public: typedef const_reference reference;
    public: typedef typename ::boost::mpl::if_<
                                ::boost::is_same<
                                    functor_type,
                                    scalar_identity<value_type>
                                >,
                                typename ExprT::reference,
                                value_type
                >::type reference;
    public: typedef const self_type const_closure_type;
    public: typedef const_closure_type closure_type;
    public: typedef unknown_storage_tag storage_category;
    public: typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
    private: typedef typename ExprT::const_iterator const_subiterator_type;
    private: typedef const value_type *const_pointer;
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: typedef indexed_const_iterator<const_closure_type, typename const_subiterator_type::iterator_category> const_iterator;
    public: typedef const_iterator iterator;
#else
    public: typedef const_iterator iterator;
#endif

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
    public: using vector_expression<self_type>::operator();
#endif


    // Construction and destruction


    public: BOOST_UBLAS_INLINE vector_unary_functor(expression_type const& e, functor_type const& f)
        : e_(e),
          f_(f)
    {
        // Empty
    }


    // Accessors
    public: BOOST_UBLAS_INLINE size_type size() const
    {
        return e_.size();
    }


    // Expression accessors
    public: BOOST_UBLAS_INLINE expression_closure_type const& expression() const
    {
        return e_;
    }


    // Element access


    public: BOOST_UBLAS_INLINE const_reference operator()(size_type i) const
    {
        return f_(e_(i));
    }


    public: BOOST_UBLAS_INLINE reference operator()(size_type i)
    {
        // precondition: assignable only if functor is scalar_identity
        BOOST_STATIC_ASSERT(
            (boost::is_same< functor_type, scalar_identity<value_type> >::value)
        );

        return e_(i);
    }


    public: BOOST_UBLAS_INLINE const_reference operator[](size_type i) const
    {
        return f_(e_[i]);
    }


    public: BOOST_UBLAS_INLINE reference operator[](size_type i)
    {
        // precondition: assignable only if functor is scalar_identity
        BOOST_STATIC_ASSERT(
            (boost::is_same< functor_type, scalar_identity<value_type> >::value)
        );

        return e_ [i];
    }


    // Closure comparison
    public: BOOST_UBLAS_INLINE bool same_closure(vector_unary_functor const& vu) const
    {
        return (*this).expression().same_closure(vu.expression());
    }


    // Iterators


    // Element lookup
    public: BOOST_UBLAS_INLINE const_iterator find(size_type i) const
    {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
        const_subiterator_type it(e_.find(i));
        return const_iterator(*this, it.index());
#else
        return const_iterator(*this, e_.find(i));
#endif
    }


    public: BOOST_UBLAS_INLINE const_iterator begin() const
    {
        return find(0); 
    }


    public: BOOST_UBLAS_INLINE const_iterator end() const
    {
        return find(size());
    }


    public: BOOST_UBLAS_INLINE const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(end());
    }


    public: BOOST_UBLAS_INLINE const_reverse_iterator rend() const
    {
        return const_reverse_iterator(begin());
    }


    // Iterator types

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: class const_iterator: public container_const_reference<vector_unary_functor>,
                                  public iterator_base_traits<typename ExprT::const_iterator::iterator_category>::template iterator_base<const_iterator, value_type>::type
    {
        public: typedef typename ExprT::const_iterator::iterator_category iterator_category;
        public: typedef typename vector_unary_functor::difference_type difference_type;
        public: typedef typename vector_unary_functor::value_type value_type;
        public: typedef typename vector_unary_functor::const_reference reference;
        public: typedef typename vector_unary_functor::const_pointer pointer;


        // Construction and destruction


        public: BOOST_UBLAS_INLINE const_iterator()
            : container_const_reference<self_type>(),
              it_()
        {
            // Empty
        }


        public: BOOST_UBLAS_INLINE const_iterator(self_type const& vu, const_subiterator_type const& it)
            : container_const_reference<self_type>(vu),
              it_(it)
        {
            // Empty
        }


        // Arithmetic


        public: BOOST_UBLAS_INLINE const_iterator& operator++()
        {
            ++it_;

            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator--()
        {
            --it_;

            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator+=(difference_type n)
        {
            it_ += n;

            return *this;
        }


        public: BOOST_UBLAS_INLINE const_iterator& operator-=(difference_type n)
        {
            it_ -= n;

            return *this;
        }


        public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator const& it) const
        {
            BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

            return it_ - it.it_;
        }


        // Dereference


        public: BOOST_UBLAS_INLINE const_reference operator*() const
        {
            return (*this)().f_(*it_);
        }


        public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
        {
            return *(*this + n);
        }


        // Index
        public: BOOST_UBLAS_INLINE size_type index() const
        {
            return it_.index();
        }

        // Assignment
        public: BOOST_UBLAS_INLINE const_iterator& operator=(const_iterator const& it)
        {
            container_const_reference<self_type>::assign(&it());
            it_ = it.it_;

            return *this;
        }


        // Comparisons


        public: BOOST_UBLAS_INLINE bool operator==(const_iterator const& it) const
        {
            BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

            return it_ == it.it_;
        }


        public: BOOST_UBLAS_INLINE bool operator<(const_iterator const& it) const
        {
            BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

            return it_ < it.it_;
        }


        private: const_subiterator_type it_;
    };
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR


    private: expression_closure_type e_;
    private: functor_type f_;
};


template <typename ExprT, typename SignatureT>
struct vector_unary_functor_traits
{
    typedef vector_unary_functor<ExprT,SignatureT> expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
    typedef expression_type result_type;
#else
    typedef typename ExprT::vector_temporary_type result_type;
#endif
};

}}} // Namespace boost::numeric::ublasx

#endif // BOOST_NUMERIC_UBLASX_EXPRESSION_VECTOR_UNARY_FUNCTOR_HPP
