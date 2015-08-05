/**
 * \file boost/numeric/ublasx/expression/matrix_binary_functor.hpp
 *
 * \brief Matrix expression for applying binary functors to a matrix expression.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_EXPRESSION_MATRIX_BINARY_FUNCTOR_HPP
#define BOOST_NUMERIC_UBLASX_EXPRESSION_MATRIX_BINARY_FUNCTOR_HPP


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
 * \brief Matrix expression for applying binary functors to a matrix expression
 *  and where the matrix expression is the left operand.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename ExprT, typename Arg2T, typename SignatureT>
class matrix_binary_functor1: public matrix_expression< matrix_binary_functor1<ExprT, Arg2T, SignatureT> >
{
#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: class const_iterator1;
	public: friend class const_iterator1;
	public: class const_iterator2;
	public: friend class const_iterator2;
#endif


	private: typedef matrix_binary_functor1<ExprT, Arg2T, SignatureT> self_type;
	//typedef F functor_type;
	public: typedef ::boost::function<SignatureT> functor_type;
	public: typedef Arg2T arg2_type;
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
	public: typedef const_reference reference;
	public: typedef const self_type const_closure_type;
	public: typedef const_closure_type closure_type;
	public: typedef typename ExprT::orientation_category orientation_category;
	public: typedef unknown_storage_tag storage_category;
	private: typedef typename ExprT::const_iterator1 const_subiterator1_type;
	private: typedef typename ExprT::const_iterator2 const_subiterator2_type;
	private: typedef const value_type *const_pointer;
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: typedef indexed_const_iterator1<const_closure_type, typename const_subiterator1_type::iterator_category> const_iterator1;
	public: typedef const_iterator1 iterator1;
	public: typedef indexed_const_iterator2<const_closure_type, typename const_subiterator2_type::iterator_category> const_iterator2;
	public: typedef const_iterator2 iterator2;
#else
	public: typedef const_iterator1 iterator1;
	public: typedef const_iterator2 iterator2;
#endif
	public: typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	public: typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	public: using matrix_expression<self_type>::operator();
#endif


	// Construction and destruction


	public: BOOST_UBLAS_INLINE matrix_binary_functor1(expression_type const& e, arg2_type const& arg2, functor_type const& f)
 		: e_(e),
		  a2_(arg2),
		  f_(f)
	{
		// Empty
	}


	// Accessors


	public: BOOST_UBLAS_INLINE size_type size1() const
	{
		return e_.size1();
	}


	public: BOOST_UBLAS_INLINE size_type size2() const
	{
		return e_.size2();
	}


	// Expression accessors
	public: BOOST_UBLAS_INLINE expression_closure_type const& expression() const
	{
		return e_;
	}


	// Element access


	public: BOOST_UBLAS_INLINE const_reference operator()(size_type i, size_type j) const
	{
		return f_(e_(i, j), a2_);
	}


	public: BOOST_UBLAS_INLINE reference operator()(size_type i, size_type j)
	{
		//return f_(e_(i, j), a2_);
		return e_(i, j);
	}


	// Closure comparison
	public: BOOST_UBLAS_INLINE bool same_closure(matrix_binary_functor1 const& mu1) const
	{
		return (*this).expression().same_closure(mu1.expression());
	}


	// Element lookup


	public: BOOST_UBLAS_INLINE const_iterator1 find1(int rank, size_type i, size_type j) const
	{
		const_subiterator1_type it1(e_.find1(rank, i, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, it1.index1(), it1.index2());
#else
		return const_iterator1(*this, it1);
#endif
	}


	public: BOOST_UBLAS_INLINE const_iterator2 find2(int rank, size_type i, size_type j) const
	{
		const_subiterator2_type it2(e_.find2(rank, i, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, it2.index1(), it2.index2());
#else
		return const_iterator2(*this, it2);
#endif
	}


	// Iterators


	public: BOOST_UBLAS_INLINE const_iterator1 begin1() const
	{
		return find1(0, 0, 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator1 end1() const
	{
		return find1(0, size1(), 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator2 begin2() const
	{
		return find2(0, 0, 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator2 end2() const
	{
		return find2(0, 0, size2());
	}


	// Reverse iterators


	public: BOOST_UBLAS_INLINE const_reverse_iterator1 rbegin1 () const
	{
		return const_reverse_iterator1(end1());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator1 rend1() const
	{
		return const_reverse_iterator1(begin1());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator2 rbegin2 () const
	{
		return const_reverse_iterator2(end2());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator2 rend2() const
	{
		return const_reverse_iterator2(begin2());
	}


	// Iterator types


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: class const_iterator1: public container_const_reference<matrix_binary_functor1>,
								   public iterator_base_traits<typename ExprT::const_iterator1::iterator_category>::template iterator_base<const_iterator1, value_type>::type
	{
		public: typedef typename ExprT::const_iterator1::iterator_category iterator_category;
		public: typedef typename matrix_binary_functor1::difference_type difference_type;
		public: typedef typename matrix_binary_functor1::value_type value_type;
		public: typedef typename matrix_binary_functor1::const_reference reference;
		public: typedef typename matrix_binary_functor1::const_pointer pointer;

		public: typedef const_iterator2 dual_iterator_type;
		public: typedef const_reverse_iterator2 dual_reverse_iterator_type;


		// Construction and destruction


		public: BOOST_UBLAS_INLINE const_iterator1()
			: container_const_reference<self_type>(),
			  it_()
		{
			// Empty
		}


		public: BOOST_UBLAS_INLINE const_iterator1(self_type const& mu, const_subiterator1_type const& it)
			: container_const_reference<self_type>(mu),
			  it_(it)
		{
			// Empty
		}


		// Arithmetic


		public: BOOST_UBLAS_INLINE const_iterator1& operator++()
		{
			++it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator--()
		{
			--it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator+=(difference_type n)
		{
			it_ += n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator-=(difference_type n)
		{
			it_ -= n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ - it.it_;
		}


		// Dereference


		public: BOOST_UBLAS_INLINE const_reference operator*() const
		{
			return (*this)().f_(*it_, (*this)().a2_);
		}


		public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 begin() const
		{
			return (*this)().find2(1, index1(), 0);
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 end() const
		{
			return (*this)().find2(1, index1(), (*this)().size2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rbegin() const
		{
			return const_reverse_iterator2(end());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rend() const
		{
			return const_reverse_iterator2(begin());
		}
#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


		// Indices


		public: BOOST_UBLAS_INLINE size_type index1() const
		{
			return it_.index1();
		}


		public: BOOST_UBLAS_INLINE size_type index2() const
		{
			return it_.index2();
		}


		// Assignment 
		public: BOOST_UBLAS_INLINE const_iterator1& operator=(const_iterator1 const& it)
		{
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;

			return *this;
		}


		// Comparisons


		public: BOOST_UBLAS_INLINE bool operator==(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ == it.it_;
		}


		public: BOOST_UBLAS_INLINE bool operator<(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ < it.it_;
		}


		private: const_subiterator1_type it_;
	};


	public: class const_iterator2: public container_const_reference<matrix_binary_functor1>,
								   public iterator_base_traits<typename ExprT::const_iterator2::iterator_category>::template iterator_base<const_iterator2, value_type>::type
	{
		public: typedef typename ExprT::const_iterator2::iterator_category iterator_category;
		public: typedef typename matrix_binary_functor1::difference_type difference_type;
		public: typedef typename matrix_binary_functor1::value_type value_type;
		public: typedef typename matrix_binary_functor1::const_reference reference;
		public: typedef typename matrix_binary_functor1::const_pointer pointer;

		public: typedef const_iterator1 dual_iterator_type;
		public: typedef const_reverse_iterator1 dual_reverse_iterator_type;


		// Construction and destruction


		public: BOOST_UBLAS_INLINE const_iterator2()
			: container_const_reference<self_type>(),
			  it_()
		{
			// Empty
		}


		public: BOOST_UBLAS_INLINE const_iterator2(self_type const& mu, const_subiterator2_type const& it)
			: container_const_reference<self_type>(mu),
			  it_(it)
		{
			// Empty
		}


		// Arithmetic


		public: BOOST_UBLAS_INLINE const_iterator2& operator++()
		{
			++it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator--()
		{
			--it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator+=(difference_type n)
		{
			it_ += n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator-=(difference_type n)
		{
			it_ -= n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ - it.it_;
		}


		// Dereference


		public: BOOST_UBLAS_INLINE const_reference operator*() const
		{
			return (*this)().f_(*it_, (*this)().a2_);
		}


		public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 begin() const
		{
			return (*this)().find1(1, 0, index2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 end() const
		{
			return (*this)().find1(1, (*this)().size1(), index2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rbegin() const
		{
			return const_reverse_iterator1(end());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rend() const
		{
			return const_reverse_iterator1(begin());
		}
#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


		// Indices


		public: BOOST_UBLAS_INLINE size_type index1() const
		{
			return it_.index1();
		}


		public: BOOST_UBLAS_INLINE size_type index2() const
		{
			return it_.index2();
		}


		// Assignment 
		public: BOOST_UBLAS_INLINE const_iterator2& operator=(const_iterator2 const& it)
		{
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;

			return *this;
		}


		// Comparisons


		public: BOOST_UBLAS_INLINE bool operator==(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ == it.it_;
		}


		public: BOOST_UBLAS_INLINE bool operator<(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ < it.it_;
		}


		private: const_subiterator2_type it_;
	};
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR


	private: expression_closure_type e_;
	private: arg2_type a2_;
	private: functor_type f_;
}; // matrix_binary_functor1

template <typename ExprT, typename Arg2T, typename SignatureT>
struct matrix_binary_functor1_traits
{
	typedef matrix_binary_functor1<ExprT, Arg2T, SignatureT> expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type; 
#else
	typedef typename ExprT::matrix_temporary_type result_type;
#endif
}; // matrix_binary_functor1_traits


/**
 * \brief Matrix expression for applying binary functors to a matrix expression
 *  and where the matrix expression is the right operand.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename Arg1T, typename ExprT, typename SignatureT>
class matrix_binary_functor2: public matrix_expression< matrix_binary_functor2<Arg1T, ExprT, SignatureT> >
{
#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: class const_iterator1;
	public: friend class const_iterator1;
	public: class const_iterator2;
	public: friend class const_iterator2;
#endif


	private: typedef matrix_binary_functor2<Arg1T, ExprT, SignatureT> self_type;
	//typedef F functor_type;
	public: typedef ::boost::function<SignatureT> functor_type;
	public: typedef Arg1T arg1_type;
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
	public: typedef const_reference reference;
	public: typedef const self_type const_closure_type;
	public: typedef const_closure_type closure_type;
	public: typedef typename ExprT::orientation_category orientation_category;
	public: typedef unknown_storage_tag storage_category;
	private: typedef typename ExprT::const_iterator1 const_subiterator1_type;
	private: typedef typename ExprT::const_iterator2 const_subiterator2_type;
	private: typedef const value_type *const_pointer;
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: typedef indexed_const_iterator1<const_closure_type, typename const_subiterator1_type::iterator_category> const_iterator1;
	public: typedef const_iterator1 iterator1;
	public: typedef indexed_const_iterator2<const_closure_type, typename const_subiterator2_type::iterator_category> const_iterator2;
	public: typedef const_iterator2 iterator2;
#else
	public: typedef const_iterator1 iterator1;
	public: typedef const_iterator2 iterator2;
#endif
	public: typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	public: typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	public: using matrix_expression<self_type>::operator();
#endif


	// Construction and destruction


	public: BOOST_UBLAS_INLINE matrix_binary_functor2(arg1_type const& arg1, expression_type const& e, functor_type const& f)
 		: a1_(arg1),
		  e_(e),
		  f_(f)
	{
		// Empty
	}


	// Accessors


	public: BOOST_UBLAS_INLINE size_type size1() const
	{
		return e_.size1();
	}


	public: BOOST_UBLAS_INLINE size_type size2() const
	{
		return e_.size2();
	}


	// Expression accessors
	public: BOOST_UBLAS_INLINE expression_closure_type const& expression() const
	{
		return e_;
	}


	// Element access


	public: BOOST_UBLAS_INLINE const_reference operator()(size_type i, size_type j) const
	{
		return f_(a1_, e_(i, j));
	}


	public: BOOST_UBLAS_INLINE reference operator()(size_type i, size_type j)
	{
		//return f_(a1_, e_(i, j));
		return e_(i, j);
	}


	// Closure comparison
	public: BOOST_UBLAS_INLINE bool same_closure(matrix_binary_functor2 const& mu1) const
	{
		return (*this).expression().same_closure(mu1.expression());
	}


	// Element lookup


	public: BOOST_UBLAS_INLINE const_iterator1 find1(int rank, size_type i, size_type j) const
	{
		const_subiterator1_type it1(e_.find1(rank, i, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, it1.index1(), it1.index2());
#else
		return const_iterator1(*this, it1);
#endif
	}


	public: BOOST_UBLAS_INLINE const_iterator2 find2(int rank, size_type i, size_type j) const
	{
		const_subiterator2_type it2(e_.find2(rank, i, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, it2.index1(), it2.index2());
#else
		return const_iterator2(*this, it2);
#endif
	}


	// Iterators


	public: BOOST_UBLAS_INLINE const_iterator1 begin1() const
	{
		return find1(0, 0, 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator1 end1() const
	{
		return find1(0, size1(), 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator2 begin2() const
	{
		return find2(0, 0, 0);
	}


	public: BOOST_UBLAS_INLINE const_iterator2 end2() const
	{
		return find2(0, 0, size2());
	}


	// Reverse iterators


	public: BOOST_UBLAS_INLINE const_reverse_iterator1 rbegin1 () const
	{
		return const_reverse_iterator1(end1());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator1 rend1() const
	{
		return const_reverse_iterator1(begin1());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator2 rbegin2 () const
	{
		return const_reverse_iterator2(end2());
	}


	public: BOOST_UBLAS_INLINE const_reverse_iterator2 rend2() const
	{
		return const_reverse_iterator2(begin2());
	}


	// Iterator types


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	public: class const_iterator1: public container_const_reference<matrix_binary_functor2>,
								   public iterator_base_traits<typename ExprT::const_iterator1::iterator_category>::template iterator_base<const_iterator1, value_type>::type
	{
		public: typedef typename ExprT::const_iterator1::iterator_category iterator_category;
		public: typedef typename matrix_binary_functor2::difference_type difference_type;
		public: typedef typename matrix_binary_functor2::value_type value_type;
		public: typedef typename matrix_binary_functor2::const_reference reference;
		public: typedef typename matrix_binary_functor2::const_pointer pointer;

		public: typedef const_iterator2 dual_iterator_type;
		public: typedef const_reverse_iterator2 dual_reverse_iterator_type;


		// Construction and destruction


		public: BOOST_UBLAS_INLINE const_iterator1()
			: container_const_reference<self_type>(),
			  it_()
		{
			// Empty
		}


		public: BOOST_UBLAS_INLINE const_iterator1(self_type const& mu, const_subiterator1_type const& it)
			: container_const_reference<self_type>(mu),
			  it_(it)
		{
			// Empty
		}


		// Arithmetic


		public: BOOST_UBLAS_INLINE const_iterator1& operator++()
		{
			++it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator--()
		{
			--it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator+=(difference_type n)
		{
			it_ += n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator1& operator-=(difference_type n)
		{
			it_ -= n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ - it.it_;
		}


		// Dereference


		public: BOOST_UBLAS_INLINE const_reference operator*() const
		{
			return (*this)().f_((*this)().a1_, *it_);
		}


		public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 begin() const
		{
			return (*this)().find2(1, index1(), 0);
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 end() const
		{
			return (*this)().find2(1, index1(), (*this)().size2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rbegin() const
		{
			return const_reverse_iterator2(end());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rend() const
		{
			return const_reverse_iterator2(begin());
		}
#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


		// Indices


		public: BOOST_UBLAS_INLINE size_type index1() const
		{
			return it_.index1();
		}


		public: BOOST_UBLAS_INLINE size_type index2() const
		{
			return it_.index2();
		}


		// Assignment 
		public: BOOST_UBLAS_INLINE const_iterator1& operator=(const_iterator1 const& it)
		{
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;

			return *this;
		}


		// Comparisons


		public: BOOST_UBLAS_INLINE bool operator==(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ == it.it_;
		}


		public: BOOST_UBLAS_INLINE bool operator<(const_iterator1 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ < it.it_;
		}


		private: const_subiterator1_type it_;
	};


	public: class const_iterator2: public container_const_reference<matrix_binary_functor2>,
								   public iterator_base_traits<typename ExprT::const_iterator2::iterator_category>::template iterator_base<const_iterator2, value_type>::type
	{
		public: typedef typename ExprT::const_iterator2::iterator_category iterator_category;
		public: typedef typename matrix_binary_functor2::difference_type difference_type;
		public: typedef typename matrix_binary_functor2::value_type value_type;
		public: typedef typename matrix_binary_functor2::const_reference reference;
		public: typedef typename matrix_binary_functor2::const_pointer pointer;

		public: typedef const_iterator1 dual_iterator_type;
		public: typedef const_reverse_iterator1 dual_reverse_iterator_type;


		// Construction and destruction


		public: BOOST_UBLAS_INLINE const_iterator2()
			: container_const_reference<self_type>(),
			  it_()
		{
			// Empty
		}


		public: BOOST_UBLAS_INLINE const_iterator2(self_type const& mu, const_subiterator2_type const& it)
			: container_const_reference<self_type>(mu),
			  it_(it)
		{
			// Empty
		}


		// Arithmetic


		public: BOOST_UBLAS_INLINE const_iterator2& operator++()
		{
			++it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator--()
		{
			--it_;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator+=(difference_type n)
		{
			it_ += n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE const_iterator2& operator-=(difference_type n)
		{
			it_ -= n;

			return *this;
		}


		public: BOOST_UBLAS_INLINE difference_type operator-(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ - it.it_;
		}


		// Dereference


		public: BOOST_UBLAS_INLINE const_reference operator*() const
		{
			return (*this)().f_((*this)().a1_, *it_);
		}


		public: BOOST_UBLAS_INLINE const_reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 begin() const
		{
			return (*this)().find1(1, 0, index2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 end() const
		{
			return (*this)().find1(1, (*this)().size1(), index2());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rbegin() const
		{
			return const_reverse_iterator1(end());
		}


		public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rend() const
		{
			return const_reverse_iterator1(begin());
		}
#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


		// Indices


		public: BOOST_UBLAS_INLINE size_type index1() const
		{
			return it_.index1();
		}


		public: BOOST_UBLAS_INLINE size_type index2() const
		{
			return it_.index2();
		}


		// Assignment 
		public: BOOST_UBLAS_INLINE const_iterator2& operator=(const_iterator2 const& it)
		{
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;

			return *this;
		}


		// Comparisons


		public: BOOST_UBLAS_INLINE bool operator==(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ == it.it_;
		}


		public: BOOST_UBLAS_INLINE bool operator<(const_iterator2 const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());

			return it_ < it.it_;
		}


		private: const_subiterator2_type it_;
	};
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR


	private: arg1_type a1_;
	private: expression_closure_type e_;
	private: functor_type f_;
}; // matrix_binary_functor2

template <typename Arg1T, typename ExprT, typename SignatureT>
struct matrix_binary_functor2_traits
{
	typedef matrix_binary_functor2<Arg1T, ExprT, SignatureT> expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type; 
#else
	typedef typename ExprT::matrix_temporary_type result_type;
#endif
}; // matrix_binary_functor2_traits


}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_EXPRESSION_MATRIX_BINARY_FUNCTOR_HPP
