/**
 * \file boost/numeric/ublasx/matrix_diagonal.hpp
 *
 * \brief Diagonal view of a matrix.
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_PROXY_MATRIX_DIAGONAL_HPP
#define BOOST_NUMERIC_UBLASX_PROXY_MATRIX_DIAGONAL_HPP

#include <algorithm>
#include <boost/mpl/if.hpp>
#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/detail/vector_assign.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/traits.hpp>
//#include <boost/numeric/ublasx/traits/array_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_const.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

/**
 * \brief Matrix based diagonal vector class
 * \tparam MatrixT A model of MatrixExpression.
 *
 * This class provides a view of a specific diagonal of a matrix of type
 * \a MatrixT.
 * The diagonal of the matrix is chosen during construction through the
 * parameter \c k passed to the constructor.
 * The parameter \c k has the following meaning:
 * - \c k = 0, the main diagonal is extracted;
 * - \c k > 0, the \c k-th diagonal above the main diagonal is extracted;
 * - \c k < 0, the \c k-th diagonal under the main diagonal is extracted;
 * .
 * Model of VectorExpression.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixT>
class matrix_diagonal: public vector_expression< matrix_diagonal<MatrixT> >
{
	public: class const_iterator;
	public: class iterator;


	private: typedef matrix_diagonal<MatrixT> self_type;
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	public: using vector_expression<self_type>::operator();
#endif // BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	public: typedef MatrixT matrix_type;
	public: typedef typename matrix_traits<MatrixT>::size_type size_type;
	public: typedef typename matrix_traits<MatrixT>::difference_type difference_type;
	public: typedef typename matrix_traits<MatrixT>::value_type value_type;
//	public: typedef typename array_type<MatrixT>::type array_type; // FIXME: not in matrix_traits
	public: typedef typename matrix_traits<MatrixT>::const_reference const_reference;
	public: typedef typename ::boost::mpl::if_<
									::boost::is_const<MatrixT>,
									typename matrix_traits<MatrixT>::const_reference,
									typename matrix_traits<MatrixT>::reference
							>::type reference;
	public: typedef typename ::boost::mpl::if_<
									::boost::is_const<MatrixT>,
									typename matrix_traits<MatrixT>::const_closure_type,
									typename matrix_traits<MatrixT>::closure_type
							>::type matrix_closure_type;
	public: typedef const self_type const_closure_type;
	public: typedef self_type closure_type;
	public: typedef typename storage_restrict_traits<
								typename matrix_traits<MatrixT>::storage_category,
								dense_proxy_tag
							>::storage_category storage_category;
	// Iterator types
	private: typedef typename matrix_traits<MatrixT>::const_iterator1 const_subiterator1_type;
	private: typedef typename matrix_traits<MatrixT>::const_iterator2 const_subiterator2_type;
	private: typedef typename ::boost::mpl::if_<
									::boost::is_const<MatrixT>,
									typename matrix_traits<MatrixT>::const_iterator1,
									typename matrix_traits<MatrixT>::iterator1
							>::type subiterator1_type;
	private: typedef typename ::boost::mpl::if_<
									::boost::is_const<MatrixT>,
									typename matrix_traits<MatrixT>::const_iterator2,
									typename matrix_traits<MatrixT>::iterator2
							>::type subiterator2_type;
	// Reverse iterator
	public: typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	public: typedef reverse_iterator_base<iterator> reverse_iterator;


	//@{ Construction and destruction

	public: BOOST_UBLAS_INLINE
		matrix_diagonal(matrix_type& data, difference_type k)
			: data_(data),
			  k_(k),
			  r_(k < 0 ? -k : 0),
			  c_(k > 0 ?  k : 0)
	{
		// Early checking of preconditions here.
		// BOOST_UBLAS_CHECK(r_ < data_.size1() && c_ < data_.size2(), bad_index());
	}


	public: BOOST_UBLAS_INLINE
		matrix_diagonal(matrix_closure_type const& data, difference_type k, int)
			: data_(data),
			  k_(k),
			  r_(k < 0 ? -k : 0),
			  c_(k > 0 ?  k : 0)
	{
		// Early checking of preconditions here.
		// BOOST_UBLAS_CHECK(r_ < data_.size1() && c_ < data_.size2(), bad_index());
	}


//	public: template <typename ExprT>
//		BOOST_UBLAS_INLINE
//		matrix_diagonal(matrix_expression<ExprT>& me, difference_type k)
//			: data_(me()),
//			  k_(k),
//			  r_(k < 0 ? -k : 0),
//			  c_(k > 0 ?  k : 0)
//	{
//		// Early checking of preconditions here.
//		// BOOST_UBLAS_CHECK(r_ < data_.size1() && c_ < data_.size2(), bad_index());
//	}

	//@} Construction and destruction

	//@{ Accessors

	public: BOOST_UBLAS_INLINE
		size_type size() const
	{
		if (k_ > 0)
		{
			return ::std::min(data_.size1(), data_.size2() - c_);
		}
		return ::std::min(data_.size1() - r_, data_.size2());
	}


	public: BOOST_UBLAS_INLINE
		difference_type offset() const
	{
		return k_;
	}

	//@} Accessors

	//@{ Storage accessors

	public: BOOST_UBLAS_INLINE
		matrix_closure_type const& data() const
	{
		return data_;
	}


	public: BOOST_UBLAS_INLINE
		matrix_closure_type& data()
	{
		return data_;
	}

	//@} Storage accessors

	//@{ Element access

#ifndef BOOST_UBLASX_PROXY_CONST_MEMBER
	public: BOOST_UBLAS_INLINE
		const_reference operator()(size_type j) const
	{
		return data_(j+r_, j+c_);
	}


	public: BOOST_UBLAS_INLINE
		reference operator()(size_type j)
	{
		return data_(j+r_, j+c_);
	}


	public: BOOST_UBLAS_INLINE
		const_reference operator[](size_type j) const
	{
		return (*this)(j);
	}


	public: BOOST_UBLAS_INLINE
		reference operator[](size_type j)
	{
		return (*this)(j);
	}
#else
	public: BOOST_UBLAS_INLINE
		reference operator()(size_type j) const
	{
		return data_(j+r_, j+c_);
	}


	public: BOOST_UBLAS_INLINE
		reference operator[](size_type j) const
	{
		return (*this)(j);
	}
#endif // BOOST_UBLASX_PROXY_CONST_MEMBER

	//@} Element access

	//@{ Assignment

	public: BOOST_UBLAS_INLINE
		matrix_diagonal& operator=(matrix_diagonal const& md)
	{
		// ISSUE need a temporary, proxy can be overlaping alias
		vector_assign<scalar_assign>(
			*this,
			typename vector_temporary_traits<matrix_type>::type(md)
		);
		return *this;
	}


	public: BOOST_UBLAS_INLINE
		matrix_diagonal& assign_temporary(matrix_diagonal& md)
	{
		// assign elements, proxied container remains the same
		vector_assign<scalar_assign>(*this, md);
		return *this;
	}


	public: template<class AE>
		BOOST_UBLAS_INLINE
		matrix_diagonal& operator=(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_assign>(
			*this,
			typename vector_temporary_traits<matrix_type>::type(ae)
		);
		return *this;
	}


	public: template<class AE>
	BOOST_UBLAS_INLINE
	matrix_diagonal& assign(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_assign>(*this, ae);
		return *this;
	}


	public: template<class AE>
		BOOST_UBLAS_INLINE
		matrix_diagonal& operator+=(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_assign>(
			*this,
			typename vector_temporary_traits<matrix_type>::type(*this + ae)
		);
		return *this;
	}


	public: template<class AE>
		BOOST_UBLAS_INLINE
		matrix_diagonal& plus_assign(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_plus_assign>(*this, ae);
		return *this;
	}


	public: template<class AE>
		BOOST_UBLAS_INLINE
		matrix_diagonal& operator-=(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_assign>(
			*this,
			typename vector_temporary_traits<matrix_type>::type(*this - ae)
		);
		return *this;
	}


	public: template<class AE>
		BOOST_UBLAS_INLINE
		matrix_diagonal &minus_assign(vector_expression<AE> const& ae)
	{
		vector_assign<scalar_minus_assign>(*this, ae);
		return *this;
	}


	public: template<class AT>
		BOOST_UBLAS_INLINE
		matrix_diagonal &operator*=(AT const& at)
	{
		vector_assign_scalar<scalar_multiplies_assign>(*this, at);
		return *this;
	}


	public: template<class AT>
		BOOST_UBLAS_INLINE
		matrix_diagonal &operator/=(AT const& at)
	{
		vector_assign_scalar<scalar_divides_assign>(*this, at);
		return *this;
	}

	//@} Assignment

	//@{ Closure comparison

	public: BOOST_UBLAS_INLINE
		bool same_closure(matrix_diagonal const& mr) const
	{
		return this->data_.same_closure(mr.data_);
	}

	//@} Closure comparison

	//@{ Comparison

	public: BOOST_UBLAS_INLINE
		bool operator==(matrix_diagonal const& mr) const
	{
		return this->data_ == mr.data_ && offset() == mr.offset();
	}

	//@} Comparison

	//@{ Swapping

	public: BOOST_UBLAS_INLINE
		void swap(matrix_diagonal mr) //FIXME: why not pass-by-reference?
	{
		if (this != &mr)
		{
			BOOST_UBLAS_CHECK(size() == mr.size(), bad_size());
			// Sparse ranges may be nonconformant now.
			// std::swap_ranges (begin(), end(), mr.begin());
			vector_swap<scalar_swap>(*this, mr);
		}
	}


	public: BOOST_UBLAS_INLINE
		friend
		void swap(matrix_diagonal mr1, matrix_diagonal mr2) //FIXME: why not pass-by-reference?
	{
		mr1.swap(mr2);
	}

	//@} Swapping

	//@{ Iterators

	public: BOOST_UBLAS_INLINE
		const_iterator find(size_type j) const
	{
		const_subiterator1_type it1(data_.find1(2, j+r_, c_));
		const_subiterator2_type it2(data_.find2(1, r_, j+c_));

		return const_iterator(*this, it1, it2);
	}


	public: BOOST_UBLAS_INLINE
		iterator find(size_type j)
	{
		subiterator1_type it1(data_.find1(2, j+r_, c_));
		subiterator2_type it2(data_.find2(1, r_, j+c_));

		return iterator(*this, it1, it2);
	}


	public: BOOST_UBLAS_INLINE
		const_iterator begin() const
	{
		return find(0);
	}


	public: BOOST_UBLAS_INLINE
		const_iterator end() const
	{
		return find(size());
	}


	public: BOOST_UBLAS_INLINE
		iterator begin()
	{
		return find(0);
	}


	public: BOOST_UBLAS_INLINE
		iterator end()
	{
		return find(size());
	}


	public: BOOST_UBLAS_INLINE
		const_reverse_iterator rbegin() const
	{
		return const_reverse_iterator(end());
	}


	public: BOOST_UBLAS_INLINE
		const_reverse_iterator rend() const
	{
		return const_reverse_iterator(begin());
	}


	public: BOOST_UBLAS_INLINE
		reverse_iterator rbegin()
	{
		return reverse_iterator(end());
	}


	public: BOOST_UBLAS_INLINE
		reverse_iterator rend()
	{
		return reverse_iterator(begin());
	}


	public: class const_iterator: 	public container_const_reference<matrix_diagonal>,
									public iterator_base_traits<typename const_subiterator1_type::iterator_category>::template iterator_base<const_iterator, value_type>::type
	{
		public: typedef typename const_subiterator1_type::value_type value_type;
		public: typedef typename const_subiterator1_type::difference_type difference_type;
		public: typedef typename const_subiterator1_type::reference reference;
		public: typedef typename const_subiterator1_type::pointer pointer;


		// Iterators cannot be different
		BOOST_STATIC_ASSERT((
			::boost::is_same<
					typename MatrixT::const_iterator1::iterator_category,
					typename MatrixT::const_iterator2::iterator_category
			>::value
		));


		// Construction and destruction
		public: BOOST_UBLAS_INLINE
			const_iterator()
			: container_const_reference<self_type>(),
			  it1_(),
			  it2_()
		{
		}


		public: BOOST_UBLAS_INLINE
			const_iterator(self_type const& mr, const_subiterator1_type const& it1, const_subiterator2_type const& it2)
			: container_const_reference<self_type>(mr),
			  it1_(it1),
			  it2_(it2)
		{
		}


		public: BOOST_UBLAS_INLINE
			const_iterator(typename self_type::iterator const& it)  // ISSUE self_type:: stops VC8 using std::iterator here
			: container_const_reference<self_type>(it()),
			  it1_(it.it1_),
			  it2_(it.it2_)
		{
		}


		// Arithmetic
		public: BOOST_UBLAS_INLINE
			const_iterator& operator++()
		{
			++it1_;
			++it2_;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			const_iterator& operator--()
		{
			--it1_;
			--it2_;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			const_iterator& operator+=(difference_type n)
		{
			it1_ += n;
			it1_ += n;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			const_iterator& operator-=(difference_type n)
		{
			it1_ -= n;
			it2_ -= n;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			difference_type operator-(const_iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure (it()), external_logic());
			return BOOST_UBLAS_SAME(it1_ - it.it1_, it2_ - it.it2_);
		}


		// Dereference
		public: BOOST_UBLAS_INLINE
			const_reference operator*() const
		{
			return (*this)().data_(it1_.index1(), it2_.index2());
		}


		public: BOOST_UBLAS_INLINE
			const_reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


		// Index
		public: BOOST_UBLAS_INLINE
			size_type index() const
		{
			if (it1_.index1() > it2_.index2())
			{
				return it2_.index2();
			}

			return it1_.index1();
		}


		// Assignment
		public: BOOST_UBLAS_INLINE
			const_iterator& operator=(const_iterator const& it)
		{
			container_const_reference<self_type>::assign(&it());
			it1_ = it.it1_;
			it2_ = it.it2_;
			return *this;
		}


		// Comparison
		public: BOOST_UBLAS_INLINE
			bool operator==(const_iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return it1_ == it.it1_ && it2_ == it.it2_;
		}


		public: BOOST_UBLAS_INLINE
			bool operator<(const_iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return it1_ < it.it1_ && it2_ < it.it2_;
		}


		private: const_subiterator1_type it1_;
		private: const_subiterator2_type it2_;
	};


	public: class iterator: 	public container_reference<matrix_diagonal>,
								public iterator_base_traits<typename subiterator1_type::iterator_category>::template iterator_base<iterator, value_type>::type
	{
		public: typedef typename subiterator1_type::value_type value_type;
		public: typedef typename subiterator1_type::difference_type difference_type;
		public: typedef typename subiterator1_type::reference reference;
		public: typedef typename subiterator1_type::pointer pointer;


		// Iterators cannot be different
		BOOST_STATIC_ASSERT((
			::boost::is_same<
					typename MatrixT::iterator1::iterator_category,
					typename MatrixT::iterator2::iterator_category
			>::value
		));


		// Construction and destruction
		public: BOOST_UBLAS_INLINE
			iterator ()
			: container_reference<self_type>(),
			  it1_(),
			  it2_()
		{
		}


		public: BOOST_UBLAS_INLINE
			iterator(self_type& mr, subiterator1_type const& it1, subiterator2_type const& it2)
			: container_reference<self_type>(mr),
			  it1_(it1),
			  it2_(it2)
		{
		}


		// Arithmetic
		public: BOOST_UBLAS_INLINE
			iterator &operator++()
		{
			++it1_;
			++it2_;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			iterator& operator--()
		{
			--it1_;
			--it2_;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			iterator& operator+=(difference_type n)
		{
			it1_ += n;
			it2_ += n;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			iterator& operator-=(difference_type n)
		{
			it1_ -= n;
			it2_ -= n;
			return *this;
		}


		public: BOOST_UBLAS_INLINE
			difference_type operator-(iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return BOOST_UBLAS_SAME(it1_ - it.it1_, it2_ - it.it2_);
		}


		// Dereference
		public: BOOST_UBLAS_INLINE
			reference operator*() const
		{
			return (*this)().data_(it1_.index1(), it2_.index2());
		}


		public: BOOST_UBLAS_INLINE
			reference operator[](difference_type n) const
		{
			return *(*this + n);
		}


		// Index
		public: BOOST_UBLAS_INLINE
			size_type index() const
		{
			return BOOST_UBLAS_SAME(it1_.index1(),  it2_.index2());
		}


		// Assignment
		public: BOOST_UBLAS_INLINE
			iterator& operator=(iterator const& it)
		{
			container_reference<self_type>::assign(&it());
			it1_ = it.it1_;
			it2_ = it.it2_;
			return *this;
		}


		// Comparison
		public: BOOST_UBLAS_INLINE
			bool operator==(iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return it1_ == it.it1_ && it2_ == it.it2_;
		}


		public: BOOST_UBLAS_INLINE
			bool operator<(iterator const& it) const
		{
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return it1_ < it.it1_ && it2_ < it.it2_;
		}


		private: subiterator1_type it1_;
		private: subiterator2_type it2_;
		private: friend class const_iterator; // See the const_interator(iterator const&) constructor
	};

	//@} Iterators

	//@{ Data members

	private: matrix_closure_type data_; ///< The underlying matrix.
	private: difference_type k_; ///< Offset from the main diagonal.
	private: size_type r_; ///< Offset from the row of the main diagonal.
	private: size_type c_; ///< Offset from the column of the main diagonal.

	//@} Data members
};

} // Namespace ublasx

namespace ublas {

// Specialize temporary traits

template <typename MatrixT>
struct vector_temporary_traits< ::boost::numeric::ublasx::matrix_diagonal<MatrixT> >: public vector_temporary_traits<MatrixT> {};

template <typename  MatrixT>
struct vector_temporary_traits< const ::boost::numeric::ublasx::matrix_diagonal<MatrixT> >: public vector_temporary_traits<MatrixT> {};

} // Namespace ublas

}} // Namespace boost::numeric


#endif // BOOST_NUMERIC_UBLASX_PROXY_MATRIX_DIAGONAL_HPP
