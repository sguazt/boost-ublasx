/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 *  \file boost/numeric/ublasx/container/generalized_diagonal_matrix.hpp
 *
 *  \brief Generalized diagonal matrix class and adaptor.
 *
 *  Copyright (c) 2009-2010, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 *  \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_CONTAINER_GENERALIZED_DIAGONAL_MATRIX_HPP
#define BOOST_NUMERIC_UBLASX_CONTAINER_GENERALIZED_DIAGONAL_MATRIX_HPP


#include <algorithm>
#include <boost/mpl/if.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/type_traits/is_const.hpp>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


namespace detail { namespace /*<unnamed>*/ {
    // Matrix resizing algorithm
    template <class L, class M>
    BOOST_UBLAS_INLINE
    void matrix_resize_preserve (M& m, M& temporary) {
        typedef L layout_type;
        typedef typename M::size_type size_type;
        const size_type msize1 (m.size1 ());        // original size
        const size_type msize2 (m.size2 ());
        const size_type size1 (temporary.size1 ());    // new size is specified by temporary
        const size_type size2 (temporary.size2 ());
        // Common elements to preserve
        const size_type size1_min = (std::min) (size1, msize1);
        const size_type size2_min = (std::min) (size2, msize2);
        // Order for major and minor sizes
        const size_type major_size = layout_type::size_M (size1_min, size2_min);
        const size_type minor_size = layout_type::size_m (size1_min, size2_min);
        // Indexing copy over major
        for (size_type major = 0; major != major_size; ++major) {
            for (size_type minor = 0; minor != minor_size; ++minor) {
                    // find indexes - use invertability of element_ functions
                const size_type i1 = layout_type::index_M(major, minor);
                const size_type i2 = layout_type::index_m(major, minor);
                //if ( triangular_type::other(i1,i2) ) {
                //    temporary.data () [triangular_type::element (layout_type (), i1, size1, i2, size2)] =
                //        m.data() [triangular_type::element (layout_type (), i1, msize1, i2, msize2)];
                //}
                temporary.data () [layout_type::element (i1, size1, i2, size2)] = m.data() [layout_type::element (i1, msize1, i2, msize2)];

            }
        }
        m.assign_temporary (temporary);
    }
}} // Namespace detail::<unnamed>


/**
 * \brief Generalized diagonal matrix.
 *
 * \tparam ValueT The type of matrix values.
 * \tparam LayoutT The matrix layout type.
 *  Default to \c row_major.
 * \tparam ArrayT The type of the array storing the matrix.
 *  Default to \c unbounded_array<ValueT>
 *
 * This class can be used to represent:
 * - <em>diagonal matrices</em>: a square matrix \f$A\f$ of order \f$n\f$ is a
 *   <em>diagonal matrix</em> if \f$a_{ij}=0\f$ for \f$i \ne j\f$,
 *   \f$i,j=0,\ldots,n\f$;
 * - <em>rectangular diagonal matrices</em>: an \f$m \times n\f$ matrix \f$A\f$
 *   is a <em>rectangular diagonal matrix</em> if \f$a_{ij}=0\f$ for
 *   \f$i \ne j\f$,\f$i=1,\ldots,m\f$ and \f$j=1,\ldots,n\f$. Thus the only
 *   entries in \f$A\f$ that may be non-zero are the \f$d_{ii}\f$,
 *   \f$i=1,\ldots,\min\{m,n\}\f$;
 * - <em>sub-diagonal matrices</em>: a square matrix \f$A\f$ of order
 *   \f$n\f$ is a <em>k-th sub-diagonal matrix</em> if \f$a_{ij}=0\f$ for
 *   \f$i \ne (j+k)\f$, \f$i=1,\ldots,n\f$ and \f$j=1,\ldots,n\f$;
 * - <em>super-diagonal matrices</em>: a square matrix \f$A\f$ of order \f$n\f$
 *   is a <em>k-th super-diagonal matrix</em> if \f$a_{ij}=0\f$ for
 *   \f$(i+k) \ne j\f$, \f$i=1,\ldots,n\f$ and \f$j=1,\ldots,n\f$;
 * - <em>rectangular sub-diagonal matrices</em>: a \f$m \times n\f$ matrix
 *   \f$A\f$ is a <em>k-th rectangular sub-diagonal matrix</em> if
 *   \f$a_{ij}=0\f$ for \f$i \ne (j+k)\f$, \f$i=1,\ldots,m\f$ and
 *   \f$j=1,\ldots,n\f$;
 * - <em>rectangular super-diagonal matrices</em>: a \f$m \times n\f$ matrix
 *   \f$A\f$ is a <em>k-th rectangular super-diagonal matrix</em> if
 *   \f$a_{ij}=0\f$ for \f$(i+k) \ne j\f$, \f$i=1,\ldots,m\f$ and
 *   \f$j=1,\ldots,n\f$;
 * .
 *
 * References:
 * - [1] H. Scneider et al,
 *       "Matrices and Linear Algebra",
 *       2nd edition, Dover Publications, Inc., 1989
 * - [2] M. Brookes,
 *       "The Matrix Reference Manual",
 *       [online] http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html, 2005
 * .
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <
    typename ValueT,
    typename LayoutT = row_major,
    typename ArrayT = unbounded_array<ValueT>
>
class generalized_diagonal_matrix: public matrix_container<generalized_diagonal_matrix<ValueT, LayoutT, ArrayT> >
{

    private: typedef ValueT *pointer;
    private: typedef LayoutT layout_type;
    private: typedef generalized_diagonal_matrix<ValueT, LayoutT, ArrayT> self_type;
    public: typedef typename ArrayT::size_type size_type;
    public: typedef typename ArrayT::difference_type difference_type;
    public: typedef ValueT value_type;
    public: typedef const ValueT &const_reference;
    public: typedef ValueT &reference;
    public: typedef ArrayT array_type;
    public: typedef const matrix_reference<const self_type> const_closure_type;
    public: typedef matrix_reference<self_type> closure_type;
    public: typedef vector<ValueT, ArrayT> vector_temporary_type;
    public: typedef matrix<ValueT, LayoutT, ArrayT> matrix_temporary_type;  // general sub-matrix
    public: typedef packed_tag storage_category;
    public: typedef typename LayoutT::orientation_category orientation_category;
    private: typedef const value_type const_value_type;
    // Iterator types
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: typedef indexed_iterator1<self_type, packed_random_access_iterator_tag> iterator1;
    public: typedef indexed_iterator2<self_type, packed_random_access_iterator_tag> iterator2;
    public: typedef indexed_const_iterator1<self_type, packed_random_access_iterator_tag> const_iterator1;
    public: typedef indexed_const_iterator2<self_type, packed_random_access_iterator_tag> const_iterator2;
#else
    public: class const_iterator1;
    public: class iterator1;
    public: class const_iterator2;
    public: class iterator2;
#endif
    public: typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
    public: typedef reverse_iterator_base1<iterator1> reverse_iterator1;
    public: typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
    public: typedef reverse_iterator_base2<iterator2> reverse_iterator2;


#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
    public: using matrix_container<self_type>::operator();
#endif


    //@{ Construction and destruction

    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix()
        : matrix_container<self_type>(),
          size1_(0),
          size2_(0),
          k_(0),
          r_(0),
          c_(0),
          data_(0)
    {
        // Empty
    }


    /**
     * \brief Create a diagonal matrix of order \a size whose non-zero elements
     *  are on diagonal \a k.
     */
    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(size_type size, difference_type k=0)
        : matrix_container<self_type>(),
          size1_(size),
          size2_(size),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          data_(size - (k >= 0 ? k : -k))
    {
        // preconditions
        BOOST_UBLAS_CHECK(size_type(k >= 0 ? k : -k) < size, bad_size());
    }


    /**
     * \brief Create a rectangular diagonal matrix of size \a size1 by \a size2
     * whose non-zero elements are on diagonal \a k.
     */
    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(size_type size1, size_type size2, difference_type k)
        : matrix_container<self_type>(),
          size1_(size1),
          size2_(size2),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          data_(std::min(size1 - r_, size2 - c_))
    {
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(size_type size, difference_type k, array_type const& data)
        : matrix_container<self_type>(),
          size1_(size),
          size2_(size),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          data_(data)
    {
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());

        size_type real_size = size - (r_ + c_);

        if (data_.size() > real_size)
        {
            data_.resize(real_size, value_type(0));
        }
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(size_type size1, size_type size2, difference_type k, array_type const& data)
        : matrix_container<self_type>(),
          size1_(size1),
          size2_(size2),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          data_(data)
    {
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());

        size_type real_size = ::std::min(size1 - r_, size2 - c_);

        if (data_.size() > real_size)
        {
            data_.resize(real_size, value_type(0));
        }
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(generalized_diagonal_matrix const& m)
        : matrix_container<self_type>(),
          size1_(m.size1_),
          size2_(m.size2_),
          k_(m.k_),
          r_(m.r_),
          c_(m.c_),
          data_(m.data_)
    {
        // Empty
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(matrix_expression<ExprT> const& me, difference_type k=0)
        : matrix_container<self_type>(),
          //size_(std::min(me().size1(), me().size2())),
          size1_(me().size1()),
          size2_(me().size2()),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          //data_(size_ - (k >= 0 ? k : -k))
          data_(::std::min(me().size1() - r_, me().size2() - c_))
    {
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());

        matrix_assign<scalar_assign>(*this, me);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix(vector_expression<ExprT> const& ve, difference_type k=0)
        : matrix_container<self_type>(),
          size1_(ve().size() + (k < 0 ? -k : 0)),
          size2_(ve().size() + (k >= 0 ? k : 0)),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0),
          data_(ve().size())
    {
        // preconditions
        BOOST_UBLAS_CHECK(r_ < size1_, bad_size());
        BOOST_UBLAS_CHECK(c_ < size2_, bad_size());

        typedef typename ExprT::size_type ve_size_type;

        ve_size_type ve_size = ve().size();

        for (
            ve_size_type i = 0;
            i < ve_size;
            ++i
        ) {
            (*this)(i+r_, i+c_) = ve()(i);
        }
    }


    //@} Construction/Destruction

    //@{ Accessors


    public: BOOST_UBLAS_INLINE
        size_type size1() const
    {
        return size1_;
    }


    public: BOOST_UBLAS_INLINE
        size_type size2() const
    {
        return size2_;
    }


    public: BOOST_UBLAS_INLINE
        difference_type offset() const
    {
        return k_;
    }


    // Storage accessors
    public: BOOST_UBLAS_INLINE
        array_type const& data() const
    {
        return data_;
    }


    public: BOOST_UBLAS_INLINE
        array_type& data()
    {
        return data_;
    }


    //@} Accessors

    //@{ Resizing


    public: BOOST_UBLAS_INLINE
        void resize(size_type size, difference_type k=0, bool preserve=true)
    {
        if (preserve)
        {
            self_type temporary(size, k);
            detail::matrix_resize_preserve<layout_type>(*this, temporary);
        }
        else
        {
            size1_ = size;
            size2_ = size;
            k_ = k;
            r_ = k < 0 ? -k : 0;
            c_ = k > 0 ?  k : 0;
            data().resize(size - (k > 0 ? k : -k));
        }
    }


    public: BOOST_UBLAS_INLINE
        void resize(size_type size1, size_type size2, difference_type k=0, bool preserve=true)
    {
        if (preserve)
        {
            self_type temporary(size1, size2, k);
            detail::matrix_resize_preserve<layout_type>(*this, temporary);
        }
        else
        {
            size1_ = size1;
            size2_ = size2;
            k_ = k;
            r_ = k < 0 ? -k : 0;
            c_ = k > 0 ?  k : 0;
            data().resize(::std::min(size1 - r_, size2 - c_));
        }
    }


    public: BOOST_UBLAS_INLINE
        void resize_packed_preserve(size_type size, difference_type k=0)
    {
        size1_ = size;
        size2_ = size;
        k_ = k;
        r_ = k < 0 ? -k : 0;
        c_ = k > 0 ?  k : 0;
        data().resize(size - (k > 0 ? k : -k), value_type());
    }


    public: BOOST_UBLAS_INLINE
        void resize_packed_preserve(size_type size1, size_type size2, difference_type k=0)
    {
        size1_ = size1;
        size2_ = size2;
        k_ = k;
        r_ = k < 0 ? -k : 0;
        c_ = k > 0 ?  k : 0;
        data().resize(::std::min(size1 - r_, size2 - c_), value_type());
    }


    //@} Resizing

    //@{ Element access


    public: BOOST_UBLAS_INLINE
        const_reference operator()(size_type i, size_type j) const
    {
        BOOST_UBLAS_CHECK(i < size1_, bad_index());
        BOOST_UBLAS_CHECK(j < size2_, bad_index());

        const difference_type dr(i-r_);
        const difference_type dc(j-c_);

        if (dr == dc)
        {
            if (k_ > 0)
            {
                return data()[layout_type::element(i, size1_, 0, 1)];
            }
            else
            {
                return data()[layout_type::element(0, 1, j, size2_)];
            }
        }

        return zero_;
    }


    public: BOOST_UBLAS_INLINE
        reference at_element(size_type i, size_type j)
    {
        BOOST_UBLAS_CHECK(i < size1_, bad_index());
        BOOST_UBLAS_CHECK(j < size2_, bad_index());

        if (k_ > 0)
        {
            return data()[layout_type::element(i, size1_, 0, 1)];
        }
        else
        {
            return data()[layout_type::element(0, 1, j, size2_)];
        }
    }


    public: BOOST_UBLAS_INLINE
        reference operator()(size_type i, size_type j)
    {
        BOOST_UBLAS_CHECK(i < size1_, bad_index());
        BOOST_UBLAS_CHECK(j < size2_, bad_index());

        const difference_type dr(i-r_);
        const difference_type dc(j-c_);

        if (dr != dc)
        {
            bad_index().raise();
        }

        if (k_ > 0)
        {
            return data()[layout_type::element(i, size1_, 0, 1)];
        }
        else
        {
            return data()[layout_type::element(0, 1, j, size2_)];
        }
    }


    //@} Element access

    //@{ Element assignment


    public: BOOST_UBLAS_INLINE
        reference insert_element(size_type i, size_type j, const_reference t)
    {
        return (operator()(i, j) = t);
    }


    public: BOOST_UBLAS_INLINE
        void erase_element(size_type i, size_type j)
    {
        operator()(i, j) = value_type/*zero*/();
    }


    //@} Element assignment

    //@{ Zeroing


    public: BOOST_UBLAS_INLINE
        void clear()
    {
        ::std::fill(data().begin(), data().end(), value_type/*zero*/());
    }


    //@} Zeroing

    //@{ Assignment


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator=(generalized_diagonal_matrix const& m)
    {
        size1_ = m.size1_;
        size2_ = m.size2_;
        k_ = m.k_;
        r_ = m.r_;
        c_ = m.c_;
        data() = m.data();

        return *this;
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& assign_temporary(generalized_diagonal_matrix& m)
    {
        swap(m);

        return *this;
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator=(matrix_expression<ExprT> const& me)
    {
        self_type temporary(me, k_);

        return assign_temporary(temporary);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& assign(matrix_expression<ExprT> const& me)
    {
        matrix_assign<scalar_assign>(*this, me);

        return *this;
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator=(vector_expression<ExprT> const& ve)
    {
        self_type temporary(ve, k_);

        return assign_temporary(temporary);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& assign(vector_expression<ExprT> const& ve)
    {
        return operator=(ve);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator+=(matrix_expression<ExprT> const& me)
    {
        self_type temporary(*this + me, k_);

        return assign_temporary(temporary);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& plus_assign(matrix_expression<ExprT> const& me)
    {
        matrix_assign<scalar_plus_assign>(*this, me);

        return *this;
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator-=(matrix_expression<ExprT> const& me)
    {
        self_type temporary(*this - me, k_);

        return assign_temporary(temporary);
    }


    public: template <typename ExprT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& minus_assign(matrix_expression<ExprT> const& me)
    {
        matrix_assign<scalar_minus_assign>(*this, me);

        return *this;
    }


    public: template <typename ScalarT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator*=(ScalarT const& se)
    {
        matrix_assign_scalar<scalar_multiplies_assign>(*this, se);

        return *this;
    }


    public: template <typename ScalarT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_matrix& operator/=(ScalarT const& se)
    {
        matrix_assign_scalar<scalar_divides_assign>(*this, se);

        return *this;
    }


    //@} Assignment

    //@{ Swapping


    public: BOOST_UBLAS_INLINE
        void swap(generalized_diagonal_matrix& m)
    {
        if (this != &m)
        {
            ::std::swap(size1_, m.size1_);
            ::std::swap(size2_, m.size2_);
            ::std::swap(k_, m.k_);
            ::std::swap(r_, m.r_);
            ::std::swap(c_, m.c_);
            data().swap(m.data());
        }
    }


    public: BOOST_UBLAS_INLINE
        friend void swap(generalized_diagonal_matrix& m1, generalized_diagonal_matrix& m2)
    {
        m1.swap(m2);
    }


    //@} Swapping

    //@{ Element lookup

    public: BOOST_UBLAS_INLINE
        const_iterator1 find1(int /*rank*/, size_type i, size_type j) const
    {
        i = ::std::min(::std::max(i, r_), r_ + ::std::min(size1_-r_, size2_-c_));

        return const_iterator1(*this, i, j);
    }


    public: BOOST_UBLAS_INLINE
        iterator1 find1(int /*rank*/, size_type i, size_type j)
    {
        i = ::std::min(::std::max(i, r_), r_ + ::std::min(size1_-r_, size2_-c_));

        return iterator1(*this, i, j);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 find2(int /*rank*/, size_type i, size_type j) const
    {
        j = ::std::min(::std::max(j, c_), c_ + ::std::min(size1_-r_, size2_-c_));

        return const_iterator2(*this, i, j);
    }


    public: BOOST_UBLAS_INLINE
        iterator2 find2(int /*rank*/, size_type i, size_type j)
    {
        j = ::std::min(::std::max(j, c_), c_ + ::std::min(size1_-r_, size2_-c_));

        return iterator2(*this, i, j);
    }


    //@} Element lookup

    //@{ Forward Iterators

    public: BOOST_UBLAS_INLINE
        const_iterator1 begin1() const
    {
        return find1(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator1 end1() const
    {
        return find1(0, r_ + ::std::min(size1_-r_, size2_-c_), c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator1 begin1()
    {
        return find1(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator1 end1()
    {
        return find1(0, r_ + ::std::min(size1_-r_, size2_-c_), c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 begin2() const
    {
        return find2(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 end2() const
    {
        return find2(0, r_, c_ + ::std::min(size1_-r_, size2_-c_));
    }


    public: BOOST_UBLAS_INLINE
        iterator2 begin2()
    {
        return find2(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator2 end2()
    {
        return find2(0, r_, c_ + ::std::min(size1_-r_, size2_-c_));
    }


    //@} Forward Iterators

    //@{ Reverse iterators

    public: BOOST_UBLAS_INLINE
        const_reverse_iterator1 rbegin1() const
    {
        return const_reverse_iterator1(end1());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator1 rend1() const
    {
        return const_reverse_iterator1(begin1());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator1 rbegin1()
    {
        return reverse_iterator1(end1());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator1 rend1()
    {
        return reverse_iterator1(begin1());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator2 rbegin2() const
    {
        return const_reverse_iterator2(end2());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator2 rend2() const
    {
        return const_reverse_iterator2(begin2());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator2 rbegin2()
    {
        return reverse_iterator2(end2());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator2 rend2()
    {
        return reverse_iterator2(begin2());
    }


    //@} Reverse iterators

    //@{ Iterators types

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR

    public: class const_iterator1:  public container_const_reference<generalized_diagonal_matrix>,
                                    public random_access_iterator_base<packed_random_access_iterator_tag, const_iterator1, value_type>
    {
        public: typedef typename generalized_diagonal_matrix::value_type value_type;
        public: typedef typename generalized_diagonal_matrix::difference_type difference_type;
        public: typedef typename generalized_diagonal_matrix::const_reference reference;
        public: typedef const typename generalized_diagonal_matrix::pointer pointer;

        public: typedef const_iterator2 dual_iterator_type;
        public: typedef const_reverse_iterator2 dual_reverse_iterator_type;

        // Construction and destruction
        public: BOOST_UBLAS_INLINE
            const_iterator1()
            : container_const_reference<self_type>(),
              it1_(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1(self_type const& m, size_type it1, size_type it2)
            : container_const_reference<self_type>(m),
              it1_(it1),
              it2_(it2)
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1(const iterator1 &it)
            : container_const_reference<self_type>(it()),
              it1_(it.it1_),
              it2_(it.it2_)
        {
        }


        // Arithmetic
        public: BOOST_UBLAS_INLINE
            const_iterator1& operator++()
        {
            ++it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator--()
        {
            --it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator+=(difference_type n)
        {
            it1_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator-=(difference_type n)
        {
            it1_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ - it.it1_;
        }


        // Dereference
        public: BOOST_UBLAS_INLINE
            const_reference operator*() const
        {
            return (*this)()(it1_, it2_);
        }


        public: BOOST_UBLAS_INLINE
            const_reference operator[](difference_type n) const
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
            difference_type k = (*this)().offset();
            //return (*this)().find2(1, it1_, it1_ + (k > 0 ? k : 0));
            return (*this)().find2(1, it1_, it1_ + k);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator2 end() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find2(1, it1_, it1_ + (k > 0 ? k : 0) + 1);
            return (*this)().find2(1, it1_, it1_ + k + 1);
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
#endif


        // Indices
        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it1_;
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_;
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            const_iterator1& operator=(const_iterator1 const& it)
        {
            container_const_reference<self_type>::assign (&it ());
            it1_ = it.it1_;
            it2_ = it.it2_;
            return *this;
        }


        // Comparison
        public: BOOST_UBLAS_INLINE
            bool operator==(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ == it.it1_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ < it.it1_;
        }


        private: size_type it1_;
        private: size_type it2_;
    };


    public: class iterator1:    public container_reference<generalized_diagonal_matrix>,
                                public random_access_iterator_base<packed_random_access_iterator_tag, iterator1, value_type>
    {
        public: typedef typename generalized_diagonal_matrix::value_type value_type;
        public: typedef typename generalized_diagonal_matrix::difference_type difference_type;
        public: typedef typename generalized_diagonal_matrix::reference reference;
        public: typedef typename generalized_diagonal_matrix::pointer pointer;

        public: typedef iterator2 dual_iterator_type;
        public: typedef reverse_iterator2 dual_reverse_iterator_type;


        // Construction and destruction
        public: BOOST_UBLAS_INLINE
            iterator1()
            : container_reference<self_type>(),
              it1_(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            iterator1(self_type& m, size_type it1, size_type it2)
            : container_reference<self_type>(m),
              it1_(it1),
              it2_(it2)
        {
        }


        // Arithmetic
        public: BOOST_UBLAS_INLINE
            iterator1& operator++()
        {
            ++it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator--()
        {
            --it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator+=(difference_type n)
        {
            it1_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator-=(difference_type n)
        {
            it1_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ - it.it1_;
        }


        // Dereference
        public: BOOST_UBLAS_INLINE
            reference operator*() const
        {
            return (*this)().at_element(it1_, it2_);
        }


        public: BOOST_UBLAS_INLINE
            reference operator[](difference_type n) const
        {
            return *(*this + n);
        }


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 begin() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find2(1, it1_, it1_ + (k > 0 ? k : 0));
            return (*this)().find2(1, it1_, it1_ + k);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 end() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find2(1, it1_, it1_ + (k > 0 ? k : 0) + 1);
            return (*this)().find2(1, it1_, it1_ + k + 1);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rbegin() const
        {
            return reverse_iterator2(end());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rend() const
        {
            return reverse_iterator2(begin());
        }
#endif


        // Indices
        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it1_;
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_;
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            iterator1& operator=(iterator1 const& it)
        {
            container_reference<self_type>::assign(&it());
            it1_ = it.it1_;
            it2_ = it.it2_;
            return *this;
        }


        // Comparison
        public: BOOST_UBLAS_INLINE
            bool operator==(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ == it.it1_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
            return it1_ < it.it1_;
        }


        private: size_type it1_;
        private: size_type it2_;
        private: friend class const_iterator1;
    };


    public: class const_iterator2:  public container_const_reference<generalized_diagonal_matrix>,
                                    public random_access_iterator_base<packed_random_access_iterator_tag, const_iterator2, value_type>
    {
        public: typedef typename generalized_diagonal_matrix::value_type value_type;
        public: typedef typename generalized_diagonal_matrix::difference_type difference_type;
        public: typedef typename generalized_diagonal_matrix::const_reference reference;
        public: typedef const typename generalized_diagonal_matrix::pointer pointer;

        public: typedef const_iterator1 dual_iterator_type;
        public: typedef const_reverse_iterator1 dual_reverse_iterator_type;

        // Construction and destruction
        public: BOOST_UBLAS_INLINE
            const_iterator2()
            : container_const_reference<self_type>(),
              it1_(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2(self_type const& m, size_type it1, size_type it2)
            : container_const_reference<self_type>(m),
              it1_(it1),
              it2_(it2)
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2(iterator2 const& it)
            : container_const_reference<self_type>(it()),
              it1_(it.it1_),
              it2_(it.it2_)
        {
        }


        // Arithmetic
        public: BOOST_UBLAS_INLINE
            const_iterator2& operator++()
        {
            ++it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator--()
        {
            --it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator+=(difference_type n)
        {
            it2_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator-=(difference_type n)
        {
            it2_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ - it.it2_;
        }


        // Dereference
        public: BOOST_UBLAS_INLINE
            const_reference operator*() const
        {
            return (*this)()(it1_, it2_);
        }


        public: BOOST_UBLAS_INLINE
            const_reference operator[](difference_type n) const
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
            difference_type k = (*this)().offset();
            //return (*this)().find1(1, it2_ - (k < 0 ? k : 0), it2_);
            return (*this)().find1(1, it2_ - k, it2_);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator1 end() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find1(1, it2_ - (k < 0 ? k : 0) + 1, it2_);
            return (*this)().find1(1, it2_ - k + 1, it2_);
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
#endif


        // Indices
        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it1_;
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_;
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            const_iterator2& operator=(const_iterator2 const& it)
        {
            container_const_reference<self_type>::assign(&it());
            it1_ = it.it1_;
            it2_ = it.it2_;

            return *this;
        }


        // Comparison
        public: BOOST_UBLAS_INLINE
            bool operator==(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ == it.it2_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ < it.it2_;
        }


        private: size_type it1_;
        private: size_type it2_;
    };


    public: class iterator2:    public container_reference<generalized_diagonal_matrix>,
                                public random_access_iterator_base<packed_random_access_iterator_tag, iterator2, value_type>
    {
        public: typedef typename generalized_diagonal_matrix::value_type value_type;
        public: typedef typename generalized_diagonal_matrix::difference_type difference_type;
        public: typedef typename generalized_diagonal_matrix::reference reference;
        public: typedef typename generalized_diagonal_matrix::pointer pointer;

        public: typedef iterator1 dual_iterator_type;
        public: typedef reverse_iterator1 dual_reverse_iterator_type;

        // Construction and destruction
        public: BOOST_UBLAS_INLINE
            iterator2()
            : container_reference<self_type>(),
              it1_(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            iterator2(self_type& m, size_type it1, size_type it2)
            : container_reference<self_type>(m),
              it1_(it1),
              it2_(it2)
        {
        }


        // Arithmetic
        public: BOOST_UBLAS_INLINE
            iterator2& operator++()
        {
            ++it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator--()
        {
            --it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator+=(difference_type n)
        {
            it2_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator-=(difference_type n)
        {
            it2_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ - it.it2_;
        }


        // Dereference
        public: BOOST_UBLAS_INLINE
            reference operator*() const
        {
            return (*this)().at_element(it1_, it2_);
        }


        public: BOOST_UBLAS_INLINE
            reference operator[](difference_type n) const
        {
            return *(*this + n);
        }


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 begin() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find1(1, it2_ - (k < 0 ? k : 0), it2_);
            return (*this)().find1(1, it2_ - k, it2_);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 end() const
        {
            difference_type k = (*this)().offset();
            //return (*this)().find1(1, it2_ - (k < 0 ? k : 0) + 1, it2_);
            return (*this)().find1(1, it2_ - k + 1, it2_);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rbegin() const
        {
            return reverse_iterator1(end());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rend() const
        {
            return reverse_iterator1(begin());
        }
#endif


        // Indices
        public: BOOST_UBLAS_INLINE
            size_type index1 () const
        {
            return it1_;
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_;
        }

        // Assignment
        public: BOOST_UBLAS_INLINE
            iterator2& operator=(iterator2 const& it)
        {
            container_reference<self_type>::assign (&it ());
            it1_ = it.it1_;
            it2_ = it.it2_;
            return *this;
        }


        // Comparison
        public: BOOST_UBLAS_INLINE
            bool operator==(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ == it.it2_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this) () == &it (), external_logic());
            BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
            return it2_ < it.it2_;
        }


        private: size_type it1_;
        private: size_type it2_;
        private: friend class const_iterator2;
    };

#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR


    //@} Iterators types


    /// The number of rows
    private: size_type size1_;
    /// The number of columns
    private: size_type size2_;
    /**
     * \brief The diagonal offset: \c 0 for the main diagonal,
     *  <code>&gt; 0</code> for super-diagonals,
     *  and <code>&lt; 0</code> for sub-diagonals.
     */
    private: difference_type k_;
    /// The starting row of the diagonal.
    private: size_type r_;
    /// The starting column of the diagonal.
    private: size_type c_;
    /// The data array.
    private: array_type data_;
    /// The zero value used as return value for elements off the diagonal.
    private: static const_value_type zero_;
};


template<
    typename ValueT,
    typename LayoutT,
    typename ArrayT
>
typename generalized_diagonal_matrix<ValueT,LayoutT,ArrayT>::const_value_type generalized_diagonal_matrix<ValueT,LayoutT,ArrayT>::zero_ = generalized_diagonal_matrix<ValueT,LayoutT,ArrayT>::value_type/*zero*/();


/**
 * \brief Generalized diagonal matrix adaptor: convert any matrix into a
 *  generalized diagonal matrix.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixT>
class generalized_diagonal_adaptor: public matrix_expression< generalized_diagonal_adaptor<MatrixT> >
{
    private: typedef generalized_diagonal_adaptor<MatrixT> self_type;

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
    public: using matrix_expression<self_type>::operator ();
#endif

    public: typedef const MatrixT const_matrix_type;
    public: typedef MatrixT matrix_type;
    public: typedef typename matrix_traits<MatrixT>::size_type size_type;
    public: typedef typename matrix_traits<MatrixT>::difference_type difference_type;
    public: typedef typename matrix_traits<MatrixT>::value_type value_type;
    public: typedef const value_type const_value_type;
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
    // Replaced by _temporary_traits to avoid type requirements on MatrixT
    //typedef typename MatrixT::vector_temporary_type vector_temporary_type;
    //typedef typename MatrixT::matrix_temporary_type matrix_temporary_type;
    public: typedef typename storage_restrict_traits<
                                typename matrix_traits<MatrixT>::storage_category,
                                packed_proxy_tag
                >::storage_category storage_category;
    public: typedef typename matrix_traits<MatrixT>::orientation_category orientation_category;


    // Construction and destruction

    public: BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor(matrix_type& data, difference_type k = 0)
        : matrix_expression<self_type>(),
          data_(data),
          k_(k),
          r_(k < 0 ? -k : 0),
          c_(k > 0 ?  k : 0)
    {
        // Empty
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor(generalized_diagonal_adaptor const& m)
        : matrix_expression<self_type>(),
          data_(m.data_),
          k_ (m.k_),
          r_(m.r_),
          c_(m.c_)
    {
        // Empty
    }


    // Accessors

    public: BOOST_UBLAS_INLINE
        size_type size1() const
    {
        return data_.size1();
    }


    public: BOOST_UBLAS_INLINE
        size_type size2() const
    {
        return data_.size2();
    }


    public: BOOST_UBLAS_INLINE
        difference_type offset() const
    {
        return k_;
    }


    // Storage accessors

    public: BOOST_UBLAS_INLINE
        const matrix_closure_type& data() const
    {
        return data_;
    }


    public: BOOST_UBLAS_INLINE
        matrix_closure_type &data ()
    {
        return data_;
    }


    // Element access


#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER

    public: BOOST_UBLAS_INLINE
        const_reference operator()(size_type i, size_type j) const
    {
        BOOST_UBLAS_CHECK(i < size1(), bad_index());
        BOOST_UBLAS_CHECK(j < size2(), bad_index());

        const difference_type dr(i-r_);
        const difference_type dc(j-c_);

        if (dr == dc)
        {
            return data()(i,j);
        }

        return zero_;
    }


    public: BOOST_UBLAS_INLINE
        reference operator()(size_type i, size_type j)
    {
        BOOST_UBLAS_CHECK(i < size1(), bad_index());
        BOOST_UBLAS_CHECK(j < size2(), bad_index());

        const difference_type dr(i-r_);
        const difference_type dc(j-c_);

        if (dr == dc)
        {
            return data()(i,j);
        }

#ifndef BOOST_UBLAS_REFERENCE_CONST_MEMBER
        bad_index().raise();
#endif
        return const_cast<reference>(zero_);
    }


#else // BOOST_UBLAS_PROXY_CONST_MEMBER


    public: BOOST_UBLAS_INLINE
        reference operator()(size_type i, size_type j) const
    {
        BOOST_UBLAS_CHECK(i < size1(), bad_index());
        BOOST_UBLAS_CHECK(j < size2(), bad_index());

        const difference_type dr(i-r_);
        const difference_type dc(j-c_);

        if (dr == dc)
        {
            return data()(i,j);
        }

#ifndef BOOST_UBLAS_REFERENCE_CONST_MEMBER
        bad_index().raise();
#endif
        return const_cast<reference>(zero_);
    }

#endif // BOOST_UBLAS_PROXY_CONST_MEMBER


    // Assignment


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator=(generalized_diagonal_adaptor const& m)
    {
        //[FIXME]
        // These instructions are to allow assignment of matrices with
        // different structure (i.e., different diagonal offset).
        // However, this change the rationale under the adaptor matrix
        // assignment (e.g., for banded_adaptor, cannot assign two matrices with
        // different bands limit).
        k_ = m.k_;
        r_ = m.r_;
        c_ = m.c_;
        //[/FIXME]

        matrix_assign<scalar_assign>(*this, m);
        return *this;
    }


    public: BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& assign_temporary(generalized_diagonal_adaptor& m)
    {
        *this = m;
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator=(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_assign>(*this, matrix<value_type>(ae));
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& assign(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_assign>(*this, ae);
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator+=(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_assign>(*this, matrix<value_type>(*this + ae));
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& plus_assign(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_plus_assign>(*this, ae);
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator-=(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_assign>(*this, matrix<value_type>(*this -ae));
        return *this;
    }


    public: template<class AE>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& minus_assign(matrix_expression<AE> const& ae)
    {
        matrix_assign<scalar_minus_assign>(*this, ae);
        return *this;
    }


    public: template<class AT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator*=(AT const &at)
    {
        matrix_assign_scalar<scalar_multiplies_assign>(*this, at);
        return *this;
    }


    public: template<class AT>
        BOOST_UBLAS_INLINE
        generalized_diagonal_adaptor& operator/=(AT const& at)
    {
        matrix_assign_scalar<scalar_divides_assign>(*this, at);
        return *this;
    }


    // Closure comparison
    public: BOOST_UBLAS_INLINE
        bool same_closure(generalized_diagonal_adaptor const& ba) const
    {
        return (*this).data().same_closure(ba.data());
    }


    // Swapping


    public: BOOST_UBLAS_INLINE
        void swap(generalized_diagonal_adaptor& m)
    {
        if (this != &m)
        {
            BOOST_UBLAS_CHECK(k_ == m.k_, bad_size());

            matrix_swap<scalar_swap>(*this, m);
        }
    }


    public: BOOST_UBLAS_INLINE
        friend void swap(generalized_diagonal_adaptor& m1, generalized_diagonal_adaptor& m2)
    {
        m1.swap(m2);
    }


    // Iterator types


    // Use the matrix MatrixT iterator
    private: typedef typename matrix_traits<MatrixT>::const_iterator1 const_subiterator1_type;
    private: typedef typename ::boost::mpl::if_<
                                ::boost::is_const<MatrixT>,
                                typename matrix_traits<MatrixT>::const_iterator1,
                                typename matrix_traits<MatrixT>::iterator1
                >::type subiterator1_type;
    private: typedef typename matrix_traits<MatrixT>::const_iterator2 const_subiterator2_type;
    private: typedef typename ::boost::mpl::if_<
                                ::boost::is_const<MatrixT>,
                                typename matrix_traits<MatrixT>::const_iterator2,
                                typename matrix_traits<MatrixT>::iterator2
                >::type subiterator2_type;


#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: typedef indexed_iterator1<self_type, packed_random_access_iterator_tag> iterator1;
    public: typedef indexed_iterator2<self_type, packed_random_access_iterator_tag> iterator2;
    public: typedef indexed_const_iterator1<self_type, packed_random_access_iterator_tag> const_iterator1;
    public: typedef indexed_const_iterator2<self_type, packed_random_access_iterator_tag> const_iterator2;
#else
    public: class const_iterator1;
    public: class iterator1;
    public: class const_iterator2;
    public: class iterator2;
#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR
    public: typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
    public: typedef reverse_iterator_base1<iterator1> reverse_iterator1;
    public: typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
    public: typedef reverse_iterator_base2<iterator2> reverse_iterator2;


    // Element lookup


    public: BOOST_UBLAS_INLINE
        const_iterator1 find1(int rank, size_type i, size_type j) const
    {
//      if (rank == 1)
//      {
//          // Safety check: avoid underflow
//          if (j < c_)
//          {
//              j = c_;
//          }
//
//          // Make sure that: r_ <= j-k_ <= i <= j-k_+1 <= r_+min(size1-r_, size2-c_)
//          i = ::std::max(r_, ::std::max(j-k_, ::std::min(i, ::std::min(j+1-k_, r_ + ::std::min(size1()-r_, size2()-c_)))));
//      }
//      else
//      {
//          // Make sure that: r_ <= i <= min(size1-r_, size2-c_)
//          i = ::std::max(r_, ::std::min(i, r_ + ::std::min(size1()-r_, size2()-c_)));
//      }

        i = ::std::min(::std::max(i, r_), r_ + ::std::min(size1()-r_, size2()-c_));

        return const_iterator1(*this, data().find1(rank, i, j));
    }


    public: BOOST_UBLAS_INLINE
        iterator1 find1(int rank, size_type i, size_type j)
    {
//      if (rank == 1)
//      {
//          // Safety check: avoid underflow
//          if (j < c_)
//          {
//              j = c_;
//          }
//
//          // Make sure that: r_ <= j-k_ <= i <= j-k_+1 <= r_+min(size1-r_, size2-c_)
//          i = ::std::max(r_, ::std::max(j-k_, ::std::min(i, ::std::min(j+1-k_, r_ + ::std::min(size1()-r_, size2()-c_)))));
//      }
//      else
//      {
//          // Make sure that: r_ <= i <= r_+min(size1-r_, size2-c_)
//          i = ::std::max(r_, ::std::min(i, r_ + ::std::min(size1()-r_, size2()-c_)));
//      }

        i = ::std::min(::std::max(i, r_), r_ + ::std::min(size1()-r_, size2()-c_));

        return iterator1(*this, data().find1(rank, i, j));
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 find2(int rank, size_type i, size_type j) const
    {
//      if (rank == 1)
//      {
//          // Safety check: avoid underflow
//          if (i < r_)
//          {
//              i = r_;
//          }
//
//          // Make sure that: c_ <= i+k_ <= j <= i+k+1 <= c_+min(size1-r_, size2-c_)
//          j = ::std::max(c_, ::std::max(i+k_, ::std::min(j, ::std::min(i+1+k_, c_ + ::std::min(size1()-r_, size2()-c_)))));
//      }
//      else
//      {
//          // Make sure that: c_ <= j <= c_+min(size1-r_, size2-c_)
//          j = ::std::max(c_, ::std::min(j, c_ + ::std::min(size1()-r_, size2()-c_)));
//      }

        j = ::std::min(::std::max(j, c_), c_ + ::std::min(size1()-r_, size2()-c_));

        return const_iterator2(*this, data().find2(rank, i, j));
    }


    public: BOOST_UBLAS_INLINE
        iterator2 find2(int rank, size_type i, size_type j)
    {
//      if (rank == 1)
//      {
//          // Safety check: avoid underflow
//          if (i < r_)
//          {
//              i = r_;
//          }
//
//          // Make sure that: c_ <= i+k_ <= j <= i+k+1 <= c_+min(size1-r_, size2-c_)
//          j = ::std::max(c_, ::std::max(i+k_, ::std::min(j, ::std::min(i+1+k_, c_ + ::std::min(size1()-r_, size2()-c_)))));
//      }
//      else
//      {
//          // Make sure that: c_ <= j <= c_+min(size1-r_, size2-c_)
//          j = ::std::max(c_, ::std::min(j, c_ + ::std::min(size1()-r_, size2()-c_)));
//      }

        j = ::std::min(::std::max(j, c_), c_ + ::std::min(size1()-r_, size2()-c_));

        return iterator2(*this, data().find2(rank, i, j));
    }


    public: BOOST_UBLAS_INLINE
        const_iterator1 begin1() const
    {
        //return find1(0, 0, 0);
        return find1(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator1 end1() const
    {
        //return find1(0, size1(), 0);
        return find1(0, r_ + ::std::min(size1()-r_, size2()-c_), c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator1 begin1()
    {
        //return find1(0, 0, 0);
        return find1(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator1 end1()
    {
        //return find1(0, size1(), 0);
        return find1(0, r_ + ::std::min(size1()-r_, size2()-c_), c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 begin2() const
    {
        //return find2(0, 0, 0);
        return find2(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        const_iterator2 end2() const
    {
        //return find2 (0, 0, size2());
        return find2(0, r_, c_ + ::std::min(size1()-r_, size2()-c_));
    }


    public: BOOST_UBLAS_INLINE
        iterator2 begin2()
    {
        //return find2(0, 0, 0);
        return find2(0, r_, c_);
    }


    public: BOOST_UBLAS_INLINE
        iterator2 end2()
    {
        //return find2(0, 0, size2());
        return find2(0, r_, c_ + ::std::min(size1()-r_, size2()-c_));
    }


    // Reverse iterators

    public: BOOST_UBLAS_INLINE
        const_reverse_iterator1 rbegin1() const
    {
        return const_reverse_iterator1(end1());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator1 rend1() const
    {
        return const_reverse_iterator1(begin1());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator1 rbegin1()
    {
        return reverse_iterator1(end1());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator1 rend1()
    {
        return reverse_iterator1(begin1());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator2 rbegin2() const
    {
        return const_reverse_iterator2(end2());
    }


    public: BOOST_UBLAS_INLINE
        const_reverse_iterator2 rend2() const
    {
        return const_reverse_iterator2(begin2());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator2 rbegin2()
    {
        return reverse_iterator2(end2());
    }


    public: BOOST_UBLAS_INLINE
        reverse_iterator2 rend2()
    {
        return reverse_iterator2(begin2());
    }


    // Iterator types: Iterators simply are indices.


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR

    public: class const_iterator1: public container_const_reference<generalized_diagonal_adaptor>,
                                   public random_access_iterator_base<
                                            typename iterator_restrict_traits<
                                                typename const_subiterator1_type::iterator_category,
                                                packed_random_access_iterator_tag
                                            >::iterator_category,
                                            const_iterator1,
                                            value_type
                                    >
    {
        public: typedef typename const_subiterator1_type::value_type value_type;
        public: typedef typename const_subiterator1_type::difference_type difference_type;
        public: typedef typename const_subiterator1_type::reference reference;
        public: typedef typename const_subiterator1_type::pointer pointer;
        public: typedef const_iterator2 dual_iterator_type;
        public: typedef const_reverse_iterator2 dual_reverse_iterator_type;


        // Construction and destruction


        public: BOOST_UBLAS_INLINE
            const_iterator1()
            : container_const_reference<self_type>(),
              it1_()
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1(self_type const& m, const_subiterator1_type const& it1)
            : container_const_reference<self_type>(m),
              it1_(it1)
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1(iterator1 const& it)
            : container_const_reference<self_type>(it()),
              it1_(it.it1_)
        {
        }


        // Arithmetic


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator++()
        {
            ++it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator--()
        {
            --it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator+=(difference_type n)
        {
            it1_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator1& operator-=(difference_type n)
        {
            it1_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ - it.it1_;
        }


        // Dereference


        public: BOOST_UBLAS_INLINE
            const_reference operator*() const
        {
            size_type i = index1();
            size_type j = index2();

            BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
            BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());

            const difference_type k = (*this)().offset();
            const difference_type dr(i - (k < 0 ? -k : 0));
            const difference_type dc(j - (k > 0 ?  k : 0));

            if (dr == dc)
            {
                return *it1_;
            }

            return (*this)()(i, j);
        }


        public: BOOST_UBLAS_INLINE
            const_reference operator[](difference_type n) const
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
            //return (*this)().find2(1, index1(), 0);
            return (*this)().find2(1, index1(), index1() + (*this)().offset());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator2 end() const
        {
            //return (*this)().find2(1, index1(), (*this)().size2());
            return (*this)().find2(1, index1(), index1() + (*this)().offset() + 1);
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
#endif


        // Indices

        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it1_.index1();
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it1_.index2();
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            const_iterator1& operator=(const_iterator1 const& it)
        {
            container_const_reference<self_type>::assign(&it());
            it1_ = it.it1_;

            return *this;
        }


        // Comparison


        public: BOOST_UBLAS_INLINE
            bool operator==(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ == it.it1_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(const_iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ < it.it1_;
        }


        private: const_subiterator1_type it1_;
    };


    public: class iterator1: public container_reference<generalized_diagonal_adaptor>,
                             public random_access_iterator_base<
                                        typename iterator_restrict_traits<
                                            typename subiterator1_type::iterator_category,
                                            packed_random_access_iterator_tag
                                        >::iterator_category,
                                        iterator1,
                                        value_type
                                >
    {
        private: friend class const_iterator1;


        public: typedef typename subiterator1_type::value_type value_type;
        public: typedef typename subiterator1_type::difference_type difference_type;
        public: typedef typename subiterator1_type::reference reference;
        public: typedef typename subiterator1_type::pointer pointer;
        public: typedef iterator2 dual_iterator_type;
        public: typedef reverse_iterator2 dual_reverse_iterator_type;


        // Construction and destruction


        public: BOOST_UBLAS_INLINE
            iterator1()
            : container_reference<self_type>(),
              it1_()
        {
        }


        public: BOOST_UBLAS_INLINE
            iterator1(self_type& m, subiterator1_type const& it1)
            : container_reference<self_type>(m),
              it1_(it1)
        {
        }


        // Arithmetic


        public: BOOST_UBLAS_INLINE
            iterator1& operator++()
        {
            ++it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator--()
        {
            --it1_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator+=(difference_type n)
        {
            it1_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator1& operator-=(difference_type n)
        {
            it1_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ - it.it1_;
        }


        // Dereference


        public: BOOST_UBLAS_INLINE
            reference operator*() const
        {
            size_type i = index1();
            size_type j = index2();

            BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
            BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());

            const difference_type k = (*this)().offset();
            const difference_type dr(i - (k < 0 ? -k : 0));
            const difference_type dc(j - (k > 0 ?  k : 0));

            if (dr == dc)
            {
                return *it1_;
            }

            return (*this)()(i, j);
        }


        public: BOOST_UBLAS_INLINE
            reference operator[](difference_type n) const
        {
            return *(*this + n);
        }


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 begin() const
        {
            //return (*this)().find2(1, index1(), 0);
            return (*this)().find2(1, index1(), index1() + (*this)().offset());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 end() const
        {
            //return (*this)().find2(1, index1(), (*this)().size2());
            return (*this)().find2(1, index1(), index1() + (*this)().offset() + 1);
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rbegin() const
        {
            return reverse_iterator2(end());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rend() const
        {
            return reverse_iterator2(begin());
        }

#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


        // Indices


        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it1_.index1();
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it1_.index2();
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            iterator1& operator=(iterator1 const& it)
        {
            container_reference<self_type>::assign(&it());
            it1_ = it.it1_;
            return *this;
        }


        // Comparison


        public: BOOST_UBLAS_INLINE
            bool operator==(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ == it.it1_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(iterator1 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it1_ < it.it1_;
        }


        private: subiterator1_type it1_;
    };


    public: class const_iterator2: public container_const_reference<generalized_diagonal_adaptor>,
                                   public random_access_iterator_base<
                                        packed_random_access_iterator_tag,
                                        const_iterator2,
                                        value_type
                                    >
    {
        public: typedef typename iterator_restrict_traits<
                                    typename const_subiterator2_type::iterator_category,
                                    packed_random_access_iterator_tag
                    >::iterator_category iterator_category;
        public: typedef typename const_subiterator2_type::value_type value_type;
        public: typedef typename const_subiterator2_type::difference_type difference_type;
        public: typedef typename const_subiterator2_type::reference reference;
        public: typedef typename const_subiterator2_type::pointer pointer;
        public: typedef const_iterator1 dual_iterator_type;
        public: typedef const_reverse_iterator1 dual_reverse_iterator_type;


        // Construction and destruction


        public: BOOST_UBLAS_INLINE
            const_iterator2()
            : container_const_reference<self_type>(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2(self_type const& m, const_subiterator2_type const& it2)
            : container_const_reference<self_type>(m),
              it2_(it2)
        {
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2(iterator2 const& it)
            : container_const_reference<self_type>(it()),
              it2_(it.it2_)
        {
        }


        // Arithmetic
        public: BOOST_UBLAS_INLINE
            const_iterator2& operator++()
        {
            ++it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator--()
        {
            --it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator+=(difference_type n)
        {
            it2_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            const_iterator2& operator-=(difference_type n)
        {
            it2_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            return it2_ - it.it2_;
        }


        // Dereference


        public: BOOST_UBLAS_INLINE
            const_reference operator*() const
        {
            size_type i = index1();
            size_type j = index2();

            BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
            BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());

            const difference_type k = (*this)().offset();
            const difference_type dr(i - (k < 0 ? -k : 0));
            const difference_type dc(j - (k > 0 ?  k : 0));

            if (dr == dc)
            {
                return *it2_;
            }

            return (*this)()(i, j);
        }


        public: BOOST_UBLAS_INLINE
            const_reference operator[](difference_type n) const
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
            //return (*this)().find1(1, 0, index2());
            return (*this)().find1(1, index2() - (*this)().offset(), index2());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator1 end() const
        {
            //return (*this)().find1(1, (*this)().size1(), index2());
            return (*this)().find1(1, index2() - (*this)().offset() + 1, index2());
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


        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it2_.index1();
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_.index2();
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            const_iterator2& operator=(const_iterator2 const& it)
        {
            container_const_reference<self_type>::assign(&it());
            it2_ = it.it2_;
            return *this;
        }


        // Comparison


        public: BOOST_UBLAS_INLINE
            bool operator==(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            return it2_ == it.it2_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(const_iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            return it2_ < it.it2_;
        }


        private: const_subiterator2_type it2_;
    };


    public: class iterator2: public container_reference<generalized_diagonal_adaptor>,
                             public random_access_iterator_base<
                                        typename iterator_restrict_traits<
                                               typename subiterator2_type::iterator_category,
                                                packed_random_access_iterator_tag
                                        >::iterator_category,
                                        iterator2,
                                        value_type
                                >
    {
        private: friend class const_iterator2;


        public: typedef typename subiterator2_type::value_type value_type;
        public: typedef typename subiterator2_type::difference_type difference_type;
        public: typedef typename subiterator2_type::reference reference;
        public: typedef typename subiterator2_type::pointer pointer;
        public: typedef iterator1 dual_iterator_type;
        public: typedef reverse_iterator1 dual_reverse_iterator_type;


        // Construction and destruction


        public: BOOST_UBLAS_INLINE
            iterator2()
            : container_reference<self_type>(),
              it2_()
        {
        }


        public: BOOST_UBLAS_INLINE
            iterator2(self_type& m, subiterator2_type const& it2)
            : container_reference<self_type>(m),
              it2_(it2)
        {
        }


        // Arithmetic


        public: BOOST_UBLAS_INLINE
            iterator2& operator++()
        {
            ++it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator--()
        {
            --it2_;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator+=(difference_type n)
        {
            it2_ += n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            iterator2& operator-=(difference_type n)
        {
            it2_ -= n;
            return *this;
        }


        public: BOOST_UBLAS_INLINE
            difference_type operator-(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
            return it2_ - it.it2_;
        }


        // Dereference


        public: BOOST_UBLAS_INLINE
            reference operator*() const
        {
            size_type i = index1();
            size_type j = index2();
            BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
            BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());

            const difference_type k = (*this)().offset();
            const difference_type dr(i - (k < 0 ? -k : 0));
            const difference_type dc(j - (k > 0 ?  k : 0));

            if (dr == dc)
            {
                return *it2_;
            }

            return (*this)()(i, j);
        }


        public: BOOST_UBLAS_INLINE
            reference operator[](difference_type n) const
        {
            return *(*this + n);
        }


#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 begin() const
        {
            //return (*this)().find1(1, 0, index2());
            return (*this)().find1(1, index2() - (*this)().offset(), index2());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 end() const
        {
            //return (*this)().find1(1, (*this)().size1(), index2());
            return (*this)().find1(1, index2() - (*this)().offset() + 1, index2());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rbegin() const
        {
            return reverse_iterator1(end());
        }


        public: BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rend() const
        {
            return reverse_iterator1(begin());
        }

#endif // BOOST_UBLAS_NO_NESTED_CLASS_RELATION


        // Indices


        public: BOOST_UBLAS_INLINE
            size_type index1() const
        {
            return it2_.index1();
        }


        public: BOOST_UBLAS_INLINE
            size_type index2() const
        {
            return it2_.index2();
        }


        // Assignment
        public: BOOST_UBLAS_INLINE
            iterator2& operator=(iterator2 const& it)
        {
            container_reference<self_type>::assign(&it());
            it2_ = it.it2_;

            return *this;
        }


        // Comparison


        public: BOOST_UBLAS_INLINE
            bool operator==(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it2_ == it.it2_;
        }


        public: BOOST_UBLAS_INLINE
            bool operator<(iterator2 const& it) const
        {
            BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());

            return it2_ < it.it2_;
        }


        private: subiterator2_type it2_;
    };

#endif // BOOST_UBLAS_USE_INDEXED_ITERATOR


    private: matrix_closure_type data_;
    private: difference_type k_;
    private: size_type r_;
    private: size_type c_;


    private: static const_value_type zero_;
};


template <typename MatrixT>
typename generalized_diagonal_adaptor<MatrixT>::const_value_type generalized_diagonal_adaptor<MatrixT>::zero_ = typename generalized_diagonal_adaptor<MatrixT>::value_type/*zero*/();

} // Namespace ublasx


namespace ublas {

// Specialization for temporary_traits


template <typename MatrixT>
struct vector_temporary_traits< ::boost::numeric::ublasx::generalized_diagonal_adaptor<MatrixT> >: vector_temporary_traits<MatrixT> {};


template <typename MatrixT>
struct vector_temporary_traits< const ::boost::numeric::ublasx::generalized_diagonal_adaptor<MatrixT> >: vector_temporary_traits<MatrixT> {};


template <typename MatrixT>
struct matrix_temporary_traits< ::boost::numeric::ublasx::generalized_diagonal_adaptor<MatrixT> >: matrix_temporary_traits<MatrixT> {};


template <typename MatrixT>
struct matrix_temporary_traits< const ::boost::numeric::ublasx::generalized_diagonal_adaptor<MatrixT> >: matrix_temporary_traits<MatrixT> {};

} // Namespace ublas

}} // Namespace boost::numeric


#endif // BOOST_NUMERIC_UBLASX_CONTAINER_GENERALIZED_DIAGONAL_MATRIX_HPP
