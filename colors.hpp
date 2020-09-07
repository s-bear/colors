// Copyright (c) 2020 Samuel B Powell

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.


#pragma once

#include <array>
#include <limits>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <numeric>

namespace color {

	//sneaky templates to force the compiler to delay evaluation in templates (e.g. for static_assert or type traits)
	template<class T> constexpr bool always_false = std::false_type::value;
	template<class T> constexpr bool always_true = std::true_type::value;

	template<class T, class U, template<class U> class class_t>
	using retype_ = class_t<T>;

	template<class T, class U, template<class U> class class_t>
	inline retype_<T, U, class_t> retype_f(const class_t<U>&) { return retype_<T, U, class_t>; }

	template<class T, class class_t>
	using retype = decltype(retype_f(std::declval<class_t>()));


	namespace cem {
		//constexpr math

		//absolute value
		template<typename T>
		constexpr T abs(T x) {
			return x < 0 ? -x : x;
		}
		
		//exp using series expansion O(N)
		template<typename T>
		constexpr T exp(T x, size_t N) {
			T y = 1 + x/(N+1);
			for (size_t i = N; i > 0; --i) y = 1 + y * (x / i);
			return y;
		}
		
		//sqrt using Newton's method, O(N)
		template<typename T>
		constexpr T sqrt(T x, T eps, size_t N) {
			if (x < 0) throw std::runtime_error("sqrt(x): x < 0");
			T y = 0.5*(1 + x);
			for (size_t i = 0; i < N; ++i) {
				y = 0.5*(y + x / y);
				if (y - x / y < eps) break;
			}
			return y;
		}
		
		//arithmetic-geometric mean, O(N)
		template<typename T>
		constexpr T agm(T x, T y, T eps, size_t N) {
			for (size_t i = 0; i < N; ++i) {
				T a = 0.5*(x + y);
				y = sqrt(x*y, eps, N);
				x = a;
				if (abs(x - y) < eps) break;
			}
			return x;
		}
		
		//log using agm & newton's method, O(N^2)
		template<typename T>
		constexpr T log(T x, size_t N) {
			//use AGM for an initial guess
			T y = 1.5707963267948966 / agm(1, 1 / (x * 64), 1e-5, N) - 5.545177444479562;
			//newton's method to refine
			for (size_t i = 0; i < N; ++i) {
				auto ey = exp(y,N);
				y += 2 * (x - ey) / (x + ey);
			}
			return y;
		}
		
		//pow, O(N^2)
		template<typename T>
		constexpr T pow(T x, T y, size_t N) {
			if (y == 0) return 1;
			if (y == 1) return x;
			if (y == 2) return x * x;
			if (y == 3) return x * x * x;
			return exp(log(x, N)*y,N);
		}
		//matrix transpose
		template<size_t R, size_t C, typename T>
		constexpr std::array<std::array<T, R>, C> trans(const std::array<std::array<T, C>, R> &A) {
			std::array<std::array<T, R>, C> B;
			for (size_t i = 0; i < R; ++i) for (size_t j = 0; j < C; ++j) B[j][i] = A[i][j];
			return B;
		}
		//in-place LUP decomposition
		template<size_t N, typename T>
		constexpr void lu(std::array<std::array<T,N>,N> &A, std::array<size_t,N+1> &P, T eps) {
			size_t i, j, k, imax;
			T maxA, absA;
			//init permutation vector
			for (i = 0; i <= N; ++i) P[i] = i;
			for (i = 0; i < N; ++i) {
				maxA = 0;
				imax = i;
				//find max in column
				for (k = i; k < N; ++k) {
					absA = abs(A[k][i]);
					if (absA > maxA) {
						maxA = absA;
						imax = k;
					}
				}
				//check tolerance
				if (maxA < eps) throw std::runtime_error("degenerate matrix");
				if (imax != i) {
					//pivot rows
					std::swap(P[i], P[imax]);
					std::swap(A[i], A[imax]);
					//for (k = 0; k < N; ++k) std::swap(A[i][k], A[imax][k]);
					P[N]++;
				}
				for (j = i + 1; j < N; ++j) {
					A[j][i] /= A[i][i];
					for (k = i + 1; k < N; ++k) A[j][k] -= A[j][i] * A[i][k];
				}
			}
		}
		
		//matrix inverse based on LUP decomposition
		template<size_t N, typename T>
		constexpr std::array<std::array<T, N>,N> inv(std::array<std::array<T, N>,N> A, T eps) {
			std::array<size_t, N + 1> P;
			lu(A, P, eps);
			std::array<std::array<T,N>, N> iA;
			for (size_t j = 0; j < N; ++j) {
				for (size_t i = 0; i < N; ++i) {
					iA[i][j] = 0;
					if (P[i] == j) iA[i][j] = 1;
					for (size_t k = 0; k < i; ++k) iA[i][j] -= A[i][k] * iA[k][j];
				}
				for (int i = N - 1; i >= 0; --i) {
					for (size_t k = i + 1; k < N; ++k) iA[i][j] -= A[i][k] * iA[k][j];
					iA[i][j] /= A[i][i];
				}
			}
			return iA;
		}
	}

	template<typename int_t, typename float_t>
	int_t quantize(float_t x, int_t max = 0, int_t min = 0) {
		if (max == 0) max = std::numeric_limits<int_t>::max();
		float_t q = std::round((max - min)*x + min);
		if (q < (float_t)min) return min;
		if (q > (float_t)max) return max;
		return (int_t)q;
	}


//we have a bunch of typedefs to reduce the boilerplate of defining color types based on std::array

#define COLOR_D_OP(color_t,arr,op) \
	constexpr color_t& operator##op(const color_t &c) noexcept(noexcept(arr[0] op c.arr[0])) { \
		for(size_t i = 0; i < std::size(arr); ++i) arr[i] op c.arr[i]; \
		return *this; }\
	template<class U> \
	constexpr color_t& operator##op(const U &s) noexcept(noexcept(arr[0] op s)) {\
		for(auto &x : arr) x op s; \
		return *this; }

#define COLOR_DEFINE_OPS(color_t,arr) COLOR_D_OP(color_t,arr,+=) COLOR_D_OP(color_t,arr,-=) COLOR_D_OP(color_t,arr,*=) COLOR_D_OP(color_t,arr,/=)

#define COLOR_STDARRAY_TYPEDEFS \
	typedef typename array_type::value_type value_type;\
	typedef typename array_type::size_type size_type;\
	typedef typename array_type::difference_type difference_type;\
	typedef typename array_type::reference reference;\
	typedef typename array_type::const_reference const_reference;\
	typedef typename array_type::pointer pointer;\
	typedef typename array_type::const_pointer const_pointer;\
	typedef typename array_type::iterator iterator;\
	typedef typename array_type::const_iterator const_iterator;\
	typedef typename array_type::reverse_iterator reverse_iterator;\
	typedef typename array_type::const_reverse_iterator const_reverse_iterator;

#define COLOR_STDARRAY_MEMBERS(arr) \
	constexpr reference front() { return arr.front(); }\
	constexpr const_reference front() const { return arr.front(); }\
	constexpr reference back() { return arr.back(); }\
	constexpr const_reference back() const { return arr.back(); }\
	constexpr pointer data() noexcept { return arr.data(); }\
	constexpr const_pointer data() const noexcept { return arr.data(); }\
	constexpr iterator begin() noexcept { return arr.begin(); }\
	constexpr const_iterator begin() const noexcept { return arr.begin(); }\
	constexpr const_iterator cbegin() const noexcept { return arr.cbegin(); }\
	constexpr iterator end() noexcept { return arr.end(); }\
	constexpr const_iterator end() const noexcept { return arr.end(); }\
	constexpr const_iterator cend() const noexcept { return arr.cend(); }\
	constexpr reverse_iterator rbegin() noexcept { return arr.rbegin(); }\
	constexpr const_reverse_iterator rbegin() const noexcept { return arr.rbegin(); }\
	constexpr const_reverse_iterator crbegin() const noexcept { return arr.crbegin(); }\
	constexpr reverse_iterator rend() noexcept { return arr.rend(); }\
	constexpr const_reverse_iterator rend() const noexcept { return arr.rend(); }\
	constexpr const_reverse_iterator crend() const noexcept { return arr.crend(); }\
	constexpr bool empty() const noexcept { return arr.empty(); }\
	constexpr size_type size() const noexcept { return arr.size(); }\
	constexpr size_type max_size() const noexcept { return arr.max_size(); }\
	void fill(const value_type& value) { arr.fill(value); }\
	constexpr reference at(size_type pos) { return arr.at(pos); }\
	constexpr const_reference at(size_type pos) const { return arr.at(pos); }\
	constexpr reference operator[](size_type pos) { return arr[pos]; }\
	constexpr const_reference operator[](size_type pos) const { return arr[pos]; }

	template<class color_t, class AlwaysVoid>
	struct traits_ { static_assert(always_false<color_t>, "color::traits_<color_t> : color type is not supported!"); };
	
	template<class color_t>
	using traits = traits_<color_t, void>;

	struct color_space {};

	template<typename T>
	struct XYZ_space_ : color_space {
		template<typename space_t>
		constexpr bool operator==(const space_t&) const { 
			return std::is_same_v<XYZ_space_, space_t>;
		}
	};
	
	template<typename T>
	struct XYZ_ {
		typedef XYZ_space_<T> colorspace;
		typedef std::array<T, 3> array_type;

		static_assert(std::is_standard_layout_v<array_type>, "XYZ_<T> : std::array<T,3> must be standard layout");
		COLOR_STDARRAY_TYPEDEFS
		union {
			array_type arr;
			struct { T X, Y, Z; };
		};

		static constexpr XYZ_ xy(const T &x, const T &y, const T &Y = T(1)) {
			return XYZ_{ Y*x / y, Y, Y*(1 - x - y)/y };
		}
		static constexpr XYZ_ D50() {
			return xy(T(0.34567), T(0.35850), T(1.0));
		}
		static constexpr XYZ_ D65() {
			return xy(T(0.31271), T(0.32902), T(1.0));
		}
		constexpr bool operator==(const XYZ_ &other) {
			return arr == other.arr;
		}

		COLOR_STDARRAY_MEMBERS(arr)
		COLOR_DEFINE_OPS(XYZ_, arr)
	};
	
	typedef XYZ_space_<float> XYZ_space;
	typedef XYZ_<float> XYZ;
	
	template<typename T>
	struct traits_<XYZ_<T>, void> {
		static_assert(std::is_floating_point_v<T>, "XYZ colors only support floating point types");
		typedef XYZ_space_<T> colorspace;
		typedef XYZ_<T> convert_type;  //The XYZ colorspace is the root of the conversion tree
		static constexpr colorspace convert_space(const colorspace&) {
			return colorspace(); //The XYZ colorspace is the root of the conversion tree
		}
	};
	

	template<class A, class B>
	constexpr auto operator+(A a, B&& b) -> std::remove_reference_t<decltype(a += b)> {
		return a += b;
	}

	template<class A, class B>
	constexpr auto operator-(A a, B&& b) -> std::remove_reference_t<decltype(a -= b)> {
		return a -= b;
	}

	template<class A, class B>
	constexpr auto operator*(A a, B&& b) -> std::remove_reference_t<decltype(a *= b)> {
		return a *= b;
	}

	template<class A, class B>
	constexpr auto operator/(A a, B&& b) -> std::remove_reference_t<decltype(a /= b)> {
		return a /= b;
	}


	//Many colorspaces use a gamma function with a power curve above some threshold and a linear section below
	template<typename T>
	struct gamma_ {
		//find alpha & beta by solving: m*beta + c == alpha*pow(beta, gamma)-alpha+1 && m == gamma*alpha*pow(beta, gamma-1)
		//defaults are for linear
		const T g = 1, m = 1, c = 0, alpha = 1, beta = 1;

		T apply(T x) const noexcept {
			if (x > beta) return alpha*std::pow(x, g) - alpha + 1;
			else return m * x + c;
		}
		constexpr T apply_ce(T x) const noexcept {
			if (x > beta) return alpha*cem::pow(x, g, 10) - alpha + 1;
			else return m * x + c;
		}
		T revert(T y) const noexcept {
			if (y > m*beta+c) return std::pow((y + alpha - 1) / alpha, T(1) / g);
			else return (y-c) / m;
		}
		constexpr T revert_ce(T y) const noexcept {
			if (y > m*beta+c) return cem::pow((y + alpha - 1) / alpha, T(1) / g, 10);
			else return (y-c) / m;
		}

		T operator()(T x) const noexcept {
			return apply(x);
		}
		constexpr bool operator==(const gamma_& other) const {
			return g == other.g && m == other.m && c == other.c && alpha == other.alpha && beta == other.beta;
		}
	};
	
	template<typename T>
	constexpr gamma_<T> linear_gamma() {
		return gamma_<T>();
	}
	template<typename T>
	constexpr gamma_<T> Lab_gamma() {
		return gamma_<T>{T(1.0/3), T(7.787037037037036), T(16)/116, T(1), T(0.008856451679035631)};
	}
	template<typename T>
	struct Lab_space_ : color_space {
		typedef T value_type;
		const XYZ_<T> white;
		static constexpr gamma_<T> gamma = Lab_gamma<T>();
		constexpr Lab_space_(XYZ_<T> white) : white(white) {}
		template<class lab_space_t>
		constexpr std::enable_if_t<std::is_base_of_v<Lab_space_, lab_space_t>, bool>
			operator==(const lab_space_t& other) const {
			return white == other.white && gamma == other.gamma;
		}
		template<class lab_space_t>
		constexpr std::enable_if_t<!std::is_base_of_v<Lab_space_, lab_space_t>, bool>
			operator==(const lab_space_t& other) const {
			return false;
		}
	};
	template<typename T>
	struct Lab_D65_space_ : public Lab_space_<T> {
		constexpr Lab_D65_space_() : Lab_space_<T>(XYZ_<T>::D65()) {}
		using Lab_space_<T>::operator==;
	};

	template<class T, class colorspace_t>
	struct Lab_ {
		typedef colorspace_t colorspace;
		typedef std::array<T, 3> array_type;

		static_assert(std::is_base_of_v<Lab_space, colorspace>, "Lab_<T, colorspace> : colorspace must be derived from Lab_space");
		static_assert(std::is_standard_layout_v<array_type>, "Lab_<T, colorspace> : std::array<T,3> must be standard layout");

		COLOR_STDARRAY_TYPEDEFS
		union {
			array_type arr;
			struct { T L, a, b; };
		};
		constexpr bool operator==(const Lab_ &other) {
			return arr == other.arr;
		}

		COLOR_STDARRAY_MEMBERS(arr)
		COLOR_DEFINE_OPS(Lab_, arr)
	};

	template<class T, class colorspace_t>
	struct traits_<Lab_<T, colorspace_t>, void> {
		static_assert(std::is_floating_point_v<T>, "Lab colorspaces only support floating point types");
		typedef colorspace_t colorspace;
		typedef XYZ_<T> convert_type;
		static constexpr XYZ_space_<T> convert_space(const colorspace&) {
			return XYZ_space_<T>();
		}
	};

	typedef Lab_space_<float> Lab_space;
	typedef Lab_D65_space_<float> Lab_D65_space;
	typedef Lab_<float, Lab_D65_space> Lab;

	template<class T, class colorspace_t>
	struct LCh_ {
		typedef colorspace_t colorspace;
		typedef std::array<T, 3> array_type;

		static_assert(std::is_base_of_v<Lab_space, colorspace>, "LCh_<T, colorspace> : colorspace must be derived from Lab_space");
		static_assert(std::is_standard_layout_v<array_type>, "LCh_<T, colorspace> : std::array<T,3> must be standard layout");

		COLOR_STDARRAY_TYPEDEFS
		union {
			array_type arr;
			struct { T L, C, h; };
		};
		constexpr bool operator==(const LCh_ &other) {
			return arr == other.arr;
		}

		COLOR_STDARRAY_MEMBERS(arr)
	};
	
	typedef LCh_<float, Lab_D65_space> LCh;

	template<class T, class colorspace_t>
	struct traits_<LCh_<T, colorspace_t>, void> {
		static_assert(std::is_floating_point_v<T>, "LCh colorspaces only support floating point types");
		typedef colorspace_t colorspace;
		typedef Lab_<T, colorspace_t> convert_type;
		static constexpr colorspace convert_space(const colorspace& cs) {
			return cs; //Lab will use the same whitepoint so we need to copy the space
		}	
	};

	//BT.601-7, BT.709-6, BT.2020-2
	template<typename T>
	constexpr gamma_<T> BT_gamma() {
		return gamma_<T>{ T(0.45), T(4.5), T(0), T(1.09929682680944), T(0.018053968510807) };
	}
	template<typename T>
	constexpr gamma_<T> sRGB_gamma() {
		return gamma_<T>{ T(1) / T(2.4), T(12.95), T(0), T(1.054910859740092), T(0.003028729163821957) };
	}
	
	template<typename T>
	struct RGB_space_ : color_space {
		typedef decltype(T(1)*1.0f) float_type; //use float if T is an int type, otherwise match type of T
		typedef XYZ_<float_type> XYZ_t;
		//primaries
		const XYZ_t red, green, blue;
		//white and black points
		const XYZ_t white, black;
		//piecewise linear - gamma transfer function parameters
		const gamma_<float_type> gamma;
		//conversion matrices
		const std::array<std::array<float_type,3>,3> M_toXYZ, M_toRGB;

		constexpr RGB_space_(XYZ_t red, XYZ_t green, XYZ_t blue, gamma_<float_type> gamma = sRGB_gamma<float_type>(), XYZ_t white = XYZ_t::D65(), XYZ_t black = { 0 })
			: red(red), green(green), blue(blue), white(white), black(black), gamma(gamma),
			M_toXYZ(cem::trans(std::array<std::array<float_type,3>,3>{red.arr, green.arr, blue.arr})), M_toRGB{ cem::inv(M_toXYZ,float_type(1e-10))}
		{
			//nothing to do here
		}
		template<typename U>
		constexpr RGB_space_(const RGB_space_<U>& other) : 
			red(other.red), green(other.green), blue(other.blue), white(other.white), black(other.black), gamma(other.gamma),
			M_toRGB(other.M_toRGB), M_toXYZ(other.M_toXYZ)
		{
			//nothing to do here
		}
		template<class rgb_space_t>
		constexpr std::enable_if_t<std::is_base_of_v<RGB_space_, rgb_space_t>, bool>
		operator==(const rgb_space_t& other) const {
			return red == other.red && green == other.green && blue == other.blue && white == other.white && black == other.black && gamma == other.gamma;
		}
		template<class rgb_space_t>
		constexpr std::enable_if_t<!std::is_base_of_v<RGB_space_, rgb_space_t>, bool>
		operator==(const rgb_space_t& other) const {
			return false;
		}
	};
	template<typename T>
	struct sRGB_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr sRGB_space_() : RGB_space_<T>(XYZ_t::xy(0.64f, 0.33f, 0.2126f), XYZ_t::xy(0.3f, 0.6f, 0.7152f), XYZ_t::xy(0.15f, 0.06f, 0.0722f)) {}
	};
	template<class T>
	struct sRGB_linear_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr sRGB_linear_space_() : RGB_space_<T>(XYZ_t::xy(0.64f, 0.33f, 0.2126f), XYZ_t::xy(0.3f, 0.6f, 0.7152f), XYZ_t::xy(0.15f, 0.06f, 0.0722f), linear_gamma<T>()) {}
	};
	template<class T>
	struct RGB_BT601_7_525_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr RGB_BT601_7_525_space_() : RGB_space_<T>(XYZ_t::xy(0.630f, 0.340f, 0.299f), XYZ_t::xy(0.310f, 0.595f, 0.587f), XYZ_t::xy(0.155f, 0.070f, 0.114f), BT_gamma<T>()) {}
	};
	template<class T>
	struct RGB_BT601_7_625_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr RGB_BT601_7_625_space_() : RGB_space_<T>(XYZ_t::xy(0.64f, 0.33f, 0.299f), XYZ_t::xy(0.29f, 0.6f, 0.587f), XYZ_t::xy(0.15f, 0.06f, 0.114f), BT_gamma<T>()) {}
	};
	template<class T>
	struct RGB_BT709_6_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr RGB_BT709_6_space_() : RGB_space_<T>(XYZ_t::xy(0.64f, 0.33f, 0.2126f), XYZ_t::xy(0.3f, 0.6f, 0.7152f), XYZ_t::xy(0.15f, 0.06f, 0.0722f), BT_gamma<T>()) {}
	};
	template<class T>
	struct RGB_BT2020_2_space_ : public RGB_space_<T> {
		using XYZ_t = typename RGB_space_<T>::XYZ_t;
		using RGB_space_<T>::operator==;
		constexpr RGB_BT2020_2_space_() : RGB_space_<T>(XYZ_t::xy(0.708f, 0.292f, 0.2627f), XYZ_t::xy(0.170f, 0.797f, 0.6780f), XYZ_t::xy(0.131f, 0.046f, 0.0593f), BT_gamma<T>()) {}
	};

	template<class T, class colorspace_t>
	struct RGB_ {
		typedef colorspace_t colorspace;
		typedef std::array<T, 3> array_type;

		static_assert(std::is_base_of_v<RGB_space_<T>, colorspace>, "RGB_t<T, colorspace> : colorspace must be derived from RGB_space");
		static_assert(std::is_standard_layout_v<array_type>, "RGB_t<T, colorspace> : std::array<T,3> must be standard layout");

		COLOR_STDARRAY_TYPEDEFS
		union {
			array_type arr;
			struct { T R,G,B; };
		};
		constexpr bool operator==(const RGB_ &other) {
			return arr == other.arr;
		}
		constexpr RGB_& clip() {
			if constexpr (!std::is_integral_v<T>) { //TODO: clip integer representations
				for (auto &x : arr) x = std::clamp(x, T(0), T(1));
			}
			return *this;
		}
		COLOR_STDARRAY_MEMBERS(arr)
	};
	
	template<class T, class colorspace_t>
	struct traits_<RGB_<T, colorspace_t>, void> {
		typedef T value_type;
		typedef decltype(T(1)*1.0f) float_type;
		typedef colorspace_t colorspace;
		typedef XYZ_<float_type> convert_type;
		static constexpr XYZ_space_<float_type> convert_space(const colorspace& cs) {
			return XYZ_space_<float_type>();
		}
	};

	typedef sRGB_space_<float> sRGB_space;
	typedef sRGB_linear_space_<float> sRGB_linear_space;

	typedef RGB_<float, sRGB_space> sRGB;
	typedef RGB_<float, sRGB_linear_space> sRGB_linear;


	template<class T, class colorspace_t>
	struct YCbCr_ {
		typedef colorspace_t colorspace;
		typedef std::array<T, 3> array_type;

		static_assert(std::is_base_of_v<RGB_space_<T>, colorspace>, "YCbCr_<T, colorspace> : colorspace must be derived from RGB_space");
		static_assert(std::is_standard_layout_v<array_type>, "YCbCr_<T, colorspace> : std::array<T,3> must be standard layout");

		COLOR_STDARRAY_TYPEDEFS
		union {
			array_type arr;
			struct { T Y, Cb, Cr; };
		};
		constexpr bool operator==(const YCbCr_ &other) {
			return arr == other.arr;
		}
		constexpr YCbCr_& clip() {
			if constexpr (!std::is_integral_v<T>) { //TODO: clip integer representations?
				Y = std::clamp(Y, T(0), T(1));
				Cb = std::clamp(Cr, T(-0.5), T(0.5));
				Cr = std::clamp(Cr, T(-0.5), T(0.5));
			}
			return *this;
		}
		COLOR_STDARRAY_MEMBERS(arr)
	};
	typedef YCbCr_<float, RGB_BT709_6_space_<float>> YCbCr;
	typedef YCbCr_<float, RGB_BT601_7_525_space_<float>> YCbCr_601_525;
	typedef YCbCr_<float, RGB_BT601_7_625_space_<float>> YCbCr_601_625;
	typedef YCbCr_<float, RGB_BT709_6_space_<float>> YCbCr_709;
	typedef YCbCr_<float, RGB_BT2020_2_space_<float>> YCbCr_2020;

	template<class T, class colorspace_t>
	struct traits_<YCbCr_<T, colorspace_t>, void> {
		typedef T value_type;
		typedef decltype(T(1)*1.0f) float_type;
		typedef colorspace_t colorspace;
		typedef retype<float_type, colorspace> convert_space_t;
		typedef RGB_<float_type, convert_space_t> convert_type;

		static constexpr convert_space_t convert_space(const colorspace& cs) {
			return convert_space_t(cs); //promote to floating point when converting
		}
	};

	template<typename color_t>
	typename traits<color_t>::colorspace space(const color_t& c) {
		return typename traits<color_t>::colorspace();
	}

	template<class color_t>
	color_t clip(color_t c) {
		return c.clip();
	}

	//single parameter convert forwards to 4-parameter with default-constructed colorspace objects
	template<class dst_t, class src_t>
	dst_t convert(const src_t &pt) {
		dst_t d;
		convert(d, space(d), pt, space(pt));
		return d;
	}

	template<class dst_t, class src_t, class src_space_t, class dst_space_t>
	void convert(dst_t& dst, const dst_space_t &dst_space, const src_t &src, const src_space_t& src_space) {
		if constexpr (std::is_same_v<dst_t, src_t>) {
			//TODO: if src_space != dst_space
			dst = src;
		}
		else {
			//convert through the parents in the conversion tree
			//TODO: this can take a much longer path through the tree than strictly necessary

			//src -> parent(src)
			typename traits<src_t>::convert_type tmp1;
			const auto tmp1_space = traits<src_t>::convert_space(src_space);
			convert(tmp1, tmp1_space, src, src_space);
			
			//parent(src) -> parent(dst)
			typename traits<dst_t>::convert_type tmp2;
			const auto tmp2_space = traits<dst_t>::convert_space(dst_space);
			convert(tmp2, tmp2_space, tmp1, tmp1_space);
			
			//parent(dst) -> dst
			convert(dst, dst_space, tmp2, tmp2_space);
		}
	}

	template<typename T, typename U, size_t N>
	auto dot(const std::array<T, N> &A, const std::array<U, N> &B) -> decltype(T(0)*U(0))
	{
		auto x = T(0)*U(0);
		for (size_t i = 0; i < N; ++i) x += A[i] * B[i];
		return x;
	}

	template<typename T, typename U, size_t R, size_t C>
	auto mult(const std::array<std::array<T, C>, R> &mat, const std::array<U, C> &vec) -> std::array<decltype(T(0)*U(0)), R>
	{
		std::array<decltype(T(0)*U(0)), R> ret;
		for (size_t i = 0; i < R; ++i) ret[i] = dot(mat[i], vec);
		return ret;
	}

	// Specific colorspace conversions //

	// XYZ <=> RGB
	template<class T, class rgb_space_t>
	void convert(RGB_<T, rgb_space_t> &dst, const rgb_space_t &dst_space, const XYZ &src, const XYZ_space& src_space) {
		//normalize to white & black points
		XYZ src_n = (dst_space.white / dst_space.white.Y) * (src - dst_space.black) / (dst_space.white - dst_space.black);
		//matrix multiply to get linear coordinates
		auto rgb = mult(dst_space.M_toRGB, src_n.arr);
		//apply gamma function
		for (auto &x : rgb) x = dst_space.gamma(x);
		//done
		dst.arr = rgb;
	}
	
	template<class T, class rgb_space_t>
	void convert(XYZ& dst, const XYZ_space& dst_space, const RGB_<T, rgb_space_t> &src, const rgb_space_t &src_space) {
		//copy
		using float_type = typename traits<RGB_<T, rgb_space_t>>::float_type;
		RGB_<float_type, rgb_space_t> rgb = src;
		//invert gamma to linearize
		for (auto &x : rgb) x = src_space.gamma.revert(x);
		//matrix multiply to get XYZ
		dst.arr = mult(src_space.M_toXYZ, rgb.arr);
		//denormalize from white & black points
		dst = (dst * (src_space.white - src_space.black)* src_space.white.Y / src_space.white) + src_space.black;
	}
	
	
	// RGB <=> YCC
	template<class T, class rgb_space_t>
	void convert(YCbCr_<T, rgb_space_t> &dst, const rgb_space_t &dst_space, const RGB_<T, rgb_space_t> &src, const rgb_space_t &src_space) {
		dst.Y = src_space.red.Y*src.R + src_space.green.Y*src.G + src_space.blue.Y*src.B;
		dst.Cb = (src.B - dst.Y) / (2 - 2 * src_space.blue.Y);
		dst.Cr = (src.R - dst.Y) / (2 - 2 * src_space.red.Y);
	}

	template<class T, class rgb_space_t>
	void convert(RGB_<T, rgb_space_t> &dst, const rgb_space_t& dst_space, const YCbCr_<T, rgb_space_t> &src, const rgb_space_t &src_space) {
		dst.B = src.Cb*(2 - 2 * src_space.blue.Y) + src.Y;
		dst.R = src.Cr*(2 - 2 * src_space.red.Y) + src.Y;
		dst.G = (src.Y - dst.R * src_space.red.Y - dst.B * src_space.blue.Y) / src_space.green.Y;
	}

	// XYZ <=> Lab
	template<class T, class lab_space_t>
	void convert(Lab_<T, lab_space_t> &dst, const lab_space_t &dst_space, const XYZ_<T> &src, const XYZ_space_<T> &src_space) {
		//normalize to white point
		auto src_n = src / dst_space.white;
		//apply gamma
		for (auto &x : src_n) x = dst_space.gamma(x);
		dst.L = T(1.16)*src_n.Y - T(0.16);
		dst.a = 5 * (src_n.X - src_n.Y);
		dst.b = 2 * (src_n.Y - src_n.Z);
	}
	template<class T, class lab_space_t>
	void convert(XYZ_<T> &dst, const XYZ_space_<T> &dst_space, const Lab_<T, lab_space_t> &src, const lab_space_t &src_space) {
		dst.Y = (src.L + T(0.16)) / T(1.16);
		dst.X = dst.Y + src.a / 5;
		dst.Z = dst.Y - src.b / 2;
		for (auto &x : dst) x = src_space.gamma.revert(x);
		dst *= src_space.white;
	}

	// Lab <=> LCh
	template<class T, class lab_space_t>
	void convert(Lab_<float, lab_space_t> &dst, const lab_space_t &dst_space, const LCh_<T, lab_space_t>& src, const lab_space_t &src_space) {
		dst.L = src.L;
		dst.a = src.C*std::cos(src.h);
		dst.b = src.C*std::sin(src.h);
	}
	template<class T, class lab_space_t>
	void convert(LCh_<T, lab_space_t> &dst, const lab_space_t &dst_space, const Lab_<T, lab_space_t>& src, const lab_space_t &src_space) {
		dst.L = src.L;
		dst.C = std::hypot(src.a, src.b);
		dst.h = std::atan2(src.b, src.a);
	}
}
