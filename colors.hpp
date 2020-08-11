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

namespace colors {

	template<typename tag_t, typename value_t> struct traits {};
	template<class C>
	using traits_helper = traits<typename C::tag, typename C::value_type>;

	namespace detail {
		//Metaprogramming helpers
		template<class A, class B>
		using enable_if_same_t = std::enable_if_t<std::is_same_v<A, B>, int>;

		template<class A, class B>
		constexpr bool same_tag_v = std::is_same_v<typename A::tag, typename B::tag>;
		template<class A, class B>
		using enable_if_same_tag_t = std::enable_if_t<!std::is_same_v<A, B> && same_tag_v<A, B>, int>;

		template<class A, class B>
		using enable_if_not_same_t = std::enable_if_t<!std::is_same_v<A, B> && !same_tag_v<A, B>, int>;

	}

	// Generic conversion functions //

	//1: no conversion

	template<typename dst_t, typename src_t, detail::enable_if_same_t<dst_t, src_t> = 0>
	constexpr dst_t convert(const src_t& src) {
		return src;
	}

	//2: no colorspace conversion, but we do have a type conversion
	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_integral_v<dst_vt> && std::is_integral_v<src_vt>, int> = 0>
	constexpr dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//both are integral: shift
		int bits = 8 * ((int)sizeof(dst_vt) - (int)sizeof(src_vt));
		if (bits == 0) return val;
		else if (bits > 0) return (dst_vt)(val) << bits;
		else return (dst_vt)(val >> (-bits));
	}
	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_floating_point_v<dst_vt> && std::is_integral_v<src_vt>, int> = 0>
	constexpr dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//convert integral to floating point
		return ((dst_vt)val) / ((dst_vt)src_max);
	}
	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_integral_v<dst_vt> && std::is_floating_point_v<src_vt>, int> = 0>
	constexpr dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//convert floating point to integral
		return (dst_vt)(val*std::nextafter((src_vt)(dst_max + 1.0), (src_vt)(0.0)));
	}

	template<typename dst_t, typename src_t, detail::enable_if_same_tag_t<dst_t, src_t> = 0>
	constexpr dst_t convert(const src_t& src) {
		dst_t dst;
		auto di = std::begin(dst);
		for (auto si = std::begin(src); si != std::end(src); ++si, ++di) {
			*di = convert_value(*si, traits_helper<src_t>::max(), traits_helper<dst_t>::max());
		}
		return dst;
	}

	//3: colorspace conversion
	// convert colorspaces by attempting src_t -> traits<src_t>::convert_type -> traits<dst_t>::convert_type -> dst_t
	template<typename dst_t, typename src_t, detail::enable_if_not_same_t<dst_t, src_t> = 0>
	constexpr dst_t convert(const src_t& src) {
		typedef typename traits_helper<src_t>::convert_type src_ct;
		typedef typename traits_helper<dst_t>::convert_type dst_ct;
		src_ct s = convert<src_ct, src_t>(src);
		dst_ct d = convert<dst_ct, src_ct>(s);
		return convert<dst_t, dst_ct>(d);
	}

	// Generic color base class, use tag to restrict copy & assign
	template<typename T, size_t N, typename tag_>
	struct color_t : public std::array<T, N> {
		typedef tag_ tag;
		constexpr color_t() : std::array<T, N>{} {}
		template<typename... Ts>
		constexpr color_t(const T& t, Ts&&... ts) : std::array<T, N>{t, std::forward<Ts>(ts)...}
		{
			static_assert(sizeof...(Ts) == N-1, "color_t<T,N,tag>(Ts&&... ts): ts must be N parameters");
		}

		template<typename... Ts>
		void store(Ts&... ts) const {
			static_assert(sizeof...(Ts) == N, "color_t<T,N,tag>::store(Ts&... ts): ts must be N parameters");
			store_helper(0, ts...);
		}
	private:
		template<typename Q, typename... Ts>
		void store_helper(size_t i, Q& q, Ts&... ts) const {
			q = this->at(i);
			store_helper(i + 1, ts...);
		}
		template<typename Q>
		void store_helper(size_t i, Q& q) const {
			q = this->at(i);
		}
	};

	template<typename T, size_t N, typename tag_, typename derived_t>
	struct color_math_t : public color_t<T, N, tag_> {
		constexpr color_math_t() : color_t<T, N, tag_>() {}
		template<typename... Ts>
		constexpr color_math_t(const T& t, Ts&&... ts) : color_t<T, N, tag_>( t, std::forward<Ts>(ts)... ) {}
		
		constexpr derived_t& operator+=(const derived_t& rhs) {
			std::transform(std::begin(rhs), std::end(rhs), std::begin(*this), std::begin(*this), std::plus<T>());
			return *static_cast<derived_t*>(this);
		}
		constexpr derived_t& operator +=(const T& rhs) {
			for (auto &x : *this) x += rhs;
			return *static_cast<derived_t*>(this);
		}

		constexpr derived_t& operator-=(const derived_t& rhs) {
			std::transform(std::begin(rhs), std::end(rhs), std::begin(*this), std::begin(*this), std::minus<T>());
			return *static_cast<derived_t*>(this);
		}
		constexpr derived_t& operator -=(const T& rhs) {
			for (auto &x : *this) x -= rhs;
			return *static_cast<derived_t*>(this);
		}

		constexpr derived_t& operator*=(const derived_t& rhs) {
			std::transform(std::begin(rhs), std::end(rhs), std::begin(*this), std::begin(*this), std::multiplies<T>());
			return *static_cast<derived_t*>(this);
		}
		constexpr derived_t& operator *=(const T& rhs) {
			for (auto &x : *this) x *= rhs;
			return *static_cast<derived_t*>(this);
		}

		constexpr derived_t& operator/=(const derived_t& rhs) {
			std::transform(std::begin(rhs), std::end(rhs), std::begin(*this), std::begin(*this), std::divides<T>());
			return *static_cast<derived_t*>(this);
		}
		constexpr derived_t& operator /=(const T& rhs) {
			for (auto &x : *this) x /= rhs;
			return *static_cast<derived_t*>(this);
		}

	};

	template<typename A, typename B>
	constexpr auto operator+(A a, B&& b) -> std::remove_reference_t<decltype(a+=b)>
	{
		return a += b;
	}
	template<typename A, typename B>
	constexpr auto operator-(A a, B&& b) -> std::remove_reference_t<decltype(a -= b)>
	{
		return a -= b;
	}
	template<typename A, typename B>
	constexpr auto operator*(A a, B&& b) -> std::remove_reference_t<decltype(a *= b)>
	{
		return a *= b;
	}
	template<typename A, typename B>
	constexpr auto operator/(A a, B&& b) -> std::remove_reference_t<decltype(a /= b)>
	{
		return a /= b;
	}


	// XYZ //

	struct xyz_tag;
	struct xyz_t : public color_math_t<float, 3, xyz_tag, xyz_t> {
		typedef color_math_t<float, 3, xyz_tag, xyz_t> color_type;
		static const xyz_t D50, D65;

		float &x = this->at(0);
		float &y = this->at(1);
		float &z = this->at(2);

		constexpr xyz_t() : color_type{} {}
		constexpr xyz_t(float x, float y, float z) : color_type{x,y,z} {}
		constexpr xyz_t& operator=(const xyz_t& other) {
			color_t::operator=(other); return *this;
		}
		
		static constexpr xyz_t xy(float x, float y) {
			return xyz_t{ x / y, 1.0f, (1 - x - y) / y };
		}
	};

	const xyz_t xyz_t::D50 = xyz_t::xy(0.34567f, 0.35850f);
	const xyz_t xyz_t::D65 = xyz_t::xy(0.31271f, 0.32902f);

	// Lab //
	struct lab_tag;
	struct lab_t : public color_math_t<float, 3, lab_tag, lab_t> {
		typedef color_math_t<float, 3, lab_tag, lab_t> color_type;

		float &L = this->at(0);
		float &a = this->at(1);
		float &b = this->at(2);

		constexpr lab_t() : color_type{} {}
		constexpr lab_t(float L, float a, float b) : color_type{L,a,b} {}
		constexpr lab_t& operator=(const lab_t& other) {
			color_t::operator=(other); return *this;
		}
		

		static constexpr float gamma(float c) {
			if (c > 0.008856452f) // (6/29)**3
				return std::pow(c, 1.0f / 3.0f);
			else
				return c * 7.787037f + 0.13793103f; // (c/3)*(29/6)**2 + (4/29)
		}
		static constexpr float gamma_inv(float c) {
			if (c > 0.20689656f) // (6/29)
				return std::pow(c, 3.0f);
			else
				return 0.12841855f*(c - 0.13793103f); //3*(6/29)**2*(c - 4/29)
		}
	};

	// LCh //
	struct lch_tag;
	struct lch_t : public color_t<float, 3, lch_tag> {
		float &L = this->at(0);
		float &C = this->at(1);
		float &h = this->at(2);
		
		constexpr lch_t() : color_t{} {}
		constexpr lch_t(float L, float C, float h) : color_t{L,C,h} {}
		constexpr lch_t& operator=(const lch_t& other) {
			color_t::operator=(other);
			return *this;
		}
	};
	
	// RGB (linearized sRGB) //
	struct rgb_tag;
	template<typename T>
	struct rgb_t : public color_math_t<T, 3, rgb_tag, rgb_t<T>> {
		typedef color_math_t<T, 3, rgb_tag, rgb_t<T>> color_type;

		T &r = this->at(0);
		T &g = this->at(1);
		T &b = this->at(2);
		
		constexpr rgb_t() : color_type{} {}
		constexpr rgb_t(T r, T g, T b) : color_type{r,g,b} {}
		template<typename Q>
		constexpr explicit rgb_t(const color_t<Q, 3, rgb_tag>& other) : color_type(convert<rgb_t<T>>(other)) {}
		template<typename Q>
		constexpr rgb_t& operator=(const color_t<Q, 3, rgb_tag>& other) {
			color_type::operator=(convert<rgb_t<T>>(other)); return *this;
		}
	};

	
	// sRGB //

	struct srgb_tag;

	template<typename T>
	struct srgb_t : public color_t<T, 3, srgb_tag> {
		typedef color_t<T, 3, srgb_tag> color_type;

		T &r = this->at(0);
		T &g = this->at(1);
		T &b = this->at(2);

		constexpr srgb_t() : color_type{} {}
		constexpr srgb_t(T r, T g, T b) : color_type{r,g,b} {}
		template<typename Q>
		constexpr explicit srgb_t(const color_t<Q, 3, srgb_tag>& other) : color_type(convert<srgb_t<T>>(other)) {}
		template<typename Q>
		constexpr srgb_t& operator=(const color_t<Q, 3, srgb_tag>& other) {
			color_type::operator=(convert<srgb_t<T>>(other)); return *this;
		}
		// sRGB "gamma" function
		static constexpr float gamma(float c) {
			if (c > 0.0031308f)
				return std::pow(1.055f*c, 1.0f / 2.4f) - 0.055f;
			else
				return 12.92f*c;
		}
		// sRGB inverse "gamma" function
		static constexpr float gamma_inv(float c) {
			if (c > 0.04045f)
				return std::pow((c + 0.055f) / 1.055f, 2.4f);
			else
				return c / 12.92f;
		}
	};

	// XYZ Traits //

	template<>
	struct traits<xyz_tag, float> {
		typedef rgb_t<float> convert_type;
	};

	// LAB Traits //
	template<>
	struct traits<lab_tag, float> {
		typedef xyz_t convert_type;
	};

	// LCh Traits //
	template<>
	struct traits<lch_tag, float> {
		typedef lab_t convert_type;
	};

	// RGB Traits //
	template<> struct traits<rgb_tag, float> {
		typedef xyz_t convert_type;
		static constexpr float min() noexcept { return 0.0f; }
		static constexpr float max() noexcept { return 1.0f; }
	};

	template<> struct traits<rgb_tag, uint8_t> {
		typedef rgb_t<float> convert_type;
		static constexpr uint8_t min() noexcept { return 0; }
		static constexpr uint8_t max() noexcept { return 0xff; }
	};

	template<> struct traits<rgb_tag, uint16_t> {
		typedef rgb_t<float> convert_type;
		static constexpr uint16_t min() noexcept { return 0; }
		static constexpr uint16_t max() noexcept { return 0xffff; }
	};

	// sRGB Traits //
	template<> struct traits<srgb_tag, float> {
		typedef rgb_t<float> convert_type;
		static constexpr float min() noexcept { return 0.0f; }
		static constexpr float max() noexcept { return 1.0f; }
	};

	template<> struct traits<srgb_tag, uint8_t> {
		typedef srgb_t<float> convert_type;
		static constexpr uint8_t min() noexcept { return 0; }
		static constexpr uint8_t max() noexcept { return 0xff; }
	};

	template<> struct traits<srgb_tag, uint16_t> {
		typedef srgb_t<float> convert_type;
		static constexpr uint16_t min() noexcept { return 0; }
		static constexpr uint16_t max() noexcept { return 0xffff; }
	};

	template<typename color_t>
	color_t clip(const color_t& c) {
		color_t r;
		auto ri = std::begin(r);
		for (auto ci = std::begin(c); ci != std::end(c); ++ci, ++ri) {
			*ri = std::max(traits_helper<color_t>::min(), std::min(traits_helper<color_t>::max(), *ci));
		}
		return r;
	}



	// Specific colorspace conversions //

	// sRGB <=> RGB (linear)
	template<>
	constexpr srgb_t<float> convert(const rgb_t<float>& src) {
		return srgb_t<float>{srgb_t<float>::gamma(src.r), srgb_t<float>::gamma(src.g), srgb_t<float>::gamma(src.b)};
	}
	template<>
	constexpr rgb_t<float> convert(const srgb_t<float>& src) {
		return rgb_t<float>{srgb_t<float>::gamma_inv(src.r), srgb_t<float>::gamma_inv(src.g), srgb_t<float>::gamma_inv(src.b)};
	}

	// XYZ <=> RGB (linear)
	template<>
	constexpr xyz_t convert(const rgb_t<float>& src) {
		float x = 0.41239080f*src.r + 0.35758434f*src.g + 0.18048079f*src.b;
		float y = 0.21263901f*src.r + 0.71516868f*src.g + 0.07219232f*src.b;
		float z = 0.01933082f*src.r + 0.11919478f*src.g + 0.95053215f*src.b;
		return xyz_t{ x,y,z };
	}
	template<>
	constexpr rgb_t<float> convert(const xyz_t& src) {
		float r =  3.24096996f*src.x - 1.53738318f*src.y - 0.49861076f*src.z;
		float g = -0.96924366f*src.x + 1.87596751f*src.y + 0.04155505f*src.z;
		float b =  0.05563008f*src.x - 0.20397696f*src.y + 1.05697152f*src.z;
		return rgb_t<float>{ r,g,b };
	}

	// XYZ <=> Lab
	template<>
	constexpr xyz_t convert(const lab_t& src) {
		float ll = (src.L + 0.16f) / 1.16f;
		return xyz_t::D65*xyz_t{lab_t::gamma_inv(ll + src.a / 5), lab_t::gamma_inv(ll), lab_t::gamma_inv(ll - src.b / 2) };
	}

	template<>
	constexpr lab_t convert(const xyz_t& src) {
		float xx = lab_t::gamma(src.x / xyz_t::D65.x);
		float yy = lab_t::gamma(src.y / xyz_t::D65.y);
		float zz = lab_t::gamma(src.z / xyz_t::D65.z);
		return lab_t{ 1.16f * yy - 0.16f, 5 * (xx - yy), 2 * (yy - zz) };
	}

	// Lab <=> LCh
	template<>
	constexpr lab_t convert(const lch_t& src) {
		return lab_t{ src.L, src.C*std::cos(src.h), src.C*std::sin(src.h) };
	}
	template<>
	constexpr lch_t convert(const lab_t& src) {
		return lch_t{ src.L, std::hypot(src.a, src.b), std::atan2(src.b, src.a) };
	}
}
