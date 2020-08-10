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

namespace colors {

	template<typename color_t> struct traits {};

	// XYZ //

	struct xyz_tag;
	struct xyz_t : public std::array<float, 3> {
		typedef xyz_tag tag;
		static const xyz_t D50, D65;

		xyz_t() {}
		xyz_t(float x, float y, float z) : std::array<float, 3>{x,y,z} {}

		float &x = this->at(0);
		float &y = this->at(1);
		float &z = this->at(2);

		static constexpr xyz_t xy(float x, float y) {
			return xyz_t{ x / y, 1.0f, (1 - x - y) / y };
		}
	};

	const xyz_t xyz_t::D50 = xyz_t::xy(0.34567f, 0.35850f);
	const xyz_t xyz_t::D65 = xyz_t::xy(0.31271f, 0.32902f);

	// Lab //
	struct lab_tag;
	struct lab_t : public std::array<float, 3> {
		typedef lab_tag tag;

		lab_t() {}
		lab_t(float L, float a, float b) : std::array<float, 3>{L,a,b} {}

		float &L = this->at(0);
		float &a = this->at(1);
		float &b = this->at(2);

		static float gamma(float c) {
			if (c > 0.008856452f) // (6/29)**3
				return std::pow(c, 1.0f / 3.0f);
			else
				return c * 7.787037f + 0.13793103f; // (c/3)*(29/6)**2 + (4/29)
		}
		static float gamma_inv(float c) {
			if (c > 0.20689656f) // (6/29)
				return std::pow(c, 3.0f);
			else
				return 0.12841855f*(c - 0.13793103f); //3*(6/29)**2*(c - 4/29)
		}
	};

	// LCh //
	struct lch_tag;
	struct lch_t : public std::array<float, 3> {
		typedef lch_tag tag;

		lch_t() {}
		lch_t(float L, float C, float h) : std::array<float, 3>{L,C,h} {}

		float &L = this->at(0);
		float &C = this->at(1);
		float &h = this->at(2);
	};
	
	// RGB (linearized sRGB) //
	struct rgb_tag;
	template<typename T>
	struct rgb_t : public std::array<T, 3> {
		typedef rgb_tag tag;

		rgb_t() {}
		rgb_t(T r, T g, T b) : std::array<T, 3>{r,g,b} {}

		T &r = this->at(0);
		T &g = this->at(1);
		T &b = this->at(2);
	};

	
	// sRGB //

	struct srgb_tag;

	template<typename T>
	struct srgb_t : public std::array<T, 3> {
		typedef srgb_tag tag;

		srgb_t() {}
		srgb_t(T r, T g, T b) : std::array<T, 3>{r,g,b} {}

		T &r = this->at(0);
		T &g = this->at(1);
		T &b = this->at(2);

		// sRGB "gamma" function
		static float gamma(float c) {
			if (c > 0.0031308f)
				return std::pow(1.055f*c, 1.0f / 2.4f) - 0.055f;
			else
				return 12.92f*c;
		}
		// sRGB inverse "gamma" function
		static float gamma_inv(float c) {
			if (c > 0.04045f)
				return std::pow((c + 0.055f) / 1.055f, 2.4f);
			else
				return c / 12.92f;
		}
	};

	// XYZ Traits //

	template<>
	struct traits<xyz_t> {
		typedef rgb_t<float> convert_type;
	};

	// LAB Traits //
	template<>
	struct traits<lab_t> {
		typedef xyz_t convert_type;
	};

	// LCh Traits //
	template<>
	struct traits<lch_t> {
		typedef lab_t convert_type;
	};

	// RGB Traits //
	template<> struct traits<rgb_t<float>> {
		typedef xyz_t convert_type;
		static constexpr float min() noexcept { return 0.0f; }
		static constexpr float max() noexcept { return 1.0f; }
	};

	template<> struct traits<rgb_t<uint8_t>> {
		typedef rgb_t<float> convert_type;
		static constexpr uint8_t min() noexcept { return 0; }
		static constexpr uint8_t max() noexcept { return 0xff; }
	};

	template<> struct traits<rgb_t<uint16_t>> {
		typedef rgb_t<float> convert_type;
		static constexpr uint16_t min() noexcept { return 0; }
		static constexpr uint16_t max() noexcept { return 0xffff; }
	};

	// sRGB Traits //
	template<> struct traits<srgb_t<float>> {
		typedef rgb_t<float> convert_type;
		static constexpr float min() noexcept { return 0.0f; }
		static constexpr float max() noexcept { return 1.0f; }
	};

	template<> struct traits<srgb_t<uint8_t>> {
		typedef srgb_t<float> convert_type;
		static constexpr uint8_t min() noexcept { return 0; }
		static constexpr uint8_t max() noexcept { return 0xff; }
	};

	template<> struct traits<srgb_t<uint16_t>> {
		typedef srgb_t<float> convert_type;
		static constexpr uint16_t min() noexcept { return 0; }
		static constexpr uint16_t max() noexcept { return 0xffff; }
	};

	template<typename color_t>
	color_t clip(const color_t& c) {
		color_t r;
		auto ri = std::begin(r);
		for (auto ci = std::begin(c); ci != std::end(c); ++ci, ++ri) {
			*ri = std::max(traits<color_t>::min(), std::min(traits<color_t>::max(), *ci));
		}
		return r;
	}

	// Generic conversion functions //

	//1: no conversion
	template<class A, class B>
	using enable_if_same_t = std::enable_if_t<std::is_same_v<A, B>, int>;

	template<typename dst_t, typename src_t, enable_if_same_t<dst_t, src_t> = 0>
	dst_t convert(const src_t& src) {
		return src;
	}

	//2: no colorspace conversion, but we do have a type conversion
	template<class A, class B>
	constexpr bool same_tag_v = std::is_same_v<typename A::tag, typename B::tag>;
	template<class A, class B>
	using enable_if_same_tag_t = std::enable_if_t<!std::is_same_v<A, B> && same_tag_v<A, B>, int>;

	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_integral_v<dst_vt> && std::is_integral_v<src_vt>, int> = 0>
	dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//both are integral: shift
		int bits = 8 * ((int)sizeof(dst_vt) - (int)sizeof(src_vt));
		if (bits == 0) return val;
		else if (bits > 0) return (dst_vt)(val) << bits;
		else return (dst_vt)(val >> (-bits));
	}
	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_floating_point_v<dst_vt> && std::is_integral_v<src_vt>, int> = 0>
	dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//convert integral to floating point
		return ((dst_vt)val) / ((dst_vt)src_max);
	}
	template<typename dst_vt, typename src_vt, std::enable_if_t<std::is_integral_v<dst_vt> && std::is_floating_point_v<src_vt>, int> = 0>
	dst_vt convert_value(const src_vt& val, const src_vt& src_max, const dst_vt& dst_max) {
		//convert floating point to integral
		return (dst_vt)(val*std::nextafter((src_vt)(dst_max + 1.0), (src_vt)(0.0)));
	}

	template<typename dst_t, typename src_t, enable_if_same_tag_t<dst_t, src_t> = 0>
	dst_t convert(const src_t& src) {
		dst_t dst;
		auto di = std::begin(dst);
		for (auto si = std::begin(src); si != std::end(src); ++si, ++di) {
			*di = convert_value(*si, traits<src_t>::max(), traits<dst_t>::max());
		}
		return dst;
	}

	//3: colorspace conversion
	template<class A, class B>
	using enable_if_not_same_t = std::enable_if_t<!std::is_same_v<A, B> && !same_tag_v<A, B>, int>;

	//3.a: convert colorspaces by attempting src_t -> traits<src_t>::convert_type -> traits<dst_t>::convert_type -> dst_t
	template<typename dst_t, typename src_t, enable_if_not_same_t<dst_t, src_t> = 0> 
	dst_t convert(const src_t& src) {
		typedef typename traits<src_t>::convert_type src_ct;
		typedef typename traits<dst_t>::convert_type dst_ct;
		src_ct s = convert<src_ct, src_t>(src);
		dst_ct d = convert<dst_ct, src_ct>(s);
		return convert<dst_t,dst_ct>(d);
	}

	// sRGB <=> RGB (linear) conversion
	template<>
	srgb_t<float> convert(const rgb_t<float>& src) {
		return srgb_t<float>{srgb_t<float>::gamma(src.r), srgb_t<float>::gamma(src.g), srgb_t<float>::gamma(src.b)};
	}
	template<>
	rgb_t<float> convert(const srgb_t<float>& src) {
		return rgb_t<float>{srgb_t<float>::gamma_inv(src.r), srgb_t<float>::gamma_inv(src.g), srgb_t<float>::gamma_inv(src.b)};
	}

	// XYZ <=> RGB (linear) conversion
	template<>
	xyz_t convert(const rgb_t<float>& src) {
		float x = 0.41239080f*src.r + 0.35758434f*src.g + 0.18048079f*src.b;
		float y = 0.21263901f*src.r + 0.71516868f*src.g + 0.07219232f*src.b;
		float z = 0.01933082f*src.r + 0.11919478f*src.g + 0.95053215f*src.b;
		return xyz_t{ x,y,z };
	}
	template<>
	rgb_t<float> convert(const xyz_t& src) {
		float r =  3.24096996f*src.x - 1.53738318f*src.y - 0.49861076f*src.z;
		float g = -0.96924366f*src.x + 1.87596751f*src.y + 0.04155505f*src.z;
		float b =  0.05563008f*src.x - 0.20397696f*src.y + 1.05697152f*src.z;
		return rgb_t<float>{ r,g,b };
	}

	// XYZ <=> Lab
	template<>
	xyz_t convert(const lab_t& src) {
		float ll = (src.L + 0.16f) / 1.16f;
		return xyz_t{
			xyz_t::D65.x*lab_t::gamma_inv(ll + src.a / 5),
			xyz_t::D65.y*lab_t::gamma_inv(ll),
			xyz_t::D65.z*lab_t::gamma_inv(ll - src.b / 2) };
	}

	template<>
	lab_t convert(const xyz_t& src) {
		//assume xyz is already normalized to the white point
		float xx = lab_t::gamma(src.x / xyz_t::D65.x);
		float yy = lab_t::gamma(src.y / xyz_t::D65.y);
		float zz = lab_t::gamma(src.z / xyz_t::D65.z);
		return lab_t{ 1.16f * yy - 0.16f, 5 * (xx - yy), 2 * (yy - zz) };
	}

	// Lab <=> LCh
	template<>
	lab_t convert(const lch_t& src) {
		return lab_t{ src.L, src.C*std::cos(src.h), src.C*std::sin(src.h) };
	}
	template<>
	lch_t convert(const lab_t& src) {
		return lch_t{ src.L, std::hypot(src.a, src.b), std::atan2(src.b, src.a) };
	}
}
