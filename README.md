# colors
C++ color-space conversions

Includes
- sRGB & linear RGB with float, 8-bit, and 16-bit elements
- CIE XYZ, Lab, LCh with float elements
- Conversion between any pair of formats
- Easily extinsible to more color spaces and more data types: all you need is a simple traits class and conversion functions to/from one of the existing spaces

E.g:
```cpp
//Increase the saturation of an sRGB color using LCh space to preserve luminance
#include "colors.hpp"
auto lch = colors::convert<colors::lch_t>(colors::srgb_t<uint8_t>(0xff, 0xcc, 0x88));
lch.C = std::pow(lch.C, 0.5);
auto rgb = colors::convert<colors::srgb_t<uint8_t>>(lch);
```

To Do:
- Aggregate initialization (`rgb_t<float>{1.0f,1.0f,1.0f}`) isn't well-supported. There's probably not much reason to be using std::array<> all over the place since it doesn't work well.
- More conveneint load/store interfaces would be good
- Maybe write some tests, ha!
- Add more spaces: CIECAM, HSV, HSL, etc? (There are some other interesting spaces out there that could be useful... e.g. there's I recall a variant of Lab that doesn't have hue shifts when changing chromaticity at the expense of being less uniform in JNDs)
- Color arithmetic: namely alpha (or other) blending. NB: Non-linear spaces (like sRGB) should not support maths as it doesn't make any sense
- Projecting colors back into the sRGB gamut along a specific vector (e.g. get a displayable color while maintaining hue)
