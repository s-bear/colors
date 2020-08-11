# colors
C++ color-space conversions in a single header

Includes
- sRGB & linear RGB with float, 8-bit, and 16-bit elements
- CIE XYZ, Lab, LCh with float elements
- Conversion between any pair of formats
- Easily extinsible to more color spaces and more data types: all you need is a simple traits class and conversion functions to/from one of the existing spaces

E.g:
```cpp
//Increase the saturation of an sRGB color using LCh space to preserve luminance
#include "colors.hpp"
uint8_t r = 0xff, g = 0xcc, b = 0x88;
auto lch = colors::convert<colors::lch_t>(colors::srgb_t<uint8_t>(r,g,b));
lch.C = std::pow(lch.C, 0.5);
colors::convert<colors::srgb_t<uint8_t>>(lch).store(r,g,b);
```

Notes:
- Converting between element types happens without an explicit call to `convert` (ie. `srgb_t<uint8_t> x(srgb_t<float>(1.0f, 1.0f, 1.0f))` works)
- Only basic arithmetic is implemented, and only on the "linear" color spaces: RGB, XYZ, and Lab

To Do:
- Maybe write some tests, ha!
- Add more spaces: CIECAM, HSV, HSL, etc? (There are some other interesting spaces out there that could be useful... e.g. there's I recall a variant of Lab that doesn't have hue shifts when changing chromaticity at the expense of being less uniform in JNDs)
- Color arithmetic: namely alpha (or other) blending.
- Projecting colors back into the sRGB gamut along a specific vector (e.g. get a displayable color while maintaining hue)
