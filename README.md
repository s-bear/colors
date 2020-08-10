# colors
C++ color-space conversions

Includes
- sRGB & linear RGB with float, 8-bit, and 16-bit elements
- CIE XYZ, Lab, LCh with float elements
- Conversion between any pair of formats

E.g:
```cpp
//Increase the saturation of an sRGB color using LCh space to preserve luminance
#include "colors.hpp"
auto lch = colors::convert<colors::lch_t>(colors::srgb_t<uint8_t>(0xff, 0xcc, 0x88));
lch.C = std::pow(lch.C, 0.5);
auto rgb = colors::convert<colors::srgb_t<uint8_t>>(lch);
```

To Do:
- Aggregate initialization isn't well-supported. There's probably not much reason to be using std::array<> all over the place since it doesn't work well.
- More conveneint load/store interfaces would be good
- Maybe write some tests, ha
