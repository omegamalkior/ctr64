/* GLideN64 -> PICA200 backend.
 *
 * Contract with GLideN64 (mirrors the OpenGL backend so the front half needs no changes):
 *   - Coordinates are GL-style: framebuffer row 0 is the bottom, NDC y = +1 is the top, texture
 *     t = 0 is row 0. PICA textures are addressed the same way, but the PICA writes window row y
 *     of a render target to memory row (storage height - 1 - y). So every draw negates clip y
 *     and mirrors viewport and scissor against the storage height; the result is a GL-layout
 *     image that samples back exactly as GLideN64 expects (and GL's winding is preserved).
 *   - GLideN64's vertex shader position math (viewport transform, quarter-pixel snap, screen-
 *     space modes, the 640x640 virtual viewport) is applied on the CPU while packing vertices;
 *     the PICA vertex shader (shaders/gfx.v.pica) only passes data through.
 *   - GL clip depth [-w, w] is remapped to PICA's [-w, 0] as z' = -(z + w) / 2, which with the
 *     depth map at scale -1 reproduces GL's 0..1 window depth exactly.
 *   - Textures have a logical (GL) size and power-of-two storage; texture coordinates are scaled
 *     by logical/storage at pack time. Storage starts in linear memory and moves to VRAM the
 *     first time the texture becomes a render target.
 *
 * M2 scope: geometry, depth, culling, blending, frame buffers and presentation. Combiners are
 * reduced to shade (or primitive) color; texturing and the TEV combiner compiler come in M3. */

#include "pica_ContextImpl.h"

#include <arm_acle.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <unordered_map>
#include <vector>

/* libctru and GLideN64 both typedef u8..s64, with different underlying types for 32 bits
 * (uint32_t is unsigned long on ARM, GLideN64 uses unsigned int). libctru's names are renamed
 * while its headers are parsed; values convert implicitly between the two. */
#define u8 ctru_u8
#define u16 ctru_u16
#define u32 ctru_u32
#define u64 ctru_u64
#define s8 ctru_s8
#define s16 ctru_s16
#define s32 ctru_s32
#define s64 ctru_s64
#define vu8 ctru_vu8
#define vu16 ctru_vu16
#define vu32 ctru_vu32
#define vu64 ctru_vu64
#define vs8 ctru_vs8
#define vs16 ctru_vs16
#define vs32 ctru_vs32
#define vs64 ctru_vs64
#include <3ds.h>
extern "C" {
#include "pica/pica.h"
#include "video.h"
}
#include "ctru_abi.h"
#include "Ctr64Prof.h"
extern "C" {
#include "prof.h"
}
#undef u8
#undef u16
#undef u32
#undef u64
#undef s8
#undef s16
#undef s32
#undef s64
#undef vu8
#undef vu16
#undef vu32
#undef vu64
#undef vs8
#undef vs16
#undef vs32
#undef vs64

#include <Combiner.h>
#include <CombinerKey.h>
#include <Config.h>
#include <DisplayWindow.h>
#include <GBI.h>
#include <Graphics/ColorBufferReader.h>
#include <Graphics/CombinerProgram.h>
#include <Graphics/Context.h>
#include <Graphics/FramebufferTextureFormats.h>
#include <Graphics/Parameters.h>
#include <Graphics/PixelBuffer.h>
#include <Graphics/ShaderProgram.h>
#include <GraphicsDrawer.h>
#include <RSP.h>
#include <FrameBuffer.h>
#include <Textures.h>
#include <gDP.h>
#include <gSP.h>

#include "pica_params.h"
#include "pica_Combiner.h"
#include "debug_crumb.h"
extern "C" bool ctr64_render_skip; /* frameskip.c */
extern "C" bool ctr64_frameskip_present(bool stale);
extern "C" bool ctr64_frame_cleared, ctr64_frame_fbtex, ctr64_frame_drew; /* frameskip.c */
extern "C" unsigned ctr64_fb_copies;
bool g_ctr64ReadRgba16; /* ColorBufferToRDRAM: read 16-bit RGBA5551 (GLideN64's converter) instead of RGBA8 bytes */
bool g_ctr64ReadPmBlue; /* with it: Paper Mario's subscreen fix, blue 1..0xFB raised by 4 first */
extern "C" const char *ctr64_allocfail_where; /* last failed allocation, for stats.log */
extern "C" unsigned ctr64_tex_evict(unsigned bytes); /* GLideN64 Textures.cpp: frees cached textures */
extern "C" unsigned ctr64_gfx_evicted; /* gpuAlloc calls that evicted textures (gln64_ctr.c) */
extern "C" unsigned ctr64_allocfail_bytes;

using namespace graphics;
using namespace pica_param;

extern "C" unsigned int ctr64_gfx_stats[8]; /* core_compat/gln64_ctr.c */
extern "C" unsigned int ctr64_gfx_fcram[2]; /* core_compat/gln64_ctr.c: draws to FCRAM color/depth */
extern "C" unsigned int ctr64_gfx_vram_moves[2]; /* KB moved into / out of VRAM (keepInVram) */
extern "C" unsigned int ctr64_gfx_reused;
extern "C" unsigned int ctr64_game_starts; /* core_host.c */
extern "C" char ctr64_rt_mode[64]; /* stats.log: the frame buffer storage this game runs with (init) */
extern "C" unsigned int ctr64_gfx_shadow[3]; /* shadow copies for sampling (shadowOf): count, KB; [2] copyByTransfer */
extern "C" unsigned int ctr64_video_frames;
/* bytes of texture / render target storage held by the backend: [0] linear, [1] VRAM,
 * [2] the largest single linear allocation, [3] storage pending release */
extern "C" unsigned int ctr64_gfx_mem[4];
enum { ST_TRI, ST_RECT, ST_CLEAR, ST_SWAP, ST_DROP, ST_FBO, ST_ALLOC_FAIL, ST_SUBDIV };
extern "C" unsigned long long ctr64_gfx_ticks[3]; /* [0] draw calls, [1] display lists, [2] present */
extern "C" int ctr64_prof_on; /* main.c: fine-grained timers on (Debug info) */

/* Accumulates the system ticks spent in a backend draw call into ctr64_gfx_ticks[0]. */
struct DrawTimer { /* only with profiling on: a tick read is a kernel call */
    unsigned long long t0 = ctr64_prof_on ? svcGetSystemTick() : 0;
    ~DrawTimer()
    {
        if (t0)
            ctr64_gfx_ticks[0] += svcGetSystemTick() - t0;
    }
};

extern "C" bool ctr64_crc_nomemo; /* gln64_ctr.c; dev switch nocrcmemo.txt */

namespace {

/* ---- textures ------------------------------------------------------------------------------ */

struct Tex {
    u16 w = 0, h = 0;   /* logical size */
    u16 tw = 0, th = 0; /* power-of-two storage */
    GPU_TEXCOLOR fmt = GPU_RGBA8;
    u8 bpp = 4;
    bool depth = false; /* D24S8 storage (depth renderbuffer / texture) */
    bool vram = false;
    void *data = nullptr;
    u32 param = GPU_TEXTURE_MAG_FILTER(GPU_NEAREST) | GPU_TEXTURE_MIN_FILTER(GPU_NEAREST) |
                GPU_TEXTURE_WRAP_S(GPU_CLAMP_TO_EDGE) | GPU_TEXTURE_WRAP_T(GPU_CLAMP_TO_EDGE);
    u32 used_frame = 0; /* pica frame number in which the GPU last read or wrote it */
    bool dirty = false; /* contents changed since last bound: clear the GPU texture cache */
    bool lazy = false;  /* storage not allocated until first use (render targets) */
    bool rt = false;    /* attached to a frame buffer: prefer VRAM */
    bool skipped = false; /* draws into it were dropped by frameskip since it was last drawn */
    bool drawn = true;    /* drawn or written since the window last presented it */
    u64 copy_fence = 0;   /* a queued GPU copy into it (stagedUpdate) is done at this fence */
    u32 draw_frame = 0;   /* pica frame of the last draw into it (VRAM placement, keepInVram) */
    u32 ndraw = 0;        /* draws since the last present */
    u32 ndraw_last = ~0u; /* draws in the last present interval it was drawn in (~0: none yet) */
    u32 gen = 0;          /* contents generation: bumped with dirty (shadow copies, shadowOf) */
    u32 nshadow = 0;      /* shadow copies of it in this present interval (rtlog.txt) */
};

struct Fbo {
    u32 color = 0, depth = 0;
    PicaTarget rt{};
    bool valid = false;
};

u8 g_morton[64];

void init_morton()
{
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            g_morton[y * 8 + x] = (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) |
                                  ((x & 4) << 2) | ((y & 4) << 3);
}

inline u32 tiled_index(u32 x, u32 y, u32 tw)
{
    return ((((y >> 3) * (tw >> 3)) + (x >> 3)) << 6) + g_morton[((y & 7) << 3) + (x & 7)];
}

/* Texel (x, y) in GL convention (y = 0 is t = 0). PICA textures and render targets keep tiled
 * row 0 at the top (t = 1), so GL row y is tiled row th - 1 - y - for textures uploaded here
 * and for images the GPU renders alike, which keeps render-to-texture GL-consistent. */
inline u32 gl_index(u32 x, u32 y, u32 tw, u32 th)
{
    return tiled_index(x, th - 1 - y, tw);
}

u16 pot(u32 v)
{
    u16 p = 8;
    while (p < v && p < 1024)
        p <<= 1;
    return p;
}

/* Storage width of a frame buffer texture or renderbuffer: the width rounded up to 32 texels.
 * With anti-aliasing a 640x480 target then takes 1.25 MB instead of 2 MB: the targets of a frame
 * fit in VRAM (2026-11-09: MK64 menus moved 4 MB per frame in and out of VRAM, OoT's pause drew
 * into an FCRAM depth buffer; HW profile4 with AA: 40-70 MB/s of target moves, MK GPU 3.2 -> 23.6
 * ms per frame). The PICA renders to any width (multiple of 8) but samples only power-of-two
 * widths on hardware (video_npot_selftest): there such a texture is sampled through a shadow
 * copy (shadowOf). The height stays a power of two: the rows of every target keep their place,
 * and a depth buffer still matches the color buffers it served before. */
bool g_slowtex; /* dev switch slowtex.txt: texture uploads by the old per-texel paths (A/B) */
bool g_potfb; /* dev switch potfb.txt: power-of-two frame buffer storage always (as before 2026-11-10) */

/* Hardware (no non-power-of-two sampling): frame buffers at exact widths only with anti-aliasing,
 * where that keeps the 2x targets in VRAM, at the price of a shadow copy per sampled frame buffer.
 * At 1x power-of-two storage fits and needs none. Hardware A/B (2026-11-10, profiles 3dslogs/10,
 * same scenes): OoT AA 97.2% exact vs 89.2% power-of-two, MK 88.7 vs 89.1, SSB 96.0 vs 97.2 (its
 * CPU freezes dominate). Decided when a frame buffer is created (the game's settings, AA
 * included, are final then; not always at context init). stats.log "frame buffer storage". */
static bool pot_storage()
{
    if (video_npot_ok)
        return false;
    const bool aa = config.frameBufferEmulation.enable && config.frameBufferEmulation.nativeResFactor > 1;
    const bool pot_choice = !aa || g_potfb;
    if (aa)
        snprintf(ctr64_rt_mode, sizeof(ctr64_rt_mode), "%s",
                 pot_choice ? "power-of-two widths" : "exact widths + shadow copies");
    return pot_choice;
}

u16 fb_width(u32 v)
{
    if (v < 64 || v > 1024 || pot_storage())
        return pot(v);
    return (u16)((v + 31) & ~31u);
}

/* Without non-power-of-two sampling (hardware) a texture stored at such a width is sampled
 * through a power-of-two shadow copy (shadowOf); the width texture coordinates refer to */
inline bool needs_shadow(const Tex &t)
{
    return !video_npot_ok && (t.tw & (t.tw - 1)) != 0;
}
inline u32 sample_w(const Tex &t)
{
    return needs_shadow(t) ? pot(t.tw) : t.tw;
}

/* GL source texel -> 0xRRGGBBAA */
u32 fetch_rgba8(const u8 *src, u32 i, u32 format, u32 type)
{
    switch (type) {
    case DT_US_4444: {
        u16 v = reinterpret_cast<const u16 *>(src)[i];
        u32 r = (v >> 12) & 15, g = (v >> 8) & 15, b = (v >> 4) & 15, a = v & 15;
        return (r * 17) << 24 | (g * 17) << 16 | (b * 17) << 8 | (a * 17);
    }
    case DT_US_5551: {
        u16 v = reinterpret_cast<const u16 *>(src)[i];
        u32 r = (v >> 11) & 31, g = (v >> 6) & 31, b = (v >> 1) & 31;
        return ((r << 3) | (r >> 2)) << 24 | ((g << 3) | (g >> 2)) << 16 | ((b << 3) | (b >> 2)) << 8 |
               ((v & 1) ? 0xFF : 0);
    }
    case DT_US_565: {
        u16 v = reinterpret_cast<const u16 *>(src)[i];
        u32 r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
        return ((r << 3) | (r >> 2)) << 24 | ((g << 2) | (g >> 4)) << 16 | ((b << 3) | (b >> 2)) << 8 | 0xFF;
    }
    default:
        break;
    }
    switch (format) {
    case CF_RGB: {
        const u8 *p = src + i * 3;
        return u32(p[0]) << 24 | u32(p[1]) << 16 | u32(p[2]) << 8 | 0xFF;
    }
    case CF_RED:
    case CF_LUMINANCE: {
        u8 l = src[i];
        return u32(l) << 24 | u32(l) << 16 | u32(l) << 8 | 0xFF;
    }
    case CF_RG: {
        const u8 *p = src + i * 2;
        return u32(p[0]) << 24 | u32(p[1]) << 16 | 0xFF;
    }
    default: {
        const u8 *p = src + i * 4;
        return u32(p[0]) << 24 | u32(p[1]) << 16 | u32(p[2]) << 8 | p[3];
    }
    }
}

/* ---- texel codecs: 0xRRGGBBAA <-> PICA storage formats (byte layouts as the PICA reads them) -- */

inline u32 expand5(u32 v) { return (v << 3) | (v >> 2); }

void store_texel(Tex &t, u32 idx, u32 rgba)
{
    const u32 r = rgba >> 24, g = (rgba >> 16) & 0xFF, b = (rgba >> 8) & 0xFF, a = rgba & 0xFF;
    u8 *p = static_cast<u8 *>(t.data);
    switch (t.fmt) {
    case GPU_RGBA4:
        reinterpret_cast<u16 *>(p)[idx] = (u16)((r >> 4) << 12 | (g >> 4) << 8 | (b >> 4) << 4 | a >> 4);
        break;
    case GPU_RGBA5551:
        reinterpret_cast<u16 *>(p)[idx] = (u16)((r >> 3) << 11 | (g >> 3) << 6 | (b >> 3) << 1 | (a >> 7));
        break;
    case GPU_RGB8:
        p[idx * 3 + 0] = (u8)b;
        p[idx * 3 + 1] = (u8)g;
        p[idx * 3 + 2] = (u8)r;
        break;
    case GPU_LA8:
        p[idx * 2 + 0] = (u8)a;
        p[idx * 2 + 1] = (u8)r;
        break;
    case GPU_LA4:
        p[idx] = (u8)((r >> 4) << 4 | a >> 4);
        break;
    default:
        reinterpret_cast<u32 *>(p)[idx] = rgba;
        break;
    }
}

u32 load_texel(const Tex &t, u32 idx)
{
    const u8 *p = static_cast<const u8 *>(t.data);
    switch (t.fmt) {
    case GPU_RGBA4: {
        const u32 v = reinterpret_cast<const u16 *>(p)[idx];
        return ((v >> 12) * 17) << 24 | (((v >> 8) & 15) * 17) << 16 | (((v >> 4) & 15) * 17) << 8 | (v & 15) * 17;
    }
    case GPU_RGBA5551: {
        const u32 v = reinterpret_cast<const u16 *>(p)[idx];
        return expand5(v >> 11) << 24 | expand5((v >> 6) & 31) << 16 | expand5((v >> 1) & 31) << 8 | ((v & 1) ? 255 : 0);
    }
    case GPU_RGB8:
        return (u32)p[idx * 3 + 2] << 24 | (u32)p[idx * 3 + 1] << 16 | (u32)p[idx * 3] << 8 | 0xFF;
    case GPU_LA8: {
        const u32 l = p[idx * 2 + 1];
        return l << 24 | l << 16 | l << 8 | p[idx * 2];
    }
    case GPU_LA4: {
        const u32 l = (p[idx] >> 4) * 17;
        return l << 24 | l << 16 | l << 8 | (p[idx] & 15) * 17;
    }
    default:
        return reinterpret_cast<const u32 *>(p)[idx];
    }
}

u8 bytes_per_texel(GPU_TEXCOLOR f)
{
    switch (f) {
    case GPU_RGBA8: return 4;
    case GPU_RGB8: return 3;
    case GPU_LA4: return 1;
    default: return 2;
    }
}

/* Whether a texel survives the format exactly. */
bool texel_fits(GPU_TEXCOLOR f, u32 rgba)
{
    const u32 r = rgba >> 24, g = (rgba >> 16) & 0xFF, b = (rgba >> 8) & 0xFF, a = rgba & 0xFF;
    switch (f) {
    case GPU_LA4: return r == g && g == b && r % 17 == 0 && a % 17 == 0;
    case GPU_LA8: return r == g && g == b;
    case GPU_RGBA5551:
        return expand5(r >> 3) == r && expand5(g >> 3) == g && expand5(b >> 3) == b && (a == 0 || a == 255);
    case GPU_RGBA4: return r % 17 == 0 && g % 17 == 0 && b % 17 == 0 && a % 17 == 0;
    case GPU_RGB8: return a == 255;
    default: return true;
    }
}

/* The smallest PICA format that stores every texel exactly (N64 textures are mostly 4/8/16-bit
 * intensity, IA or RGBA5551 data that GLideN64 expands to RGBA8888). */
GPU_TEXCOLOR pick_format(const std::vector<u32> &px)
{
    static const GPU_TEXCOLOR order[] = {GPU_LA4, GPU_LA8, GPU_RGBA5551, GPU_RGBA4, GPU_RGB8};
    bool ok[5] = {true, true, true, true, true};
    int left = 5;
    for (u32 v : px) {
        for (int i = 0; i < 5; i++)
            if (ok[i] && !texel_fits(order[i], v)) {
                ok[i] = false;
                left--;
            }
        if (!left)
            return GPU_RGBA8;
    }
    for (int i = 0; i < 5; i++)
        if (ok[i])
            return order[i];
    return GPU_RGBA8;
}

/* ---- bulk texel paths (uploads of whole images: frame buffer copies, N64 textures) ----------
 * Format-specialized loops: one conversion pass over the source, then tiled stores with the
 * Morton offsets of each storage row precomputed. */

/* GL source pixels -> 0xRRGGBBAA, one specialized loop per source layout */
void fetch_all(const u8 *src, u32 n, u32 format, u32 type, u32 *out)
{
    if (type == DT_US_4444 || type == DT_US_5551 || type == DT_US_565) {
        for (u32 i = 0; i < n; i++)
            out[i] = fetch_rgba8(src, i, format, type);
        return;
    }
    if (format == CF_RGBA || (format != CF_RGB && format != CF_RED && format != CF_LUMINANCE && format != CF_RG)) {
        /* bytes R, G, B, A -> 0xRRGGBBAA: a byte swap of the little-endian word */
        const u32 *s = reinterpret_cast<const u32 *>(src);
        for (u32 i = 0; i < n; i++)
            out[i] = __builtin_bswap32(s[i]);
        return;
    }
    for (u32 i = 0; i < n; i++)
        out[i] = fetch_rgba8(src, i, format, type);
}

bool fits_all(GPU_TEXCOLOR f, const u32 *px, u32 n)
{
    if (f == GPU_RGBA8)
        return true;
    for (u32 i = 0; i < n; i++)
        if (!texel_fits(f, px[i]))
            return false;
    return true;
}

template <GPU_TEXCOLOR F> inline void put_texel(u8 *p, u32 idx, u32 rgba);
template <> inline void put_texel<GPU_RGBA8>(u8 *p, u32 idx, u32 v) { reinterpret_cast<u32 *>(p)[idx] = v; }
template <> inline void put_texel<GPU_RGB8>(u8 *p, u32 idx, u32 v)
{
    u8 *d = p + idx * 3;
    d[0] = (u8)(v >> 8); d[1] = (u8)(v >> 16); d[2] = (u8)(v >> 24);
}
template <> inline void put_texel<GPU_RGBA5551>(u8 *p, u32 idx, u32 v)
{
    reinterpret_cast<u16 *>(p)[idx] = (u16)((v >> 16 & 0xF800) | (v >> 13 & 0x07C0) | (v >> 10 & 0x003E) | (v >> 7 & 1));
}
template <> inline void put_texel<GPU_RGBA4>(u8 *p, u32 idx, u32 v)
{
    reinterpret_cast<u16 *>(p)[idx] = (u16)((v >> 16 & 0xF000) | (v >> 12 & 0x0F00) | (v >> 8 & 0x00F0) | (v >> 4 & 0x000F));
}
template <> inline void put_texel<GPU_LA8>(u8 *p, u32 idx, u32 v)
{
    p[idx * 2] = (u8)v;
    p[idx * 2 + 1] = (u8)(v >> 24);
}
template <> inline void put_texel<GPU_LA4>(u8 *p, u32 idx, u32 v) { p[idx] = (u8)((v >> 24 & 0xF0) | (v >> 4 & 0x0F)); }

/* px (w x h, GL rows) -> texture rect at (x0, y0) */
template <GPU_TEXCOLOR F> void write_rect(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h)
{
    u8 *p = static_cast<u8 *>(t.data);
    const u32 tiles_x = t.tw >> 3;
    const u32 xe = x0 >= t.tw ? 0 : (w < t.tw - x0 ? w : t.tw - x0);
    for (u32 y = 0; y < h && y0 + y < t.th; y++) {
        const u32 r = t.th - 1 - (y0 + y); /* storage row (row 0 = top) */
        const u32 rowbase = ((r >> 3) * tiles_x) << 6;
        const u8 *m = &g_morton[(r & 7) << 3];
        const u32 *s = px + (size_t)y * w;
        for (u32 x = 0; x < xe; x++) {
            const u32 tx = x0 + x;
            put_texel<F>(p, rowbase + ((tx >> 3) << 6) + m[tx & 7], s[x]);
        }
    }
}

/* ---- whole-texture uploads (init2DTexture): the hot path of new screens ----------------------
 * MK64's menu screens load ~100-200 textures in one frame: on hardware 60-80 ms of the video
 * thread per screen change went to fetch_all + pick_format + write_rect + pad_edges (Azahar VI
 * 1284 of the MK menus: 97 ms fetch/pick, 107 ms swizzle of 255 ms texture loads). Same output,
 * byte for byte: one branch-free scan for the format, and the storage written tile by tile in
 * its own order (2x2 quads = 4 consecutive texels), the power-of-two padding taken from clamped
 * source rows/columns instead of a second pass. */

/* fetch_all for the 16-bit sources, without the per-texel switch */
void fetch_all_fast(const u8 *src, u32 n, u32 format, u32 type, u32 *out)
{
    const u16 *s = reinterpret_cast<const u16 *>(src);
    switch (type) {
    case DT_US_5551:
        for (u32 i = 0; i < n; i++) {
            const u32 v = s[i];
            const u32 r = (v >> 11) & 31, g = (v >> 6) & 31, b = (v >> 1) & 31;
            out[i] = ((r << 3) | (r >> 2)) << 24 | ((g << 3) | (g >> 2)) << 16 | ((b << 3) | (b >> 2)) << 8 |
                     (0u - (v & 1)) >> 24;
        }
        return;
    case DT_US_4444:
        for (u32 i = 0; i < n; i++) {
            const u32 v = s[i];
            /* nibbles 0xRGBA -> bytes 0xR0G0B0A0, each byte n * 17 = n << 4 | n */
            const u32 e = (v & 0xF000) << 16 | (v & 0x0F00) << 12 | (v & 0x00F0) << 8 | (v & 0x000F) << 4;
            out[i] = e | e >> 4;
        }
        return;
    case DT_US_565:
        for (u32 i = 0; i < n; i++) {
            const u32 v = s[i];
            const u32 r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
            out[i] = ((r << 3) | (r >> 2)) << 24 | ((g << 2) | (g >> 4)) << 16 | ((b << 3) | (b >> 2)) << 8 | 0xFF;
        }
        return;
    default:
        fetch_all(src, n, format, type, out);
    }
}

/* pick_format in one branch-free pass. Per texel 0xRRGGBBAA:
 *   gray (LA8)      r == g == b          (v ^ v >> 8) & 0x00FFFF00 == 0
 *   nibbles (RGBA4) every byte n * 17    (v ^ v >> 4) & 0x0F0F0F0F == 0
 *   LA4             gray and nibbles
 *   RGBA5551        r, g, b == expand5(c >> 3), i.e. c & 7 == c >> 5:  (v ^ v >> 5) & 0x07070700 == 0,
 *                   and a == 0 or 255:  (a + 1) & 0xFE == 0
 *   RGB8            a == 255 */
GPU_TEXCOLOR pick_format_fast(const u32 *px, u32 n)
{
    u32 gray = 0, nib = 0, c5 = 0, a5 = 0, aand = 0xFF;
    for (u32 i = 0; i < n;) {
        const u32 end = n - i > 256 ? i + 256 : n;
        for (; i < end; i++) {
            const u32 v = px[i];
            gray |= v ^ (v >> 8);
            nib |= v ^ (v >> 4);
            c5 |= v ^ (v >> 5);
            a5 |= v + 1;
            aand &= v;
        }
        if ((gray & 0x00FFFF00) && (nib & 0x0F0F0F0F) && ((c5 & 0x07070700) || (a5 & 0xFE)) && (aand & 0xFF) != 0xFF)
            return GPU_RGBA8; /* nothing smaller fits any more */
    }
    gray &= 0x00FFFF00;
    nib &= 0x0F0F0F0F;
    if (!gray && !nib)
        return GPU_LA4;
    if (!gray)
        return GPU_LA8;
    if (!(c5 & 0x07070700) && !(a5 & 0xFE))
        return GPU_RGBA5551;
    if (!nib)
        return GPU_RGBA4;
    if ((aand & 0xFF) == 0xFF)
        return GPU_RGB8;
    return GPU_RGBA8;
}

template <GPU_TEXCOLOR F> inline u32 conv_texel(u32 v);
template <> inline u32 conv_texel<GPU_RGBA8>(u32 v) { return v; }
template <> inline u32 conv_texel<GPU_RGB8>(u32 v) { return v; }
template <> inline u32 conv_texel<GPU_RGBA5551>(u32 v)
{
    return (v >> 16 & 0xF800) | (v >> 13 & 0x07C0) | (v >> 10 & 0x003E) | (v >> 7 & 1);
}
template <> inline u32 conv_texel<GPU_RGBA4>(u32 v)
{
    return (v >> 16 & 0xF000) | (v >> 12 & 0x0F00) | (v >> 8 & 0x00F0) | (v >> 4 & 0x000F);
}
template <> inline u32 conv_texel<GPU_LA8>(u32 v) { return (v & 0xFF) | (v >> 24) << 8; }
template <> inline u32 conv_texel<GPU_LA4>(u32 v) { return (v >> 24 & 0xF0) | (v >> 4 & 0x0F); }

/* one 2x2 quad: texels (x, r), (x+1, r), (x, r+1), (x+1, r+1) of the storage, at o (4 texels);
 * RAW: the values are already in the storage format (16-bit formats only) */
template <GPU_TEXCOLOR F, bool RAW = false> inline void put_quad(u8 *o, u32 t0, u32 t1, u32 t2, u32 t3)
{
    if (F == GPU_RGBA8) {
        u32 *d = reinterpret_cast<u32 *>(o);
        d[0] = t0; d[1] = t1; d[2] = t2; d[3] = t3;
    } else if (F == GPU_RGB8) {
        const u32 t[4] = {t0, t1, t2, t3};
        for (int k = 0; k < 4; k++) {
            o[k * 3 + 0] = (u8)(t[k] >> 8);
            o[k * 3 + 1] = (u8)(t[k] >> 16);
            o[k * 3 + 2] = (u8)(t[k] >> 24);
        }
    } else if (F == GPU_LA4) {
        *reinterpret_cast<u32 *>(o) =
            conv_texel<F>(t0) | conv_texel<F>(t1) << 8 | conv_texel<F>(t2) << 16 | conv_texel<F>(t3) << 24;
    } else if (RAW) {
        u32 *d = reinterpret_cast<u32 *>(o);
        d[0] = t0 | t1 << 16;
        d[1] = t2 | t3 << 16;
    } else {
        u32 *d = reinterpret_cast<u32 *>(o);
        d[0] = conv_texel<F>(t0) | conv_texel<F>(t1) << 16;
        d[1] = conv_texel<F>(t2) | conv_texel<F>(t3) << 16;
    }
}

/* the whole storage (tw x th) from px (w x h GL rows, stride = row length in texels >= w):
 * texels past the right/bottom edge repeat the last column/row (= pad_edges) */
template <GPU_TEXCOLOR F, bool RAW = false, typename T = u32> void write_full(Tex &t, const T *px, u32 w, u32 h, u32 stride)
{
    /* storage quad q of a tile: x = q bits 0, 2 -> x bits 1, 2; y = q bits 1, 3 -> y bits 1, 2
     * (g_morton: x0 y0 x1 y1 x2 y2 from the lowest bit) */
    static const u8 qx[16] = {0, 2, 0, 2, 4, 6, 4, 6, 0, 2, 0, 2, 4, 6, 4, 6};
    static const u8 qy[16] = {0, 0, 2, 2, 0, 0, 2, 2, 4, 4, 6, 6, 4, 4, 6, 6};
    const u32 bpp = F == GPU_RGBA8 ? 4 : F == GPU_RGB8 ? 3 : F == GPU_LA4 ? 1 : 2;
    u8 *o = static_cast<u8 *>(t.data);
    const u32 tiles_x = t.tw >> 3, tiles_y = t.th >> 3;
    const bool xpad = w < t.tw;
    for (u32 ty = 0; ty < tiles_y; ty++) {
        const T *row[8];
        for (u32 ry = 0; ry < 8; ry++) {
            u32 y = t.th - 1 - (ty * 8 + ry); /* GL row of storage row ty * 8 + ry */
            if (y >= h)
                y = h - 1;
            row[ry] = px + (size_t)y * stride;
        }
        for (u32 tx = 0; tx < tiles_x; tx++) {
            const u32 x0 = tx * 8;
            if (!xpad || x0 + 8 <= w) {
                for (u32 q = 0; q < 16; q++) {
                    const u32 x = x0 + qx[q];
                    const T *a = row[qy[q]], *b = row[qy[q] + 1];
                    put_quad<F, RAW>(o, a[x], a[x + 1], b[x], b[x + 1]);
                    o += 4 * bpp;
                }
            } else {
                for (u32 q = 0; q < 16; q++) {
                    u32 xa = x0 + qx[q], xb = xa + 1;
                    if (xa >= w)
                        xa = w - 1;
                    if (xb >= w)
                        xb = w - 1;
                    const T *a = row[qy[q]], *b = row[qy[q] + 1];
                    put_quad<F, RAW>(o, a[xa], a[xb], b[xa], b[xb]);
                    o += 4 * bpp;
                }
            }
        }
    }
}

void write_full_any(Tex &t, const u32 *px, u32 w, u32 h, u32 stride)
{
    switch (t.fmt) {
    case GPU_RGB8: write_full<GPU_RGB8>(t, px, w, h, stride); break;
    case GPU_RGBA5551: write_full<GPU_RGBA5551>(t, px, w, h, stride); break;
    case GPU_RGBA4: write_full<GPU_RGBA4>(t, px, w, h, stride); break;
    case GPU_LA8: write_full<GPU_LA8>(t, px, w, h, stride); break;
    case GPU_LA4: write_full<GPU_LA4>(t, px, w, h, stride); break;
    default: write_full<GPU_RGBA8>(t, px, w, h, stride); break;
    }
}

/* write_rect with the same result (texel (x0 + x, y0 + y) = px[y * w + x], x < tw - x0, y < th - y0;
 * texels outside the rect untouched), tile by tile: tiles inside the rect by 2x2 quads in storage
 * order, edge tiles texel by texel (OoT's pause menu with AA updates a 320x240 rect of a 640x512 frame
 * buffer texture every frame: 11.9 ms per frame on hardware by write_rect) */
template <GPU_TEXCOLOR F> void write_rect_fast(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h)
{
    static const u8 qx[16] = {0, 2, 0, 2, 4, 6, 4, 6, 0, 2, 0, 2, 4, 6, 4, 6};
    static const u8 qy[16] = {0, 0, 2, 2, 0, 0, 2, 2, 4, 4, 6, 6, 4, 4, 6, 6};
    if (x0 >= t.tw || y0 >= t.th || !w || !h)
        return;
    const u32 xe = w < t.tw - x0 ? w : t.tw - x0; /* columns written */
    const u32 he = h < t.th - y0 ? h : t.th - y0; /* rows written */
    const u32 bpp = F == GPU_RGBA8 ? 4 : F == GPU_RGB8 ? 3 : F == GPU_LA4 ? 1 : 2;
    u8 *p = static_cast<u8 *>(t.data);
    const u32 tiles_x = t.tw >> 3;
    const u32 r_lo = t.th - (y0 + he), r_hi = t.th - 1 - y0; /* storage rows */
    const u32 c_lo = x0, c_hi = x0 + xe - 1;
    for (u32 tr = r_lo >> 3; tr <= (r_hi >> 3); tr++) {
        for (u32 tc = c_lo >> 3; tc <= (c_hi >> 3); tc++) {
            u8 *o = p + ((tr * tiles_x + tc) << 6) * bpp;
            const bool inside = tr * 8 >= r_lo && tr * 8 + 7 <= r_hi && tc * 8 >= c_lo && tc * 8 + 7 <= c_hi;
            if (inside) {
                for (u32 q = 0; q < 16; q++) {
                    const u32 r = tr * 8 + qy[q], x = tc * 8 + qx[q] - x0;
                    const u32 *a = px + (size_t)(t.th - 1 - r - y0) * w + x; /* storage row r */
                    const u32 *b = px + (size_t)(t.th - 2 - r - y0) * w + x; /* storage row r + 1 */
                    put_quad<F>(o + q * 4 * bpp, a[0], a[1], b[0], b[1]);
                }
            } else {
                for (u32 i = 0; i < 64; i++) {
                    const u32 lx = (i & 1) | ((i >> 1) & 2) | ((i >> 2) & 4);
                    const u32 ly = ((i >> 1) & 1) | ((i >> 2) & 2) | ((i >> 3) & 4);
                    const u32 r = tr * 8 + ly, x = tc * 8 + lx;
                    if (r < r_lo || r > r_hi || x < c_lo || x > c_hi)
                        continue;
                    put_texel<F>(o, i, px[(size_t)(t.th - 1 - r - y0) * w + (x - x0)]);
                }
            }
        }
    }
}

/* Replicate the right and bottom edges into the power-of-two padding, so that clamping at the
 * storage edge matches clamping at the logical edge (non-power-of-two clamp tiles). */
template <u32 BPP> void pad_edges_bpp(Tex &t)
{
    struct Unit { u8 b[BPP]; };
    Unit *p = static_cast<Unit *>(t.data);
    for (u32 y = 0; y < t.h; y++) {
        const Unit v = p[gl_index(t.w - 1, y, t.tw, t.th)];
        for (u32 x = t.w; x < t.tw; x++)
            p[gl_index(x, y, t.tw, t.th)] = v;
    }
    for (u32 y = t.h; y < t.th; y++)
        for (u32 x = 0; x < t.tw; x++)
            p[gl_index(x, y, t.tw, t.th)] = p[gl_index(x, t.h - 1, t.tw, t.th)];
}

void pad_edges(Tex &t)
{
    switch (t.bpp) {
    case 1: pad_edges_bpp<1>(t); break;
    case 2: pad_edges_bpp<2>(t); break;
    case 3: pad_edges_bpp<3>(t); break;
    default: pad_edges_bpp<4>(t); break;
    }
}

/* ---- programs ------------------------------------------------------------------------------- */

enum class Kind { Combiner, Copy, Clear, Text, Noop };

class ContextImplPica;
ContextImplPica *g_ctx;

void set_current_program(const CombinerProgram *p, Kind k);

class PicaCombinerProgram : public CombinerProgram {
public:
    PicaCombinerProgram(const CombinerKey &key, u32 inputs) : m_key(key), m_inputs(inputs)
    {
        pica_cmb::compile(key.getMux(), key.getCycleType(), m_tev);
    }
    const pica_cmb::Program &tev() const { return m_tev; }
    void activate() override { set_current_program(this, Kind::Combiner); }
    /* GLideN64 calls update() (not activate()) before each draw; the GL backend binds there too. */
    void update(bool) override { set_current_program(this, Kind::Combiner); }
    const CombinerKey &getKey() const override { return m_key; }
    bool usesTexture() const override
    {
        return (m_inputs & ((1u << G_GCI_TEXEL0) | (1u << G_GCI_TEXEL0_ALPHA) | (1u << G_GCI_TEXEL1) |
                            (1u << G_GCI_TEXEL1_ALPHA))) != 0;
    }
    bool usesTile(u32 t) const override
    {
        return t == 0 ? (m_inputs & ((1u << G_GCI_TEXEL0) | (1u << G_GCI_TEXEL0_ALPHA))) != 0
                      : (m_inputs & ((1u << G_GCI_TEXEL1) | (1u << G_GCI_TEXEL1_ALPHA))) != 0;
    }
    bool usesShade() const override
    {
        return (m_inputs & ((1u << G_GCI_SHADE) | (1u << G_GCI_SHADE_ALPHA))) != 0;
    }
    bool usesLOD() const override { return (m_inputs & (1u << G_GCI_LOD_FRACTION)) != 0; }
    bool usesHwLighting() const override { return false; }
    bool getBinaryForm(std::vector<char> &) override { return false; }

private:
    CombinerKey m_key;
    u32 m_inputs;
    pica_cmb::Program m_tev;
};

class PicaSpecialProgram : public ShaderProgram {
public:
    explicit PicaSpecialProgram(Kind k) : m_kind(k) {}
    void activate() override { set_current_program(this, m_kind); }

private:
    Kind m_kind;
};

class PicaTexrectDrawerProgram : public TexrectDrawerShaderProgram {
public:
    void activate() override { set_current_program(this, Kind::Copy); }
    void setTextureSize(u32, u32) override {}
    void setEnableAlphaTest(int) override {}
};

class PicaTextProgram : public TextDrawerShaderProgram {
public:
    void activate() override { set_current_program(this, Kind::Text); }
    void setTextColor(float *) override {}
};

/* Same input analysis as the GLSL backend (glsl_CombinerProgramBuilder.cpp), including its
 * first/second-cycle parameter corrections. */
u32 correct_first(u32 p) { return p == G_GCI_TEXEL1 ? G_GCI_TEXEL0 : p == G_GCI_TEXEL1_ALPHA ? G_GCI_TEXEL0_ALPHA : p; }
u32 correct_first_2cyc(u32 p) { return p == G_GCI_COMBINED ? G_GCI_HALF : p; }
u32 correct_second(u32 p)
{
    switch (p) {
    case G_GCI_TEXEL0: return G_GCI_TEXEL1;
    case G_GCI_TEXEL1: return G_GCI_TEXEL0;
    case G_GCI_TEXEL0_ALPHA: return G_GCI_TEXEL1_ALPHA;
    case G_GCI_TEXEL1_ALPHA: return G_GCI_TEXEL0_ALPHA;
    default: return p;
    }
}

u32 collect_inputs(const CombinerStage &st, u32 (*fix)(u32))
{
    u32 in = 0;
    for (u32 i = 0; i < st.numOps; i++) {
        const CombinerOp &op = st.op[i];
        u32 p[3] = {op.param1, op.param2, op.param3};
        for (u32 k = 0; k < 3; k++) {
            u32 v = fix ? fix(p[k]) : p[k];
            if (v < 32)
                in |= 1u << v;
        }
    }
    return in;
}

/* ---- frame buffer readback ------------------------------------------------------------------ */

/* One GPU readback: the rows of a render target untiled into a linear buffer, and the format
 * GLideN64 wants them in */
struct ReadJob {
    s32 x0 = 0, y0 = 0;
    u32 w = 0, h = 0, hh = 0, ww = 0, s0 = 0, s1 = 0, tw = 0, th = 0;
    bool rgba16 = false, pm_blue = false;
};

/* A linear buffer the GPU untiles into, and the read queued into it (if any) */
struct AsyncReadback {
    u32 *lin = nullptr;
    size_t bytes = 0;
    u64 fence = 0;
    bool pending = false; /* queued, not yet waited for */
    bool valid = false;   /* lin holds (or will hold) job */
    ReadJob job;
    bool reserve(size_t need);
    void finish();
    ~AsyncReadback();
};

class PicaColorBufferReader : public ColorBufferReader {
public:
    explicit PicaColorBufferReader(CachedTexture *tex) : ColorBufferReader(tex) {}
    AsyncReadback m_async; /* copyToRDRAM double-buffer mode */
    void cleanUp() override {}
    /* The base class would read into its own two buffers (allocated per frame buffer, which the
     * patched constructor no longer does) and then copy the first w*h*bytes onto themselves, as
     * stride = width and the formats are integer here: the texels are returned as read, from one
     * buffer for all readers (they are used one at a time). */
    const u8 *readPixels(s32 x0, s32 y0, u32 width, u32 height, u32 size, bool sync) override;

private:
    const u8 *_readPixels(const ReadColorBufferParams &params, u32 &heightOffset, u32 &stride) override;
};

class PicaPixelReadBuffer : public PixelReadBuffer {
public:
    explicit PicaPixelReadBuffer(size_t size) : m_data(size, 0) {}
    void readPixels(s32, s32, u32, u32, Parameter, Parameter) override {}
    void *getDataRange(u32 offset, u32 range) override
    {
        return offset + range <= m_data.size() ? m_data.data() + offset : nullptr;
    }
    void closeReadBuffer() override {}
    void bind() override {}
    void unbind() override {}

private:
    std::vector<u8> m_data;
};

struct PicaFbTexFormats : public FramebufferTextureFormats {
    PicaFbTexFormats()
    {
        colorInternalFormat = internalcolorFormat::RGBA8;
        colorFormat = colorFormat::RGBA;
        colorType = datatype::UNSIGNED_BYTE;
        colorFormatBytes = 4;
        monochromeInternalFormat = internalcolorFormat::RGBA8;
        monochromeFormat = colorFormat::RGBA;
        monochromeType = datatype::UNSIGNED_BYTE;
        monochromeFormatBytes = 4;
        depthInternalFormat = internalcolorFormat::DEPTH;
        depthFormat = colorFormat::DEPTH;
        depthType = datatype::UNSIGNED_INT;
        depthFormatBytes = 4;
        depthImageInternalFormat = internalcolorFormat::RGBA8;
        depthImageFormat = colorFormat::RGBA;
        depthImageType = datatype::UNSIGNED_BYTE;
        depthImageFormatBytes = 4;
        lutInternalFormat = internalcolorFormat::RGBA8;
        lutFormat = colorFormat::RGBA;
        lutType = datatype::UNSIGNED_BYTE;
        lutFormatBytes = 4;
        fontInternalFormat = internalcolorFormat::RGBA8;
        fontFormat = colorFormat::RGBA;
        fontType = datatype::UNSIGNED_BYTE;
        fontFormatBytes = 4;
    }
};

/* ---- vertex format -------------------------------------------------------------------------- */

struct GVtx {
    float x, y, z, w;
    u8 c[4];
    float s0, t0, s1, t1;
};

const PicaAttrib k_layout[4] = {
    {GPU_FLOAT, 4},
    {GPU_UNSIGNED_BYTE, 4},
    {GPU_FLOAT, 2},
    {GPU_FLOAT, 2},
};

inline u8 to_u8(float v)
{
    /* round to nearest and clamp to 0..255 (float->int conversion truncates; USAT clamps) */
    return (u8)__usat((int)(v * 255.0f + 0.5f), 8);
}

/* floorf without the libm call (VFPv2 has no rounding instruction); |v| < 2^31 */
inline float fast_floor(float v)
{
    const float t = (float)(int)v;
    return t > v ? t - 1.0f : t;
}

/* Decals (ZMODE_DEC) use GL polygon offset (factor -3, units -3): offset = -3 * max depth slope
 * - 3 * r. The PICA's depth offset is a per-draw constant, so the slope term is computed on the
 * CPU from the draw's triangles (poly_offset_for_draw); this is the floor for flat or
 * screen-aligned geometry (r taken as 2^-16, the PICA interpolating depth with less than 24 bits). */
const float k_poly_offset = -3.0f / 65536.0f;
constexpr float k_poly_factor = -3.0f;

#if defined(CTR64_RA)
constexpr u32 WINDOW_W = 320, WINDOW_H = 240; /* RetroArch core: the 4:3 image (pica_DisplayWindow.cpp) */
#else
constexpr u32 WINDOW_W = 400, WINDOW_H = 240;
#endif
constexpr u32 TEX_WINDOW_COLOR = 0xFFFF0001u, TEX_WINDOW_DEPTH = 0xFFFF0002u;

/* ---- the context ---------------------------------------------------------------------------- */

struct TexXform;

class ContextImplPica : public graphics::ContextImpl {
public:
    /* lifecycle */
    void init() override;
    void destroy() override;

    /* state */
    void setClampMode(ClampMode mode) override { m_clamp = mode; }
    ClampMode getClampMode() override { return m_clamp; }
    void enable(EnableParam p, bool on) override;
    u32 isEnabled(EnableParam p) override;
    void cullFace(CullModeParam mode) override { m_cullMode = u32(mode); }
    void enableDepthWrite(bool on) override { m_depthWrite = on; }
    void setDepthCompare(CompareParam mode) override { m_depthFunc = u32(mode); }
    void setViewport(s32 x, s32 y, s32 w, s32 h) override { m_vp[0] = x; m_vp[1] = y; m_vp[2] = w; m_vp[3] = h; }
    void setScissor(s32 x, s32 y, s32 w, s32 h) override { m_sc[0] = x; m_sc[1] = y; m_sc[2] = w; m_sc[3] = h; }
    void setBlending(BlendParam s, BlendParam d) override { setBlendingSeparate(s, d, s, d); }
    void setBlendingSeparate(BlendParam sc, BlendParam dc, BlendParam sa, BlendParam da) override
    {
        m_blend[0] = u32(sc); m_blend[1] = u32(dc); m_blend[2] = u32(sa); m_blend[3] = u32(da);
    }
    void setBlendColor(f32 r, f32 g, f32 b, f32 a) override { m_blendColor = pica_const_color(to_u8(r), to_u8(g), to_u8(b), to_u8(a)); }
    void clearColorBuffer(f32 r, f32 g, f32 b, f32 a) override;
    void clearDepthBuffer() override;
    void setPolygonOffset(f32, f32) override {}

    /* textures */
    ObjectHandle createTexture(Parameter) override;
    void deleteTexture(ObjectHandle name) override;
    void init2DTexture(const Context::InitTextureParams &p) override;
    void update2DTexture(const Context::UpdateTextureDataParams &p) override;
    void setTextureParameters(const Context::TexParameters &p) override;
    void bindTexture(const Context::BindTextureParameters &p) override;
    void setTextureUnpackAlignment(s32 a) override { m_unpack = a; }
    s32 getTextureUnpackAlignment() const override { return m_unpack; }
    s32 getMaxTextureSize() const override { return 1024; }
    f32 getMaxAnisotropy() const override { return 0.0f; }
    void bindImageTexture(const Context::BindImageTextureParameters &) override {}
    u32 convertInternalTextureFormat(u32 f) const override { return f; }
    void textureBarrier() override {}

    /* frame buffers */
    FramebufferTextureFormats *getFramebufferTextureFormats() override { return new PicaFbTexFormats; }
    ObjectHandle createFramebuffer() override;
    void deleteFramebuffer(ObjectHandle name) override;
    void bindFramebuffer(BufferTargetParam target, ObjectHandle name) override;
    void addFrameBufferRenderTarget(const Context::FrameBufferRenderTarget &p) override;
    ObjectHandle createRenderbuffer() override { return createTexture(Parameter()); }
    void initRenderbuffer(const Context::InitRenderbufferParams &p) override;
    bool blitFramebuffers(const Context::BlitFramebuffersParams &) override { return false; }
    void setDrawBuffers(u32) override {}
    PixelReadBuffer *createPixelReadBuffer(size_t size) override { return new PicaPixelReadBuffer(size); }
    ColorBufferReader *createColorBufferReader(CachedTexture *tex) override { return new PicaColorBufferReader(tex); }

    /* programs */
    bool isCombinerProgramBuilderObsolete() override { return false; }
    void resetCombinerProgramBuilder() override {}
    CombinerProgram *createCombinerProgram(Combiner &color, Combiner &alpha, const CombinerKey &key) override;
    bool saveShadersStorage(const Combiners &) override { return false; }
    bool loadShadersStorage(Combiners &) override { return false; }
    ShaderProgram *createDepthFogShader() override { return new PicaSpecialProgram(Kind::Noop); }
    TexrectDrawerShaderProgram *createTexrectDrawerDrawShader() override { return new PicaTexrectDrawerProgram; }
    ShaderProgram *createTexrectDrawerClearShader() override { return new PicaSpecialProgram(Kind::Clear); }
    ShaderProgram *createTexrectUpscaleCopyShader() override { return new PicaSpecialProgram(Kind::Copy); }
    ShaderProgram *createTexrectColorAndDepthUpscaleCopyShader() override { return nullptr; }
    ShaderProgram *createTexrectDownscaleCopyShader() override { return new PicaSpecialProgram(Kind::Copy); }
    ShaderProgram *createTexrectColorAndDepthDownscaleCopyShader() override { return nullptr; }
    ShaderProgram *createGammaCorrectionShader() override { return new PicaSpecialProgram(Kind::Copy); }
    ShaderProgram *createFXAAShader() override { return new PicaSpecialProgram(Kind::Copy); }
    TextDrawerShaderProgram *createTextDrawerShader() override { return new PicaTextProgram; }
    void resetShaderProgram() override { m_program = nullptr; m_kind = Kind::Noop; }

    /* drawing */
    void drawTriangles(const Context::DrawTriangleParameters &p) override;
    void drawRects(const Context::DrawRectParameters &p) override;
    void drawLine(f32, SPVertex *) override {}
    f32 getMaxLineWidth() override { return 1.0f; }

    /* misc */
    bool isSupported(SpecialFeatures) const override { return false; }
    s32 getMaxMSAALevel() override { return 0; }
    bool isError() const override { return false; }
    bool isFramebufferError() const override { return false; }

    /* backend internals */
    void setProgram(const CombinerProgram *p, Kind k) { m_program = p; m_kind = k; }
    void present();
    void endVI();
    bool skipDraw(bool clear = false);
    const u8 *readTexture(CachedTexture *tex, s32 x0, s32 y0, u32 w, u32 h, std::vector<u8> &out);
    const u8 *readTextureAsync(CachedTexture *tex, s32 x0, s32 y0, u32 w, u32 h, AsyncReadback &a,
                               std::vector<u8> &out);

private:
    Tex *tex(u32 id)
    {
        /* a few lookups per draw, mostly of the same ids: the last two found are kept (map nodes
         * do not move; m_tex.erase/clear forget them) */
        if (id == m_texHit[0].id && id)
            return m_texHit[0].t;
        if (id == m_texHit[1].id && id)
            return m_texHit[1].t;
        auto it = m_tex.find(id);
        if (it == m_tex.end())
            return nullptr;
        m_texHit[1] = m_texHit[0];
        m_texHit[0] = TexHit{id, &it->second};
        return &it->second;
    }
    struct TexHit { u32 id; Tex *t; };
    TexHit m_texHit[2] = {{0, nullptr}, {0, nullptr}};
    void forgetTexHits() { m_texHit[0] = m_texHit[1] = TexHit{0, nullptr}; }
    Fbo &drawFbo() { return m_drawFbo == 0 ? m_window : m_fbo[m_drawFbo]; }
    void allocStorage(Tex &t, bool vram, bool reuse = false, bool zero = true);
    void moveToVram(u32 id, bool copy = true);
    bool evictToFcram(u32 id);
    void keepInVram(Fbo &f, int use);
    bool inFlight(const Tex &t) const;
    void releaseStorage(Tex &t);
    void collectPending(bool all);
    void *gpuAlloc(u32 bytes, bool vram_first, bool &is_vram);
    void storageChanged(u32 id);
    void ensureStorage(u32 id);
    void writeTexels(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h);
    bool stagedUpdate(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h);
    void rebuildFbo(Fbo &f);
    bool applyState(bool depth_allowed);
    /* drawTriangles: the render mode (othermode low word) of triangles drawn into the N64 Z
     * buffer's address, and the VIs since; 0 = none (Pilotwings 64 shadow) */
    u32 m_zPassMode = 0;
    u32 m_zPassAge = 0;
    void bindUnit(int unit, bool n64_filter = false);
    bool setupCombiner(bool rect);
    enum { USE_DRAW, USE_COLOR_CLEARED, USE_DEPTH_ONLY }; /* how materialize's caller uses the FBO */
    void materialize(Fbo &f, int use = USE_DRAW);
    float poly_offset_for_draw(const GVtx *v, const u16 *idx, u32 count, GPU_Primitive_t prim);
    GVtx *linearizeShade(GVtx *v, u32 &n, bool fog_alpha);
    const char *stateString()
    {
        static char b[200];
        const Tex *t0 = tex(m_unit[0]), *t1 = tex(m_unit[1]);
        snprintf(b, sizeof(b), "depth %d func %u write %d poff %d blend %d %u/%u fbo %u(%ux%u) t0 %u:%ux%u%s fmt%d t1 %u:%ux%u%s",
                 m_enDepth, m_depthFunc, m_depthWrite, m_enPolyOffset, m_enBlend, m_blend[0], m_blend[1], m_drawFbo, (unsigned)drawFbo().rt.width, (unsigned)drawFbo().rt.height,
                 m_unit[0], t0 ? t0->w : 0, t0 ? t0->h : 0, t0 && t0->data ? "" : "(null)", t0 ? (int)t0->fmt : -1,
                 m_unit[1], t1 ? t1->w : 0, t1 ? t1->h : 0, t1 && t1->data ? "" : "(null)");
        return b;
    }
    void texture_xforms(TexXform tx[2]);
    void *shadowOf(u32 id, Tex &t, bool &copied);
    bool copyByTransfer(const Context::DrawRectParameters &p);
    bool m_xferCopy = true; /* copyByTransfer (dev switch noxfercopy.txt: off) */
    void *m_xferTemp[2] = {nullptr, nullptr};
    u32 m_xferTempBytes[2] = {0, 0};
    struct Shadow {
        void *p;
        u16 tw, th;
        u8 bpp;
        u32 owner, gen;
        const void *src;
    };
    std::vector<Shadow> m_shadows; /* one per storage size: the GPU copies in queue order */
    void ensureFrame()
    {
        if (!pica_in_frame()) {
            pica_frame_begin();
            collectPending(false);
        }
    }

    std::unordered_map<u32, Tex> m_tex;
    std::vector<u32> m_scratch; /* upload staging, reused (0xRRGGBBAA texels) */
    std::vector<GVtx> m_shadeScratch; /* linearizeShade() output, reused */
    std::vector<GVtx> m_idxScratch;   /* drawTriangles: indexed vertices before the expansion */
    std::vector<u32> m_idxUsed;       /* drawTriangles: m_idxGen = vertex referred to by this draw */
    u32 m_idxGen = 0;
    std::unordered_map<u32, Fbo> m_fbo;
    u32 m_nextTex = 1, m_nextFbo = 1;
    Fbo m_window;
    u32 m_drawFbo = 0, m_readFbo = 0;
    u32 m_unit[2] = {0, 0};
    struct Pending {
        void *ptr;
        bool vram;
        u32 frame;
        u32 bytes;
    };
    static u32 storageBytes(const Tex &t) { return (u32)t.tw * t.th * (t.depth ? 4 : t.bpp); }
    void freeNow(void *p, bool vram, u32 bytes);
    std::vector<Pending> m_pending; /* storage freed while the GPU may still use it */

    ClampMode m_clamp = ClampMode::ClippingEnabled;
    bool m_enBlend = false, m_enCull = false, m_enDepth = false, m_enScissor = false, m_enPolyOffset = false;
    bool m_enDepthClamp = false;
    bool m_vramLru = true; /* keepInVram (dev switch novramlru.txt: off) */
    FILE *m_rtLog = nullptr; /* RTLOG dev switch rtlog.txt: render targets per present */
    u32 m_cullMode = CM_BACK;
    bool m_depthWrite = true;
    u32 m_depthFunc = CMP_LESS;
    s32 m_vp[4] = {0, 0, WINDOW_W, WINDOW_H};
    s32 m_sc[4] = {0, 0, WINDOW_W, WINDOW_H};
    u32 m_blend[4] = {BL_ONE, BL_ZERO, BL_ONE, BL_ZERO};
    u32 m_blendColor = 0;
    s32 m_unpack = 4;

    const CombinerProgram *m_program = nullptr;
    Kind m_kind = Kind::Noop;
    float m_cover = 0; /* frame buffer area drawn this frame, in pixels (noteCoverage) */
    void noteCoverage(const GVtx *v, u32 n);
    bool m_showStale = false; /* the window copied a frame buffer with dropped draws (skipDraw) */
    bool m_showSame = false;  /* the window copied the buffer on screen, unchanged since */
    u32 m_winSrc = 0, m_winCopies = 0; /* this present's window copies */
    u32 m_shown = 0;          /* buffer on screen when the last present showed one copy */
    bool m_winClear = false;  /* a window clear waiting for the window's first draw kept */
    u32 m_winClearColor = 0;
    void clearTarget(Fbo &f, u32 rgba);

    shaderProgram_s m_shader{};
    u32 *m_half = nullptr; /* 8x8 texture of 0.5 for the combiner (texture unit 2) */
};

void set_current_program(const CombinerProgram *p, Kind k)
{
    if (g_ctx)
        g_ctx->setProgram(p, k);
}

/* ---- lifecycle ------------------------------------------------------------------------------ */

void ContextImplPica::init()
{
    CRUMB("gfx: init");
    g_ctx = this;
    init_morton();

    shaderProgramInit(&m_shader);
    DVLB_s *lib = video_shader_library();
    if (shaderProgramSetVsh(&m_shader, ctru_dvle(lib, 1)) < 0)
        printf("ctr64: gfx.v.pica missing from the shader library\n");

    m_half = static_cast<u32 *>(linearMemAlign(8 * 8 * 4, 0x80));
    if (m_half) {
        for (int i = 0; i < 64; i++)
            m_half[i] = 0x80808080;
        pica_cache_flush(m_half, 8 * 8 * 4);
    }

    if (FILE *rl = fopen("sdmc:/3ds/ctr64/rtlog.txt", "r")) {
        fclose(rl);
        m_rtLog = fopen("sdmc:/3ds/ctr64/rtlog.log", "w");
    }
    if (FILE *nx = fopen("sdmc:/3ds/ctr64/noxfercopy.txt", "r")) {
        fclose(nx);
        m_xferCopy = false;
    }
    if (FILE *pf = fopen("sdmc:/3ds/ctr64/potfb.txt", "r")) {
        fclose(pf);
        g_potfb = true;
    }
    if (FILE *st = fopen("sdmc:/3ds/ctr64/slowtex.txt", "r")) {
        fclose(st);
        g_slowtex = true;
    }
    if (FILE *cm = fopen("sdmc:/3ds/ctr64/nocrcmemo.txt", "r")) {
        fclose(cm);
        ctr64_crc_nomemo = true;
    }
    if (FILE *nv = fopen("sdmc:/3ds/ctr64/novramlru.txt", "r")) {
        fclose(nv);
        m_vramLru = false;
    }
    /* GLideN64's default frame buffer: the 400x240 window, presented to the top LCD. With frame
     * buffer emulation it only receives the finished frame once per present: in FCRAM when
     * anti-aliasing needs the VRAM for the game's 2x targets (keepInVram). */
    const bool win_vram = !(m_vramLru && config.frameBufferEmulation.enable && config.frameBufferEmulation.nativeResFactor > 1);
    Tex &c = m_tex[TEX_WINDOW_COLOR];
    c.w = WINDOW_W; c.h = WINDOW_H; c.tw = 512; c.th = 256;
    allocStorage(c, win_vram);
    Tex &d = m_tex[TEX_WINDOW_DEPTH];
    d.w = WINDOW_W; d.h = WINDOW_H; d.tw = 512; d.th = 256; d.depth = true;
    allocStorage(d, win_vram);
    m_window.color = TEX_WINDOW_COLOR;
    m_window.depth = TEX_WINDOW_DEPTH;
    rebuildFbo(m_window);
}

void ContextImplPica::destroy()
{
    CRUMB("gfx: destroy");
    pica_flush();
    pica_wait_idle();
    collectPending(true);
    for (auto &kv : m_tex) {
        Tex &t = kv.second;
        if (t.data)
            freeNow(t.data, t.vram, storageBytes(t));
    }
    m_tex.clear();
    forgetTexHits();
    m_fbo.clear();
    if (m_half)
        linearFree(m_half);
    m_half = nullptr;
    pica_forget_program(&m_shader); /* a later context may be allocated at the same address */
    shaderProgramFree(&m_shader);
    if (g_ctx == this)
        g_ctx = nullptr;
}

/* ---- state ---------------------------------------------------------------------------------- */

void ContextImplPica::enable(EnableParam p, bool on)
{
    CRUMB("gfx: enable");
    switch (u32(p)) {
    case EN_BLEND: m_enBlend = on; break;
    case EN_CULL_FACE: m_enCull = on; break;
    case EN_DEPTH_TEST: m_enDepth = on; break;
    case EN_SCISSOR_TEST: m_enScissor = on; break;
    case EN_POLYGON_OFFSET_FILL: m_enPolyOffset = on; break;
    case EN_DEPTH_CLAMP: m_enDepthClamp = on; break;
    default: break;
    }
}

u32 ContextImplPica::isEnabled(EnableParam p)
{
    CRUMB("gfx: isEnabled");
    switch (u32(p)) {
    case EN_BLEND: return m_enBlend;
    case EN_CULL_FACE: return m_enCull;
    case EN_DEPTH_TEST: return m_enDepth;
    case EN_SCISSOR_TEST: return m_enScissor;
    case EN_POLYGON_OFFSET_FILL: return m_enPolyOffset;
    case EN_DEPTH_CLAMP: return m_enDepthClamp;
    default: return 0;
    }
}

void ContextImplPica::clearColorBuffer(f32 r, f32 g, f32 b, f32 a)
{
    CRUMB("gfx: clearColorBuffer");
    if (m_drawFbo != 0) /* an N64 frame buffer, not the window GLideN64 clears at each present */
        ctr64_frame_cleared = true;
    m_enScissor = true;
    const u32 rgba = pica_rgba8(to_u8(r), to_u8(g), to_u8(b), to_u8(a));
    if (m_drawFbo == 0) { /* done with the first window draw kept (skipDraw) or at the present */
        m_winClear = true;
        m_winClearColor = rgba;
        return;
    }
    if (!skipDraw(true))
        clearTarget(drawFbo(), rgba);
}

/* The whole buffer (the fill engine ignores the scissor). The GL backend re-enables scissor after
 * a clear, which clearColorBuffer does: not here, a deferred window clear runs at a later draw. */
void ContextImplPica::clearTarget(Fbo &f, u32 rgba)
{
    materialize(f, USE_COLOR_CLEARED);
    if (!f.valid)
        return;
    ensureFrame();
    ctr64_gfx_stats[ST_CLEAR]++;
    pica_clear(&f.rt, rgba, false, 0);
    if (Tex *t = tex(f.color)) {
        t->dirty = true;
        t->gen++;
        t->used_frame = pica_frame_number();
    }
}

void ContextImplPica::clearDepthBuffer()
{
    CRUMB("gfx: clearDepthBuffer");
    if (skipDraw(true))
        return;
    Fbo &f = drawFbo();
    materialize(f, USE_DEPTH_ONLY); /* the color buffer moves at its next use: often a full clear */
    if (!f.valid || !f.rt.depth)
        return;
    ensureFrame();
    pica_clear_depth(&f.rt, 0xFFFFFF);
    m_enScissor = true;
}

/* ---- textures ------------------------------------------------------------------------------- */

ObjectHandle ContextImplPica::createTexture(Parameter)
{
    CRUMB("gfx: createTexture");
    u32 id = m_nextTex++;
    m_tex[id] = Tex();
    return ObjectHandle(id);
}

void ContextImplPica::deleteTexture(ObjectHandle name)
{
    CRUMB("gfx: deleteTexture");
    u32 id = u32(name);
    Tex *t = tex(id);
    if (!t || id >= TEX_WINDOW_COLOR)
        return;
    releaseStorage(*t);
    m_tex.erase(id);
    forgetTexHits();
    for (auto &u : m_unit)
        if (u == id)
            u = 0;
    storageChanged(id); /* a frame buffer it was attached to drops the freed storage */
}

/* ---- GPU memory lifetime ---------------------------------------------------------------------
 * The GPU runs up to a frame behind the CPU (two command slots), so texture memory the GPU may
 * still read or write is never reused in place:
 *   - freed storage goes to m_pending and is released once the last frame that used it is done
 *     (frame n is complete when frame n + 2 begins);
 *   - CPU updates to a texture still in flight go to fresh storage ("orphaning"), except for
 *     render targets, whose GPU-written contents must be kept: those wait for the GPU. */

bool ContextImplPica::inFlight(const Tex &t) const
{
    return t.data && t.used_frame + 2 > pica_frame_number();
}

extern "C" void *ctr64_xlinear_alloc(u32 size, u32 align);

/* libctru's linear allocator treats a pool with no free block left as never initialised: the
 * next linearAlloc re-creates it over the whole linear heap and hands out memory that is in use
 * (the screens, command lists, textures). Seen with anti-aliasing in the Mario Kart 64 menus:
 * hundreds of small textures took the last bytes, then garbage on both screens and a crash.
 * Every allocation here leaves this much free. */
static const u32 k_linear_margin = 256 * 1024;

static void *linear_alloc_safe(u32 bytes, u32 align)
{
#if defined(CTR64_RA)
    /* RetroArch replaces libctru's linear allocator (os/ctr/ctr_linear.c): it has no such
     * re-initialisation, and its linearSpaceFree() counts only the never-used space at the top,
     * not freed blocks, so the check below would refuse memory that is there */
    return linearMemAlign(bytes, align);
#else
    if ((u64)bytes + k_linear_margin + align > linearSpaceFree())
        return nullptr;
    return linearMemAlign(bytes, align);
#endif
}
extern "C" bool ctr64_xlinear_free(void *p);

void ContextImplPica::freeNow(void *p, bool vram, u32 bytes)
{
    ctr64_gfx_mem[vram ? 1 : 0] -= bytes;
    if (vram)
        vramFree(p);
    else if (!ctr64_xlinear_free(p))
        linearFree(p);
}

void ContextImplPica::releaseStorage(Tex &t)
{
    if (!t.data)
        return;
    const u32 bytes = storageBytes(t);
    if (inFlight(t)) {
        m_pending.push_back({t.data, t.vram, t.used_frame, bytes});
        ctr64_gfx_mem[3] += bytes;
    } else {
        freeNow(t.data, t.vram, bytes);
    }
    t.data = nullptr;
    t.vram = false;
}

void ContextImplPica::collectPending(bool all)
{
    const u32 now = pica_frame_number();
    size_t keep = 0;
    for (size_t i = 0; i < m_pending.size(); i++) {
        Pending &p = m_pending[i];
        if (all || p.frame + 2 <= now) {
            freeNow(p.ptr, p.vram, p.bytes);
            ctr64_gfx_mem[3] -= p.bytes;
        } else
            m_pending[keep++] = p;
    }
    m_pending.resize(keep);
}

void *ContextImplPica::gpuAlloc(u32 bytes, bool vram_first, bool &is_vram)
{
    const char *where = ctr64_crumb;
    /* a size that failed after a full GPU drain this frame fails again without another drain
     * (a game that asks again every draw would otherwise wait for the GPU hundreds of times) */
    static u32 fail_frame = ~0u, fail_bytes = ~0u;
    bool drained = fail_frame == pica_frame_number() && bytes >= fail_bytes;
    for (int attempt = 0; attempt < 8; attempt++) {
        void *p = vram_first ? vramAlloc(bytes) : nullptr;
        is_vram = p != nullptr;
        if (!p)
            p = linear_alloc_safe(bytes, 0x80); /* the PICA can also render to and sample FCRAM */
        if (!p)
            p = ctr64_xlinear_alloc(bytes, 0x80); /* main.c: New 3DS reserve (anti-aliasing) */
        if (p) {
            ctr64_gfx_mem[is_vram ? 1 : 0] += bytes;
            if (!is_vram && bytes > ctr64_gfx_mem[2])
                ctr64_gfx_mem[2] = bytes;
            if (bytes >= fail_bytes)
                fail_frame = ~0u; /* memory came back */
            return p;
        }
        /* out of memory: drop least recently used textures from GLideN64's cache (Textures.cpp),
         * then let the GPU finish and release everything pending */
        const bool evicted = ctr64_tex_evict(bytes) != 0;
        if (evicted)
            ctr64_gfx_evicted++;
        if (drained && !evicted)
            break; /* nothing new to free: one GPU drain per attempt that can help */
        drained = true;
        CTR64_PROF(29);
        pica_flush();
        pica_wait_idle();
        collectPending(true);
    }
    if (bytes < fail_bytes || fail_frame != pica_frame_number()) {
        fail_frame = pica_frame_number();
        fail_bytes = bytes;
    }
    ctr64_gfx_stats[ST_ALLOC_FAIL]++;
    { /* development aid: with memlog.txt, the GPU memory held at the first failure -> memlog.log */
        static bool done;
        FILE *ml;
        if (!done && (ml = fopen("sdmc:/3ds/ctr64/memlog.txt", "r"))) {
            fclose(ml);
            done = true;
            if ((ml = fopen("sdmc:/3ds/ctr64/memlog.log", "w"))) {
                fprintf(ml, "fail %u bytes at %s, linear free %u\n", bytes, where, (unsigned)linearSpaceFree());
                for (auto &kv : m_tex)
                    if (kv.second.data)
                        fprintf(ml, "tex %u %ux%u store %ux%u bpp %u depth %d vram %d rt %d lazy %d used %u drawn_frame %u bytes %u\n",
                                kv.first, kv.second.w, kv.second.h, kv.second.tw, kv.second.th, kv.second.bpp, kv.second.depth,
                                kv.second.vram, kv.second.rt, kv.second.lazy, kv.second.used_frame, kv.second.draw_frame, storageBytes(kv.second));
                for (auto &p : m_pending)
                    fprintf(ml, "pending %u vram %d frame %u\n", p.bytes, p.vram, p.frame);
                fclose(ml);
            }
        }
    }
    ctr64_allocfail_where = where;
    ctr64_allocfail_bytes = bytes;
    return nullptr;
}

void ContextImplPica::allocStorage(Tex &t, bool vram, bool reuse, bool zero)
{
    CRUMB("gfx: allocStorage");
    const u32 bytes = storageBytes(t);
    if (reuse) {
        /* Storage of the same size released in the last frames (still pending: the GPU may be
         * reading it) is taken over instead of allocating anew: the GPU runs its queue in order,
         * so it is done with the old contents before it zero-fills them for the new owner, and
         * the CPU never touches it. OoT's pause menu with anti-aliasing re-creates a 2 MB frame
         * buffer texture every frame: each allocation then found no room until the GPU drained
         * (HW profile 2026-11-09: 845 ms of allocStorage waits in its slow seconds). */
        int best = -1;
        for (size_t i = 0; i < m_pending.size(); i++)
            if (m_pending[i].bytes == bytes && (best < 0 || (m_pending[i].vram && !m_pending[best].vram) ||
                                                (m_pending[i].vram == m_pending[best].vram && vram == m_pending[i].vram &&
                                                 m_pending[i].frame < m_pending[best].frame)))
                best = (int)i;
        if (best >= 0 && (m_pending[best].vram || !vram || vramSpaceFree() < bytes)) {
            t.data = m_pending[best].ptr;
            t.vram = m_pending[best].vram;
            ctr64_gfx_mem[3] -= bytes;
            m_pending.erase(m_pending.begin() + best);
            pica_fill_zero(t.data, bytes);
            t.used_frame = 0;
            t.dirty = true;
            t.gen++;
            ctr64_gfx_reused++;
            return;
        }
    }
    t.data = gpuAlloc(bytes, vram, t.vram);
    if (t.data && !t.vram) {
        if (zero) {
            memset(t.data, 0, bytes);
            pica_cache_flush(t.data, bytes);
        }
    } else if (t.data) {
        pica_fill_zero(t.data, bytes); /* VRAM holds what the last app left (SM64's first frame) */
    }
    t.used_frame = 0;
    t.dirty = true;
    t.gen++;
}

void ContextImplPica::storageChanged(u32 id)
{
    if (m_window.color == id || m_window.depth == id)
        rebuildFbo(m_window);
    for (auto &kv : m_fbo)
        if (kv.second.color == id || kv.second.depth == id)
            rebuildFbo(kv.second);
}

void ContextImplPica::moveToVram(u32 id, bool copy)
{
    CRUMB("gfx: moveToVram");
    Tex *t = tex(id);
    if (!t || t->vram || !t->data)
        return;
    const u32 bytes = storageBytes(*t);
    void *v = vramAlloc(bytes);
    if (!v)
        return; /* keep rendering to FCRAM */
    ctr64_gfx_mem[1] += bytes;
    /* a GPU copy, queued after the rendering so far (a CPU store to VRAM faults on hardware);
     * the old storage stays allocated until that copy has run (in flight, freed 2 frames on) */
    if (copy)
        pica_copy(v, t->data, bytes);
    t->used_frame = pica_frame_number();
    releaseStorage(*t);
    t->data = v;
    t->vram = true;
    t->dirty = true;
    t->gen++;
    storageChanged(id);
}

/* Render targets in FCRAM are much slower to draw into than in VRAM. With anti-aliasing a
 * 640x480 target takes 2 MB (1024x512 storage) and the 6 MB of VRAM holds only two of a game's
 * color buffers + depth buffer, so most drawing went to FCRAM (2026-11-09, SSB intro in Azahar:
 * every draw had an FCRAM depth buffer, 65% an FCRAM color buffer; none at 1x). The targets drawn
 * now are kept in VRAM: a target not in VRAM displaces the one drawn longest ago (not in this
 * frame), which a GPU copy moves to FCRAM. VRAM is only ever written by the GPU, in queue order,
 * so the displaced storage is reusable at once. Dev switch novramlru.txt: off. */
bool ContextImplPica::evictToFcram(u32 id)
{
    Tex *t = tex(id);
    if (!t || !t->vram || !t->data)
        return false;
    const u32 bytes = storageBytes(*t);
    void *l = linear_alloc_safe(bytes, 0x80);
    if (!l)
        l = ctr64_xlinear_alloc(bytes, 0x80);
    if (!l)
        return false;
    ctr64_gfx_mem[0] += bytes;
    pica_cache_flush(l, bytes); /* no dirty CPU lines left to overwrite what the GPU writes */
    pica_copy_from_vram(l, t->data, bytes); /* after everything drawn into it so far */
    freeNow(t->data, true, bytes);
    t->data = l;
    t->vram = false;
    t->dirty = true;
    t->gen++;
    t->used_frame = pica_frame_number();
    ctr64_gfx_vram_moves[1] += bytes >> 10;
    storageChanged(id);
    return true;
}

void ContextImplPica::keepInVram(Fbo &f, int use)
{
    const u32 now = pica_frame_number();
    const u32 ids[2] = {f.depth, f.color}; /* the depth buffer first: read and written by every draw */
    for (int k = 0; k < (use == USE_DEPTH_ONLY ? 1 : 2); k++) {
        Tex *t = tex(ids[k]);
        if (!t || !t->data)
            continue;
        t->draw_frame = now;
        if (use == USE_DRAW)
            t->ndraw++;
        if (t->vram || !t->rt || !m_vramLru)
            continue;
        const bool copy = !(k == 1 && use == USE_COLOR_CLEARED); /* a full clear replaces the contents */
        const u32 bytes = storageBytes(*t);
        /* A color target drawn at most twice per present (GLideN64's native-resolution copies of
         * a 2x frame buffer for the RDRAM: one copy draw each) is not worth the VRAM another needs:
         * it is drawn in FCRAM, even when VRAM is free now (MK64 with anti-aliasing: three 1x copy
         * targets next to three 2x color buffers kept the VRAM one target short, 1.6 MB moved per
         * frame, Azahar 2026-11-10) */
        if (!t->depth && t->ndraw_last <= 2)
            continue;
        moveToVram(ids[k], copy);
        while (!t->vram) {
            /* the VRAM target drawn longest ago, not in this frame and not this FBO's; among
             * those, rarely drawn targets first */
            u32 victim = 0, oldest = now;
            bool victim_rare = false;
            for (auto &kv : m_tex) {
                const Tex &v = kv.second;
                const bool rare = !v.depth && v.ndraw_last <= 2;
                /* a rare target drawn in this frame too: it is drawn in FCRAM from here on (MK64
                 * menus with AA: its three 1x copy targets, one drawn per frame, kept the VRAM one
                 * 2x buffer short, 1.6 MB moved per frame) */
                if (!v.vram || !v.rt || !v.data || kv.first == f.color || kv.first == f.depth ||
                    (v.draw_frame >= now && !rare))
                    continue;
                if ((rare && !victim_rare) || (rare == victim_rare && v.draw_frame < oldest)) {
                    oldest = v.draw_frame;
                    victim = kv.first;
                    victim_rare = rare;
                }
            }
            if (!victim || !evictToFcram(victim))
                break; /* all drawn in this frame: this one stays in FCRAM */
            moveToVram(ids[k], copy);
            t = tex(ids[k]);
        }
        if (t->vram)
            ctr64_gfx_vram_moves[0] += copy ? bytes >> 10 : 0;
    }
}

void ContextImplPica::init2DTexture(const Context::InitTextureParams &p)
{
    CTR64_PROF(25);
    CRUMB("gfx: init2DTexture");
    const u32 id = u32(p.handle);
    Tex *t = tex(id);
    if (!t || p.mipMapLevel != 0)
        return;
    releaseStorage(*t);
    t->w = (u16)p.width;
    t->h = (u16)p.height;
    t->tw = pot(p.width);
    t->th = pot(p.height);
    t->depth = u32(p.internalFormat) == IF_DEPTH;
    t->lazy = false;
    if (p.data && !t->depth && u32(p.dataType) == DT_US_5551 && !g_slowtex && p.width && p.height) {
        /* RGBA5551 source (95% of the MK64 menu textures): unless it is all gray (LA8/LA4 would
         * be smaller), pick_format chooses RGBA5551 and the stored values are the source values:
         * copied into the tiles as they are (same bytes as the general path below) */
        CTR64_PROF(30);
        const u16 *src = static_cast<const u16 *>(p.data);
        const u32 n = (u32)p.width * p.height;
        u32 i = 0;
        while (i < n && !((src[i] ^ (src[i] >> 5)) & 0x07FE)) /* r == g == b: gray */
            i++;
        if (i < n) {
            t->fmt = GPU_RGBA5551;
            t->bpp = 2;
            {
                CTR64_PROF(26);
                allocStorage(*t, false, false, false);
            }
            storageChanged(id);
            if (!t->data)
                return;
            {
                CTR64_PROF(27);
                write_full<GPU_RGBA5551, true, u16>(*t, src, p.width, p.height, p.width);
            }
            t->drawn = true;
            {
                CTR64_PROF(31);
                pica_cache_flush(t->data, storageBytes(*t));
            }
            t->dirty = true;
            t->gen++;
            return;
        }
    }
    if (p.data && !t->depth) {
        /* N64 texture data: store it in the smallest format that keeps every texel exact */
        std::vector<u32> &px = m_scratch;
        /* a bogus size (e.g. a negative tile width, see GLideN64 Textures.cpp) must cost one
         * missing texture, not an abort: the 3DS heap cannot take what a PC shrugs off */
        try {
            px.resize((size_t)p.width * p.height);
        } catch (const std::bad_alloc &) {
            FILE *lf = fopen("sdmc:/3ds/ctr64/texwarn.log", "a");
            if (lf) {
                fprintf(lf, "init2DTexture %ux%u: no memory, texture skipped\n", (unsigned)p.width, (unsigned)p.height);
                fclose(lf);
            }
            return;
        }
        {
            CTR64_PROF(30);
            fetch_all_fast(static_cast<const u8 *>(p.data), (u32)px.size(), u32(p.format), u32(p.dataType), px.data());
            t->fmt = g_slowtex ? pick_format(px) : pick_format_fast(px.data(), (u32)px.size());
        }
        t->bpp = bytes_per_texel(t->fmt);
        {
            CTR64_PROF(26);
            allocStorage(*t, false, false, g_slowtex); /* every byte is written below */
        }
        storageChanged(id);
        if (!t->data)
            return;
        CTR64_PROF(27);
        if (g_slowtex || !p.width || !p.height) {
            writeTexels(*t, px.data(), 0, 0, p.width, p.height);
            return;
        }
        switch (t->fmt) {
        case GPU_RGB8: write_full<GPU_RGB8>(*t, px.data(), p.width, p.height, p.width); break;
        case GPU_RGBA5551: write_full<GPU_RGBA5551>(*t, px.data(), p.width, p.height, p.width); break;
        case GPU_RGBA4: write_full<GPU_RGBA4>(*t, px.data(), p.width, p.height, p.width); break;
        case GPU_LA8: write_full<GPU_LA8>(*t, px.data(), p.width, p.height, p.width); break;
        case GPU_LA4: write_full<GPU_LA4>(*t, px.data(), p.width, p.height, p.width); break;
        default: write_full<GPU_RGBA8>(*t, px.data(), p.width, p.height, p.width); break;
        }
        t->drawn = true;
        {
            CTR64_PROF(31);
            pica_cache_flush(t->data, storageBytes(*t));
        }
        t->dirty = true;
        t->gen++;
        return;
    }
    switch (u32(p.internalFormat)) {
    case IF_RGBA4: t->fmt = GPU_RGBA4; break;
    case IF_RGB5_A1: t->fmt = GPU_RGBA5551; break;
    default: t->fmt = GPU_RGBA8; break;
    }
    t->bpp = t->depth ? 4 : bytes_per_texel(t->fmt);
    t->tw = fb_width(p.width);
    /* No data: a render target or a texture filled later. Storage is allocated on first use,
     * so targets GLideN64 creates but never draws to (e.g. the 640x580 TexrectDrawer buffer
     * at native resolution) cost nothing. */
    t->lazy = true;
    storageChanged(id);
}

void ContextImplPica::ensureStorage(u32 id)
{
    Tex *t = tex(id);
    if (!t || t->data || !t->lazy)
        return;
    allocStorage(*t, t->rt, true);
    t->lazy = !t->data;
    storageChanged(id);
}

void ContextImplPica::writeTexels(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h)
{
    t.drawn = true;
    if (g_slowtex) {
        switch (t.fmt) {
        case GPU_RGB8: write_rect<GPU_RGB8>(t, px, x0, y0, w, h); break;
        case GPU_RGBA5551: write_rect<GPU_RGBA5551>(t, px, x0, y0, w, h); break;
        case GPU_RGBA4: write_rect<GPU_RGBA4>(t, px, x0, y0, w, h); break;
        case GPU_LA8: write_rect<GPU_LA8>(t, px, x0, y0, w, h); break;
        case GPU_LA4: write_rect<GPU_LA4>(t, px, x0, y0, w, h); break;
        default: write_rect<GPU_RGBA8>(t, px, x0, y0, w, h); break;
        }
    } else {
        switch (t.fmt) {
        case GPU_RGB8: write_rect_fast<GPU_RGB8>(t, px, x0, y0, w, h); break;
        case GPU_RGBA5551: write_rect_fast<GPU_RGBA5551>(t, px, x0, y0, w, h); break;
        case GPU_RGBA4: write_rect_fast<GPU_RGBA4>(t, px, x0, y0, w, h); break;
        case GPU_LA8: write_rect_fast<GPU_LA8>(t, px, x0, y0, w, h); break;
        case GPU_LA4: write_rect_fast<GPU_LA4>(t, px, x0, y0, w, h); break;
        default: write_rect_fast<GPU_RGBA8>(t, px, x0, y0, w, h); break;
        }
    }
    const bool padded = x0 == 0 && y0 == 0 && w >= t.w && h >= t.h && (t.w < t.tw || t.h < t.th);
    if (padded)
        pad_edges(t);
    if (!t.vram) {
        /* flush the rows of 8x8 tiles written (storage rows run bottom-up) */
        const u32 r_lo = y0 + h >= t.th ? 0 : t.th - (y0 + h), r_hi = y0 >= t.th ? 0 : t.th - 1 - y0;
        const u32 tile_row_bytes = (u32)(t.tw >> 3) * 64 * t.bpp;
        if (padded || h == 0)
            pica_cache_flush(t.data, storageBytes(t));
        else
            pica_cache_flush(static_cast<u8 *>(t.data) + (r_lo >> 3) * tile_row_bytes,
                                  ((r_hi >> 3) - (r_lo >> 3) + 1) * tile_row_bytes);
    }
    t.dirty = true;
    t.gen++;
}

/* Updates the 8x8 tiles that the rect (x0, y0, w, h) touches through a staging block: the texels
 * are written into the block (tile rows of only the touched tiles) and a GPU copy into the
 * texture, with a row gap for the untouched tiles, is queued after the draws recorded so far, so
 * the CPU never waits for the GPU (with anti-aliasing each wait was ~7 ms, 50 per second in the
 * OoT pause menu). Partly covered edge tiles are first read from the texture (the GPU only reads
 * it). false: not possible here (padded update, no memory) - the caller waits instead. */
bool ContextImplPica::stagedUpdate(Tex &t, const u32 *px, u32 x0, u32 y0, u32 w, u32 h)
{
    if (t.vram || t.depth || !h || !w || y0 >= t.th || x0 >= t.tw)
        return false;
    if (x0 == 0 && y0 == 0 && w >= t.w && h >= t.h && (t.w < t.tw || t.h < t.th))
        return false; /* edge padding writes the whole texture */
    CRUMB("gfx: stagedUpdate"); /* waits from here on are this path's (stats.log "waits") */
    if (x0 + w > t.tw)
        return false; /* rows of px are w long: clipping w here would shear them (the caller's
                         path writes the clipped rect with the right stride) */
    if (y0 + h > t.th)
        h = t.th - y0;
    const u32 r_lo = t.th - (y0 + h), r_hi = t.th - 1 - y0; /* storage rows */
    const u32 tr0 = r_lo >> 3, ntr = (r_hi >> 3) - tr0 + 1;
    const u32 tx0 = x0 >> 3, ntx = ((x0 + w + 7) >> 3) - tx0;
    const u32 tile_bytes = 64u * t.bpp;
    const u32 run = ntx * tile_bytes;                  /* touched bytes per tile row */
    const u32 row_bytes = (u32)(t.tw >> 3) * tile_bytes;
    if (run % 16 || (row_bytes - run) % 16)
        return false;
    const u32 bytes = ntr * run;
    bool v = false;
    u8 *stage = static_cast<u8 *>(gpuAlloc(bytes, false, v));
    if (!stage)
        return false;
    u8 *dst = static_cast<u8 *>(t.data) + tr0 * row_bytes + tx0 * tile_bytes;
    const bool partial = (x0 & 7) || ((x0 + w) & 7) || (r_lo & 7) || ((r_hi + 1) & 7);
    if (partial) {
        /* the edge tiles are read from the texture: an earlier staged copy must have landed.
         * (Aligned updates never wait: copies run in queue order. On hardware that copy sits
         * behind a frame of drawing, ~18 ms with anti-aliasing.) */
        pica_wait_fence(t.copy_fence);
        pica_cache_invalidate(dst, (ntr - 1) * row_bytes + run);
        for (u32 r = 0; r < ntr; r++)
            memcpy(stage + r * run, dst + r * row_bytes, run);
    }
    /* the copy below rewrites the texture behind the cache: drop any cached lines there */
    pica_cache_invalidate(dst, (ntr - 1) * row_bytes + run);
    Tex s = t;
    s.tw = (u16)(ntx * 8);
    s.data = stage - tr0 * run; /* storage tile row tr lands at stage + (tr - tr0) * run */
    if (x0 == 0 && y0 == 0 && w == t.tw && h == t.th && !g_slowtex)
        write_full_any(s, px, w, h, w); /* the whole texture (frame buffer copies, AA): tiled writer */
    else
        writeTexels(s, px, x0 - tx0 * 8, y0, w, h); /* flushes the stage */
    pica_cache_flush(stage, bytes);
    pica_copy_gap(dst, stage, bytes, run / 16, (row_bytes - run) / 16);
    t.copy_fence = pica_fence();
    t.used_frame = pica_frame_number();
    t.dirty = true;
    t.gen++;
    t.drawn = true;
    m_pending.push_back({stage, false, pica_frame_number(), bytes});
    ctr64_gfx_mem[3] += bytes;
    return true;
}

/* Textures up to this size are orphaned when updated while in flight; larger ones wait. */
const u32 k_orphan_max_bytes = 1024 * 1024;

void ContextImplPica::update2DTexture(const Context::UpdateTextureDataParams &p)
{
    CRUMB("gfx: update2DTexture");
    ctr64_fb_copies++;
    CTR64_PROF(15);
    const u32 id = u32(p.handle);
    Tex *t = tex(id);
    if (!t || !p.data || p.mipMapLevel != 0 || t->depth)
        return;
    if (!t->data && t->lazy && !t->rt && needs_shadow(*t)) {
        /* a texture without data that turns out to be filled by the CPU, not drawn into
         * (GLideN64's RDRAM frame buffer copy: OoT's pause menu, 640x480 with AA): sampled with a
         * 1.25 MB shadow copy per draw on hardware at its exact width (5 per frame, ~6 MB of GPU
         * copies per frame). Power-of-two storage instead: sampled as it is. */
        t->tw = pot(t->w);
    }
    ensureStorage(id);
    if (!t->data)
        return;
    std::vector<u32> &px = m_scratch;
    px.resize((size_t)p.width * p.height);
    Ctr64ProfScope fetch_scope(36);
    fetch_all_fast(static_cast<const u8 *>(p.data), (u32)px.size(), u32(p.format), u32(p.dataType), px.data());
    const bool fits = fits_all(t->fmt, px.data(), (u32)px.size());
    fetch_scope.end();
    const bool full = p.x == 0 && p.y == 0 && p.width >= t->w && p.height >= t->h;
    if (inFlight(*t) || !fits) {
        if (fits && (t->vram || storageBytes(*t) >= k_orphan_max_bytes)) {
            /* a render target, whose GPU-written contents must survive, or a texture too big to
             * double up (the RDRAM frame buffer copy, updated every frame in the OoT pause menu:
             * a second 2 MB block rarely exists in linear memory). A big texture is updated by a
             * GPU copy queued after the draws that read it; otherwise wait for the GPU. */
            {
                CTR64_PROF(37);
                if (!t->vram && stagedUpdate(*t, px.data(), p.x, p.y, p.width, p.height))
                    return;
            }
            Ctr64ProfScope wait_scope(38);
            pica_flush();
            pica_wait_idle();
        } else {
            /* new storage (keeping the old for frames still in flight), widened to RGBA8 if the
             * update does not fit the current format */
            Tex n = *t;
            if (!fits) {
                n.fmt = GPU_RGBA8;
                n.bpp = 4;
            }
            n.data = gpuAlloc(storageBytes(n), t->vram, n.vram);
            if (!n.data)
                return;
            if (!full) {
                if (t->vram || inFlight(*t)) {
                    pica_flush();
                    pica_wait_idle(); /* read back what the GPU may still be writing */
                }
                const u32 count = (u32)t->tw * t->th;
                if (n.fmt == t->fmt)
                    memcpy(n.data, t->data, storageBytes(n));
                else
                    for (u32 i = 0; i < count; i++)
                        store_texel(n, i, load_texel(*t, i));
            }
            releaseStorage(*t);
            *t = n;
            t->used_frame = 0;
            storageChanged(id);
        }
    }
    Ctr64ProfScope write_scope(39);
    if (full && !t->vram && !g_slowtex) {
        /* the whole image (GLideN64's RDRAM frame buffer copies: OoT's pause menu, 320x240 per
         * frame, was 11.9 ms on hardware by writeTexels): the tiled writer of init2DTexture, same
         * bytes (texels past the logical size come from the edge, as pad_edges did) */
        const u32 w = p.width < t->w ? p.width : t->w, h = p.height < t->h ? p.height : t->h;
        switch (t->fmt) {
        case GPU_RGB8: write_full<GPU_RGB8>(*t, px.data(), w, h, p.width); break;
        case GPU_RGBA5551: write_full<GPU_RGBA5551>(*t, px.data(), w, h, p.width); break;
        case GPU_RGBA4: write_full<GPU_RGBA4>(*t, px.data(), w, h, p.width); break;
        case GPU_LA8: write_full<GPU_LA8>(*t, px.data(), w, h, p.width); break;
        case GPU_LA4: write_full<GPU_LA4>(*t, px.data(), w, h, p.width); break;
        default: write_full<GPU_RGBA8>(*t, px.data(), w, h, p.width); break;
        }
        t->drawn = true;
        pica_cache_flush(t->data, storageBytes(*t));
        t->dirty = true;
        t->gen++;
        return;
    }
    writeTexels(*t, px.data(), p.x, p.y, p.width, p.height);
}

void ContextImplPica::setTextureParameters(const Context::TexParameters &p)
{
    CRUMB("gfx: setTextureParameters");
    Tex *t = tex(u32(p.handle));
    u32 unit = u32(p.textureUnitIndex);
    if (unit < 2)
        m_unit[unit] = u32(p.handle);
    if (!t)
        return;
    auto filt = [](Parameter v, GPU_TEXTURE_FILTER_PARAM cur) {
        if (!v.isValid())
            return cur;
        u32 f = u32(v);
        return (f == TP_LINEAR || f == TP_LINEAR_MIP_NEAREST) ? GPU_LINEAR : GPU_NEAREST;
    };
    auto wrap = [](Parameter v, GPU_TEXTURE_WRAP_PARAM cur) {
        if (!v.isValid())
            return cur;
        u32 w = u32(v);
        return w == TP_REPEAT ? GPU_REPEAT : w == TP_MIRROR ? GPU_MIRRORED_REPEAT : GPU_CLAMP_TO_EDGE;
    };
    GPU_TEXTURE_FILTER_PARAM mag = (t->param & GPU_TEXTURE_MAG_FILTER(1)) ? GPU_LINEAR : GPU_NEAREST;
    GPU_TEXTURE_FILTER_PARAM min = (t->param & GPU_TEXTURE_MIN_FILTER(1)) ? GPU_LINEAR : GPU_NEAREST;
    GPU_TEXTURE_WRAP_PARAM ws = (GPU_TEXTURE_WRAP_PARAM)((t->param >> 12) & 3);
    GPU_TEXTURE_WRAP_PARAM wt = (GPU_TEXTURE_WRAP_PARAM)((t->param >> 8) & 3);
    t->param = GPU_TEXTURE_MAG_FILTER(filt(p.magFilter, mag)) | GPU_TEXTURE_MIN_FILTER(filt(p.minFilter, min)) |
               GPU_TEXTURE_WRAP_S(wrap(p.wrapS, ws)) | GPU_TEXTURE_WRAP_T(wrap(p.wrapT, wt));
}

void ContextImplPica::bindTexture(const Context::BindTextureParameters &p)
{
    CRUMB("gfx: bindTexture");
    u32 unit = u32(p.textureUnitIndex);
    if (unit < 2)
        m_unit[unit] = u32(p.texture);
}

/* ---- frame buffers -------------------------------------------------------------------------- */

ObjectHandle ContextImplPica::createFramebuffer()
{
    CRUMB("gfx: createFramebuffer");
    u32 id = m_nextFbo++;
    m_fbo[id] = Fbo();
    return ObjectHandle(id);
}

void ContextImplPica::deleteFramebuffer(ObjectHandle name)
{
    CRUMB("gfx: deleteFramebuffer");
    u32 id = u32(name);
    m_fbo.erase(id);
    if (m_drawFbo == id)
        m_drawFbo = 0;
    if (m_readFbo == id)
        m_readFbo = 0;
}

void ContextImplPica::bindFramebuffer(BufferTargetParam target, ObjectHandle name)
{
    CRUMB("gfx: bindFramebuffer");
    u32 id = u32(name);
    if (id != 0 && m_fbo.find(id) == m_fbo.end())
        id = 0;
    u32 t = u32(target);
    ctr64_gfx_stats[ST_FBO]++;
    if (t == BT_FRAMEBUFFER || t == BT_DRAW)
        m_drawFbo = id;
    if (t == BT_FRAMEBUFFER || t == BT_READ)
        m_readFbo = id;
}

void ContextImplPica::rebuildFbo(Fbo &f)
{
    CRUMB("gfx: rebuildFbo");
    Tex *c = tex(f.color);
    f.valid = c && c->data && !c->depth;
    if (!f.valid)
        return;
    memset(&f.rt, 0, sizeof(f.rt));
    f.rt.color = c->data;
    f.rt.width = c->tw;
    f.rt.height = c->th;
    f.rt.color_fmt = c->fmt == GPU_RGBA4 ? GPU_RB_RGBA4 : c->fmt == GPU_RGBA5551 ? GPU_RB_RGBA5551 : GPU_RB_RGBA8;
    f.rt.depth_fmt = GPU_RB_DEPTH24_STENCIL8;
    Tex *d = tex(f.depth);
    /* The PICA has one frame buffer size; a depth buffer is usable if it is at least as large. */
    if (d && d->data && d->depth && (u32)d->tw * d->th >= (u32)c->tw * c->th)
        f.rt.depth = d->data;
}

void ContextImplPica::addFrameBufferRenderTarget(const Context::FrameBufferRenderTarget &p)
{
    CRUMB("gfx: addFrameBufferRenderTarget");
    u32 id = u32(p.bufferHandle);
    Fbo *f = id == 0 ? &m_window : (m_fbo.count(id) ? &m_fbo[id] : nullptr);
    if (!f)
        return;
    u32 tid = u32(p.textureHandle);
    if (Tex *t = tex(tid))
        t->rt = true;
    if (u32(p.attachment) == BA_DEPTH) {
        f->depth = tid;
    } else if (u32(p.attachment) == BA_COLOR0) {
        f->color = tid;
    }
    moveToVram(tid); /* rebuilds the FBOs that use it if the storage moves */
    rebuildFbo(*f);
}

void ContextImplPica::initRenderbuffer(const Context::InitRenderbufferParams &p)
{
    CRUMB("gfx: initRenderbuffer");
    Tex *t = tex(u32(p.handle));
    if (!t)
        return;
    releaseStorage(*t);
    t->w = (u16)p.width;
    t->h = (u16)p.height;
    t->tw = fb_width(p.width);
    t->th = pot(p.height);
    t->depth = true;
    t->bpp = 4;
    t->rt = true;
    t->lazy = true; /* allocated when first drawn to */
    storageChanged(u32(p.handle));
}

/* ---- programs ------------------------------------------------------------------------------- */

CombinerProgram *ContextImplPica::createCombinerProgram(Combiner &color, Combiner &alpha, const CombinerKey &key)
{
    u32 inputs = 0;
    if (key.getCycleType() != G_CYC_2CYCLE) {
        inputs |= collect_inputs(alpha.stage[0], correct_first);
        inputs |= collect_inputs(color.stage[0], correct_first);
    } else {
        inputs |= collect_inputs(alpha.stage[0], correct_first_2cyc);
        inputs |= collect_inputs(color.stage[0], correct_first_2cyc);
        if (alpha.numStages == 2)
            inputs |= collect_inputs(alpha.stage[1], correct_second);
        if (color.numStages == 2)
            inputs |= collect_inputs(color.stage[1], correct_second);
    }
    return new PicaCombinerProgram(key, inputs);
}

/* ---- drawing -------------------------------------------------------------------------------- */

GPU_BLENDFACTOR blend_factor(u32 f)
{
    switch (f) {
    case BL_ZERO: return GPU_ZERO;
    case BL_ONE: return GPU_ONE;
    case BL_SRC_ALPHA: return GPU_SRC_ALPHA;
    case BL_DST_ALPHA: return GPU_DST_ALPHA;
    case BL_ONE_MINUS_SRC_ALPHA: return GPU_ONE_MINUS_SRC_ALPHA;
    case BL_CONSTANT_ALPHA: return GPU_CONSTANT_ALPHA;
    case BL_ONE_MINUS_CONSTANT_ALPHA: return GPU_ONE_MINUS_CONSTANT_ALPHA;
    default: return GPU_ONE; /* dual-source factors are never requested (feature reported off) */
    }
}

void ContextImplPica::materialize(Fbo &f, int use)
{
    ensureStorage(f.color);
    ensureStorage(f.depth);
    keepInVram(f, use);
}

bool ContextImplPica::applyState(bool depth_allowed)
{
    CRUMB("gfx: applyState");
    Fbo &f = drawFbo();
    materialize(f);
    if (!f.valid)
        return false;
    if (m_enCull && m_cullMode == CM_FRONT_AND_BACK)
        return false;

    ensureFrame();
    pica_bind_target(&f.rt);
    pica_bind_program(&m_shader);
    pica_vertex_layout(k_layout, 4, sizeof(GVtx));
    pica_viewport((u16)m_vp[0], (u16)m_vp[1], (u16)m_vp[2], (u16)m_vp[3]);

    /* GL clips drawing to the frame buffer's logical size; with scissor on, to the scissor box */
    Tex *c = tex(f.color);
    s32 x0 = 0, y0 = 0, x1 = c->w, y1 = c->h;
    if (m_enScissor) {
        x0 = m_sc[0] > 0 ? m_sc[0] : 0;
        y0 = m_sc[1] > 0 ? m_sc[1] : 0;
        x1 = m_sc[0] + m_sc[2] < c->w ? m_sc[0] + m_sc[2] : c->w;
        y1 = m_sc[1] + m_sc[3] < c->h ? m_sc[1] + m_sc[3] : c->h;
        if (x1 <= x0 || y1 <= y0)
            return false;
    }
    pica_scissor(GPU_SCISSOR_NORMAL, (u16)x0, (u16)y0, (u16)x1, (u16)y1);

    const bool depth = depth_allowed && m_enDepth && f.rt.depth;
    const GPU_TESTFUNC func = m_depthFunc == CMP_LEQUAL ? GPU_LEQUAL : m_depthFunc == CMP_LESS ? GPU_LESS : GPU_ALWAYS;
    pica_depth_test(depth, depth ? func : GPU_ALWAYS, GPU_WRITE_COLOR | ((depth && m_depthWrite) ? GPU_WRITE_DEPTH : 0));
    pica_depth_map(-1.0f, (depth && m_enPolyOffset) ? k_poly_offset : 0.0f);

    /* Measured against GLideN64's GL output: for the same window coordinates the PICA's cull
     * modes select the opposite winding to GL's (GL culls back = clockwise faces here). */
    pica_cull(!m_enCull ? GPU_CULL_NONE : m_cullMode == CM_FRONT ? GPU_CULL_BACK_CCW : GPU_CULL_FRONT_CCW);

    if (m_enBlend) {
        pica_blend(GPU_BLEND_ADD, GPU_BLEND_ADD, blend_factor(m_blend[0]), blend_factor(m_blend[1]),
                   blend_factor(m_blend[2]), blend_factor(m_blend[3]));
        pica_blend_color(m_blendColor);
    } else {
        pica_blend_off();
    }

    if (c) {
        c->dirty = true; /* rendered to: sampling it later must refetch */
        c->gen++;
        c->used_frame = pica_frame_number();
    }
    if (Tex *d = tex(f.depth))
        d->used_frame = pica_frame_number();
    return true;
}

void ContextImplPica::bindUnit(int unit, bool n64_filter)
{
    CRUMB("gfx: bindUnit");
    ensureStorage(m_unit[unit]); /* sampling a target never drawn to: zeroed storage */
    Tex *t = tex(m_unit[unit]);
    if (!t || !t->data || t->depth) {
        pica_texture(unit, NULL, 0, 0, GPU_RGBA8, 0);
        return;
    }
    u32 param = t->param;
    if (n64_filter) {
        /* GLideN64 keeps N64 textures at GL_NEAREST and filters in the fragment shader when
         * the RDP's texture filter is not point sampling (uTextureFilterMode): use the PICA's
         * bilinear filter for that. */
        const bool bilerp = gDP.otherMode.cycleType != G_CYC_COPY && gDP.otherMode.textureFilter != G_TF_POINT;
        const GPU_TEXTURE_FILTER_PARAM f = bilerp ? GPU_LINEAR : GPU_NEAREST;
        param = (param & ~(GPU_TEXTURE_MAG_FILTER(1) | GPU_TEXTURE_MIN_FILTER(1))) | GPU_TEXTURE_MAG_FILTER(f) |
                GPU_TEXTURE_MIN_FILTER(f);
        /* N64 clamp applies at the tile's bounds, the mask wraps inside them (GLideN64's
         * clampWrapMirror). GLideN64 asks for CLAMP_TO_EDGE whenever clamp is on; with a tile
         * larger than the mask period the texture repeats instead (SSB's wallpaper, table). */
        const CachedTexture *ct = textureCache().current[unit];
        const gDPTile *tile = gSP.textureTile[unit];
        if (ct && tile && ct->frameBufferTexture == CachedTexture::fbNone && tile->textureMode == TEXTUREMODE_NORMAL &&
            gDP.otherMode.cycleType != G_CYC_COPY) {
            if (ct->clampS && ct->maskS) {
                u32 tileWidth = ((tile->lrs - tile->uls) & 0x03FF) + 1;
                if (tile->size > ct->size)
                    tileWidth <<= tile->size - ct->size;
                if (tileWidth > (1u << tile->originalMaskS))
                    param = (param & ~GPU_TEXTURE_WRAP_S(3)) |
                            GPU_TEXTURE_WRAP_S(ct->mirrorS ? GPU_MIRRORED_REPEAT : GPU_REPEAT);
            }
            if (ct->clampT && ct->maskT) {
                const u32 tileHeight = ((tile->lrt - tile->ult) & 0x03FF) + 1;
                if (tileHeight > (1u << tile->originalMaskT))
                    param = (param & ~GPU_TEXTURE_WRAP_T(3)) |
                            GPU_TEXTURE_WRAP_T(ct->mirrorT ? GPU_MIRRORED_REPEAT : GPU_REPEAT);
            }
        }
    }
    if (n64_filter && t->rt)
        ctr64_frame_fbtex = true; /* a frame buffer read as a texture (frameskip.c) */
    bool copied = false;
    if (needs_shadow(*t)) {
        void *sh = shadowOf(m_unit[unit], *t, copied);
        if (!sh) {
            pica_texture(unit, NULL, 0, 0, GPU_RGBA8, 0);
            return;
        }
        pica_texture(unit, sh, pot(t->tw), t->th, t->fmt, param);
    } else {
        pica_texture(unit, t->data, t->tw, t->th, t->fmt, param);
    }
    if (t->dirty || copied) {
        pica_texture_flush_cache();
        t->dirty = false;
    }
    t->used_frame = pica_frame_number();
}

/* The power-of-two shadow of a texture stored at a non-power-of-two width, up to date: a GPU copy
 * of each 8-row strip of tiles into the wider strip of the shadow (tiles keep their place, the
 * columns past the storage width stay zero, as in power-of-two storage). One shadow per storage
 * size, shared: the GPU runs copies and draws in queue order, so a copy for another texture
 * comes after the draws that sampled the previous one. */
void *ContextImplPica::shadowOf(u32 id, Tex &t, bool &copied)
{
    const u16 sw = pot(t.tw);
    Shadow *s = nullptr;
    for (Shadow &x : m_shadows)
        if (x.tw == sw && x.th == t.th && x.bpp == t.bpp) {
            s = &x;
            break;
        }
    if (!s) {
        const u32 bytes = (u32)sw * t.th * t.bpp;
        void *p = linear_alloc_safe(bytes, 0x80);
        if (!p)
            p = ctr64_xlinear_alloc(bytes, 0x80);
        if (!p)
            return nullptr;
        memset(p, 0, bytes);
        pica_cache_flush(p, bytes);
        ctr64_gfx_mem[0] += bytes;
        m_shadows.push_back(Shadow{p, sw, t.th, t.bpp, 0, 0, nullptr});
        s = &m_shadows.back();
    }
    if (s->owner == id && s->gen == t.gen && s->src == t.data)
        return s->p;
    const u32 run = (u32)t.tw * 8 * t.bpp, out = (u32)sw * 8 * t.bpp;
    pica_copy_gap(s->p, t.data, run * (t.th / 8), run / 16, (out - run) / 16);
    s->owner = id;
    s->gen = t.gen;
    s->src = t.data;
    copied = true;
    t.nshadow++;
    ctr64_gfx_shadow[0]++;
    ctr64_gfx_shadow[1] += (run * (t.th / 8)) >> 10;
    return s->p;
}

/* Texture coordinate transform of one tile (UTextureParamsFast + calcTexCoord + the
 * fragment shader's +0.0001), followed by logical -> storage scaling for the PICA texture. */
struct TexXform {
    float ss = 1, st = 1, os = 0, ot = 0, cs = 1, ct = 1, rs = 1, rt = 1;
    /* (((v * ss - os) * cs) + 0.0001) * rs, folded to v * as + bs by bake() */
    float as = 1, bs = 0.0001f, at = 1, bt = 0.0001f;
    void bake()
    {
        as = ss * cs * rs;
        bs = (0.0001f - os * cs) * rs;
        at = st * ct * rt;
        bt = (0.0001f - ot * ct) * rt;
    }
    float s(float v) const { return v * as + bs; }
    float t(float v) const { return v * at + bt; }
};

/* GLideN64's fog usage for the current draw (UFog): 1 = fog factor in shade alpha, 2 = in rgb */
int fog_usage()
{
    if (RSP.LLE)
        return 0;
    int usage = (gSP.geometryMode & G_FOG) ? 1 : 0;
    if (GBI.getMicrocodeType() == F3DAM) {
        const s16 fogMode = ((gSP.geometryMode >> 13) & 9) + 0xFFF8;
        usage = fogMode == 0 ? 1 : fogMode > 0 ? 2 : usage;
    }
    return usage;
}

/* Development aid: sdmc:/3ds/ctr64/drawlog.txt holds "from [to]" swap numbers; every draw of those
 * frames is
 * written to sdmc:/3ds/ctr64/drawlog.log with its combiner, state and window-space bounding box. */
u32 g_swap_count = 1; /* the swap that will present the draws being made */
FILE *draw_log()
{
    static int from = -2, to = -1;
    static FILE *f;
    if (from == -2) {
        from = -1;
        if (FILE *c = fopen("sdmc:/3ds/ctr64/drawlog.txt", "r")) {
            if (fscanf(c, "%d %d", &from, &to) < 1)
                from = -1;
            if (to < from)
                to = from;
            fclose(c);
        }
    }
    if ((int)g_swap_count < from || (int)g_swap_count > to) {
        if (f && (int)g_swap_count > to) {
            fclose(f);
            f = nullptr;
            from = to = -1;
        }
        return nullptr;
    }
    if (!f)
        f = fopen("sdmc:/3ds/ctr64/drawlog.log", "w");
    return f;
}

void log_draw(const char *what, const CombinerProgram *prog, int kind, const GVtx *v, u32 n, const s32 vp[4],
              const char *state)
{
    FILE *f = draw_log();
    if (!f)
        return;
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (u32 i = 0; i < n; i++) {
        if (v[i].w <= 0.0f)
            continue;
        /* NDC -> window pixels of the current viewport (GL: y up) */
        const float x = vp[0] + (v[i].x / v[i].w + 1.0f) * 0.5f * vp[2];
        const float y = vp[1] + (v[i].y / v[i].w + 1.0f) * 0.5f * vp[3];
        x0 = x < x0 ? x : x0; x1 = x > x1 ? x : x1;
        y0 = y < y0 ? y : y0; y1 = y > y1 ? y : y1;
    }
    unsigned long long mux = kind == 0 && prog ? prog->getKey().getMux() : 0;
    fprintf(f, "%s kind %d mux %016llx cyc %u n %u bbox %.0f,%.0f-%.0f,%.0f fog %d othermode %08x %s prim %.2f,%.2f,%.2f,%.2f env %.2f,%.2f,%.2f,%.2f\n",
            what, kind, mux, (unsigned)gDP.otherMode.cycleType, (unsigned)n, x0, y0, x1, y1, fog_usage(),
            (unsigned)gDP.otherMode.l, state, gDP.primColor.r, gDP.primColor.g, gDP.primColor.b, gDP.primColor.a,
            gDP.envColor.r, gDP.envColor.g, gDP.envColor.b, gDP.envColor.a);
    /* addresses: color image, depth image, and texture 0 (fb: a frame buffer texture) */
    const CachedTexture *t0 = textureCache().current[0];
    fprintf(f, "  ci %08x zi %08x t0 %08x fb %d\n", (unsigned)gDP.colorImage.address, (unsigned)gDP.depthImageAddress,
            t0 ? (unsigned)t0->address : 0u, t0 ? (int)t0->frameBufferTexture : -1);
}

/* Adds a draw's screen bounding box to the area drawn this frame; once that reaches 90% of the
 * frame buffer, the frame counts as redrawing its whole image (frameskip.c). SM64 never clears,
 * its skybox covers the screen; Paper Mario's name entry only redraws what changes. */
void ContextImplPica::noteCoverage(const GVtx *v, u32 n)
{
    if (m_drawFbo == 0)
        return;
    /* a draw sampling a frame buffer (Jet Force Gemini's round view, motion blur) composes an
     * image drawn earlier: skipping the next frame would leave that source empty */
    if (!ctr64_frame_fbtex && m_program && m_kind == Kind::Combiner) {
        TextureCache &cache = textureCache();
        for (u32 t = 0; t < 2; t++)
            if (m_program->usesTile(t) && cache.current[t] &&
                cache.current[t]->frameBufferTexture != CachedTexture::fbNone)
                ctr64_frame_fbtex = true;
    }
    if (ctr64_frame_cleared)
        return;
    const Tex *c = tex(drawFbo().color);
    if (!c || !c->w || !c->h)
        return;
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (u32 i = 0; i < n; i++) {
        if (v[i].w <= 0.0f)
            return; /* crosses the eye plane: leave it out */
        const float rw = 1.0f / v[i].w;
        const float x = v[i].x * rw, y = v[i].y * rw;
        x0 = x < x0 ? x : x0; x1 = x > x1 ? x : x1;
        y0 = y < y0 ? y : y0; y1 = y > y1 ? y : y1;
    }
    const float W = c->w, H = c->h;
    auto px = [&](float ndc, float o, float s, float lim) {
        const float p = o + (ndc + 1.0f) * 0.5f * s;
        return p < 0.0f ? 0.0f : p > lim ? lim : p;
    };
    const float w = px(x1, m_vp[0], m_vp[2], W) - px(x0, m_vp[0], m_vp[2], W);
    const float h = px(y1, m_vp[1], m_vp[3], H) - px(y0, m_vp[1], m_vp[3], H);
    m_cover += w * h;
    if (m_cover >= 0.9f * W * H)
        ctr64_frame_cleared = true;
}

/* Frameskip: in a skipped frame the draws into N64 frame buffers are dropped and the buffers
 * marked. The window, GLideN64's copy of the displayed buffer, is always drawn: the N64 shows a
 * buffer drawn in an earlier frame (double or triple buffering), so whether a present shows
 * a fresh image depends on the buffer it copies, not on whether its own frame was skipped. A
 * present copying a marked buffer is dropped (present()). Returns whether to drop this draw. */
bool ContextImplPica::skipDraw(bool clear)
{
    const Tex *src = !clear && m_kind == Kind::Copy ? tex(m_unit[0]) : nullptr;
    if (m_drawFbo == 0) {
        /* the window keeps the image last presented, so a frame dump shows the screen */
        if ((src && src->skipped) || m_showStale) {
            m_showStale = true;
            return true;
        }
        if (src) {
            m_winSrc = m_unit[0];
            /* the image on screen again (SM64 presents every VI, draws every other) */
            if (++m_winCopies == 1 && m_winSrc == m_shown && !src->drawn) {
                m_showSame = true;
                return true;
            }
        }
        if (m_showSame)
            return true;
        if (m_winClear) {
            m_winClear = false;
            clearTarget(m_window, m_winClearColor);
        }
        return false;
    }
    Tex *c = tex(drawFbo().color);
    ctr64_frame_drew = true;
    if (ctr64_render_skip) {
        if (c)
            c->skipped = true;
        return true;
    }
    if (c) {
        c->skipped = src && src->skipped; /* a copy of a marked buffer is as stale */
        c->drawn = true;
    }
    return false;
}

void ContextImplPica::texture_xforms(TexXform tx[2])
{
    TextureCache &cache = textureCache();
    for (u32 t = 0; t < 2; t++) {
        TexXform &x = tx[t];
        const gDPTile *tile = gSP.textureTile[t];
        CachedTexture *ct = cache.current[t];
        if (!tile || !ct)
            continue;
        float fuls = 0, fult = 0;
        if (tile->textureMode != TEXTUREMODE_BGIMAGE && tile->textureMode != TEXTUREMODE_FRAMEBUFFER_BG) {
            fuls = tile->fuls;
            fult = tile->fult;
            if (tile->frameBufferAddress > 0 && frameBufferList().getBuffer(tile->frameBufferAddress)) {
                if (tile->masks > 0 && tile->clamps == 0)
                    fuls = float(tile->uls % (1 << tile->masks));
                if (tile->maskt > 0 && tile->clampt == 0)
                    fult = float(tile->ult % (1 << tile->maskt));
            }
        }
        f32 shiftS = 1.0f, shiftT = 1.0f;
        getTextureShiftScale(t, cache, shiftS, shiftT);
        x.ss = shiftS;
        x.st = shiftT;
        /* (v*shift - texOffset + cacheOffset) * cacheScale */
        x.os = fuls - ct->offsetS;
        x.ot = fult - ct->offsetT;
        x.cs = ct->scaleS;
        x.ct = ct->scaleT;
        /* texel u of the cached texture is storage texel u: normalise by the storage size. (Not
         * w/tw: with a wrap mask GLideN64's cache scale is 1/pow2(width), e.g. 1/128 for the
         * 80-texel strips of the SM64 title background.) */
        if (Tex *p = tex(m_unit[t]))
            if (p->tw && x.cs > 0.0f && x.ct > 0.0f) {
                if (ct->frameBufferTexture != CachedTexture::fbNone) {
                    /* a frame buffer: cache scale = render scale / width, so u * cs * width is the
                     * storage texel (twice u when rendering at 2x) */
                    x.rs = float(ct->width) / sample_w(*p);
                    x.rt = float(ct->height) / p->th;
                } else {
                    x.rs = 1.0f / (x.cs * sample_w(*p));
                    x.rt = 1.0f / (x.ct * p->th);
                }
            }
    }
    tx[0].bake();
    tx[1].bake();
}

bool ContextImplPica::setupCombiner(bool rect)
{
    CRUMB("gfx: setupCombiner");
    (void)rect;
    PicaTev tev[pica_cmb::MAX_STAGES];
    for (auto &t : tev)
        pica_tev_default(&t);
    u8 buf_rgb = 0, buf_a = 0;
    u32 buf_color = 0xFFFFFFFF;
    bool alpha_test = false;
    u8 alpha_ref = 0;

    switch (m_kind) {
    case Kind::Copy:
    case Kind::Text:
        bindUnit(0);
        pica_texture(1, NULL, 0, 0, GPU_RGBA8, 0);
        pica_texture(2, NULL, 0, 0, GPU_RGBA8, 0);
        tev[0].src_rgb = tev[0].src_alpha = GPU_TEVSOURCES(GPU_TEXTURE0, 0, 0);
        break;
    case Kind::Clear:
        pica_texture(0, NULL, 0, 0, GPU_RGBA8, 0);
        pica_texture(1, NULL, 0, 0, GPU_RGBA8, 0);
        pica_texture(2, NULL, 0, 0, GPU_RGBA8, 0);
        tev[0].src_rgb = tev[0].src_alpha = GPU_TEVSOURCES(GPU_CONSTANT, 0, 0);
        tev[0].color = 0;
        break;
    default: {
        const pica_cmb::Program &pr = static_cast<const PicaCombinerProgram *>(m_program)->tev();
        int n = pr.n;
        for (int i = 0; i < n; i++) {
            const pica_cmb::Stage &s = pr.st[i];
            PicaTev &t = tev[i];
            t.src_rgb = GPU_TEVSOURCES(s.src_rgb[0], s.src_rgb[1], s.src_rgb[2]);
            t.src_alpha = GPU_TEVSOURCES(s.src_a[0], s.src_a[1], s.src_a[2]);
            t.op_rgb = GPU_TEVOPERANDS(s.op_rgb[0], s.op_rgb[1], s.op_rgb[2]);
            t.op_alpha = GPU_TEVOPERANDS(s.op_a[0], s.op_a[1], s.op_a[2]);
            t.func_rgb = s.func_rgb;
            t.func_alpha = s.func_a;
            t.scale_rgb = s.scale_rgb;
            t.scale_alpha = s.scale_a;
            t.color = pica_cmb::constant(s.k_rgb, s.k_a);
        }
        buf_rgb = pr.buf_rgb_mask;
        buf_a = pr.buf_a_mask;
        pr.uses_tex[0] ? bindUnit(0, true) : pica_texture(0, NULL, 0, 0, GPU_RGBA8, 0);
        pr.uses_tex[1] ? bindUnit(1, true) : pica_texture(1, NULL, 0, 0, GPU_RGBA8, 0);
        /* unit 2: the constant 0.5 the compiler uses as a second per-stage constant */
        if (pr.uses_half)
            pica_texture(2, m_half, 8, 8, GPU_RGBA8,
                         GPU_TEXTURE_MAG_FILTER(GPU_NEAREST) | GPU_TEXTURE_MIN_FILTER(GPU_NEAREST));
        else
            pica_texture(2, NULL, 0, 0, GPU_RGBA8, 0);

        /* Fragment-shader tail of the GLSL backend (legacy blending): coverage as alpha, fog. */
        const u32 cycle = gDP.otherMode.cycleType;
        const bool cvg_alpha = cycle <= G_CYC_2CYCLE && gDP.otherMode.alphaCvgSel && !gDP.otherMode.cvgXAlpha;
        const bool fog = config.generalEmulation.enableLegacyBlending != 0 && fog_usage() == 1;
        if ((cvg_alpha || fog) && n < pica_cmb::MAX_STAGES) {
            PicaTev &t = tev[n++];
            if (fog) {
                t.src_rgb = GPU_TEVSOURCES(GPU_CONSTANT, GPU_PREVIOUS, GPU_PRIMARY_COLOR);
                t.op_rgb = GPU_TEVOPERANDS(GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_ALPHA);
                t.func_rgb = GPU_INTERPOLATE;
            }
            if (cvg_alpha)
                t.src_alpha = GPU_TEVSOURCES(GPU_CONSTANT, 0, 0);
            t.color = pica_const_color(to_u8(gDP.fogColor.r), to_u8(gDP.fogColor.g), to_u8(gDP.fogColor.b), 255);
        }

        /* Alpha compare (UAlphaTestInfo + ShaderAlphaTest); cvgXAlpha discards below 0.125. */
        float ref = -1.0f;
        if (cycle == G_CYC_COPY) {
            if (gDP.otherMode.alphaCompare & G_AC_THRESHOLD)
                ref = 0.5f;
        } else if (cycle != G_CYC_FILL && (gDP.otherMode.alphaCompare & G_AC_THRESHOLD)) {
            ref = gDP.blendColor.a;
            if (cvg_alpha) {
                if (0.125f < ref)
                    return false; /* the test compares the constant 0.125: every fragment fails */
                ref = -1.0f;
            }
        }
        if (cycle <= G_CYC_2CYCLE && gDP.otherMode.cvgXAlpha && ref < 0.125f)
            ref = 0.125f;
        if (ref > 0.0f) {
            alpha_test = true;
            alpha_ref = to_u8(ref);
        }
        break;
    }
    }
    for (int i = 0; i < pica_cmb::MAX_STAGES; i++)
        pica_tev(i, &tev[i]);
    pica_tev_buffer(buf_rgb, buf_a, buf_color);
    pica_alpha_test(alpha_test, GPU_GEQUAL, alpha_ref);
    return true;
}


void ContextImplPica::drawTriangles(const Context::DrawTriangleParameters &p)
{
    DrawTimer timer;
    CRUMB("gfx: drawTriangles");
    const u32 mode = u32(p.mode);
    if (mode == DM_LINES || !p.vertices || !p.verticesCount || m_kind == Kind::Noop || skipDraw())
        return;
    {
        CTR64_PROF(32);
        /* Triangles drawn into the game's own Z buffer (color image = depth image address).
         * Pilotwings 64 draws the glider's shadow there (blended black: the Z values under the
         * shadow move nearer), then the shadow mesh into the frame in the same render mode,
         * with a depth test against that Z buffer which then fails over the ground. Without N64
         * depth compare GLideN64 gives the Z address a separate color buffer, the Z buffer
         * never changes, and the second pass drew a large dark shape over the ground and out to
         * the horizon over the sea. Both passes are dropped (the second: triangles in the same
         * depth-tested, non-writing render mode in that game frame), as the PC build shows the
         * scene with N64 depth compare. */
        if (config.frameBufferEmulation.N64DepthCompare == 0 && gDP.otherMode.depthCompare &&
            !gDP.otherMode.depthUpdate) {
            if (gDP.colorImage.address == gDP.depthImageAddress && gDP.depthImageAddress != 0) {
                m_zPassMode = gDP.otherMode.l;
                m_zPassAge = 0;
                return;
            }
            if (m_zPassMode && gDP.otherMode.l == m_zPassMode)
                return;
        }
        if (!applyState(true)) {
            ctr64_gfx_stats[ST_DROP]++;
            return;
        }
        if (!setupCombiner(false))
            return;
    }
    ctr64_gfx_stats[ST_TRI]++;
    {   /* draws into FCRAM render targets (stats.log "fcram draws"): slower than VRAM */
        const Fbo &f = drawFbo();
        const Tex *c = tex(f.color), *d = tex(f.depth);
        if (c && !c->vram) ctr64_gfx_fcram[0]++;
        if (f.rt.depth && d && !d->vram) ctr64_gfx_fcram[1]++;
    }
    /* Indexed triangle lists (renderAndDrawTriangles: each kept vertex once) are converted once
     * per vertex into m_idxScratch, then expanded into a plain list in the pool (below), so the GPU
     * gets the same non-indexed draws as before. */
    const bool expand = p.elementsCount && p.elements && u32(p.elementsType) == DT_UNSIGNED_SHORT &&
                        mode == DM_TRIANGLES;
    GVtx *v;
    if (expand) {
        pica_reserve(4096, p.elementsCount * sizeof(GVtx) + 256);
        if (m_idxScratch.size() < p.verticesCount) {
            m_idxScratch.resize(p.verticesCount);
            m_idxUsed.resize(p.verticesCount, 0u);
        }
        v = m_idxScratch.data();
        /* only the vertices the elements refer to are converted (the others: culled triangles) */
        if (++m_idxGen == 0) {
            std::fill(m_idxUsed.begin(), m_idxUsed.end(), 0u);
            m_idxGen = 1;
        }
        const u16 *e = static_cast<const u16 *>(p.elements);
        for (u32 i = 0; i < p.elementsCount; i++)
            m_idxUsed[e[i]] = m_idxGen;
    } else {
        pica_reserve(4096, p.verticesCount * sizeof(GVtx) + p.elementsCount * 2 + 256);
        v = static_cast<GVtx *>(pica_frame_alloc(p.verticesCount * sizeof(GVtx), 16));
        if (!v)
            return;
    }

    /* GLideN64 vertex shader math (VertexShaderTriangle + VertexShaderEnd), per vertex */
    const float vs0 = gSP.viewport.vscale[0], vs1 = -gSP.viewport.vscale[1];
    const float vt0 = gSP.viewport.vtrans[0], vt1 = gSP.viewport.vtrans[1];
    const bool native = config.frameBufferEmulation.nativeResFactor == 1 && config.video.multisampling == 0;
    const float offset = (config.generalEmulation.enableInaccurateTextureCoordinates == 0 && native) ? 0.5f : 0.0f;
    const int fogUsage = fog_usage();
    const float fogMul = gSP.fog.multiplierf, fogOff = gSP.fog.offsetf;
    const bool clampZ = m_clamp != ClampMode::ClippingEnabled;
    TexXform tx[2];
    texture_xforms(tx);
    const bool persp = (RSP.LLE || GBI.isTexturePersp()) ? gDP.otherMode.texturePersp != 0 : true;
    const float tscale_s = gSP.texture.scales, tscale_t = gSP.texture.scalet;

    Ctr64ProfScope vtx_scope(33);
    for (u32 i = 0; i < p.verticesCount; i++) {
        if (expand && m_idxUsed[i] != m_idxGen)
            continue;
        const SPVertex &s = i < p.verticesBase ? p.vertices[i] : p.vertices2[i - p.verticesBase];
        GVtx &o = v[i];
        float x = s.x, y = s.y, z = s.z, w = s.w;
        const u32 mod = s.modify;

        if (mod & MODIFY_XY) {
            x *= w;
            y *= w;
        } else {
            x = x * vs0 + vt0 * w;
            y = y * vs1 + vt1 * w;
            x = fast_floor(x * 4.0f) * 0.25f;
            y = fast_floor(y * 4.0f) * 0.25f;
        }
        if (mod & MODIFY_Z)
            z *= w;
        x += offset * w;
        y += offset * w;
        o.x = (x - 0.5f * SCREEN_SIZE_DIM * w) / (0.5f * SCREEN_SIZE_DIM);
        o.y = (y - 0.5f * SCREEN_SIZE_DIM * w) / (0.5f * SCREEN_SIZE_DIM);
        float zp = -0.5f * (z + w);
        if (clampZ) {
            if (zp > 0.0f) zp = 0.0f;
            if (zp < -w) zp = -w;
        }
        o.z = zp;
        o.w = w;

        const float *col = p.flatColors ? &s.flat_r : &s.r;
        o.c[0] = to_u8(col[0]); o.c[1] = to_u8(col[1]); o.c[2] = to_u8(col[2]); o.c[3] = to_u8(col[3]);
        if (fogUsage) {
            float fp = (s.z < -s.w && !(mod & MODIFY_Z)) ? (-fogMul + fogOff) : (s.z / s.w * fogMul + fogOff);
            u8 f8 = to_u8(fp < 0.0f ? 0.0f : fp > 1.0f ? 1.0f : fp);
            if (fogUsage == 1)
                o.c[3] = f8;
            else
                o.c[0] = o.c[1] = o.c[2] = f8;
        }
        float ts = s.s * tscale_s, tt = s.t * tscale_t;
        if (!persp && !(mod & MODIFY_ST)) {
            ts *= 0.5f;
            tt *= 0.5f;
        }
        o.s0 = tx[0].s(ts); o.t0 = tx[0].t(tt);
        o.s1 = tx[1].s(ts); o.t1 = tx[1].t(tt);
    }

    vtx_scope.end();
    u32 count = p.verticesCount;
    const void *elements = p.elements;
    u32 elementsCount = p.elementsCount;
    if (expand) { /* room reserved above (no pica_reserve between: it may restart the pool) */
        const u16 *e = static_cast<const u16 *>(elements);
        GVtx *flat = static_cast<GVtx *>(pica_frame_alloc(elementsCount * sizeof(GVtx), 16));
        if (!flat)
            return;
        for (u32 i = 0; i < elementsCount; i++)
            flat[i] = v[e[i]];
        v = flat;
        count = elementsCount;
        elements = nullptr;
        elementsCount = 0;
    }
    if (!(elementsCount && elements) && mode == DM_TRIANGLES && m_kind == Kind::Combiner && m_program &&
        m_program->usesShade()) {
        CTR64_PROF(34);
        v = linearizeShade(v, count, fogUsage == 1 && config.generalEmulation.enableLegacyBlending != 0);
    }
    CTR64_PROF(35);

    if (draw_log())
        log_draw("tri", m_program, (int)m_kind, v, count, m_vp, stateString());
    noteCoverage(v, count);
    pica_vertex_buffer(v);

    GPU_Primitive_t prim_type = mode == DM_TRIANGLE_STRIP ? GPU_TRIANGLE_STRIP : mode == DM_TRIANGLE_FAN ? GPU_TRIANGLE_FAN : GPU_TRIANGLES;
    u16 *idx = nullptr;
    if (elementsCount && elements) {
        idx = static_cast<u16 *>(pica_frame_alloc(elementsCount * 2, 16));
        if (!idx)
            return;
        if (u32(p.elementsType) == DT_UNSIGNED_BYTE) {
            const u8 *e = static_cast<const u8 *>(elements);
            for (u32 i = 0; i < elementsCount; i++)
                idx[i] = e[i];
        } else {
            memcpy(idx, elements, elementsCount * 2);
        }
    }
    if (m_enPolyOffset && m_enDepth)
        pica_depth_map(-1.0f, poly_offset_for_draw(v, idx, idx ? elementsCount : count, prim_type));
    if (idx)
        pica_draw_elements(prim_type, idx, elementsCount);
    else
        pica_draw_arrays(prim_type, 0, count);
}

/* ---- screen-linear shade -------------------------------------------------------------------
 * The N64 RDP interpolates shade linearly in screen space, and so does GLideN64 with clipping
 * enabled (vShadeColorNoperspective). The PICA interpolates vertex colours perspective-
 * correctly, which differs visibly on large triangles spanning a wide depth range (floors near
 * the camera). Such triangles are subdivided at screen-space edge midpoints until the w ratio in
 * each piece is small enough: position, depth and texture coordinates are interpolated in clip
 * space (t = wa / (wa + wb) projects to the screen midpoint), shade by the plain average, and the
 * fog factor, which GL interpolates perspective-correctly, in clip space.
 * Along an edge with w ratio r the two weights differ by at most (r - 1) / (2 (r + 1)), so the
 * shade error is bounded by that times the largest channel difference between the vertices. */
const float k_shade_max_error = 1.5f; /* colour levels (of 255) */
const int k_shade_max_depth = 4;

inline GVtx screen_midpoint(const GVtx &a, const GVtx &b, bool fog_alpha)
{
    const float t = a.w / (a.w + b.w);
    GVtx m;
    m.x = a.x + (b.x - a.x) * t;
    m.y = a.y + (b.y - a.y) * t;
    m.z = a.z + (b.z - a.z) * t;
    m.w = a.w + (b.w - a.w) * t;
    m.s0 = a.s0 + (b.s0 - a.s0) * t;
    m.t0 = a.t0 + (b.t0 - a.t0) * t;
    m.s1 = a.s1 + (b.s1 - a.s1) * t;
    m.t1 = a.t1 + (b.t1 - a.t1) * t;
    for (int k = 0; k < 3; k++)
        m.c[k] = (u8)((a.c[k] + b.c[k] + 1) >> 1);
    m.c[3] = fog_alpha ? (u8)(a.c[3] + (b.c[3] - a.c[3]) * t + 0.5f) : (u8)((a.c[3] + b.c[3] + 1) >> 1);
    return m;
}

/* Whether perspective-correct shade can be off by more than k_shade_max_error over the triangle
 * (alpha ignored when it carries the fog factor, which GL interpolates perspective-correctly too). */
inline bool shade_needs_split(const GVtx &a, const GVtx &b, const GVtx &c, bool fog_alpha)
{
    const float lo = std::min(a.w, std::min(b.w, c.w)), hi = std::max(a.w, std::max(b.w, c.w));
    if (lo <= 0.0f)
        return false;
    int dc = 0;
    for (int k = 0, n = fog_alpha ? 3 : 4; k < n; k++) {
        const int mn = std::min(a.c[k], std::min(b.c[k], c.c[k])), mx = std::max(a.c[k], std::max(b.c[k], c.c[k]));
        dc = std::max(dc, mx - mn);
    }
    return dc * (hi - lo) > 2.0f * k_shade_max_error * (hi + lo);
}

void subdivide_for_shade(const GVtx &a, const GVtx &b, const GVtx &c, int depth, bool fog_alpha,
                         std::vector<GVtx> &out)
{
    if (depth == 0 || !shade_needs_split(a, b, c, fog_alpha)) {
        out.push_back(a);
        out.push_back(b);
        out.push_back(c);
        return;
    }
    const GVtx ab = screen_midpoint(a, b, fog_alpha), bc = screen_midpoint(b, c, fog_alpha),
               ca = screen_midpoint(c, a, fog_alpha);
    subdivide_for_shade(a, ab, ca, depth - 1, fog_alpha, out);
    subdivide_for_shade(ab, b, bc, depth - 1, fog_alpha, out);
    subdivide_for_shade(ca, bc, c, depth - 1, fog_alpha, out);
    subdivide_for_shade(ab, bc, ca, depth - 1, fog_alpha, out);
}

/* Replaces v[0..n) (a triangle list) by its subdivision when any triangle needs it; returns the
 * vertex buffer to draw and its count. */
GVtx *ContextImplPica::linearizeShade(GVtx *v, u32 &n, bool fog_alpha)
{
    bool any = false;
    for (u32 i = 0; i + 2 < n && !any; i += 3)
        any = shade_needs_split(v[i], v[i + 1], v[i + 2], fog_alpha);
    if (!any)
        return v;
    std::vector<GVtx> &out = m_shadeScratch;
    out.clear();
    for (u32 i = 0; i + 2 < n; i += 3) {
        subdivide_for_shade(v[i], v[i + 1], v[i + 2], k_shade_max_depth, fog_alpha, out);
    }
    /* far more than any real draw: drawn as it is (a restarted pool could not hold it, and v
     * would then point into reused memory) */
    if (out.size() * sizeof(GVtx) > 1024 * 1024)
        return v;
    pica_reserve(512, out.size() * sizeof(GVtx) + 256); /* may restart the pool: v is not used below */
    GVtx *nv = static_cast<GVtx *>(pica_frame_alloc(out.size() * sizeof(GVtx), 16));
    if (!nv)
        return v;
    memcpy(nv, out.data(), out.size() * sizeof(GVtx));
    n = (u32)out.size();
    ctr64_gfx_stats[ST_SUBDIV] += n / 3;
    return nv;
}

/* GL polygon offset for a decal draw: factor * max over its triangles of max(|dz/dx|, |dz/dy|)
 * in window space (depth 0..1 per pixel), plus the constant floor. */
float ContextImplPica::poly_offset_for_draw(const GVtx *v, const u16 *idx, u32 count, GPU_Primitive_t prim)
{
    const float hw = 0.5f * m_vp[2], hh = 0.5f * m_vp[3];
    auto win = [&](u32 i, float &x, float &y, float &z) {
        const GVtx &q = v[idx ? idx[i] : i];
        const float iw = q.w != 0.0f ? 1.0f / q.w : 0.0f;
        x = q.x * iw * hw;
        y = q.y * iw * hh;
        z = -q.z * iw; /* window depth with the depth map at scale -1 */
    };
    float slope = 0.0f;
    const u32 ntri = prim == GPU_TRIANGLES ? count / 3 : count >= 3 ? count - 2 : 0;
    for (u32 t = 0; t < ntri; t++) {
        u32 i0, i1, i2;
        if (prim == GPU_TRIANGLES) { i0 = t * 3; i1 = i0 + 1; i2 = i0 + 2; }
        else if (prim == GPU_TRIANGLE_FAN) { i0 = 0; i1 = t + 1; i2 = t + 2; }
        else { i0 = t; i1 = t + 1; i2 = t + 2; }
        float x0, y0, z0, x1, y1, z1, x2, y2, z2;
        win(i0, x0, y0, z0);
        win(i1, x1, y1, z1);
        win(i2, x2, y2, z2);
        const float ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
        const float bx = x2 - x0, by = y2 - y0, bz = z2 - z0;
        const float det = ax * by - ay * bx;
        if (fabsf(det) < 1e-6f)
            continue;
        const float dzdx = fabsf((az * by - ay * bz) / det);
        const float dzdy = fabsf((ax * bz - az * bx) / det);
        const float m = dzdx > dzdy ? dzdx : dzdy;
        if (m > slope)
            slope = m;
    }
    if (slope > 0.01f) /* edge-on triangles: keep the offset bounded */
        slope = 0.01f;
    return k_poly_factor * slope + k_poly_offset;
}

void ContextImplPica::drawRects(const Context::DrawRectParameters &p)
{
    DrawTimer timer;
    CRUMB("gfx: drawRects");
    /* a fill or rectangle over the whole N64 frame buffer redraws it (frameskip.c) */
    if (p.vertices && p.verticesCount && m_kind == Kind::Combiner && m_drawFbo != 0 && !ctr64_frame_cleared) {
        if (Tex *c = tex(drawFbo().color)) {
            float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
            for (u32 i = 0; i < p.verticesCount; i++) {
                const RectVertex &s = p.vertices[i];
                x0 = s.x < x0 ? s.x : x0; x1 = s.x > x1 ? s.x : x1;
                y0 = s.y < y0 ? s.y : y0; y1 = s.y > y1 ? s.y : y1;
            }
            if (x0 <= 0.0f && y0 <= 0.0f && x1 >= c->w - 1 && y1 >= c->h - 8)
                ctr64_frame_cleared = true;
        }
    }
    if (!p.vertices || !p.verticesCount || m_kind == Kind::Noop || skipDraw())
        return;
    const bool special = m_kind != Kind::Combiner;
    if (!applyState(!special)) {
        ctr64_gfx_stats[ST_DROP]++;
        return;
    }
    if (m_kind == Kind::Copy && copyByTransfer(p)) {
        ctr64_gfx_stats[ST_RECT]++;
        return;
    }
    if (m_rtLog && p.verticesCount >= 4) { /* rtlog.txt: draws sampling a frame buffer that needs a shadow */
        const Tex *s = tex(m_unit[0]), *d = tex(drawFbo().color);
        if (s && s->rt && needs_shadow(*s))
            fprintf(m_rtLog, "  S kind %d src %u %ux%u/%ux%u dst %u %ux%u/%ux%u blend %d depth %d sc %d %d,%d,%d,%d vp %d,%d,%d,%d"
                             " v %.3f,%.3f %.3f,%.3f w %.3f st %.4f,%.4f %.4f,%.4f\n",
                    (int)m_kind, (unsigned)m_unit[0], s->w, s->h, s->tw, s->th, (unsigned)drawFbo().color, d ? d->w : 0,
                    d ? d->h : 0, d ? d->tw : 0, d ? d->th : 0, (int)m_enBlend, (int)m_enDepth, (int)m_enScissor,
                    (int)m_sc[0], (int)m_sc[1], (int)m_sc[2], (int)m_sc[3], m_vp[0], m_vp[1], m_vp[2], m_vp[3],
                    p.vertices[0].x, p.vertices[0].y, p.vertices[3].x, p.vertices[3].y, p.vertices[0].w,
                    p.vertices[0].s0, p.vertices[0].t0, p.vertices[3].s0, p.vertices[3].t0);
    }
    if (!setupCombiner(true))
        return;
    ctr64_gfx_stats[ST_RECT]++;
    pica_reserve(4096, p.verticesCount * sizeof(GVtx) + 256);

    GVtx *v = static_cast<GVtx *>(pica_frame_alloc(p.verticesCount * sizeof(GVtx), 16));
    if (!v)
        return;

    /* Texture coordinates are normalized to the logical texture size; scale them into storage. */
    float ss0 = 1.0f, st0 = 1.0f, ss1 = 1.0f, st1 = 1.0f;
    if (Tex *t = tex(m_unit[0]))
        if (t->tw) { ss0 = (float)t->w / sample_w(*t); st0 = (float)t->h / t->th; }
    if (Tex *t = tex(m_unit[1]))
        if (t->tw) { ss1 = (float)t->w / sample_w(*t); st1 = (float)t->h / t->th; }
    /* N64 texture rectangles: coordinates come normalised by GLideN64's cache scale, which is
     * 1/pow2(width) with a wrap mask (MK64's title: 320-texel strips, mask 512), so texel u is at
     * u * scale; normalise by the storage size as for triangles (texture_xforms) */
    if (!special) {
        TextureCache &cache = textureCache();
        for (u32 u = 0; u < 2; u++) {
            const CachedTexture *ct = cache.current[u];
            Tex *t = tex(m_unit[u]);
            if (!ct || !t || !t->tw || ct->scaleS <= 0.0f || ct->scaleT <= 0.0f || t->w != ct->width)
                continue;
            if (ct->frameBufferTexture != CachedTexture::fbNone) { /* see texture_xforms */
                (u ? ss1 : ss0) = float(ct->width) / sample_w(*t);
                (u ? st1 : st0) = float(ct->height) / t->th;
                continue;
            }
            (u ? ss1 : ss0) = 1.0f / (ct->scaleS * sample_w(*t));
            (u ? st1 : st0) = 1.0f / (ct->scaleT * t->th);
        }
    }

    const bool native = config.frameBufferEmulation.nativeResFactor == 1 && config.video.multisampling == 0;
    const float offset = (config.generalEmulation.enableInaccurateTextureCoordinates == 0 && native) ? 0.5f : 0.0f;
    const u8 rc[4] = {to_u8(gDP.rectColor.r), to_u8(gDP.rectColor.g), to_u8(gDP.rectColor.b), to_u8(gDP.rectColor.a)};

    for (u32 i = 0; i < p.verticesCount; i++) {
        const RectVertex &s = p.vertices[i];
        GVtx &o = v[i];
        float x = s.x, y = s.y, w = s.w;
        if (!special) {
            /* N64 pixel space through the 640x640 virtual viewport (VertexShaderEnd) */
            x = (x + offset * w - 0.5f * SCREEN_SIZE_DIM * w) / (0.5f * SCREEN_SIZE_DIM);
            y = (y + offset * w - 0.5f * SCREEN_SIZE_DIM * w) / (0.5f * SCREEN_SIZE_DIM);
        }
        o.x = x;
        o.y = y;
        o.z = -0.5f * (s.z + w);
        o.w = w;
        memcpy(o.c, special ? (const u8[4]){255, 255, 255, 255} : rc, 4);
        /* + 0.0001: ShaderFragmentCorrectTexCoords */
        o.s0 = (s.s0 + 0.0001f) * ss0; o.t0 = (s.t0 + 0.0001f) * st0;
        o.s1 = (s.s1 + 0.0001f) * ss1; o.t1 = (s.t1 + 0.0001f) * st1;
    }

    if (draw_log())
        log_draw("rect", m_program, (int)m_kind, v, p.verticesCount, m_vp, stateString());
    if (!special)
        noteCoverage(v, p.verticesCount);
    pica_vertex_buffer(v);
    const u32 mode = u32(p.mode);
    pica_draw_arrays(mode == DM_TRIANGLES ? GPU_TRIANGLES : mode == DM_TRIANGLE_FAN ? GPU_TRIANGLE_FAN : GPU_TRIANGLE_STRIP,
                     0, p.verticesCount);
}

/* A copy draw of a frame buffer at exactly half size into a tile-aligned rectangle - the window
 * copy and GLideN64's native-resolution copies with anti-aliasing - done by the transfer engine
 * (2x2 average = bilinear sampling at exactly half size, up to rounding) instead of a draw that
 * samples it: on hardware a texture stored at a non-power-of-two width is only sampled through a
 * shadow copy of the whole of it (Azahar 2026-11-10: 40-150 MB of shadow copies per 10 s, MK
 * 684 MB). Queued in order with the draws. Only for copies the sampling path would shadow;
 * anything else (blending, depth, scissor cutting into it, flips, unaligned) is drawn as before.
 * Dev switch noxfercopy.txt: off. */
bool ContextImplPica::copyByTransfer(const Context::DrawRectParameters &p)
{
    if (!m_xferCopy || p.verticesCount != 4 || m_enBlend || m_enDepth)
        return false;
    Tex *s = tex(m_unit[0]);
    Tex *d = tex(drawFbo().color);
    if (!s || !d || s == d || !s->data || !d->data || s->depth || d->depth || s->fmt != GPU_RGBA8 ||
        d->fmt != GPU_RGBA8 || !needs_shadow(*s))
        return false;
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, s0 = 1e9f, s1 = -1e9f, t0 = 1e9f, t1 = -1e9f;
    for (u32 i = 0; i < 4; i++) {
        const RectVertex &v = p.vertices[i];
        if (v.w < 0.9999f || v.w > 1.0001f)
            return false;
        x0 = std::min(x0, v.x); x1 = std::max(x1, v.x);
        y0 = std::min(y0, v.y); y1 = std::max(y1, v.y);
        s0 = std::min(s0, v.s0); s1 = std::max(s1, v.s0);
        t0 = std::min(t0, v.t0); t1 = std::max(t1, v.t0);
    }
    /* an axis-aligned rectangle, texture coordinates running the same way as the positions */
    for (u32 i = 0; i < 4; i++) {
        const RectVertex &v = p.vertices[i];
        const bool lx = v.x < 0.5f * (x0 + x1), ly = v.y < 0.5f * (y0 + y1);
        if ((v.x != x0 && v.x != x1) || (v.y != y0 && v.y != y1) || lx != (v.s0 < 0.5f * (s0 + s1)) ||
            ly != (v.t0 < 0.5f * (t0 + t1)))
            return false;
    }
    auto whole = [](float f, int &out) {
        out = (int)lroundf(f);
        return fabsf(f - (float)out) < 0.01f;
    };
    int dx0, dx1, dy0, dy1, sx0, sx1, sy0, sy1;
    if (!whole(m_vp[0] + (x0 + 1.0f) * 0.5f * m_vp[2], dx0) || !whole(m_vp[0] + (x1 + 1.0f) * 0.5f * m_vp[2], dx1) ||
        !whole(m_vp[1] + (y0 + 1.0f) * 0.5f * m_vp[3], dy0) || !whole(m_vp[1] + (y1 + 1.0f) * 0.5f * m_vp[3], dy1) ||
        !whole(s0 * s->w, sx0) || !whole(s1 * s->w, sx1) || !whole(t0 * s->h, sy0) || !whole(t1 * s->h, sy1))
        return false;
    const int w = dx1 - dx0, h = dy1 - dy0;
    /* whole storage rows of the source (the engine reads full rows), tile-aligned everywhere */
    if (w <= 0 || h <= 0 || sx0 != 0 || sx1 != s->tw || sx1 - sx0 != 2 * w || sy1 - sy0 != 2 * h || h % 8 || w % 8)
        return false;
    const int r0 = (int)s->th - sy1, q0 = (int)d->th - dy1;
    if (r0 < 0 || sy0 < 0 || r0 % 8 || dx0 < 0 || dy0 < 0 || dx1 > d->tw || q0 < 0 || q0 % 8 || dx0 % 8)
        return false;
    if (m_enScissor && (m_sc[0] > dx0 || m_sc[1] > dy0 || m_sc[0] + m_sc[2] < dx1 || m_sc[1] + m_sc[3] < dy1))
        return false;
    const u32 bytes = (u32)w * h * 4;
    const bool direct = dx0 == 0 && w == d->tw;
    for (int k = 0; k < (direct ? 1 : 2); k++)
        if (m_xferTempBytes[k] < bytes) {
            if (m_xferTemp[k]) { /* rare (a larger copy): the old one may be in a queued transfer */
                pica_flush();
                pica_wait_idle();
                freeNow(m_xferTemp[k], false, m_xferTempBytes[k]);
            }
            m_xferTemp[k] = linear_alloc_safe(bytes, 0x80);
            if (!m_xferTemp[k])
                m_xferTemp[k] = ctr64_xlinear_alloc(bytes, 0x80);
            m_xferTempBytes[k] = m_xferTemp[k] ? bytes : 0;
            if (!m_xferTemp[k])
                return false;
            ctr64_gfx_mem[0] += bytes;
        }
    const u8 *src = static_cast<const u8 *>(s->data) + (size_t)r0 * s->tw * 4;
    u8 *dst = static_cast<u8 *>(d->data) + (size_t)q0 * d->tw * 4 + (size_t)(dx0 / 8) * 64 * 4;
    pica_downscale2_rgba8(src, s->tw, (u16)(2 * h), m_xferTemp[0]);
    if (direct) {
        pica_tile_rgba8(m_xferTemp[0], (u16)w, (u16)h, dst);
    } else {
        pica_tile_rgba8(m_xferTemp[0], (u16)w, (u16)h, m_xferTemp[1]);
        pica_copy_gap(dst, m_xferTemp[1], bytes, (u32)w * 8 * 4 / 16, (u32)(d->tw - w) * 8 * 4 / 16);
    }
    s->used_frame = pica_frame_number();
    ctr64_gfx_shadow[2]++;
    return true;
}

/* ---- presentation and readback -------------------------------------------------------------- */

/* Development aid: sdmc:/3ds/ctr64/dumpframes.txt lists swap numbers; at each, the window
 * buffer is written to sdmc:/3ds/ctr64/frame_<n>.tga (bottom-up rows, like the texture). */
static bool dump_wanted(u32 swap)
{
    static std::vector<u32> wanted;
    static bool loaded;
    if (!loaded) {
        loaded = true;
        if (FILE *f = fopen("sdmc:/3ds/ctr64/dumpframes.txt", "r")) {
            unsigned n;
            while (fscanf(f, "%u", &n) == 1)
                wanted.push_back(n);
            fclose(f);
        }
    }
    for (u32 n : wanted)
        if (n == swap)
            return true;
    return false;
}

static void write_tga(const Tex &t0, const char *path)
{
    if (!t0.data || t0.depth)
        return;
    /* a VRAM target goes through a GPU copy into linear memory first: Azahar does not keep its
     * emulated VRAM up to date for CPU reads (frames dumped from VRAM came out all zero) */
    Tex t = t0;
    void *lin = nullptr;
    if (t0.vram) {
        const u32 bytes = (u32)t0.tw * t0.th * 4;
        lin = linearAlloc(bytes);
        if (!lin)
            return;
        pica_copy_from_vram(lin, t0.data, bytes);
        pica_flush();
        pica_wait_idle();
        GSPGPU_InvalidateDataCache(lin, bytes);
        t.data = lin;
        t.vram = false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        if (lin)
            linearFree(lin);
        return;
    }
    const u8 hdr[18] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (u8)t.w, (u8)(t.w >> 8), (u8)t.h, (u8)(t.h >> 8), 32, 8};
    fwrite(hdr, 1, sizeof(hdr), f);
    /* TGA rows are bottom-up like GL rows */
    std::vector<u8> row(t.w * 4);
    for (u32 y = 0; y < t.h; y++) {
        for (u32 x = 0; x < t.w; x++) {
            u32 v = load_texel(t, gl_index(x, y, t.tw, t.th));
            u8 *d = &row[x * 4];
            d[0] = v >> 8; d[1] = v >> 16; d[2] = v >> 24; d[3] = v; /* BGRA */
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    if (lin)
        linearFree(lin);
}

void ContextImplPica::present()
{
    struct PresentTimer {
        unsigned long long t0 = svcGetSystemTick();
        ~PresentTimer() { ctr64_gfx_ticks[2] += svcGetSystemTick() - t0; }
    } timer;
    CRUMB("gfx: present");
    m_cover = 0;
    const bool stale = m_showStale, same = m_showSame && !stale;
    const u32 src = m_winSrc, copies = m_winCopies;
    m_showStale = m_showSame = false;
    m_winSrc = m_winCopies = 0;
    if (m_winClear && !stale && !same)
        clearTarget(m_window, m_winClearColor);
    m_winClear = false;
    if (!ctr64_frameskip_present(stale) || same)
        return; /* the window holds a buffer with dropped draws or the image on screen: it stays */
    m_shown = copies == 1 ? src : 0;
    if (Tex *t = tex(src))
        t->drawn = false;
    if (m_rtLog)
        fprintf(m_rtLog, "P %u:", (unsigned)pica_frame_number());
    for (auto &kv : m_tex) {
        Tex &t = kv.second;
        if (m_rtLog && t.rt && t.data && (t.ndraw || t.draw_frame + 1 >= pica_frame_number()))
            fprintf(m_rtLog, " %u%s %ux%u/%ux%u %s n%u s%u", (unsigned)kv.first, t.depth ? "d" : "c", t.w, t.h, t.tw, t.th,
                    t.vram ? "V" : "F", (unsigned)t.ndraw, (unsigned)t.nshadow);
        t.nshadow = 0;
        if (t.ndraw)
            t.ndraw_last = t.ndraw;
        t.ndraw = 0;
    }
    if (m_rtLog) {
        fprintf(m_rtLog, " | moves %u/%u KB, shadow copies %u (%u KB), transfer copies %u\n", ctr64_gfx_vram_moves[0],
                ctr64_gfx_vram_moves[1], ctr64_gfx_shadow[0], ctr64_gfx_shadow[1], ctr64_gfx_shadow[2]);
        fflush(m_rtLog);
    }
    ctr64_gfx_stats[ST_SWAP]++;
    ensureFrame();
    Tex *c = tex(TEX_WINDOW_COLOR);
    pica_texture_flush_cache();
    video_draw_fullscreen(c->data, c->tw, c->th, GPU_RGBA8, c->w, c->h, false, true);
    video_present();
}

/* End of an N64 VI (after GLideN64's VI_UpdateScreen, which presents only when there is a new
 * image). Frames are counted here, per VI, so frame dumps and input scripts line up with the
 * Windows reference build whether or not this VI presented. */
void ContextImplPica::endVI()
{
    const u32 vi = ++ctr64_video_frames;
    g_swap_count = vi + 1;
    if (m_zPassMode && ++m_zPassAge > 4) /* the second pass comes in the same game frame */
        m_zPassMode = 0;
    Tex *c = tex(TEX_WINDOW_COLOR);
    if (dump_wanted(vi)) {
        ensureFrame();
        pica_flush();
        pica_wait_idle();
        char path[64];
        snprintf(path, sizeof(path), "sdmc:/3ds/ctr64/frame_%u.tga", (unsigned)vi);
        write_tga(*c, path);
        static int with_fbos = -1; /* sdmc:/3ds/ctr64/dumpfbo.txt: also dump every FBO */
        if (with_fbos < 0) {
            FILE *f = fopen("sdmc:/3ds/ctr64/dumpfbo.txt", "r");
            with_fbos = f != nullptr;
            if (f)
                fclose(f);
        }
        for (auto &kv : m_fbo)
            if (Tex *t = with_fbos ? tex(kv.second.color) : nullptr) {
                snprintf(path, sizeof(path), "sdmc:/3ds/ctr64/frame_%u_fb%u_%ux%u.tga", (unsigned)vi,
                         (unsigned)kv.first, (unsigned)t->w, (unsigned)t->h);
                write_tga(*t, path);
            }
    }

    /* Detailed profiler: a screenshot of the top screen every SHOT_EVERY VIs, next to the
     * profile (prof.c writes it). The GPU untiles the window into linear memory at one VI and
     * the copy is handed over at the next, so no VI waits for the GPU. */
    if (ctr64_profiler >= 2) {
        enum { SHOT_EVERY = 180 };
        static u8 *lin;
        static u64 fence;
        static u32 lin_bytes;
        if (lin) {
            pica_wait_fence(fence); /* queued a VI ago: done by now */
            pica_cache_invalidate(lin, lin_bytes);
            /* tiled row 0 is the top (gl_index): the image's top row is tiled row th - h */
            prof_shot(vi - 1, lin + (size_t)(c->th - c->h) * c->tw * 4, c->tw);
            linearFree(lin);
            lin = nullptr;
        }
        if (vi % SHOT_EVERY == 0 && c->data && c->w == 400 && c->h == 240 && prof_shot_ready()) {
            lin_bytes = (u32)c->tw * c->th * 4;
            lin = (u8 *)linearAlloc(lin_bytes);
            if (lin) {
                ensureFrame();
                pica_untile_rgba8_async(c->data, c->tw, c->th, lin);
                fence = pica_fence();
            }
        }
    }

    /* Development aid: sdmc:/3ds/ctr64/freezeframe.txt holds a VI number at which the
     * emulation stops and that frame stays on screen (for emulator screenshots). */
    static int freeze_at = -2;
    if (freeze_at == -2) {
        freeze_at = -1;
        if (FILE *f = fopen("sdmc:/3ds/ctr64/freezeframe.txt", "r")) {
            if (fscanf(f, "%d", &freeze_at) != 1)
                freeze_at = -1;
            fclose(f);
        }
    }
    if ((int)vi == freeze_at)
        while (aptMainLoop()) {
            ensureFrame();
            video_draw_fullscreen(c->data, c->tw, c->th, GPU_RGBA8, c->w, c->h, false, true);
            video_present();
        }
}


template <bool PmBlue>
static inline u16 to_rgba5551(u32 v) /* 0xRRGGBBAA */
{
    if (PmBlue) {
        const u32 b = (v >> 8) & 0xFF;
        if (b > 0x00 && b <= 0xFB)
            v += 0x400;
    }
    return (u16)(((v >> 16) & 0xF800) | ((v >> 13) & 0x07C0) | ((v >> 10) & 0x3E) | ((v & 0xFF) != 0));
}

/* RGBA8 render targets: the GPU untiles the 8-row strips holding GL rows y0..y0+h-1 (storage
 * rows th-y0-h..th-1-y0) into linear memory; false if this read can't take that path */
static bool plan_read(const Tex *t, s32 x0, s32 y0, u32 w, u32 h, ReadJob &j)
{
    if (!t || !t->data || t->depth || t->fmt != GPU_RGBA8 || x0 < 0 || y0 < 0 || (u32)x0 >= t->tw ||
        (u32)y0 >= t->th)
        return false;
    j.x0 = x0; j.y0 = y0; j.w = w; j.h = h;
    j.tw = t->tw; j.th = t->th;
    j.hh = std::min<u32>(h, t->th - y0);
    j.ww = std::min<u32>(w, t->tw - x0);
    j.s0 = (t->th - y0 - j.hh) & ~7u;
    j.s1 = (t->th - y0 + 7) & ~7u;
    j.rgba16 = g_ctr64ReadRgba16;
    j.pm_blue = g_ctr64ReadPmBlue;
    return true;
}

static size_t read_bytes(const ReadJob &j)
{
    return (size_t)j.tw * (j.s1 - j.s0) * 4;
}

static void queue_untile(const Tex *t, const ReadJob &j, u32 *lin)
{
    pica_untile_rgba8_async(static_cast<const u8 *>(t->data) + (size_t)j.s0 * j.tw * 4, j.tw, j.s1 - j.s0, lin);
}

/* The untiled strips in lin to GLideN64's layout (w*h, RGBA5551 or R,G,B,A bytes, rows from y0) */
static const u8 *convert_read(const u32 *lin, const ReadJob &j, std::vector<u8> &out)
{
    CTR64_PROF(21);
    const u32 w = j.w;
    if (j.rgba16) {
        out.resize((size_t)w * j.h * 2);
        for (u32 y = 0; y < j.hh; y++) {
            const u32 *src = lin + (size_t)(j.th - 1 - j.y0 - y - j.s0) * j.tw + j.x0;
            u16 *dst = reinterpret_cast<u16 *>(&out[(size_t)y * w * 2]);
            if (j.pm_blue)
                for (u32 x = 0; x < j.ww; x++)
                    dst[x] = to_rgba5551<true>(src[x]);
            else {
                u32 x = 0;
                if ((reinterpret_cast<uintptr_t>(dst) & 3) == 0) /* two pixels per word store */
                    for (; x + 2 <= j.ww; x += 2)
                        *reinterpret_cast<u32 *>(dst + x) =
                            to_rgba5551<false>(src[x]) | ((u32)to_rgba5551<false>(src[x + 1]) << 16);
                for (; x < j.ww; x++)
                    dst[x] = to_rgba5551<false>(src[x]);
            }
            std::fill(dst + j.ww, dst + w, (u16)0);
        }
        std::fill(out.begin() + (size_t)j.hh * w * 2, out.end(), 0);
        return out.data();
    }
    out.resize((size_t)w * j.h * 4);
    for (u32 y = 0; y < j.hh; y++) {
        const u32 *src = lin + (size_t)(j.th - 1 - j.y0 - y - j.s0) * j.tw + j.x0;
        u32 *dst = reinterpret_cast<u32 *>(&out[(size_t)y * w * 4]);
        for (u32 x = 0; x < j.ww; x++)
            dst[x] = __builtin_bswap32(src[x]); /* 0xRRGGBBAA -> bytes R, G, B, A */
        std::fill(dst + j.ww, dst + w, 0u);
    }
    std::fill(out.begin() + (size_t)j.hh * w * 4, out.end(), 0);
    return out.data();
}

bool AsyncReadback::reserve(size_t need)
{
    if (need <= bytes)
        return true;
    finish();
    if (lin)
        linearFree(lin);
    lin = static_cast<u32 *>(linear_alloc_safe((u32)need, 128));
    bytes = lin ? need : 0;
    return lin != nullptr;
}

void AsyncReadback::finish()
{
    if (pending) {
        pica_wait_fence(fence);
        pica_cache_invalidate(lin, (u32)read_bytes(job));
        pending = false;
    }
}

AsyncReadback::~AsyncReadback()
{
    finish(); /* the GPU must not write into freed memory */
    if (lin)
        linearFree(lin);
}

/* copyToRDRAM "double buffer": return the image this frame buffer held at its previous copy
 * (as GLideN64's PBO readers do) and queue this one, so the copy never waits for the GPU to
 * finish drawing. nullptr when there is no matching earlier read (GLideN64 then skips the copy). */
const u8 *ContextImplPica::readTextureAsync(CachedTexture *ct, s32 x0, s32 y0, u32 w, u32 h, AsyncReadback &a,
                                            std::vector<u8> &out)
{
    CTR64_PROF(19);
    Tex *t = ct ? tex(u32(ct->name)) : nullptr;
    ReadJob j;
    if (!plan_read(t, x0, y0, w, h, j)) {
        a.finish();
        a.valid = false;
        return readTexture(ct, x0, y0, w, h, out);
    }
    ctr64_fb_copies++;
    const u8 *result = nullptr;
    if (a.pending || a.valid) {
        a.finish();
        const ReadJob &p = a.job;
        if (a.valid && p.x0 == j.x0 && p.y0 == j.y0 && p.w == j.w && p.h == j.h && p.rgba16 == j.rgba16)
            result = convert_read(a.lin, p, out);
    }
    if (!a.reserve(read_bytes(j))) {
        a.valid = false;
        return result;
    }
    {
        CTR64_PROF(20);
        queue_untile(t, j, a.lin);
    }
    a.fence = pica_fence();
    a.job = j;
    a.pending = true;
    a.valid = true;
    return result;
}

const u8 *ContextImplPica::readTexture(CachedTexture *ct, s32 x0, s32 y0, u32 w, u32 h, std::vector<u8> &out)
{
    CTR64_PROF(19);
    ctr64_fb_copies++;
    Tex *t = ct ? tex(u32(ct->name)) : nullptr;
    const bool rgba16 = g_ctr64ReadRgba16;
    const u32 bpp = rgba16 ? 2 : 4;
    out.resize((size_t)w * h * bpp); /* zeroed below only where nothing is read */
    if (!t || !t->data || t->depth) {
        std::fill(out.begin(), out.end(), 0);
        return out.data();
    }
    ReadJob j;
    if (plan_read(t, x0, y0, w, h, j)) {
        /* one linear buffer for all synchronous reads (never destroyed: no GPU wait at exit) */
        static AsyncReadback &s_sync = *new AsyncReadback;
        if (s_sync.reserve(read_bytes(j))) {
            {
                CTR64_PROF(20);
                queue_untile(t, j, s_sync.lin);
                s_sync.fence = pica_fence();
                s_sync.job = j;
                s_sync.pending = true;
                s_sync.finish();
            }
            return convert_read(s_sync.lin, j, out);
        }
    }
    pica_flush();
    pica_wait_idle();
    std::fill(out.begin(), out.end(), 0);
    for (u32 y = 0; y < h; y++) {
        u32 ty = y0 + y;
        if (ty >= t->th)
            break;
        for (u32 x = 0; x < w; x++) {
            u32 tx = x0 + x;
            if (tx >= t->tw)
                break;
            u32 v = load_texel(*t, gl_index(tx, ty, t->tw, t->th));
            if (rgba16) {
                reinterpret_cast<u16 *>(out.data())[(size_t)y * w + x] =
                    g_ctr64ReadPmBlue ? to_rgba5551<true>(v) : to_rgba5551<false>(v);
                continue;
            }
            u8 *d = &out[((size_t)y * w + x) * 4];
            d[0] = v >> 24; d[1] = v >> 16; d[2] = v >> 8; d[3] = v;
        }
    }
    return out.data();
}

std::vector<u8> g_readback; /* shared by all PicaColorBufferReaders */

const u8 *PicaColorBufferReader::readPixels(s32 x0, s32 y0, u32 width, u32 height, u32, bool sync)
{
    if (!g_ctx)
        return nullptr;
    if (!sync)
        return g_ctx->readTextureAsync(m_pTexture, x0, y0, width, height, m_async, g_readback);
    m_async.finish();
    m_async.valid = false;
    return g_ctx->readTexture(m_pTexture, x0, y0, width, height, g_readback);
}

const u8 *PicaColorBufferReader::_readPixels(const ReadColorBufferParams &params, u32 &heightOffset, u32 &stride)
{
    heightOffset = 0;
    stride = params.width;
    return g_ctx ? g_ctx->readTexture(m_pTexture, params.x0, params.y0, params.width, params.height, g_readback)
                 : nullptr;
}

} // namespace

namespace pica {

graphics::ContextImpl *createContextImpl()
{
    return new ContextImplPica;
}

void presentWindow()
{
    if (g_ctx)
        g_ctx->present();
}

} // namespace pica

/* the frame number GLideN64 code ages things by (DepthBuffer.cpp evictOld) */
extern "C" unsigned ctr64_frame_now(void)
{
    return pica_frame_number();
}

extern "C" void ctr64_gfx_end_vi(void)
{
    if (g_ctx)
        g_ctx->endVI();
}

/* GLideN64 dropped triangles of a skipped frame before reaching the backend (GraphicsDrawer) */
extern "C" void ctr64_gfx_draw_skipped(void)
{
    if (g_ctx)
        g_ctx->skipDraw();
}

namespace pica {

} // namespace pica
