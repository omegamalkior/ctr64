#include "video.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ctr64_shaders_shbin.h"
#include "ctru_abi.h"

static const PicaAttrib k_layout[] = {
    {GPU_FLOAT, 3},
    {GPU_UNSIGNED_BYTE, 4},
    {GPU_FLOAT, 2},
};

static struct {
    DVLB_s *dvlb;
    shaderProgram_s prog;
    int u_proj;
    PicaTarget screen;
    float proj_screen[4][4];
} V;

bool video_init(void)
{
    if (!pica_init())
        return false;
    /* The top screen image is drawn once per frame: in FCRAM, so its 375 KB of VRAM bank B
     * holds part of a frame buffer instead (MK64 menus with anti-aliasing: three 2x color
     * buffers + depth + the 2D buffer fit only without it, else 1.25 MB moved per frame). */
    if (!pica_target_create_linear(&V.screen, 240, 400, GPU_RB_RGBA8) &&
        !pica_target_create(&V.screen, 240, 400, GPU_RB_RGBA8, false, VRAM_ALLOC_B))
        return false;
    V.dvlb = DVLB_ParseFile((u32 *)ctr64_shaders_shbin, ctr64_shaders_shbin_size);
    shaderProgramInit(&V.prog);
    shaderProgramSetVsh(&V.prog, ctru_dvle(V.dvlb, 0));
    V.u_proj = shaderInstanceGetUniformLocation(V.prog.vertexShader, "projection");
    pica_mtx_ortho_tilt(V.proj_screen, 0, 400, 240, 0, 0.0f, 1.0f);
    pica_fill_selftest();
    video_npot_selftest();
    return true;
}

void video_exit(void)
{
    pica_exit();
    shaderProgramFree(&V.prog);
    DVLB_Free(V.dvlb);
    pica_target_destroy(&V.screen);
}

shaderProgram_s *video_program(void) { return &V.prog; }
DVLB_s *video_shader_library(void) { return V.dvlb; }
int video_projection_uniform(void) { return V.u_proj; }
const PicaTarget *video_screen(void) { return &V.screen; }

void video_use_program(const float proj[4][4])
{
    pica_bind_program(&V.prog);
    pica_uniform_matrix(V.u_proj, proj);
    pica_vertex_layout(k_layout, 3, sizeof(VideoVertex));
}

bool video_batch_begin(VideoBatch *b, int cap)
{
    b->v = pica_frame_alloc(sizeof(VideoVertex) * cap, 16);
    b->n = 0;
    b->cap = b->v ? cap : 0;
    return b->v != NULL;
}

void video_batch_vertex(VideoBatch *b, float x, float y, u32 rgb, float u, float v)
{
    if (b->n >= b->cap)
        return;
    VideoVertex *p = &b->v[b->n++];
    p->x = x;
    p->y = y;
    p->z = -0.5f;
    p->r = rgb >> 16;
    p->g = rgb >> 8;
    p->b = rgb;
    p->a = 0xFF;
    p->u = u;
    p->v = v;
}

void video_batch_quad4(VideoBatch *b, float x0, float y0, float x1, float y1, u32 tl, u32 tr, u32 bl,
                       u32 br, float u0, float v0, float u1, float v1)
{
    video_batch_vertex(b, x0, y0, tl, u0, v0);
    video_batch_vertex(b, x0, y1, bl, u0, v1);
    video_batch_vertex(b, x1, y1, br, u1, v1);
    video_batch_vertex(b, x0, y0, tl, u0, v0);
    video_batch_vertex(b, x1, y1, br, u1, v1);
    video_batch_vertex(b, x1, y0, tr, u1, v0);
}

void video_batch_quad(VideoBatch *b, float x0, float y0, float x1, float y1, u32 rgb)
{
    video_batch_quad4(b, x0, y0, x1, y1, rgb, rgb, rgb, rgb, 0, 0, 0, 0);
}

void video_batch_draw(VideoBatch *b)
{
    if (!b->n)
        return;
    pica_vertex_buffer(b->v);
    pica_draw_arrays(GPU_TRIANGLES, 0, b->n);
}

static void draw_screen_quad(const void *tex, u16 tw, u16 th, GPU_TEXCOLOR fmt, u16 w, u16 h,
                             bool top_is_t0, bool linear, float x0, float x1)
{
    PicaTev tev;
    VideoBatch b;

    pica_clear(&V.screen, pica_rgba8(0, 0, 0, 0xFF), false, 0);
    pica_bind_target(&V.screen);
    pica_viewport(0, 0, V.screen.width, V.screen.height);
    /* GLideN64 may have left any fragment state behind */
    pica_scissor(GPU_SCISSOR_DISABLE, 0, 0, 0, 0);
    pica_depth_test(false, GPU_ALWAYS, GPU_WRITE_COLOR);
    pica_depth_map(-1.0f, 0.0f);
    pica_cull(GPU_CULL_NONE);
    pica_alpha_test(false, GPU_ALWAYS, 0);
    pica_blend_off();
    video_use_program(V.proj_screen);
    if (!tex)
        return;

    GPU_TEXTURE_FILTER_PARAM f = linear ? GPU_LINEAR : GPU_NEAREST;
    pica_texture(0, tex, tw, th, fmt,
                 GPU_TEXTURE_MAG_FILTER(f) | GPU_TEXTURE_MIN_FILTER(f) |
                     GPU_TEXTURE_WRAP_S(GPU_CLAMP_TO_EDGE) | GPU_TEXTURE_WRAP_T(GPU_CLAMP_TO_EDGE));
    pica_tev_default(&tev);
    tev.src_rgb = GPU_TEVSOURCES(GPU_TEXTURE0, 0, 0);
    tev.src_alpha = GPU_TEVSOURCES(GPU_CONSTANT, 0, 0); /* opaque regardless of N64 alpha */
    tev.color = 0xFFFFFFFF;
    pica_tev(0, &tev);

    float u1 = (float)w / tw, vh = (float)h / th;
    float vt = top_is_t0 ? 0.0f : vh, vb = top_is_t0 ? vh : 0.0f;
    if (video_batch_begin(&b, 6)) {
        video_batch_quad4(&b, x0, 0, x1, 240, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0, vt, u1, vb);
        video_batch_draw(&b);
    }
}

/* Can the GPU sample a texture whose width is not a power of two? Frame buffer textures could
 * then be stored 640 texels wide instead of 1024 (anti-aliasing: 1.25 MB instead of 2 MB, so the
 * targets of a frame fit in VRAM). Azahar samples any width; the real PICA is checked here: a
 * 96x32 texture of distinct texels is drawn onto the screen target with nearest filtering and
 * every texel must show up, and nothing else. A wrong row pitch repeats or misses texels.
 * Dev switch nonpot.txt: off. Result in core.log (video_npot_report). */
bool video_npot_ok;
char video_npot_report[96] = "non-power-of-two texture self-test not run";

void video_npot_selftest(void)
{
    enum { TW = 96, TH = 32, ALLOC_W = 128 };
    FILE *off = fopen("sdmc:/3ds/ctr64/nonpot.txt", "r");
    if (off) {
        fclose(off);
        snprintf(video_npot_report, sizeof(video_npot_report), "non-power-of-two textures: off (nonpot.txt)");
        return;
    }
    const u32 scr_bytes = 240 * 400 * 4;
    u32 *tex = linearMemAlign(ALLOC_W * TH * 4, 128); /* room for a pot pitch: reads stay inside */
    u32 *scr = linearMemAlign(scr_bytes, 128);
    u8 *seen = malloc(TW * TH);
    if (!tex || !scr || !seen) {
        snprintf(video_npot_report, sizeof(video_npot_report), "non-power-of-two self-test: no memory");
        goto done;
    }
    memset(tex, 0, ALLOC_W * TH * 4);
    memset(seen, 0, TW * TH);
    for (u32 y = 0; y < TH; y++)
        for (u32 x = 0; x < TW; x++) {
            u32 m = 0; /* morton order inside the 8x8 tile */
            for (int b = 0; b < 3; b++)
                m |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1);
            tex[((y >> 3) * (TW >> 3) + (x >> 3)) * 64 + m] = pica_rgba8((u8)x, (u8)y, 0x80, 0xFF);
        }
    pica_cache_flush(tex, ALLOC_W * TH * 4);
    pica_frame_begin();
    draw_screen_quad(tex, TW, TH, GPU_RGBA8, TW, TH, false, false, 0, 384);
    pica_copy_from_vram(scr, V.screen.color, scr_bytes);
    pica_clear(&V.screen, pica_rgba8(0, 0, 0, 0xFF), false, 0);
    pica_frame_end(&V.screen, GFX_TOP);
    pica_wait_idle();
    pica_cache_invalidate(scr, scr_bytes);
    u32 bad = 0, texels = 0, distinct = 0;
    const u32 black = pica_rgba8(0, 0, 0, 0xFF);
    for (u32 i = 0; i < 240 * 400; i++) {
        const u32 v = scr[i];
        if (v == black)
            continue;
        const u32 r = v >> 24, g = (v >> 16) & 0xFF, b = (v >> 8) & 0xFF, a = v & 0xFF;
        if (b != 0x80 || a != 0xFF || r >= TW || g >= TH) {
            bad++;
            continue;
        }
        texels++;
        if (!seen[g * TW + r]) {
            seen[g * TW + r] = 1;
            distinct++;
        }
    }
    video_npot_ok = bad == 0 && distinct == TW * TH && texels >= 240 * 384 * 9 / 10;
    snprintf(video_npot_report, sizeof(video_npot_report),
             "non-power-of-two textures: %s (%lu/%u texels seen, %lu wrong pixels)",
             video_npot_ok ? "usable" : "not usable", (unsigned long)distinct, TW * TH, (unsigned long)bad);
done:
    free(seen);
    if (scr)
        linearFree(scr);
    if (tex)
        linearFree(tex);
}

void video_draw_to_screen(const void *tex, u16 tw, u16 th, GPU_TEXCOLOR fmt, u16 w, u16 h,
                          bool top_is_t0, bool linear)
{
    /* 4:3 image centered on the 400x240 screen */
    draw_screen_quad(tex, tw, th, fmt, w, h, top_is_t0, linear, 40, 360);
}

#if !defined(CTR64_RA) /* RetroArch core: ra/ra_glue.c (the image goes to RetroArch) */
void video_draw_fullscreen(const void *tex, u16 tw, u16 th, GPU_TEXCOLOR fmt, u16 w, u16 h,
                           bool top_is_t0, bool linear)
{
    draw_screen_quad(tex, tw, th, fmt, w, h, top_is_t0, linear, 0, 400);
}

void video_present(void)
{
    pica_frame_end(&V.screen, GFX_TOP);
}
#endif
