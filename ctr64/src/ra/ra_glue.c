/* ctr64 as a RetroArch 3DS core (built with CTR64_RA, CMake option CTR64_RETROARCH_CORE).
 *
 * RetroArch is the frontend: it owns the screens, the menu, input, audio output and save states.
 * The core is the same Mupen64Plus-Next + GLideN64-on-PICA200 as the ctr64 app. This file stands
 * in for the app's frontend pieces (main.c, core_host.c, video.c, prof.c):
 *
 *  - video: GLideN64 renders with ctr64's PICA backend as in the app; where the app presents the
 *    window texture on the top screen, the transfer engine converts it into a linear RGB565 image
 *    (in GPU-visible memory, so RetroArch's ctr driver copies it to its texture with the GPU)
 *    which retro_run hands to RetroArch.
 *  - the GPU: RetroArch draws with libctru's command buffer globals and submits transfers itself.
 *    The core's command buffer state is swapped in for retro_run and out again afterwards, and the
 *    frame is finished on the GPU before RetroArch gets it. GLideN64 runs on the emulation thread
 *    (no graphics thread), so the GPU is only touched inside retro_run.
 *  - audio: the game's samples at its own rate, resampled (linear) to 44.1 kHz for RetroArch.
 *  - ROMs: read here (need_fullpath) into a buffer the core adopts; ROMs over 16 MB on an Old 3DS
 *    are streamed from the SD card (rom_stream.c), as in the app.
 */
#include <3ds.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "libretro.h"

#include "ctru_abi.h"
#include "debug_crumb.h"
#include "pica/pica.h"
#include "rom_stream.h"
#include "settings.h"

#define DATA_DIR "sdmc:/3ds/ctr64"

/* ---- symbols the core expects from the ctr64 app ------------------------------------------ */
unsigned long long ctr64_alist_ticks[32];
unsigned long long ctr64_prof_audio_wait_ticks;
int ctr64_prof_on;
int ctr64_profiler;
const char *volatile ctr64_crumb;
unsigned long long ctr64_ticks(void) { return svcGetSystemTick(); }
int ctr64_aa_ready(void) { return 0; } /* no anti-aliasing reserve (RetroArch owns the memory) */
void *ctr64_xlinear_alloc(u32 size, u32 align) { (void)size; (void)align; return NULL; }
bool ctr64_xlinear_free(void *p) { (void)p; return false; }
u32 ctr64_xlinear_free_bytes(void) { return 0; }
/* the standard (buffered) save state format is used with RetroArch: never called */
void ctr64_state_fill(void *dst, size_t bytes) { (void)dst; (void)bytes; }
void ctr64_state_put(const void *src, size_t bytes) { (void)src; (void)bytes; }
bool prof_shot_ready(void) { return false; }
void prof_shot(unsigned vi, const u8 *rgba, unsigned pitch) { (void)vi; (void)rgba; (void)pitch; }

static unsigned g_runs, g_frames_out;
static u64 g_stat_t0, g_run_t0, g_run_ticks;
static unsigned g_stat_runs, g_stat_frames, g_stat_sec;
static FILE *g_stats;
static u64 g_wait_ticks;
static FILE *g_log;
void ctr64_log_line(const char *line)
{
    if (!g_log) {
        mkdir("sdmc:/3ds", 0777);
        mkdir(DATA_DIR, 0777);
        g_log = fopen(DATA_DIR "/retroarch.log", "w");
    }
    if (g_log) {
        fputs(line, g_log);
        if (!*line || line[strlen(line) - 1] != '\n')
            fputc('\n', g_log);
        fflush(g_log);
    }
}
static void logf_(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    ctr64_log_line(buf);
}

/* ---- GPU state shared with RetroArch ------------------------------------------------------- */
/* RetroArch's ctr video driver finishes its GPU work at the end of each frame when this is set
 * (gfx/drivers/ctr_gfx.c in the ctr64 RetroArch build): libctru's GX queue counts GSP interrupts,
 * so the core's and RetroArch's GPU work must not overlap */
bool ctr_core_gpu_exclusive = true;

extern u32 *gpuCmdBuf;
extern u32 gpuCmdBufSize, gpuCmdBufOffset;
typedef struct { u32 *buf, size, off; } CmdState;
static CmdState g_ra_cmd, g_core_cmd;
static void cmd_save(CmdState *s) { s->buf = gpuCmdBuf; s->size = gpuCmdBufSize; s->off = gpuCmdBufOffset; }
static void cmd_load(const CmdState *s) { gpuCmdBuf = s->buf; gpuCmdBufSize = s->size; gpuCmdBufOffset = s->off; }
void pica_ra_bind(bool on); /* pica.c: binds/unbinds ctr64's GX command queue */
void pica_untile_rgb565_async(const void *src, u16 width, u16 rows, void *dst); /* pica.c */

static void gpu_enter(void)
{
    cmd_save(&g_ra_cmd);
    cmd_load(&g_core_cmd);
    pica_ra_bind(true);
}
static void gpu_leave(void)
{
    cmd_save(&g_core_cmd);
    cmd_load(&g_ra_cmd);
    pica_ra_bind(false);
}

/* ---- core options the 3DS needs (the ctr64 app's k_overrides, core_host.c) -------------- */
static const struct { const char *key, *value; } k_forced[] = {
    {"mupen64plus-cpucore", "dynamic_recompiler"},
    {"mupen64plus-rsp-plugin", "hle"},
    {"mupen64plus-rdp-plugin", "gliden64"},
    {"mupen64plus-ThreadedRenderer", "False"},
    {"mupen64plus-EnableFBEmulation", "True"},
    {"mupen64plus-EnableNativeResFactor", "1"},
    {"mupen64plus-MultiSampling", "0"},
    {"mupen64plus-FXAA", "0"},
    {"mupen64plus-EnableCopyColorToRDRAM", "Off"},
    {"mupen64plus-EnableCopyDepthToRDRAM", "Off"},
    {"mupen64plus-EnableCopyAuxToRDRAM", "False"},
    {"mupen64plus-EnableN64DepthCompare", "False"},
    {"mupen64plus-EnableShadersStorage", "False"},
    {"mupen64plus-EnableLODEmulation", "False"},
    {"mupen64plus-EnableHWLighting", "False"},
    {"mupen64plus-EnableLegacyBlending", "True"},
    {"mupen64plus-EnableFragmentDepthWrite", "False"},
    {"mupen64plus-EnableInaccurateTextureCoordinates", "True"},
    {"mupen64plus-EnableOverscan", "Disabled"},
    {"mupen64plus-txHiresEnable", "False"},
    {"mupen64plus-txFilterMode", "None"},
    {"mupen64plus-txEnhancementMode", "None"},
    {"mupen64plus-aspect", "4:3"},
    {"mupen64plus-43screensize", "320x240"},
    {"mupen64plus-MaxTxCacheSize", "600"},
};

const char *ctr64_ra_forced_option(const char *key)
{
    for (size_t i = 0; i < sizeof(k_forced) / sizeof(k_forced[0]); i++)
        if (!strcmp(key, k_forced[i].key))
            return k_forced[i].value;
    return NULL;
}

/* ---- video -------------------------------------------------------------------------------- */
static bool g_video_ready;

static struct {
    const void *tex;
    u16 tw, th, w, h;
} g_src;
static u16 *g_out[2];
static u32 g_out_bytes;
static int g_out_cur;
static bool g_frame_ready;
static u64 g_frame_fence;
static const void *g_frame;
static unsigned g_fw = 320, g_fh = 240;
static size_t g_fpitch = 640;

void video_draw_fullscreen(const void *tex, u16 tw, u16 th, GPU_TEXCOLOR fmt, u16 w, u16 h, bool top_is_t0,
                           bool linear)
{
    (void)fmt; (void)top_is_t0; (void)linear;
    g_src.tex = tex;
    g_src.tw = tw;
    g_src.th = th;
    g_src.w = w;
    g_src.h = h;
}

void video_present(void)
{
    if (g_src.tex) {
        const u32 bytes = (u32)g_src.tw * g_src.th * 2;
        if (bytes > g_out_bytes) {
            for (int i = 0; i < 2; i++) {
                if (g_out[i])
                    linearFree(g_out[i]);
                g_out[i] = linearMemAlign(bytes, 128);
            }
            g_out_bytes = (g_out[0] && g_out[1]) ? bytes : 0;
        }
        if (g_out_bytes) {
            g_out_cur ^= 1;
            u16 *dst = g_out[g_out_cur];
            pica_untile_rgb565_async(g_src.tex, g_src.tw, g_src.th, dst);
            g_frame_fence = pica_fence();
            /* untiled (no flip), the image's top row is row th - h, rows going down */
            g_frame = dst + (size_t)(g_src.th - g_src.h) * g_src.tw;
            g_fw = g_src.w;
            g_fh = g_src.h;
            g_fpitch = (size_t)g_src.tw * 2;
            g_frame_ready = true;
        }
        g_src.tex = NULL;
    }
    pica_frame_end(NULL, GFX_TOP);
}

void ctr64_ra_geometry(struct retro_game_geometry *g)
{
    g->base_width = 320;
    g->base_height = 240;
    g->max_width = 320; /* the image is GLideN64's 320x240 window whatever the game's resolution */
    g->max_height = 240;
    g->aspect_ratio = 4.0f / 3.0f;
}

static bool video_start(void)
{
    if (g_video_ready)
        return true;
    cmd_save(&g_ra_cmd);
    logf_("before pica_init: linear free %u KB, vram free %u KB", (unsigned)linearSpaceFree() / 1024,
          (unsigned)vramSpaceFree() / 1024);
    /* video.c as in the app: pica_init, the shaders, the GPU fill and non-power-of-two texture
     * self-tests (nothing is presented: pica_frame_end leaves the screens to RetroArch) */
    extern bool video_init(void);
    if (!video_init()) {
        logf_("video_init failed");
        cmd_load(&g_ra_cmd);
        return false;
    }
    pica_wait_idle();
    cmd_save(&g_core_cmd);
    cmd_load(&g_ra_cmd);
    pica_ra_bind(false);
    g_video_ready = true;
    extern char pica_fill_report[96];
    extern char video_npot_report[96];
    logf_("%s", pica_fill_report);
    logf_("%s", video_npot_report);
    return true;
}

/* ---- ROM ------------------------------------------------------------------------------------ */
void *ctr64_ra_load_rom(const char *path, size_t *size_out)
{
    settings_defaults(&g_settings);
    /* GLideN64 on the emulation thread: the GPU is only used inside retro_run. (Its own thread, as
     * in the app, was measured no faster here: it has to be drained at the end of every retro_run.) */
    g_settings.threaded_video = false;
    g_settings.threaded_audio = false;
    g_settings.frameskip = 0;          /* RetroArch has its own frame skipping */
    g_settings.speed_limit = false;    /* RetroArch paces the frames */
    g_settings.remember_blocks = false;
    g_settings.aspect = ASPECT_STRETCH; /* the window is the 4:3 image itself (pica_DisplayWindow) */
    g_settings.anti_aliasing = false;
    settings_publish();

    if (!video_start())
        return NULL;
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        logf_("cannot open %s", path ? path : "(no path)");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0x1000 || size > 0xFC00000) {
        fclose(f);
        logf_("%s: not an N64 ROM (%ld bytes)", path, size);
        return NULL;
    }
    mkdir("sdmc:/3ds", 0777);
    mkdir(DATA_DIR, 0777);
    bool stream = rom_stream_wanted(size, DATA_DIR);
    void *data = stream ? NULL : malloc(size);
    if (!stream && !data)
        stream = true;
    if (stream) {
        data = (void *)rom_stream_open(f, size, DATA_DIR, NULL); /* takes f */
        logf_("ROM streamed from the SD card: %s", data ? "ok" : "FAILED");
    } else if (fread(data, 1, size, f) != (size_t)size) {
        fclose(f);
        free(data);
        logf_("%s: read error", path);
        return NULL;
    } else {
        fclose(f);
    }
    *size_out = (size_t)size;
    struct mallinfo mi = mallinfo();
    logf_("ROM %s: %ld KB, heap used %u KB, linear free %u KB, vram free %u KB", path, size / 1024,
          mi.uordblks / 1024, (unsigned)linearSpaceFree() / 1024, (unsigned)vramSpaceFree() / 1024);
    return data;
}

extern retro_hw_context_reset_t ctr64_core_context_reset; /* core_compat/glsm_ctr.c */

/* stall watchdog (development aid): logs ctr64's breadcrumb when retro_run stops returning */
static volatile unsigned g_runs_seen;
static volatile bool g_in_run;
static void watchdog(void *arg)
{
    (void)arg;
    unsigned last = ~0u, stuck = 0;
    while (1) {
        svcSleepThread(1000000000LL);
        if (g_runs_seen == last && g_in_run) {
            if (++stuck <= 3)
                logf_("stall %us in run %u at: %s", stuck, last, ctr64_crumb ? ctr64_crumb : "?");
        } else
            stuck = 0;
        last = g_runs_seen;
    }
}

static bool dev_file(const char *name)
{
    char path[64];
    snprintf(path, sizeof(path), DATA_DIR "/%s", name);
    FILE *f = fopen(path, "r");
    if (f)
        fclose(f);
    return f != NULL;
}

static bool g_stats_on, g_dump_on;

bool ctr64_ra_after_load(void)
{
    /* the core starts the emulator (and GLideN64) in its context reset */
    gpu_enter();
    if (ctr64_core_context_reset)
        ctr64_core_context_reset();
    gpu_leave();
    logf_("loaded: linear free %u KB, vram free %u KB", (unsigned)linearSpaceFree() / 1024,
          (unsigned)vramSpaceFree() / 1024);
    g_stats_on = dev_file("stats.txt");  /* stats.log: one line a second */
    g_dump_on = dev_file("ra_dump.txt"); /* ra_dump.raw: one RGB565 image after run 200 */
    if (dev_file("watchdog.txt")) {
        s32 prio = 0x30;
        svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
        threadCreate(watchdog, NULL, 8192, prio - 2, -2, true);
    }
    return true;
}

/* ---- each frame ------------------------------------------------------------------------------- */
void ctr64_ra_run_begin(void)
{
    g_run_t0 = svcGetSystemTick();
    g_in_run = true;
    gpu_enter();
    g_frame_ready = false;
}

static void stats_tick(void)
{
    /* "t <s> fps <retro_runs/s> emu <ms per retro_run> images <new images/s>" */
    const u64 now = svcGetSystemTick();
    g_run_ticks += now - g_run_t0;
    g_stat_runs++;
    g_stat_frames += g_frame_ready;
    if (!g_stat_t0) {
        g_stat_t0 = now;
        return;
    }
    if (now - g_stat_t0 < SYSCLOCK_ARM11)
        return;
    const double wall = (double)(now - g_stat_t0) / SYSCLOCK_ARM11;
    if (!g_stats)
        g_stats = fopen(DATA_DIR "/stats.log", "w");
    if (g_stats) {
        fprintf(g_stats, "t %u fps %.1f emu %.2f images %.1f gpuwait %.2f\n", ++g_stat_sec, g_stat_runs / wall,
                (double)g_run_ticks * 1000.0 / SYSCLOCK_ARM11 / g_stat_runs, g_stat_frames / wall,
                (double)g_wait_ticks * 1000.0 / SYSCLOCK_ARM11 / g_stat_runs);
        g_wait_ticks = 0;
        fflush(g_stats);
    }
    g_stat_t0 = now;
    g_stat_runs = g_stat_frames = 0;
    g_run_ticks = 0;
}

bool ctr64_ra_run_end(const void **frame, unsigned *w, unsigned *h, size_t *pitch)
{
    /* everything the core queued is done before RetroArch touches the GPU (it copies the frame
     * with the GPU right after; see ctr_core_gpu_exclusive) */
    (void)g_frame_fence;
    extern void ctr64_gfx_drain(void);
    const u64 tw = svcGetSystemTick();
    ctr64_gfx_drain(); /* the graphics thread, if one runs */
    pica_wait_idle();
    g_wait_ticks += svcGetSystemTick() - tw;
    gpu_leave();
    g_runs++;
    g_runs_seen = g_runs;
    g_in_run = false;
    if (g_frame_ready)
        g_frames_out++;
    if (g_runs <= 3)
        logf_("run %u: %u images, %ux%u; linear free %u KB", g_runs, g_frames_out, g_fw, g_fh,
              (unsigned)linearSpaceFree() / 1024);
    if (g_stats_on)
        stats_tick();
    static bool dumped;
    if (g_dump_on && g_frame_ready && g_runs >= 200 && !dumped) {
        dumped = true;
        pica_cache_invalidate(g_out[g_out_cur], g_out_bytes);
        FILE *o = fopen(DATA_DIR "/ra_dump.raw", "wb");
        if (o) {
            for (unsigned y = 0; y < g_fh; y++)
                fwrite((const u8 *)g_frame + y * g_fpitch, 2, g_fw, o);
            fclose(o);
        }
    }
    *w = g_fw;
    *h = g_fh;
    *pitch = g_fpitch;
    *frame = g_frame_ready ? g_frame : NULL;
    return g_frame_ready;
}

/* ---- audio: the game's rate -> 44.1 kHz (linear interpolation, 16.16 fixed point) ---------- */
extern retro_audio_sample_batch_t audio_batch_cb;
static int16_t g_aout[2048 * 2];
static uint32_t g_apos;            /* 16.16 position into the input, relative to g_aprev */
static int16_t g_aprev[2];         /* last input frame of the previous call */

void ctr64_audio_push(const int16_t *in, size_t count, unsigned rate)
{
    if (!audio_batch_cb || !count || !rate)
        return;
    const uint32_t step = (uint32_t)(((uint64_t)rate << 16) / 44100);
    size_t n = 0;
    /* sample k of the input: k = 0 is g_aprev, k >= 1 is in[k - 1] */
    while (1) {
        const uint32_t k = g_apos >> 16;
        if (k >= count)
            break;
        const uint32_t fr = g_apos & 0xFFFF;
        const int16_t *a = k == 0 ? g_aprev : &in[(k - 1) * 2];
        const int16_t *b = &in[k * 2];
        g_aout[n * 2] = (int16_t)(a[0] + (((int32_t)(b[0] - a[0]) * (int32_t)fr) >> 16));
        g_aout[n * 2 + 1] = (int16_t)(a[1] + (((int32_t)(b[1] - a[1]) * (int32_t)fr) >> 16));
        if (++n == 2048) {
            audio_batch_cb(g_aout, n);
            n = 0;
        }
        g_apos += step;
    }
    if (n)
        audio_batch_cb(g_aout, n);
    g_apos -= (uint32_t)count << 16;
    g_aprev[0] = in[(count - 1) * 2];
    g_aprev[1] = in[(count - 1) * 2 + 1];
}
