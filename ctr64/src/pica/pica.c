#include "pica_internal.h"
#include "debug_crumb.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>

#define SLOT_COUNT 2
#define CMDBUF_BYTES (512 * 1024)
#define POOL_BYTES (2 * 1024 * 1024)
/* GX queue entries. The queue is a plain array: it can only be emptied (queue_recycle) when the
 * GPU has done everything in it, and a full queue makes the next command wait for that - a whole
 * GPU drain mid-frame. With anti-aliasing a frame queues many transfers (shadow copies, target
 * moves, copies by transfer), 128 entries filled every few frames while the GPU was never idle at
 * a frame start: OoT on hardware waited 10-50 ms in bindUnit/stagedUpdate/moveToVram/allocStorage
 * every ~12 VIs (2026-11-10 profiles). Now 2048 entries (64 KB), emptied whenever the GPU is found
 * idle with a quarter in use (queue_reserve) or an eighth at a frame start. */
#define QUEUE_MAX 2048
#define TOP_FB_BYTES (240 * 400 * 3)

/* Display transfer used for presenting: tiled RGBA8 render target -> linear BGR8 LCD buffer. */
#define PRESENT_FLAGS                                                                     \
    (GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |      \
     GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | \
     GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

typedef struct {
    u32 *cmd;
    u8 *pool;
    size_t pool_used;
    size_t pool_flushed;
    u64 fence; /* absolute queue position after which this slot's memory is free again */
} Slot;

static struct {
    Slot slots[SLOT_COUNT];
    int cur;
    bool in_frame;

    gxCmdQueue_s queue;
    u64 base; /* entries dropped by queue clears, so fences stay monotonic */

    u32 frame_no; /* frames begun */
    bool swap_pending;
    u64 present_fence;
    u8 *top_fb[3];  /* presentation buffers, see top_buffers() */
    u8 *top_own[2]; /* the ones allocated here */
    int top_next;   /* buffer the next transfer goes to */
    unsigned present_set; /* framebuffer register set used by the last present */

    aptHookCookie apt_cookie;

    u64 tick_begin, tick_last_begin;
    float wait_ms;
    u32 cmd_bytes;
    u64 frame_first_entry;
    PicaStats stats;
} P;

static float ticks_to_ms(u64 ticks)
{
    return (float)ticks * 1000.0f / (float)SYSCLOCK_ARM11;
}

/* ---- queue / fences ------------------------------------------------------------------ */

unsigned long long pica_prof_ticks[2]; /* profiler: [0] waits for the GPU, [1] waits for VBlank */

static u64 queue_completed(void)
{
    return P.base + *(volatile u16 *)&P.queue.lastEntry;
}

static u64 queue_enqueued(void)
{
    return P.base + P.queue.numEntries;
}

/* GPU waits attributed to the breadcrumb that was current when they started (profiling) */
static struct {
    const char *where;
    u32 count;
    float ms;
} g_waits[8];

void pica_wait_report(FILE *f)
{
    for (int i = 0; i < 8 && g_waits[i].where; i++)
        fprintf(f, " %s %lu/%.2f", g_waits[i].where, (unsigned long)g_waits[i].count, g_waits[i].ms);
    memset(g_waits, 0, sizeof(g_waits));
}

static void note_wait(const char *where, float ms)
{
    for (int i = 0; i < 8; i++)
        if (!g_waits[i].where || g_waits[i].where == where) {
            g_waits[i].where = where;
            g_waits[i].count++;
            g_waits[i].ms += ms;
            return;
        }
}

static void wait_fence(u64 fence)
{
    if (queue_completed() >= fence)
        return;
    const char *prev = ctr64_crumb;
    CRUMB("pica: wait_fence");
    u64 t0 = svcGetSystemTick();
    int n = 0;
    while (queue_completed() < fence) {
        svcSleepThread(100000); /* 0.1 ms; GX completion is interrupt driven */
        if (++n % 20000 == 0) /* 2 s: report a stuck queue */
            printf("\x1b[25;0Hfence stuck: want %llu done %llu (base %llu last %u num %u max %u cur %u)\n",
                   fence, queue_completed(), P.base, P.queue.lastEntry, P.queue.numEntries,
                   P.queue.maxEntries, P.queue.curEntry);
    }
    P.wait_ms += ticks_to_ms(svcGetSystemTick() - t0);
    pica_prof_ticks[0] += svcGetSystemTick() - t0;
    P.stats.fence_waits++;
    P.stats.fence_ms += ticks_to_ms(svcGetSystemTick() - t0);
    note_wait(prev ? prev : "?", ticks_to_ms(svcGetSystemTick() - t0));
    ctr64_crumb = prev;
}

/* Drops completed entries so the fixed-size queue never overflows. Only legal when idle. */
unsigned pica_queue_drains; /* full GX queue: waited for the GPU to finish everything (stats.log) */
static void queue_recycle(void)
{
    gxCmdQueueWait(&P.queue, -1);
    P.base += P.queue.numEntries; /* before Stop: stopping an idle queue clears it */
    gxCmdQueueStop(&P.queue);
    gxCmdQueueClear(&P.queue);
    gxCmdQueueRun(&P.queue);
}

/* profiler (prof.c): running totals in ticks. GPU idle: from the queue's completion callback
 * (all queued work done) to the next command added. */
static volatile unsigned long long g_idle_ticks, g_idle_since;
static volatile bool g_gpu_idle;

static void queue_done_cb(gxCmdQueue_s *q)
{
    (void)q;
    if (!g_gpu_idle) {
        g_idle_since = svcGetSystemTick();
        g_gpu_idle = true;
    }
}

static void gpu_busy_again(void)
{
    if (g_gpu_idle) {
        g_gpu_idle = false;
        g_idle_ticks += svcGetSystemTick() - g_idle_since;
    }
}

unsigned long long pica_prof_gpu_idle(void)
{
    const unsigned long long t = g_idle_ticks;
    return g_gpu_idle ? t + (svcGetSystemTick() - g_idle_since) : t;
}

unsigned pica_prof_presents(void)
{
    return P.stats.frames;
}

static void queue_reserve(int entries)
{
    gpu_busy_again();
    if (P.queue.numEntries > QUEUE_MAX / 4 && queue_completed() == queue_enqueued())
        queue_recycle(); /* idle: free */
    if (P.queue.numEntries + entries <= P.queue.maxEntries)
        return;
    pica_queue_drains++;
    wait_fence(queue_enqueued());
    queue_recycle();
}

void pica_cache_flush(const void *p, u32 size)
{
    static bool direct = true;
    if (direct && R_SUCCEEDED(svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)p, size)))
        return;
    direct = false;
    GSPGPU_FlushDataCache(p, size);
}

void pica_cache_invalidate(const void *p, u32 size)
{
    static bool direct = true;
    if (direct && R_SUCCEEDED(svcInvalidateProcessDataCache(CUR_PROCESS_HANDLE, (u32)p, size)))
        return;
    direct = false;
    GSPGPU_InvalidateDataCache(p, size);
}

/* Closes the command list recorded so far and queues it for execution. */
static void submit_commands(void)
{
    Slot *s = &P.slots[P.cur];
    u32 *list;
    u32 words;

    if (!gpuCmdBufOffset)
        return;
    pica_state_flush_framebuffer();
    GPUCMD_Split(&list, &words);

    if (s->pool_used > s->pool_flushed) {
        pica_cache_flush(s->pool + s->pool_flushed, s->pool_used - s->pool_flushed);
        s->pool_flushed = s->pool_used;
    }
    queue_reserve(1);
    GX_ProcessCommandList(list, words * 4, GX_CMDLIST_FLUSH);
    P.cmd_bytes += words * 4;
}

static void apt_hook(APT_HookType type, void *param)
{
    (void)param;
    if (type == APTHOOK_ONSUSPEND || type == APTHOOK_ONEXIT)
        wait_fence(queue_enqueued());
}

/* ---- device ---------------------------------------------------------------------------- */

bool pica_init(void)
{
    memset(&P, 0, sizeof(P));
    for (int i = 0; i < SLOT_COUNT; i++) {
        P.slots[i].cmd = linearMemAlign(CMDBUF_BYTES, 16);
        P.slots[i].pool = linearMemAlign(POOL_BYTES, 128);
        if (!P.slots[i].cmd || !P.slots[i].pool)
            return false;
    }
    P.queue.maxEntries = QUEUE_MAX;
    P.queue.entries = malloc(QUEUE_MAX * sizeof(gxCmdEntry_s));
    if (!P.queue.entries)
        return false;
    GX_BindQueue(&P.queue);
    gxCmdQueueSetCallback(&P.queue, queue_done_cb, NULL);
    gxCmdQueueRun(&P.queue);
    aptHook(&P.apt_cookie, apt_hook, NULL);
    P.cur = SLOT_COUNT - 1;
    return true;
}

void pica_exit(void)
{
    wait_fence(queue_enqueued());
    gxCmdQueueWait(&P.queue, -1);
    gxCmdQueueStop(&P.queue);
    gxCmdQueueClear(&P.queue);
    GX_BindQueue(NULL);
    aptUnhook(&P.apt_cookie);
    free(P.queue.entries);
    if (P.top_fb[2]) /* hand the screen back to libctru's buffers before freeing ours */
        gfxScreenSwapBuffers(GFX_TOP, false);
    for (int i = 0; i < 2; i++)
        if (P.top_own[i])
            linearFree(P.top_own[i]);
    for (int i = 0; i < SLOT_COUNT; i++) {
        linearFree(P.slots[i].cmd);
        linearFree(P.slots[i].pool);
    }
    memset(&P, 0, sizeof(P));
}

void pica_wait_idle(void)
{
    wait_fence(queue_enqueued());
}

const PicaStats *pica_stats(void)
{
    return &P.stats;
}

void pica_stats_reset_vblank(void)
{
    P.stats.vblank_waits = 0;
    P.stats.vblank_ms = 0;
    P.stats.fence_waits = 0;
    P.stats.fence_ms = 0;
}

/* ---- frames ---------------------------------------------------------------------------- */

void pica_frame_begin(void)
{
    u64 now = svcGetSystemTick();

    if (P.tick_last_begin)
        P.stats.frame_ms = ticks_to_ms(now - P.tick_last_begin);
    P.tick_last_begin = now;
    P.wait_ms = 0;
    P.cmd_bytes = 0;

    P.cur = (P.cur + 1) % SLOT_COUNT;
    Slot *s = &P.slots[P.cur];
    wait_fence(s->fence); /* the GPU is done with everything this slot recorded before */
    P.frame_no++; /* so frame frame_no - 2 (and everything before) is complete */

    if (queue_completed() == queue_enqueued() && P.queue.numEntries > QUEUE_MAX / 8)
        queue_recycle();

    s->pool_used = s->pool_flushed = 0;
    GPUCMD_SetBuffer(s->cmd, CMDBUF_BYTES / 4, 0);
    pica_state_reset();
    P.frame_first_entry = queue_enqueued();
    P.tick_begin = svcGetSystemTick();
    P.in_frame = true;
}

/* Three top-screen buffers (BGR8, FCRAM, 240x400 as the LCD scans them): libctru's two from
 * gfxInit, plus one more. */
static bool top_buffers(void)
{
    if (P.top_fb[2])
        return true;
    P.top_fb[0] = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);
    gfxScreenSwapBuffers(GFX_TOP, false);
    P.top_fb[1] = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);
    if (P.top_fb[1] == P.top_fb[0])
        P.top_fb[1] = P.top_own[0] = linearAlloc(TOP_FB_BYTES);
    P.top_fb[2] = P.top_own[1] = linearAlloc(TOP_FB_BYTES);
    if (!P.top_fb[1] || !P.top_fb[2]) {
        P.top_fb[2] = NULL;
        return false;
    }
    return true;
}

void pica_frame_end(const PicaTarget *screen, gfxScreen_t lcd)
{
    Slot *s = &P.slots[P.cur];

    if (!P.in_frame)
        return;
    submit_commands();
    P.stats.cpu_ms = ticks_to_ms(svcGetSystemTick() - P.tick_begin);

    /* Triple-buffered presentation. The previous frame's transfer is presented once it has
     * landed; GSP latches it at the next VBlank. Only if the present before that is still
     * unlatched (we are running faster than 60 Hz) do we wait for VBlank. That keeps the
     * buffer shown on screen, the one waiting to be latched and the one transferred into next
     * distinct, so the CPU never blocks on VBlank while emulation runs below full speed. */
    if (P.swap_pending) {
        wait_fence(P.present_fence);
        if (gspIsPresentPending(GSP_SCREEN_TOP)) {
            CRUMB("pica: vblank");
            u64 t0 = svcGetSystemTick();
            gspWaitForVBlank();
            P.stats.vblank_waits++;
            P.stats.vblank_ms += ticks_to_ms(svcGetSystemTick() - t0);
            pica_prof_ticks[1] += svcGetSystemTick() - t0;
        }
        P.present_set ^= 1;
        gspPresentBuffer(GSP_SCREEN_TOP, P.present_set, P.top_fb[P.top_next], P.top_fb[P.top_next],
                         240 * 3, GSP_BGR8_OES | 0x40 | 0x100);
        P.top_next = (P.top_next + 1) % 3;
        P.swap_pending = false;
    }

#if defined(CTR64_RA)
    screen = NULL; /* RetroArch core: RetroArch owns the screens (video.c's self-test frame stays off) */
#endif
    if (screen && lcd == GFX_TOP && top_buffers()) {
        u32 dim = GX_BUFFER_DIM(screen->width, screen->height);
        queue_reserve(1);
        GX_DisplayTransfer((u32 *)screen->color, dim, (u32 *)P.top_fb[P.top_next], dim, PRESENT_FLAGS);
        P.present_fence = queue_enqueued();
        P.swap_pending = true;
    }

    s->fence = queue_enqueued();
    GPUCMD_SetBuffer(NULL, 0, 0);
    P.in_frame = false;

    P.stats.frames++;
    P.stats.fence_wait_ms = P.wait_ms;
    P.stats.cmd_bytes = P.cmd_bytes;
    P.stats.queue_entries = (u32)(queue_enqueued() - P.frame_first_entry);
}

u32 pica_frame_number(void)
{
    return P.frame_no;
}

bool pica_in_frame(void)
{
    return P.in_frame;
}

void pica_flush(void)
{
    if (P.in_frame)
        submit_commands();
}

void pica_untile_rgba8(const void *src, u16 width, u16 rows, void *dst)
{
    pica_flush();
    queue_reserve(1);
    GX_DisplayTransfer((u32 *)src, GX_BUFFER_DIM(width, rows), (u32 *)dst, GX_BUFFER_DIM(width, rows),
                       GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
                           GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) |
                           GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
    wait_fence(queue_enqueued());
    pica_cache_invalidate(dst, (u32)width * rows * 4);
}

/* As pica_untile_rgba8, but only queued (after everything submitted so far). Before reading dst:
 * pica_wait_fence(pica_fence() taken right after this call), then pica_cache_invalidate. */
void pica_untile_rgba8_async(const void *src, u16 width, u16 rows, void *dst)
{
    pica_flush();
    queue_reserve(1);
    GX_DisplayTransfer((u32 *)src, GX_BUFFER_DIM(width, rows), (u32 *)dst, GX_BUFFER_DIM(width, rows),
                       GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
                           GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) |
                           GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
}

#if defined(CTR64_RA)
/* RetroArch core: the presented image as linear RGB565 for RetroArch (ra/ra_glue.c) */
void pica_untile_rgb565_async(const void *src, u16 width, u16 rows, void *dst)
{
    pica_flush();
    queue_reserve(1);
    GX_DisplayTransfer((u32 *)src, GX_BUFFER_DIM(width, rows), (u32 *)dst, GX_BUFFER_DIM(width, rows),
                       GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
                           GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) |
                           GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB565) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
}

/* RetroArch core: ctr64's GX queue is libctru's current one only inside retro_run */
void pica_ra_bind(bool on)
{
    GX_BindQueue(on ? &P.queue : NULL);
}
#endif

/* 2x2 box downscale of a tiled RGBA8 image (width x rows, multiples of 16) into a linear
 * (width/2) x (rows/2) image, queued after everything submitted so far: the transfer engine's
 * supersampling resolve, the average bilinear sampling at exactly half size gives */
void pica_downscale2_rgba8(const void *src, u16 width, u16 rows, void *dst)
{
    pica_flush();
    queue_reserve(1);
    GX_DisplayTransfer((u32 *)src, GX_BUFFER_DIM(width, rows), (u32 *)dst, GX_BUFFER_DIM(width / 2, rows / 2),
                       GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
                           GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) |
                           GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_XY));
}

/* linear RGBA8 (width x rows, multiples of 8) to tiled, queued after everything submitted so far */
void pica_tile_rgba8(const void *src, u16 width, u16 rows, void *dst)
{
    pica_flush();
    queue_reserve(1);
    GX_DisplayTransfer((u32 *)src, GX_BUFFER_DIM(width, rows), (u32 *)dst, GX_BUFFER_DIM(width, rows),
                       GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(1) | GX_TRANSFER_RAW_COPY(0) |
                           GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) |
                           GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
}

u64 pica_fence(void)
{
    return queue_enqueued();
}

void pica_wait_fence(u64 fence)
{
    wait_fence(fence);
}

void pica_reserve(u32 cmd_bytes, size_t pool_bytes)
{
    Slot *s = &P.slots[P.cur];
    u32 cmd_used = (u32)(gpuCmdBuf - s->cmd + gpuCmdBufOffset) * 4;

    if (cmd_used + cmd_bytes + 256 <= CMDBUF_BYTES && s->pool_used + pool_bytes + 128 <= POOL_BYTES)
        return;
    /* Out of room: hand everything to the GPU, wait for it, then reuse the slot from the start.
     * The GPU keeps its register state, so the shadow state stays valid. */
    submit_commands();
    wait_fence(queue_enqueued());
    GPUCMD_SetBuffer(s->cmd, CMDBUF_BYTES / 4, 0);
    s->pool_used = s->pool_flushed = 0;
}

void *pica_frame_alloc(size_t size, size_t align)
{
    Slot *s = &P.slots[P.cur];
    size_t at = (s->pool_used + align - 1) & ~(align - 1);

    if (at + size > POOL_BYTES)
        return NULL;
    s->pool_used = at + size;
    return s->pool + at;
}

/* ---- render targets ---------------------------------------------------------------------- */

static u32 color_bytes_per_pixel(GPU_COLORBUF fmt)
{
    switch (fmt) {
    case GPU_RB_RGBA8: return 4;
    case GPU_RB_RGB8: return 3;
    default: return 2;
    }
}

static u32 depth_bytes_per_pixel(GPU_DEPTHBUF fmt)
{
    switch (fmt) {
    case GPU_RB_DEPTH24_STENCIL8: return 4;
    case GPU_RB_DEPTH24: return 3;
    default: return 2;
    }
}

bool pica_target_create(PicaTarget *t, u16 width, u16 height, GPU_COLORBUF color_fmt,
                        bool with_depth, vramAllocPos bank)
{
    memset(t, 0, sizeof(*t));
    t->width = width;
    t->height = height;
    t->color_fmt = color_fmt;
    t->depth_fmt = GPU_RB_DEPTH24_STENCIL8;
    t->color = vramAllocAt((u32)width * height * color_bytes_per_pixel(color_fmt), bank);
    if (!t->color)
        return false;
    if (with_depth) {
        u32 size = (u32)width * height * depth_bytes_per_pixel(t->depth_fmt);
        /* Opposite bank first: color and depth then stream from different VRAM banks. */
        t->depth = vramAllocAt(size, bank ^ VRAM_ALLOC_ANY);
        if (!t->depth)
            t->depth = vramAllocAt(size, VRAM_ALLOC_ANY);
        if (!t->depth) {
            vramFree(t->color);
            t->color = NULL;
            return false;
        }
    }
    return true;
}

static bool is_vram(const void *p);

/* A color-only target in FCRAM (linear memory): for one drawn once per frame, leaving the VRAM to
 * the game's frame buffers */
bool pica_target_create_linear(PicaTarget *t, u16 width, u16 height, GPU_COLORBUF color_fmt)
{
    memset(t, 0, sizeof(*t));
    t->width = width;
    t->height = height;
    t->color_fmt = color_fmt;
    t->depth_fmt = GPU_RB_DEPTH24_STENCIL8;
    const u32 bytes = (u32)width * height * color_bytes_per_pixel(color_fmt);
    t->color = linearMemAlign(bytes, 0x80);
    if (!t->color)
        return false;
    memset(t->color, 0, bytes);
    pica_cache_flush(t->color, bytes);
    return true;
}

void pica_target_destroy(PicaTarget *t)
{
    if (t->color && !is_vram(t->color))
        linearFree(t->color);
    else if (t->color)
        vramFree(t->color);
    if (t->depth)
        vramFree(t->depth);
    memset(t, 0, sizeof(*t));
}

static u16 fill_control(u32 bytes_per_pixel)
{
    /* bit 0 starts the fill; bits 8-9 select 16/24/32-bit pixels */
    return BIT(0) | ((bytes_per_pixel == 4 ? 2 : bytes_per_pixel == 3 ? 1 : 0) << 8);
}

/* GX_MemoryFill into FCRAM (render targets that do not fit in VRAM, e.g. anti-aliasing) is
 * tested at start-up (pica_fill_selftest): Azahar fills any memory; if the real GPU leaves FCRAM
 * untouched, FCRAM is filled by copying instead: a VRAM scratch block is filled, copied to the
 * start of the target, then the filled part is copied after itself, doubling each time. */
#define FILL_SCRATCH_BYTES (64 * 1024)
static int g_fcram_fill = 1;   /* 1: GX_MemoryFill works on FCRAM; 0: copy fallback */
static void *g_fill_scratch;   /* VRAM */
char pica_fill_report[96] = "fill self-test not run";

static bool is_vram(const void *p)
{
    return (u32)p >= 0x1F000000 && (u32)p < 0x1F600000;
}

/* fills [p, p + bytes) with value in fill_control(bpp) units, queued after the work so far */
static void fill_any(void *p, u32 bytes, u32 value, u32 bpp)
{
    if (!bytes)
        return;
    if (is_vram(p) || g_fcram_fill || !g_fill_scratch) {
        queue_reserve(1);
        GX_MemoryFill((u32 *)p, value, (u32 *)((u8 *)p + bytes), fill_control(bpp), NULL, 0, NULL, 0);
        return;
    }
    u32 first = bytes < FILL_SCRATCH_BYTES ? bytes : FILL_SCRATCH_BYTES;
    queue_reserve(2);
    GX_MemoryFill((u32 *)g_fill_scratch, value, (u32 *)((u8 *)g_fill_scratch + first), fill_control(bpp), NULL, 0, NULL, 0);
    GX_TextureCopy((u32 *)g_fill_scratch, 0, (u32 *)p, 0, first, GX_TRANSFER_RAW_COPY(1));
    for (u32 done = first; done < bytes;) {
        u32 n = done < bytes - done ? done : bytes - done;
        queue_reserve(1);
        GX_TextureCopy((u32 *)p, 0, (u32 *)((u8 *)p + done), 0, n, GX_TRANSFER_RAW_COPY(1));
        done += n;
    }
}

/* Does GX_MemoryFill reach FCRAM on this GPU? Logged in core.log (pica_fill_report). The
 * dev switch file nofcramfill.txt forces the copy fallback (to test it in Azahar). */
void pica_fill_selftest(void)
{
    const u32 bytes = 4096;
    u32 *buf = linearMemAlign(bytes, 128);
    if (!g_fill_scratch)
        g_fill_scratch = vramAlloc(FILL_SCRATCH_BYTES);
    if (!buf) {
        snprintf(pica_fill_report, sizeof(pica_fill_report), "fill self-test: no memory");
        return;
    }
    memset(buf, 0, bytes);
    pica_cache_flush(buf, bytes);
    queue_reserve(1);
    GX_MemoryFill(buf, 0x5A3C1E0Fu, buf + bytes / 4, fill_control(4), NULL, 0, NULL, 0);
    wait_fence(queue_enqueued());
    pica_cache_invalidate(buf, bytes);
    u32 good = 0;
    for (u32 i = 0; i < bytes / 4; i++)
        good += buf[i] == 0x5A3C1E0Fu;
    g_fcram_fill = good == bytes / 4;
    FILE *f = fopen("sdmc:/3ds/ctr64/nofcramfill.txt", "r");
    if (f) {
        fclose(f);
        g_fcram_fill = 0;
    }
    /* check the fallback too */
    memset(buf, 0, bytes);
    pica_cache_flush(buf, bytes);
    const int keep = g_fcram_fill;
    g_fcram_fill = 0;
    fill_any(buf, bytes, 0x0F1E3C5Au, 4);
    g_fcram_fill = keep;
    wait_fence(queue_enqueued());
    pica_cache_invalidate(buf, bytes);
    u32 good2 = 0;
    for (u32 i = 0; i < bytes / 4; i++)
        good2 += buf[i] == 0x0F1E3C5Au;
    snprintf(pica_fill_report, sizeof(pica_fill_report), "GPU fill into FCRAM: %lu/%lu words; copy fallback %lu/%lu; using %s",
             (unsigned long)good, (unsigned long)(bytes / 4), (unsigned long)good2, (unsigned long)(bytes / 4),
             g_fcram_fill ? "fill" : "copies");
    linearFree(buf);
}

void pica_clear(const PicaTarget *t, u32 rgba, bool clear_depth, u32 depth24)
{
    u32 pixels = (u32)t->width * t->height;
    u32 cbpp = color_bytes_per_pixel(t->color_fmt);
    u32 *c0 = (u32 *)t->color, *c1 = (u32 *)((u8 *)t->color + pixels * cbpp);

    /* Draws already recorded must reach the GPU before the fill engine touches the target. */
    submit_commands();
    const bool both_vram = is_vram(t->color) && (!clear_depth || !t->depth || is_vram(t->depth));
    if (clear_depth && t->depth) {
        u32 dbpp = depth_bytes_per_pixel(t->depth_fmt);
        u32 *d0 = (u32 *)t->depth, *d1 = (u32 *)((u8 *)t->depth + pixels * dbpp);
        u32 dval = t->depth_fmt == GPU_RB_DEPTH24_STENCIL8 ? (depth24 & 0xFFFFFF) : depth24;
        if (both_vram || g_fcram_fill) {
            queue_reserve(1);
            GX_MemoryFill(c0, rgba, c1, fill_control(cbpp), d0, dval, d1, fill_control(dbpp));
        } else {
            fill_any(c0, pixels * cbpp, rgba, cbpp);
            fill_any(d0, pixels * dbpp, dval, dbpp);
        }
    } else {
        fill_any(c0, pixels * cbpp, rgba, cbpp);
    }
}

/* GPU copy of bytes (a multiple of 16) from src to dst, queued after everything submitted so
 * far. VRAM must be written this way: a CPU store to VRAM is a data abort on hardware
 * (Azahar allows it). */
void pica_copy(void *dst, const void *src, u32 bytes)
{
    pica_cache_flush(src, bytes);
    submit_commands();
    queue_reserve(1);
    GX_TextureCopy((u32 *)src, 0, (u32 *)dst, 0, bytes, GX_TRANSFER_RAW_COPY(1));
}

void pica_copy_from_vram(void *dst, const void *src, u32 bytes)
{
    submit_commands();
    queue_reserve(1);
    GX_TextureCopy((u32 *)src, 0, (u32 *)dst, 0, bytes, GX_TRANSFER_RAW_COPY(1));
}

/* GPU copy of bytes from contiguous src into dst, run16 * 16 bytes at a time with gap16 * 16
 * bytes skipped in dst after each run; queued after everything submitted so far */
void pica_copy_gap(void *dst, const void *src, u32 bytes, u32 run16, u32 gap16)
{
    submit_commands();
    queue_reserve(1);
    GX_TextureCopy((u32 *)src, GX_BUFFER_DIM(run16, 0), (u32 *)dst, GX_BUFFER_DIM(run16, gap16), bytes,
                   GX_TRANSFER_RAW_COPY(1));
}

void pica_fill_zero(void *p, u32 bytes)
{
    submit_commands();
    fill_any(p, bytes, 0, 4);
}

void pica_clear_depth(const PicaTarget *t, u32 depth24)
{
    if (!t->depth)
        return;
    u32 pixels = (u32)t->width * t->height;
    u32 dbpp = depth_bytes_per_pixel(t->depth_fmt);
    u32 *d0 = (u32 *)t->depth, *d1 = (u32 *)((u8 *)t->depth + pixels * dbpp);

    (void)d1;
    submit_commands();
    fill_any(d0, pixels * dbpp, depth24 & 0xFFFFFF, dbpp);
}

/* ---- matrices ---------------------------------------------------------------------------- */

void pica_mtx_ortho(float m[4][4], float l, float r, float b, float t, float n, float f)
{
    memset(m, 0, sizeof(float) * 16);
    /* PICA clip-space depth range is [-1, 0]. */
    m[0][0] = 2.0f / (r - l);
    m[0][3] = (l + r) / (l - r);
    m[1][1] = 2.0f / (t - b);
    m[1][3] = (b + t) / (b - t);
    m[2][2] = 1.0f / (n - f);
    m[2][3] = 0.5f * (n + f) / (n - f) - 0.5f;
    m[3][3] = 1.0f;
}

void pica_mtx_ortho_tilt(float m[4][4], float l, float r, float b, float t, float n, float f)
{
    memset(m, 0, sizeof(float) * 16);
    m[0][1] = 2.0f / (t - b);
    m[0][3] = (b + t) / (b - t);
    m[1][0] = 2.0f / (l - r);
    m[1][3] = (l + r) / (r - l);
    m[2][2] = 1.0f / (n - f);
    m[2][3] = 0.5f * (n + f) / (n - f) - 0.5f;
    m[3][3] = 1.0f;
}
