# ctr64 - Nintendo 3DS frontend. Configured through the top-level CMakeLists with the 3ds-release preset.
enable_language(ASM CXX)

set(CMAKE_C_STANDARD 11)

# -DCTR64_RETROARCH_CORE=ON: the RetroArch 3DS core (libretro_ctr.a for RetroArch's Makefile.ctr)
# instead of the ctr64 app; everything is built with CTR64_RA (see src/ra/ra_glue.c)
option(CTR64_RETROARCH_CORE "build the RetroArch 3DS core instead of the app" OFF)
if(CTR64_RETROARCH_CORE)
  add_compile_definitions(CTR64_RA=1)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/core.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/gliden64.cmake)

# PICA200 vertex shaders (picasso) -> one .shbin, embedded into the executable.
# DVLE[0] = basic (UI, presentation), DVLE[1] = gfx (GLideN64 backend)
ctr_add_shader_library(ctr64_shaders shaders/basic.v.pica shaders/gfx.v.pica)
dkp_add_embedded_binary_library(ctr64_shaders_bin ctr64_shaders)

if(CTR64_RETROARCH_CORE)
  add_library(ctr64_ra STATIC
    src/ra/ra_glue.c
    src/video.c
    src/settings.c
    src/frameskip.c
    src/gfx_async.c
    src/audio_async.c
    src/rom_stream.c
    src/ctru_abi.c
    src/pica/pica.c
    src/pica/pica_state.c)
  target_include_directories(ctr64_ra PRIVATE src ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/third_party/mupen64plus-libretro-nx/custom
    ${CMAKE_SOURCE_DIR}/third_party/mupen64plus-libretro-nx/mupen64plus-core/subprojects/md5)
  target_compile_options(ctr64_ra PRIVATE -Wall -Wno-unused-parameter -O2)
  set_source_files_properties(src/ctru_abi.c PROPERTIES COMPILE_OPTIONS -fshort-enums)
  target_link_libraries(ctr64_ra PRIVATE ctr64_shaders_bin ctr64_core)
  # one archive for RetroArch: every object of the core, GLideN64, the glue and the shaders
  set(RA_LIB ${CMAKE_CURRENT_BINARY_DIR}/libretro_ctr.a)
  add_custom_command(OUTPUT ${RA_LIB}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${RA_LIB}
    COMMAND sh -c "printf 'CREATE ${RA_LIB}\\nADDLIB $<TARGET_FILE:ctr64_core>\\nADDLIB $<TARGET_FILE:ctr64_gliden64>\\nADDLIB $<TARGET_FILE:ctr64_ra>\\nSAVE\\nEND\\n' | ${CMAKE_AR} -M"
    DEPENDS ctr64_core ctr64_gliden64 ctr64_ra
    VERBATIM)
  add_custom_target(retroarch_core ALL DEPENDS ${RA_LIB})
  return()
endif()

add_executable(ctr64
  src/main.c
  src/prof.c
  src/bench.c
  src/memcpy_prof.c
  src/crash.c
  src/frameskip.c
  src/gfx_async.c
  src/audio_async.c
  src/menu.c
  src/settings.c
  src/audio_ndsp.c
  src/core_host.c
  src/ctru_abi.c
  src/gpu_test.c
  src/rom_browser.c
  src/rom_title.c
  src/rom_stream.c
  src/ui/ui.c
  src/video.c
  src/pica/pica.c
  src/pica/pica_state.c)

target_include_directories(ctr64 PRIVATE src ${CMAKE_SOURCE_DIR}/common)
# gfx_async.c uses the core's plugin headers (custom/ holds mupen64plus-next_common.h)
set_source_files_properties(src/gfx_async.c src/audio_async.c PROPERTIES INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/third_party/mupen64plus-libretro-nx/custom)
# rom_stream.c computes the ROM's MD5 with the core's md5.c
set_source_files_properties(src/rom_stream.c PROPERTIES INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/third_party/mupen64plus-libretro-nx/mupen64plus-core/subprojects/md5)
# libctru's own struct layouts (1-byte enums); see src/ctru_abi.c
set_source_files_properties(src/ctru_abi.c PROPERTIES COMPILE_OPTIONS -fshort-enums)
target_compile_options(ctr64 PRIVATE -Wall -Wextra -Wno-unused-parameter -O2)
target_link_libraries(ctr64 PRIVATE ctr64_shaders_bin ctr64_gliden64 ctr64_core ctr64_gliden64 stdc++)
if(CTR64_PGO STREQUAL "gen") # instrumented compiler (core.cmake)
  target_link_libraries(ctr64 PRIVATE gcov)
  target_compile_definitions(ctr64 PRIVATE CTR64_PGO_GEN=1)
endif()
# The core uses int-sized enums (see core.cmake); libctru's structs shared with ctr64 code
# do not depend on enum storage size, so the mixed-enum-size link is intentional.
target_link_options(ctr64 PRIVATE -Wl,--no-enum-size-warning -Wl,--wrap=abort -Wl,--wrap=__system_allocateHeaps -Wl,--wrap=_sbrk_r)
# development: bytes copied by memcpy per caller (src/memcpy_prof.c, sdmc:/3ds/ctr64/memcpy.log)
option(CTR64_MEMCPY_PROF "count memcpy bytes per caller" OFF)
if(CTR64_MEMCPY_PROF)
  target_compile_definitions(ctr64 PRIVATE CTR64_MEMCPY_PROF=1)
  target_link_options(ctr64 PRIVATE -Wl,--wrap=memcpy)
endif()

# Release version: here, in the CIA's title version (makerom -ver, scripts/build-3ds.ps1) and
# shown on the game list and in core.log.
set(CTR64_VERSION "0.9.1")
target_compile_definitions(ctr64 PRIVATE CTR64_VERSION="${CTR64_VERSION}")

ctr_generate_smdh(ctr64.smdh
  NAME "ctr64"
  DESCRIPTION "N64 emulator ${CTR64_VERSION}"
  AUTHOR "ctr64"
  ICON "${CMAKE_CURRENT_SOURCE_DIR}/cia/icon.png")  # 48x48, from tools/icon/make_final.py

ctr_create_3dsx(ctr64 SMDH ctr64.smdh)
