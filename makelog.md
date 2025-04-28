  nvidia@tegra-ubuntu:~/driveHalo-py/Open3D/build$ make -j$(nproc)
[  0%] Creating directories for 'ext_turbojpeg'
[  0%] Creating directories for 'ext_jsoncpp'
[  0%] Creating directories for 'ext_assimp'
[  0%] Creating directories for 'open3d_downloads'
[  0%] Creating directories for 'ext_filament'
[  0%] Creating directories for 'ext_tbb'
[  0%] Creating directories for 'ext_zlib'
[  0%] Creating directories for 'ext_openblas'
[  1%] Performing download step (git clone) for 'ext_assimp'
[  1%] Performing download step (git clone) for 'ext_jsoncpp'
[  1%] No download step for 'ext_turbojpeg'
[  1%] Performing download step (git clone) for 'ext_filament'
[  1%] Performing download step (git clone) for 'ext_zlib'
[  1%] Performing download step for 'open3d_downloads'
[  1%] Performing download step (git clone) for 'ext_tbb'
[  1%] Performing download step (git clone) for 'ext_openblas'
[  1%] No update step for 'ext_turbojpeg'
Cloning into 'ext_assimp'...
Cloning into 'ext_jsoncpp'...
Cloning into 'ext_zlib'...
Cloning into 'ext_filament'...
Cloning into 'ext_tbb'...
Cloning into 'ext_openblas'...
[  1%] No patch step for 'ext_turbojpeg'
[  2%] Performing configure step for 'ext_turbojpeg'
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- CMAKE_BUILD_TYPE = Release
-- VERSION = 2.0.3, BUILD = 20250428
-- 64-bit build (arm64)
-- CMAKE_INSTALL_PREFIX = /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install
-- CMAKE_INSTALL_BINDIR = bin (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin)
-- CMAKE_INSTALL_DATAROOTDIR = share (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share)
-- CMAKE_INSTALL_DOCDIR = share/doc/libjpeg-turbo (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo)
-- CMAKE_INSTALL_INCLUDEDIR = include (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include)
-- CMAKE_INSTALL_LIBDIR = lib (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/lib)
-- CMAKE_INSTALL_MANDIR = share/man (/home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man)
-- Shared libraries disabled (ENABLE_SHARED = 0)
-- Static libraries enabled (ENABLE_STATIC = 1)
-- 12-bit JPEG support disabled (WITH_12BIT = 0)
-- Arithmetic decoding support enabled (WITH_ARITH_DEC = 1)
-- Arithmetic encoding support enabled (WITH_ARITH_ENC = 1)
-- TurboJPEG API library enabled (WITH_TURBOJPEG = 1)
-- TurboJPEG Java wrapper disabled (WITH_JAVA = 0)
-- In-memory source/destination managers enabled (WITH_MEM_SRCDST = 1)
-- Emulating libjpeg API/ABI v6.2 (WITH_JPEG7 = 0, WITH_JPEG8 = 0)
-- libjpeg API shared library version = 62.3.0
-- Compiler flags =  -O3 -DNDEBUG
-- Linker flags =  
-- Looking for sys/types.h
-- Looking for sys/types.h - found
-- Looking for stdint.h
-- Looking for stdint.h - found
-- Looking for stddef.h
-- Looking for stddef.h - found
-- Check size of size_t
-- Check size of size_t - done
-- Check size of unsigned long
-- Check size of unsigned long - done
-- Performing Test HAVE_BUILTIN_CTZL
-- Performing Test HAVE_BUILTIN_CTZL - Success
-- Looking for include file locale.h
-- Looking for include file locale.h - found
-- Looking for include file stdlib.h
-- Looking for include file stdlib.h - found
-- Looking for include file sys/types.h
-- Looking for include file sys/types.h - found
-- Looking for memset
-- Looking for memset - found
-- Looking for memcpy
-- Looking for memcpy - found
-- Check size of unsigned char
-- Check size of unsigned char - done
-- Check size of unsigned short
-- Check size of unsigned short - done
-- Performing Test INCOMPLETE_TYPES
-- Performing Test INCOMPLETE_TYPES - Success
-- Compiler supports pointers to undefined structures.
-- Performing Test RIGHT_SHIFT_IS_UNSIGNED
-- Performing Test RIGHT_SHIFT_IS_UNSIGNED - Failed
-- Performing Test __CHAR_UNSIGNED__
HEAD is now at 9059f5c Roll version numbers for 1.9.4 release (#1223)
[  2%] No update step for 'ext_jsoncpp'
[  2%] Performing patch step for 'ext_jsoncpp'
-- Performing Test __CHAR_UNSIGNED__ - Success
-- Performing Test INLINE_WORKS
[  2%] Performing configure step for 'ext_jsoncpp'
-- Performing Test INLINE_WORKS - Success
-- INLINE = __inline__ __attribute__((always_inline)) (FORCE_INLINE = 1)
-- Performing Test HAVE_VERSION_SCRIPT
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Performing Test HAVE_VERSION_SCRIPT - Success
-- Linker supports GNU-style version scripts
-- CMAKE_EXECUTABLE_SUFFIX = 
-- SIMD extensions: None (WITH_SIMD = 0)
-- FLOATTEST = 64bit
-- RPM architecture = aarch64, DEB architecture = arm64
-- Configuring done (1.7s)
-- Detecting CXX compiler ABI info - done
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_CXX_COMPILER
    WITH_CRT_DLL


-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/turbojpeg/src/ext_turbojpeg-build
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- JsonCpp Version: 1.9.4
-- Looking for C++ include clocale
[  2%] Performing build step for 'ext_turbojpeg'
[  1%] Building C object CMakeFiles/simd.dir/jsimd_none.c.o
-- Looking for C++ include clocale - found
-- Looking for localeconv
[  1%] Built target simd
[  2%] Building C object CMakeFiles/rdjpgcom.dir/rdjpgcom.c.o
-- Looking for localeconv - found
-- Looking for C++ include sys/types.h
-- Looking for C++ include sys/types.h - found
-- Looking for C++ include stdint.h
[  2%] Linking C executable rdjpgcom
[  2%] Built target rdjpgcom
[  3%] Building C object CMakeFiles/wrjpgcom.dir/wrjpgcom.c.o
-- Looking for C++ include stdint.h - found
-- Looking for C++ include stddef.h
-- Looking for C++ include stddef.h - found
-- Check size of lconv
[  4%] Linking C executable wrjpgcom
[  4%] Built target wrjpgcom
[  4%] Building C object md5/CMakeFiles/md5cmp.dir/md5cmp.c.o
-- Check size of lconv - done
-- Performing Test HAVE_DECIMAL_POINT
[  5%] Building C object md5/CMakeFiles/md5cmp.dir/md5.c.o
HEAD is now at cacf7f1 zlib 1.2.11
[  2%] No update step for 'ext_zlib'
-- Performing Test HAVE_DECIMAL_POINT - Success
-- Configuring done (1.0s)
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_C_COMPILER


[  2%] No patch step for 'ext_zlib'
-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/src/ext_jsoncpp-build
[  2%] Performing build step for 'ext_jsoncpp'
[  2%] Performing configure step for 'ext_zlib'
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


[  5%] Building C object md5/CMakeFiles/md5cmp.dir/md5hl.c.o
[ 25%] Building CXX object src/lib_json/CMakeFiles/jsoncpp_static.dir/json_reader.cpp.o
-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
[  6%] Linking C executable md5cmp
[  6%] Built target md5cmp
[  6%] Building C object CMakeFiles/jpeg-static.dir/jcapimin.c.o
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Looking for sys/types.h
[  7%] Building C object CMakeFiles/jpeg-static.dir/jcapistd.c.o
-- Looking for sys/types.h - found
-- Looking for stdint.h
[  8%] Building C object CMakeFiles/jpeg-static.dir/jccoefct.c.o
-- Looking for stdint.h - found
-- Looking for stddef.h
-- Looking for stddef.h - found
-- Check size of off64_t
[  8%] Building C object CMakeFiles/jpeg-static.dir/jccolor.c.o
-- Check size of off64_t - done
-- Looking for fseeko
-- Looking for fseeko - found
-- Looking for unistd.h
-- Looking for unistd.h - found
-- Renaming
--     /home/nvidia/driveHalo-py/Open3D/build/zlib/src/ext_zlib/zconf.h
-- to 'zconf.h.included' because this file is included with zlib
-- but CMake generates it automatically in the build directory.
-- Configuring done (0.9s)
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_CXX_COMPILER


-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/zlib/src/ext_zlib-build
[  3%] Performing build step for 'ext_zlib'
[  2%] Building C object CMakeFiles/zlib.dir/adler32.o
[  5%] Building C object CMakeFiles/zlib.dir/compress.o
[  7%] Building C object CMakeFiles/zlib.dir/crc32.o
[  9%] Building C object CMakeFiles/jpeg-static.dir/jcdctmgr.c.o
[ 10%] Building C object CMakeFiles/zlib.dir/deflate.o
[ 10%] Building C object CMakeFiles/jpeg-static.dir/jchuff.c.o
[ 12%] Building C object CMakeFiles/zlib.dir/gzclose.o
[ 15%] Building C object CMakeFiles/zlib.dir/gzlib.o
[ 17%] Building C object CMakeFiles/zlib.dir/gzread.o
[ 20%] Building C object CMakeFiles/zlib.dir/gzwrite.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/RGBD/normal_map.npy
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/normal_map.npy
HEAD is now at 141b0e3 quoting fixes, quenched warnings on MSVC
[  3%] No update step for 'ext_tbb'
[ 22%] Building C object CMakeFiles/zlib.dir/inflate.o
[  3%] Performing patch step for 'ext_tbb'
HEAD is now at 141b0e3 quoting fixes, quenched warnings on MSVC
[  3%] Performing configure step for 'ext_tbb'
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- The CXX compiler identification is GNU 9.4.0
[ 50%] Building CXX object src/lib_json/CMakeFiles/jsoncpp_static.dir/json_value.cpp.o
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test SUPPORTS_MRTM
-- Performing Test SUPPORTS_MRTM - Failed
-- Performing Test SUPPORTS_FLIFETIME
-- Performing Test SUPPORTS_FLIFETIME - Success
-- Configuring done (0.5s)
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_C_COMPILER
    TBB_BUILD_TBBMALLOC_PROXYC


-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/tbb/src/ext_tbb-build
[  3%] Performing build step for 'ext_tbb'
[ 25%] Building C object CMakeFiles/zlib.dir/infback.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/RGBD/raycast_vtx_004.npy
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/raycast_vtx_004.npy
[  2%] Building CXX object CMakeFiles/tbb_static.dir/src/old/concurrent_queue_v2.cpp.o
[ 27%] Building C object CMakeFiles/zlib.dir/inftrees.o
[ 30%] Building C object CMakeFiles/zlib.dir/inffast.o
[ 32%] Building C object CMakeFiles/zlib.dir/trees.o
[  4%] Building CXX object CMakeFiles/tbb_static.dir/src/old/concurrent_vector_v2.cpp.o
[ 35%] Building C object CMakeFiles/zlib.dir/uncompr.o
[ 37%] Building C object CMakeFiles/zlib.dir/zutil.o
[ 40%] Linking C shared library libz.so
[ 40%] Built target zlib
[ 42%] Building C object CMakeFiles/zlibstatic.dir/adler32.o
[  6%] Building CXX object CMakeFiles/tbb_static.dir/src/old/spin_rw_mutex_v2.cpp.o
[ 45%] Building C object CMakeFiles/zlibstatic.dir/compress.o
[ 47%] Building C object CMakeFiles/zlibstatic.dir/crc32.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/tests/bunnyData.pts
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/bunnyData.pts
[ 10%] Building C object CMakeFiles/jpeg-static.dir/jcicc.c.o
[ 50%] Building C object CMakeFiles/zlibstatic.dir/deflate.o
[ 11%] Building C object CMakeFiles/jpeg-static.dir/jcinit.c.o
[ 12%] Building C object CMakeFiles/jpeg-static.dir/jcmainct.c.o
[  8%] Building CXX object CMakeFiles/tbb_static.dir/src/old/task_v2.cpp.o
[ 12%] Building C object CMakeFiles/jpeg-static.dir/jcmarker.c.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/RGBD/vertex_map.npy
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/vertex_map.npy
[ 10%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/arena.cpp.o
[ 75%] Building CXX object src/lib_json/CMakeFiles/jsoncpp_static.dir/json_writer.cpp.o
[ 13%] Building C object CMakeFiles/jpeg-static.dir/jcmaster.c.o
[ 52%] Building C object CMakeFiles/zlibstatic.dir/gzclose.o
[ 55%] Building C object CMakeFiles/zlibstatic.dir/gzlib.o
[ 14%] Building C object CMakeFiles/jpeg-static.dir/jcomapi.c.o
[ 14%] Building C object CMakeFiles/jpeg-static.dir/jcparam.c.o
[ 57%] Building C object CMakeFiles/zlibstatic.dir/gzread.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/tests/point_cloud_sample1.pts
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/point_cloud_sample1.pts
[ 15%] Building C object CMakeFiles/jpeg-static.dir/jcphuff.c.o
[ 60%] Building C object CMakeFiles/zlibstatic.dir/gzwrite.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/tests/point_cloud_sample2.pts
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/point_cloud_sample2.pts
[ 62%] Building C object CMakeFiles/zlibstatic.dir/inflate.o
[ 16%] Building C object CMakeFiles/jpeg-static.dir/jcprepct.c.o
[ 16%] Building C object CMakeFiles/jpeg-static.dir/jcsample.c.o
[download_utils.py] Downloaded https://github.com/intel-isl/open3d_downloads/raw/master/tests/cube.obj
        to /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/cube.obj
[  3%] Performing update step for 'open3d_downloads'
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/raycast_vtx_004.npy already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/normal_map.npy already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/bunnyData.pts already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/point_cloud_sample1.pts already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/point_cloud_sample2.pts already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/tests/cube.obj already exists, skipped.
[download_utils.py] /home/nvidia/driveHalo-py/Open3D/examples/test_data/open3d_downloads/RGBD/vertex_map.npy already exists, skipped.
[  3%] No patch step for 'open3d_downloads'
[  3%] No configure step for 'open3d_downloads'
[ 17%] Building C object CMakeFiles/jpeg-static.dir/jctrans.c.o
[ 12%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/cache_aligned_allocator.cpp.o
[  4%] No build step for 'open3d_downloads'
[  4%] No install step for 'open3d_downloads'
[  4%] Completed 'open3d_downloads'
[  4%] Built target open3d_downloads
[ 18%] Building C object CMakeFiles/jpeg-static.dir/jdapimin.c.o
[ 19%] Building C object CMakeFiles/turbojpeg-static.dir/jcapimin.c.o
[ 65%] Building C object CMakeFiles/zlibstatic.dir/infback.o
[ 19%] Building C object CMakeFiles/jpeg-static.dir/jdapistd.c.o
[ 20%] Building C object CMakeFiles/turbojpeg-static.dir/jcapistd.c.o
[ 20%] Building C object CMakeFiles/turbojpeg-static.dir/jccoefct.c.o
[ 21%] Building C object CMakeFiles/jpeg-static.dir/jdatadst.c.o
[ 67%] Building C object CMakeFiles/zlibstatic.dir/inftrees.o
[ 22%] Building C object CMakeFiles/jpeg-static.dir/jdatasrc.c.o
[ 14%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/concurrent_hash_map.cpp.o
[ 23%] Building C object CMakeFiles/turbojpeg-static.dir/jccolor.c.o
[ 23%] Building C object CMakeFiles/jpeg-static.dir/jdcoefct.c.o
[ 70%] Building C object CMakeFiles/zlibstatic.dir/inffast.o
[100%] Linking CXX static library ../../lib/libjsoncpp.a
[ 72%] Building C object CMakeFiles/zlibstatic.dir/trees.o
[ 24%] Building C object CMakeFiles/jpeg-static.dir/jdcolor.c.o
[100%] Built target jsoncpp_static
[  4%] Performing install step for 'ext_jsoncpp'
[100%] Built target jsoncpp_static
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/lib/pkgconfig/jsoncpp.pc
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/lib/cmake/jsoncpp/jsoncppConfig.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/lib/cmake/jsoncpp/jsoncppConfig-release.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/lib/cmake/jsoncpp/jsoncppConfigVersion.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/lib/libjsoncpp.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/allocator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/assertions.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/config.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/forwards.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/json.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/json_features.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/reader.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/value.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/version.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/jsoncpp/include/json/writer.h
[  4%] Completed 'ext_jsoncpp'
[ 16%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/concurrent_monitor.cpp.o
[  4%] Built target ext_jsoncpp
[ 25%] Building C object CMakeFiles/jpeg-static.dir/jddctmgr.c.o
[ 26%] Building C object CMakeFiles/turbojpeg-static.dir/jcdctmgr.c.o
[ 75%] Building C object CMakeFiles/zlibstatic.dir/uncompr.o
[ 26%] Building C object CMakeFiles/turbojpeg-static.dir/jchuff.c.o
[ 77%] Building C object CMakeFiles/zlibstatic.dir/zutil.o
[ 80%] Linking C static library libz.a
[ 80%] Built target zlibstatic
[ 82%] Building C object CMakeFiles/example.dir/test/example.o
[  5%] Creating directories for 'ext_stdgpu'
[  5%] Performing download step (git clone) for 'ext_stdgpu'
Cloning into 'ext_stdgpu'...
[ 18%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/concurrent_queue.cpp.o
[ 85%] Linking C executable example
[ 85%] Built target example
[ 87%] Building C object CMakeFiles/minigzip.dir/test/minigzip.o
[ 26%] Building C object CMakeFiles/jpeg-static.dir/jdhuff.c.o
[ 90%] Linking C executable minigzip
[ 90%] Built target minigzip
[ 92%] Building C object CMakeFiles/example64.dir/test/example.o
[ 95%] Linking C executable example64
[ 95%] Built target example64
[ 97%] Building C object CMakeFiles/minigzip64.dir/test/minigzip.o
[100%] Linking C executable minigzip64
[100%] Built target minigzip64
[  5%] Performing install step for 'ext_zlib'
[ 27%] Building C object CMakeFiles/jpeg-static.dir/jdicc.c.o
[ 40%] Built target zlib
[ 20%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/concurrent_vector.cpp.o
[ 80%] Built target zlibstatic
[ 85%] Built target example
[ 90%] Built target minigzip
[ 95%] Built target example64
[100%] Built target minigzip64
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/lib/libz.so.1.2.11
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/lib/libz.so.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/lib/libz.so
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/lib/libz.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/include/zconf.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/include/zlib.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/share/man/man3/zlib.3
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/zlib/share/pkgconfig/zlib.pc
[  5%] Completed 'ext_zlib'
[  5%] Built target ext_zlib
[ 22%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/condition_variable.cpp.o
[ 28%] Building C object CMakeFiles/jpeg-static.dir/jdinput.c.o
[ 28%] Building C object CMakeFiles/jpeg-static.dir/jdmainct.c.o
[ 29%] Building C object CMakeFiles/jpeg-static.dir/jdmarker.c.o
[ 30%] Building C object CMakeFiles/jpeg-static.dir/jdmaster.c.o
[ 25%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/critical_section.cpp.o
[ 30%] Building C object CMakeFiles/jpeg-static.dir/jdmerge.c.o
[ 31%] Building C object CMakeFiles/jpeg-static.dir/jdphuff.c.o
[ 27%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/dynamic_link.cpp.o
[ 29%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/governor.cpp.o
[ 32%] Building C object CMakeFiles/jpeg-static.dir/jdpostct.c.o
[ 32%] Building C object CMakeFiles/jpeg-static.dir/jdsample.c.o
[ 31%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbbmalloc/backend.cpp.o
[ 33%] Building C object CMakeFiles/jpeg-static.dir/jdtrans.c.o
[ 34%] Building C object CMakeFiles/jpeg-static.dir/jerror.c.o
[ 34%] Building C object CMakeFiles/jpeg-static.dir/jfdctflt.c.o
[ 35%] Building C object CMakeFiles/jpeg-static.dir/jfdctfst.c.o
[ 36%] Building C object CMakeFiles/jpeg-static.dir/jfdctint.c.o
[ 33%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/itt_notify.cpp.o
[ 36%] Building C object CMakeFiles/jpeg-static.dir/jidctflt.c.o
[ 37%] Building C object CMakeFiles/jpeg-static.dir/jidctfst.c.o
[ 38%] Building C object CMakeFiles/jpeg-static.dir/jidctint.c.o
[ 35%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/market.cpp.o
[ 39%] Building C object CMakeFiles/turbojpeg-static.dir/jcicc.c.o
[ 37%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbbmalloc/large_objects.cpp.o
[ 40%] Building C object CMakeFiles/turbojpeg-static.dir/jcinit.c.o
[ 40%] Building C object CMakeFiles/turbojpeg-static.dir/jcmainct.c.o
[ 41%] Building C object CMakeFiles/turbojpeg-static.dir/jcmarker.c.o
[ 42%] Building C object CMakeFiles/turbojpeg-static.dir/jcmaster.c.o
[ 42%] Building C object CMakeFiles/turbojpeg-static.dir/jcomapi.c.o
[ 43%] Building C object CMakeFiles/turbojpeg-static.dir/jcparam.c.o
[ 39%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/mutex.cpp.o
[ 41%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbbmalloc/backref.cpp.o
[ 44%] Building C object CMakeFiles/turbojpeg-static.dir/jcphuff.c.o
[ 43%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/observer_proxy.cpp.o
[ 44%] Building C object CMakeFiles/turbojpeg-static.dir/jcprepct.c.o
[ 45%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbbmalloc/tbbmalloc.cpp.o
[ 45%] Building C object CMakeFiles/turbojpeg-static.dir/jcsample.c.o
[ 45%] Building C object CMakeFiles/jpeg-static.dir/jidctred.c.o
[ 46%] Building C object CMakeFiles/jpeg-static.dir/jquant1.c.o
[ 47%] Building C object CMakeFiles/turbojpeg-static.dir/jctrans.c.o
[ 47%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbbmalloc/frontend.cpp.o
[ 47%] Building C object CMakeFiles/turbojpeg-static.dir/jdapimin.c.o
[ 48%] Building C object CMakeFiles/turbojpeg-static.dir/jdapistd.c.o
[ 49%] Building C object CMakeFiles/jpeg-static.dir/jquant2.c.o
[ 50%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/pipeline.cpp.o
[ 50%] Building C object CMakeFiles/turbojpeg-static.dir/jdatadst.c.o
[ 50%] Building C object CMakeFiles/turbojpeg-static.dir/jdatasrc.c.o
[ 51%] Building C object CMakeFiles/turbojpeg-static.dir/jdcoefct.c.o
[ 51%] Building C object CMakeFiles/jpeg-static.dir/jutils.c.o
[ 52%] Building C object CMakeFiles/turbojpeg-static.dir/jdcolor.c.o
[ 53%] Building C object CMakeFiles/jpeg-static.dir/jmemmgr.c.o
[ 54%] Building C object CMakeFiles/jpeg-static.dir/jmemnobs.c.o
[ 54%] Building C object CMakeFiles/jpeg-static.dir/jaricom.c.o
[ 52%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/private_server.cpp.o
[ 55%] Building C object CMakeFiles/jpeg-static.dir/jcarith.c.o
HEAD is now at 6005da9 unordered_map,unordered_set: Improve performance of insert and erase
[ 54%] Building CXX object CMakeFiles/tbbmalloc_static.dir/src/tbb/itt_notify.cpp.o
[  5%] No update step for 'ext_stdgpu'
[  5%] No patch step for 'ext_stdgpu'
[ 56%] Building C object CMakeFiles/jpeg-static.dir/jdarith.c.o
[  5%] Performing configure step for 'ext_stdgpu'
loading initial cache file /home/nvidia/driveHalo-py/Open3D/build/stdgpu/tmp/ext_stdgpu-cache-Release.cmake
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
[ 56%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/queuing_mutex.cpp.o
[ 58%] Linking CXX static library libtbbmalloc_static.a
[ 56%] Linking C static library libjpeg.a
[ 58%] Built target tbbmalloc_static
[ 60%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/queuing_rw_mutex.cpp.o
[ 56%] Built target jpeg-static
[ 57%] Building C object CMakeFiles/turbojpeg-static.dir/jddctmgr.c.o
[ 57%] Building C object CMakeFiles/turbojpeg-static.dir/jdhuff.c.o
[ 58%] Building C object CMakeFiles/turbojpeg-static.dir/jdicc.c.o
[ 58%] Building C object CMakeFiles/turbojpeg-static.dir/jdinput.c.o
[ 62%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/reader_writer_lock.cpp.o
[ 64%] Building CXX object CMakeFiles/tbbmalloc_proxy_static.dir/src/tbbmalloc/proxy.cpp.o
[ 59%] Building C object CMakeFiles/turbojpeg-static.dir/jdmainct.c.o
[ 66%] Building CXX object CMakeFiles/tbbmalloc_proxy_static.dir/src/tbbmalloc/tbb_function_replacement.cpp.o
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/context.c.o
[ 60%] Building C object CMakeFiles/turbojpeg-static.dir/jdmarker.c.o
-- The CUDA compiler identification is NVIDIA 11.4.315
-- Detecting CUDA compiler ABI info
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/init.c.o
[ 68%] Linking CXX static library libtbbmalloc_proxy_static.a
[ 68%] Built target tbbmalloc_proxy_static
[ 60%] Building C object CMakeFiles/cjpeg-static.dir/cjpeg.c.o
[ 70%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/recursive_mutex.cpp.o
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/input.c.o
[ 60%] Building C object CMakeFiles/turbojpeg-static.dir/jdmaster.c.o
[ 61%] Building C object CMakeFiles/cjpeg-static.dir/cdjpeg.c.o
[ 61%] Building C object CMakeFiles/cjpeg-static.dir/rdgif.c.o
[ 63%] Building C object CMakeFiles/turbojpeg-static.dir/jdmerge.c.o
[ 63%] Building C object CMakeFiles/cjpeg-static.dir/rdppm.c.o
[ 72%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/scheduler.cpp.o
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/monitor.c.o
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Appended optimization flag (-O3,/O2) implicitly
-- Created device flags : $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wsign-compare,-Wconversion,-Wfloat-equal>
-- Created test device flags : $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>
-- Detected user-provided CCs : 87-real
-- Created host flags : $<$<COMPILE_LANGUAGE:CXX>:-Wall;-pedantic;-Wextra;-Wshadow;-Wsign-compare;-Wconversion;-Wfloat-equal;-Wundef;-Wdouble-promotion;-O3>
-- Created test host flags : $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
-- Found thrust: /usr/local/cuda/targets/aarch64-linux/include (found suitable version "1.12.1", minimum required is "1.9.2")
-- Found CUDAToolkit: /usr/local/cuda/targets/aarch64-linux/include (found suitable version "11.4.315", minimum required is "10.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
[ 64%] Building C object CMakeFiles/cjpeg-static.dir/rdswitch.c.o
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
[ 65%] Building C object CMakeFiles/turbojpeg-static.dir/jdphuff.c.o
[ 65%] Building C object CMakeFiles/cjpeg-static.dir/rdbmp.c.o
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/vulkan.c.o
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/window.c.o
[ 65%] Building C object CMakeFiles/turbojpeg-static.dir/jdpostct.c.o
-- Could NOT find Doxygen (missing: DOXYGEN_EXECUTABLE) (Required is at least version "1.9.1")
-- 
-- ************************ stdgpu Configuration Summary *************************
-- 
-- General:
--   Version                                   :   1.3.0
--   System                                    :   Linux
--   Build type                                :   Release
-- 
-- Build:
--   STDGPU_BACKEND                            :   STDGPU_BACKEND_CUDA
--   STDGPU_BUILD_SHARED_LIBS                  :   OFF
--   STDGPU_SETUP_COMPILER_FLAGS               :   ON
--   STDGPU_TREAT_WARNINGS_AS_ERRORS           :   OFF
--   STDGPU_ANALYZE_WITH_CLANG_TIDY            :   OFF
--   STDGPU_ANALYZE_WITH_CPPCHECK              :   OFF
-- 
-- Configuration:
--   STDGPU_ENABLE_CONTRACT_CHECKS             :   OFF
--   STDGPU_USE_32_BIT_INDEX                   :   ON
-- 
-- Examples:
--   STDGPU_BUILD_EXAMPLES                     :   OFF
-- 
-- Tests:
--   STDGPU_BUILD_TESTS                        :   OFF
--   STDGPU_BUILD_TEST_COVERAGE                :   OFF
-- 
-- Documentation:
--   Doxygen                                   :   NO
-- 
-- *******************************************************************************
-- 
-- Configuring done (2.6s)
-- Generating done (0.0s)
-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/src/ext_stdgpu-build
[  6%] Performing build step for 'ext_stdgpu'
[ 67%] Building C object CMakeFiles/cjpeg-static.dir/rdtarga.c.o
[ 67%] Building C object CMakeFiles/turbojpeg-static.dir/jdsample.c.o
[ 20%] Building CXX object src/stdgpu/CMakeFiles/stdgpu.dir/impl/iterator.cpp.o
[ 68%] Linking C executable cjpeg-static
[ 68%] Built target cjpeg-static
[ 40%] Building CXX object src/stdgpu/CMakeFiles/stdgpu.dir/impl/memory.cpp.o
[ 69%] Building C object CMakeFiles/turbojpeg-static.dir/jdtrans.c.o
[  6%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/x11_init.c.o
[ 69%] Building C object CMakeFiles/turbojpeg-static.dir/jerror.c.o
[ 70%] Building C object CMakeFiles/turbojpeg-static.dir/jfdctflt.c.o
[ 71%] Building C object CMakeFiles/turbojpeg-static.dir/jfdctfst.c.o
[ 71%] Building C object CMakeFiles/turbojpeg-static.dir/jfdctint.c.o
[ 72%] Building C object CMakeFiles/turbojpeg-static.dir/jidctflt.c.o
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/x11_monitor.c.o
[ 73%] Building C object CMakeFiles/turbojpeg-static.dir/jidctfst.c.o
[ 73%] Building C object CMakeFiles/turbojpeg-static.dir/jidctint.c.o
[ 75%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/semaphore.cpp.o
[ 74%] Building C object CMakeFiles/turbojpeg-static.dir/jidctred.c.o
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/x11_window.c.o
[ 74%] Building C object CMakeFiles/djpeg-static.dir/djpeg.c.o
[ 77%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/spin_mutex.cpp.o
[ 60%] Building CXX object src/stdgpu/CMakeFiles/stdgpu.dir/impl/limits.cpp.o
[ 75%] Building C object CMakeFiles/djpeg-static.dir/cdjpeg.c.o
[ 76%] Building C object CMakeFiles/djpeg-static.dir/rdcolmap.c.o
[ 80%] Building CXX object src/stdgpu/CMakeFiles/stdgpu.dir/cuda/impl/memory.cpp.o
[ 76%] Building C object CMakeFiles/djpeg-static.dir/rdswitch.c.o
[ 77%] Building C object CMakeFiles/djpeg-static.dir/wrgif.c.o
[ 79%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/spin_rw_mutex.cpp.o
[ 78%] Building C object CMakeFiles/djpeg-static.dir/wrppm.c.o
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/xkb_unicode.c.o
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/posix_time.c.o
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/posix_thread.c.o
[ 78%] Building C object CMakeFiles/djpeg-static.dir/wrbmp.c.o
[100%] Linking CXX static library libstdgpu.a
[100%] Built target stdgpu
[  7%] Performing install step for 'ext_stdgpu'
[ 81%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/task.cpp.o
[100%] Built target stdgpu
[  7%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/glx_context.c.o
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/cuda/FindCUDAToolkit.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/libstdgpu.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/atomic.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/unordered_map.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/unordered_set.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/queue_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/stack_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/bitset_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/memory.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/atomic.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/memory.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/platform_check.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/impl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/impl/atomic_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cuda/platform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/utility.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/mutex_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/stack.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/queue.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/vector_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/unordered_map_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/contract.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/unordered_set_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/algorithm.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cstddef.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/vector.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/memory.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/platform_check.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/impl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/impl/atomic_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/atomic.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/hip/platform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/atomic_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/type_traits.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/bit_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/platform_check.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/mutex_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/ranges_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/limits_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/unordered_base_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/stack_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/functional_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/utility_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/memory_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/unordered_set_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/deque_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/queue_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/vector_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/unordered_map_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/iterator_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/bitset_detail.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/unordered_base.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/algorithm_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/impl/cmath_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/compiler.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/atomic_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/attribute.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/functional.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/platform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/deque_fwd
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/limits.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/cmath.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/ranges.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/deque.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/iterator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/bitset.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/mutex.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/memory.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/platform_check.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/impl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/impl/atomic_detail.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/atomic.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/openmp/platform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/bit.h
-- Up-to-date: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include
-- Up-to-date: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/include/stdgpu/config.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/stdgpu-dependencies.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/Findthrust.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/stdgpu-targets.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/stdgpu-targets-release.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/stdgpu-config.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/stdgpu/lib/cmake/stdgpu/stdgpu-config-version.cmake
[  7%] Completed 'ext_stdgpu'
[  7%] Built target ext_stdgpu
[  8%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/egl_context.c.o
[ 79%] Building C object CMakeFiles/djpeg-static.dir/wrtarga.c.o
[  8%] Building CXX object cpp/tools/CMakeFiles/EncodeShader.dir/EncodeShader.cpp.o
[ 80%] Building C object CMakeFiles/turbojpeg-static.dir/jquant1.c.o
[ 81%] Linking C executable djpeg-static
[ 81%] Built target djpeg-static
[  8%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/osmesa_context.c.o
[ 82%] Building C object CMakeFiles/jpegtran-static.dir/jpegtran.c.o
[  8%] Building C object 3rdparty/GLFW/src/CMakeFiles/glfw.dir/linux_joystick.c.o
[ 83%] Building C object CMakeFiles/jpegtran-static.dir/cdjpeg.c.o
[ 83%] Building C object CMakeFiles/turbojpeg-static.dir/jquant2.c.o
[  8%] Linking CXX executable ../../bin/EncodeShader
[ 83%] Building C object CMakeFiles/jpegtran-static.dir/rdswitch.c.o
[  8%] Built target EncodeShader
[ 84%] Building C object CMakeFiles/jpegtran-static.dir/transupp.c.o
In file included from /usr/include/string.h:495,
                 from /home/nvidia/driveHalo-py/Open3D/3rdparty/GLFW/src/linux_joystick.c:38:
In function ‘strncpy’,
    inlined from ‘openJoystickDevice’ at /home/nvidia/driveHalo-py/Open3D/3rdparty/GLFW/src/linux_joystick.c:231:5:
/usr/include/aarch64-linux-gnu/bits/string_fortified.h:106:10: warning: ‘__builtin_strncpy’ specified bound 4096 equals destination size [-Wstringop-truncation]
  106 |   return __builtin___strncpy_chk (__dest, __src, __len, __bos (__dest));
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[ 83%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/task_group_context.cpp.o
[  8%] Linking C static library ../../../lib/Release/libglfw3.a
[  8%] Building C object CMakeFiles/3rdparty_glew.dir/3rdparty/glew/src/glew.c.o
[  8%] Built target glfw
[  8%] Building C object CMakeFiles/3rdparty_lzf.dir/3rdparty/liblzf/liblzf/lzf_c.c.o
[  8%] Building C object CMakeFiles/3rdparty_lzf.dir/3rdparty/liblzf/liblzf/lzf_d.c.o
[  8%] Linking C static library lib/Release/libOpen3D_3rdparty_lzf.a
[  8%] Built target 3rdparty_lzf
[ 85%] Building C object CMakeFiles/turbojpeg-static.dir/jutils.c.o
[  8%] Creating directories for 'ext_libpng'
[  8%] Performing download step (git clone) for 'ext_libpng'
Cloning into 'ext_libpng'...
[ 86%] Building C object CMakeFiles/turbojpeg-static.dir/jmemmgr.c.o
[ 86%] Building C object CMakeFiles/turbojpeg-static.dir/jmemnobs.c.o
[ 87%] Building C object CMakeFiles/turbojpeg-static.dir/jaricom.c.o
[ 85%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/tbb_main.cpp.o
[ 88%] Building C object CMakeFiles/turbojpeg-static.dir/jcarith.c.o
[ 88%] Building C object CMakeFiles/turbojpeg-static.dir/jdarith.c.o
[ 89%] Linking C executable jpegtran-static
[ 89%] Built target jpegtran-static
[ 87%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/tbb_misc.cpp.o
[ 89%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/tbb_misc_ex.cpp.o
[ 90%] Building C object CMakeFiles/turbojpeg-static.dir/turbojpeg.c.o
[ 91%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/tbb_statistics.cpp.o
[ 93%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/tbb_thread.cpp.o
[ 91%] Building C object CMakeFiles/turbojpeg-static.dir/transupp.c.o
[ 95%] Building CXX object CMakeFiles/tbb_static.dir/src/tbb/x86_rtm_rw_mutex.cpp.o
[ 97%] Building CXX object CMakeFiles/tbb_static.dir/src/rml/client/rml_tbb.cpp.o
[  8%] Building C object CMakeFiles/3rdparty_rply.dir/3rdparty/rply/rply/rply.c.o
HEAD is now at 63b03efc2 Merge pull request #2667 from xianyi/develop
[100%] Linking CXX static library libtbb_static.a
[  8%] No update step for 'ext_openblas'
[  9%] No patch step for 'ext_openblas'
[  9%] No configure step for 'ext_openblas'
[  9%] Performing build step for 'ext_openblas'
[100%] Built target tbb_static
[ 10%] Performing install step for 'ext_tbb'
[ 10%] Linking C static library lib/Release/libOpen3D_3rdparty_rply.a
[ 79%] Built target tbb_static
[ 91%] Building C object CMakeFiles/turbojpeg-static.dir/jdatadst-tj.c.o
[ 93%] Built target tbbmalloc_static
[ 10%] Built target 3rdparty_rply
[100%] Built target tbbmalloc_proxy_static
Install the project...
[ 10%] Building C object CMakeFiles/3rdparty_tinyfiledialogs.dir/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c.o
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/lib/libtbb_static.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/lib/libtbbmalloc_static.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/lib/libtbbmalloc_proxy_static.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/aligned_space.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/info.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_lru_cache.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_do.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_allocator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_invoke.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_scan.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_body_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_item_buffer_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_concurrent_unordered_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_tbb_strings.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_trace_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_tbb_windef.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_x86_eliding_mutex_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_node_set_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_types_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_warning_suppress_enable_notice.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_allocator_traits.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_concurrent_queue_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_indexer_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_range_iterator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_aggregator_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_node_handle_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_deprecated_header_message_guard.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_node_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_template_helpers.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_x86_rtm_rw_mutex_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_concurrent_skip_list_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_tbb_hash_compare_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_cache_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_streaming_node.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_tagged_buffer_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_tbb_trace_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_nodes_deduction.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_async_msg_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_warning_suppress_disable_notice.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_flow_graph_join_impl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/internal/_mutex_padding.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_config.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_stddef.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/task_scheduler_init.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_for.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/iterators.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/scalable_allocator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/queuing_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/partitioner.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/combinable.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/flow_graph_opencl_node.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/task_group.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_while.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/enumerable_thread_specific.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_map.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/global_control.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_hash_map.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/flow_graph_abstractions.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/reader_writer_lock.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/blocked_range3d.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/null_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/index.html
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/atomic.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/cache_aligned_allocator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/spin_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/memory_pool.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_profiling.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/blocked_range.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_vector.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/critical_section.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_machine.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/task_scheduler_observer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/recursive_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/aggregator.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_priority_queue.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tick_count.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_disable_exceptions.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_set.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/blocked_range2d.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_exception.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_reduce.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/blocked_rangeNd.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbbmalloc_proxy.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/runtime_loader.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/spin_rw_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/pipeline.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_unordered_set.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_for_each.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/parallel_sort.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_queue.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/windows_api.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/gcc_generic.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/icc_generic.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/linux_ia32.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/windows_ia32.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/gcc_ia32_common.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/windows_intel64.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/msvc_armv7.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/sunos_sparc.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/msvc_ia32_common.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/mac_ppc.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/linux_common.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/macos_common.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/gcc_arm.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/linux_intel64.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/ibm_aix51.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/mic_common.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/linux_ia64.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/machine/gcc_itsx.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/compat
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/compat/tuple
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/compat/condition_variable
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/compat/ppl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/compat/thread
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/null_rw_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/queuing_rw_mutex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/tbb_thread.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/task.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/flow_graph.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/concurrent_unordered_map.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/tbb/task_arena.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/lib/cmake/tbb/TBBConfigVersion.cmake
[ 10%] Completed 'ext_tbb'
[ 10%] Built target ext_tbb
[ 10%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/global_r.c.o
[ 92%] Building C object CMakeFiles/turbojpeg-static.dir/jdatasrc-tj.c.o
[ 93%] Building C object CMakeFiles/turbojpeg-static.dir/rdbmp.c.o
OpenBLAS: Detecting fortran compiler failed. Cannot compile LAPACK. Only compile BLAS.
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
/home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c: In function ‘tkinter2Present’:
/home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:3161:34: warning: ‘%s’ directive writing up to 255 bytes into a region of size between 240 and 255 [-Wformat-overflow=]
 3161 |   sprintf ( lPythonCommand , "%s %s" , gPython2Name , lPythonParams ) ;
      |                                  ^~                   ~~~~~~~~~~~~~
In file included from /usr/include/stdio.h:867,
                 from /home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:87:
/usr/include/aarch64-linux-gnu/bits/stdio2.h:36:10: note: ‘__builtin___sprintf_chk’ output between 2 and 272 bytes into a destination of size 256
   36 |   return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   37 |       __bos (__s), __fmt, __va_arg_pack ());
      |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:3168:34: warning: ‘%s’ directive writing up to 255 bytes into a region of size between 240 and 255 [-Wformat-overflow=]
 3168 |   sprintf ( lPythonCommand , "%s %s" , gPython2Name , lPythonParams ) ;
      |                                  ^~                   ~~~~~~~~~~~~~
In file included from /usr/include/stdio.h:867,
                 from /home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:87:
/usr/include/aarch64-linux-gnu/bits/stdio2.h:36:10: note: ‘__builtin___sprintf_chk’ output between 2 and 272 bytes into a destination of size 256
   36 |   return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   37 |       __bos (__s), __fmt, __va_arg_pack ());
      |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:3178:34: warning: ‘%s’ directive writing up to 255 bytes into a region of size between 240 and 255 [-Wformat-overflow=]
 3178 |   sprintf ( lPythonCommand , "%s %s" , gPython2Name , lPythonParams ) ;
      |                                  ^~                   ~~~~~~~~~~~~~
In file included from /usr/include/stdio.h:867,
                 from /home/nvidia/driveHalo-py/Open3D/3rdparty/tinyfiledialogs/include/tinyfiledialogs/tinyfiledialogs.c:87:
/usr/include/aarch64-linux-gnu/bits/stdio2.h:36:10: note: ‘__builtin___sprintf_chk’ output between 2 and 272 bytes into a destination of size 256
   36 |   return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   37 |       __bos (__s), __fmt, __va_arg_pack ());
      |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[ 93%] Building C object CMakeFiles/turbojpeg-static.dir/rdppm.c.o
[ 10%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/stat_r.c.o
[ 94%] Building C object CMakeFiles/turbojpeg-static.dir/wrbmp.c.o
[ 10%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/geom2_r.c.o
[ 95%] Building C object CMakeFiles/turbojpeg-static.dir/wrppm.c.o
[ 10%] Linking C static library lib/Release/libOpen3D_3rdparty_tinyfiledialogs.a
[ 10%] Built target 3rdparty_tinyfiledialogs
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/poly2_r.c.o
[ 95%] Linking C static library libturbojpeg.a
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/merge_r.c.o
[ 95%] Built target turbojpeg-static
[ 96%] Building C object CMakeFiles/tjunittest-static.dir/tjunittest.c.o
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/libqhull_r.c.o
[ 96%] Building C object CMakeFiles/tjunittest-static.dir/tjutil.c.o
[ 97%] Building C object CMakeFiles/tjunittest-static.dir/md5/md5.c.o
[ 98%] Building C object CMakeFiles/tjunittest-static.dir/md5/md5hl.c.o
[ 98%] Linking C executable tjunittest-static
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/geom_r.c.o
[ 98%] Built target tjunittest-static
[ 99%] Building C object CMakeFiles/tjbench-static.dir/tjbench.c.o
[ 99%] Building C object CMakeFiles/tjbench-static.dir/tjutil.c.o
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/poly_r.c.o
[ 11%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/qset_r.c.o
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/mem_r.c.o
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/random_r.c.o
[100%] Linking C executable tjbench-static
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/usermem_r.c.o
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/userprintf_r.c.o
[100%] Built target tjbench-static
[ 12%] Linking C static library lib/Release/libOpen3D_3rdparty_glew.a
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/user_r.c.o
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/io_r.c.o
[ 12%] Performing install step for 'ext_turbojpeg'
[  1%] Built target simd
[ 12%] Built target 3rdparty_glew
[  2%] Built target rdjpgcom
[ 12%] Building CXX object CMakeFiles/3rdparty_imgui.dir/3rdparty/imgui/imgui_demo.cpp.o
[ 12%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/rboxlib_r.c.o
[  4%] Built target wrjpgcom
[  6%] Built target md5cmp
[ 40%] Built target jpeg-static
[ 80%] Built target turbojpeg-static
[ 85%] Built target cjpeg-static
[ 91%] Built target djpeg-static
[ 95%] Built target jpegtran-static
[ 98%] Built target tjunittest-static
[100%] Built target tjbench-static
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/lib/libturbojpeg.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/tjbench
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include/turbojpeg.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/lib/libjpeg.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/cjpeg
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/djpeg
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/jpegtran
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/rdjpgcom
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/bin/wrjpgcom
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/README.ijg
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/README.md
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/example.txt
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/tjexample.c
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/libjpeg.txt
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/structure.txt
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/usage.txt
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/wizard.txt
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/doc/libjpeg-turbo/LICENSE.md
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man/man1/cjpeg.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man/man1/djpeg.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man/man1/jpegtran.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man/man1/rdjpgcom.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/share/man/man1/wrjpgcom.1
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/lib/pkgconfig/libjpeg.pc
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/lib/pkgconfig/libturbojpeg.pc
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include/jconfig.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include/jerror.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include/jmorecfg.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libjpeg-turbo-install/include/jpeglib.h
[ 12%] Completed 'ext_turbojpeg'
[ 12%] Built target ext_turbojpeg
[ 12%] Generating /home/nvidia/driveHalo-py/Open3D/cpp/open3d/visualization/shader/Shader.h
[ 12%] Built target ShaderFileTarget
[ 12%] Building CXX object CMakeFiles/3rdparty_imgui.dir/3rdparty/imgui/imgui_draw.cpp.o
[ 12%] Building CXX object CMakeFiles/3rdparty_imgui.dir/3rdparty/imgui/imgui_widgets.cpp.o
[ 13%] Building C object CMakeFiles/3rdparty_qhull_r.dir/3rdparty/qhull/src/libqhull_r/userprintf_rbox_r.c.o
[ 13%] Linking C static library lib/Release/libOpen3D_3rdparty_qhull_r.a
[ 13%] Built target 3rdparty_qhull_r
[ 13%] Building CXX object CMakeFiles/3rdparty_imgui.dir/3rdparty/imgui/imgui.cpp.o
[ 13%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/Coordinates.cpp.o
[ 13%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/PointCoordinates.cpp.o
[ 13%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/Qhull.cpp.o
[ 13%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullFacet.cpp.o
[ 13%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullFacetList.cpp.o
ar: `u' modifier ignored since `D' is the default (see `U')
ar: creating ../CUSTOM_LIB_NAME
[ 14%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullFacetSet.cpp.o
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
[ 14%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullHyperplane.cpp.o
[ 14%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullPoint.cpp.o
[ 14%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullPointSet.cpp.o
[ 14%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullPoints.cpp.o
[ 15%] Linking CXX static library lib/Release/libOpen3D_3rdparty_imgui.a
[ 15%] Built target 3rdparty_imgui
[ 15%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullQh.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullRidge.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullSet.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullStat.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullVertex.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/QhullVertexSet.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/RboxPoints.cpp.o
[ 16%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/RoadError.cpp.o
[ 17%] Building CXX object CMakeFiles/3rdparty_qhullcpp.dir/3rdparty/qhull/src/libqhullcpp/RoadLogEvent.cpp.o
HEAD is now at a40189cf8 Release libpng version 1.6.37
[ 17%] Linking CXX static library lib/Release/libOpen3D_3rdparty_qhullcpp.a
[ 17%] No update step for 'ext_libpng'
[ 17%] No patch step for 'ext_libpng'
[ 17%] Performing configure step for 'ext_libpng'
CMake Deprecation Warning at CMakeLists.txt:21 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at CMakeLists.txt:22 (cmake_policy):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- The C compiler identification is GNU 9.4.0
-- The ASM compiler identification is GNU
-- Found assembler: /usr/bin/cc
-- Detecting C compiler ABI info
[ 17%] Built target 3rdparty_qhullcpp
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Performing Test HAVE_LD_VERSION_SCRIPT
-- Performing Test HAVE_LD_VERSION_SCRIPT - Success
-- Symbol prefix: 
-- Configuring done (0.7s)
-- Generating done (0.0s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_CXX_COMPILER
    PNG_EXECUTABLES


-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/libpng/src/ext_libpng-build
[ 18%] Performing build step for 'ext_libpng'
[  8%] Generating scripts/pnglibconf.c
[ 10%] Generating pngprefix.h
[  8%] Generating scripts/symbols.out
[ 10%] Generating pnglibconf.c
[ 13%] Generating pnglibconf.out
[ 16%] Generating scripts/symbols.chk
[ 21%] Generating scripts/prefix.out
[ 21%] Generating pnglibconf.h
[ 27%] Generating scripts/sym.out
[ 27%] Generating scripts/vers.out
[ 29%] Generating scripts/intprefix.out
[ 32%] Generating libpng.sym
[ 35%] Generating libpng.vers
[ 35%] Built target genfiles
[ 43%] Building C object CMakeFiles/png_static.dir/pngget.c.o
[ 43%] Building C object CMakeFiles/png_static.dir/pngerror.c.o
[ 43%] Building C object CMakeFiles/png_static.dir/pngpread.c.o
[ 45%] Building C object CMakeFiles/png_static.dir/png.c.o
[ 48%] Building C object CMakeFiles/png_static.dir/pngmem.c.o
[ 51%] Building C object CMakeFiles/png_static.dir/pngread.c.o
[ 54%] Building C object CMakeFiles/png_static.dir/pngrio.c.o
[ 56%] Building C object CMakeFiles/png_static.dir/pngrtran.c.o
[ 59%] Building C object CMakeFiles/png_static.dir/pngrutil.c.o
[ 62%] Building C object CMakeFiles/png_static.dir/pngset.c.o
[ 64%] Building C object CMakeFiles/png_static.dir/pngtrans.c.o
[ 67%] Building C object CMakeFiles/png_static.dir/pngwio.c.o
[ 70%] Building C object CMakeFiles/png_static.dir/pngwrite.c.o
[ 72%] Building C object CMakeFiles/png_static.dir/pngwtran.c.o
[ 75%] Building C object CMakeFiles/png_static.dir/pngwutil.c.o
[ 78%] Building C object CMakeFiles/png_static.dir/arm/arm_init.c.o
[ 81%] Building ASM object CMakeFiles/png_static.dir/arm/filter_neon.S.o
[ 83%] Building C object CMakeFiles/png_static.dir/arm/filter_neon_intrinsics.c.o
[ 86%] Building C object CMakeFiles/png_static.dir/arm/palette_neon_intrinsics.c.o
[ 89%] Linking C static library libpng16.a
[100%] Built target png_static
[ 18%] Performing install step for 'ext_libpng'
[ 35%] Built target genfiles
[100%] Built target png_static
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/libpng16.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/libpng.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/png.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/pngconf.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/pnglibconf.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/libpng16/png.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/libpng16/pngconf.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/include/libpng16/pnglibconf.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/bin/libpng-config
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/bin/libpng16-config
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/share/man/man3/libpng.3
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/share/man/man3/libpngpf.3
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/share/man/man5/png.5
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/pkgconfig/libpng.pc
-- Up-to-date: /home/nvidia/driveHalo-py/Open3D/build/libpng/bin/libpng-config
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/pkgconfig/libpng16.pc
-- Up-to-date: /home/nvidia/driveHalo-py/Open3D/build/libpng/bin/libpng16-config
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/libpng/libpng16.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/libpng/lib/libpng/libpng16-release.cmake
[ 18%] Completed 'ext_libpng'
[ 18%] Built target ext_libpng
ar: `u' modifier ignored since `D' is the default (see `U')
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
ar: `u' modifier ignored since `D' is the default (see `U')
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
ar: `u' modifier ignored since `D' is the default (see `U')
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
ar: `u' modifier ignored since `D' is the default (see `U')

 OpenBLAS build complete. (BLAS CBLAS)

  OS               ... Linux             
  Architecture     ... arm64               
  BINARY           ... 64bit                 
  C compiler       ... GCC  (cmd & version : cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0)
  Library Name     ... CUSTOM_LIB_NAME (Multi-threading; Max num-threads is 8)

To install the library, you can run "make PREFIX=/path/to/your/installation install".

[ 18%] Performing install step for 'ext_openblas'
make[4]: warning: -j8 forced in submake: resetting jobserver mode.
Generating openblas_config.h in /home/nvidia/driveHalo-py/Open3D/build/openblas/include
Generating f77blas.h in /home/nvidia/driveHalo-py/Open3D/build/openblas/include
Generating cblas.h in /home/nvidia/driveHalo-py/Open3D/build/openblas/include
Copying the static library to /home/nvidia/driveHalo-py/Open3D/build/openblas/lib
Generating openblas.pc in /home/nvidia/driveHalo-py/Open3D/build/openblas/lib/pkgconfig
Generating OpenBLASConfig.cmake in /home/nvidia/driveHalo-py/Open3D/build/openblas/lib/cmake/openblas
Generating OpenBLASConfigVersion.cmake in /home/nvidia/driveHalo-py/Open3D/build/openblas/lib/cmake/openblas
Install OK!
[ 19%] Completed 'ext_openblas'
[ 19%] Built target ext_openblas
[ 20%] Creating directories for 'ext_faiss'
[ 20%] Performing download step (git clone) for 'ext_faiss'
Cloning into 'ext_faiss'...
HEAD is now at 8f0c6b04b Update Version.cpp
[ 20%] Performing patch-copy step for 'ext_assimp'
Copying patch files for Obj loader into assimp source
[ 21%] No update step for 'ext_assimp'
[ 21%] No patch step for 'ext_assimp'
[ 21%] Performing configure step for 'ext_assimp'
CMake Deprecation Warning at CMakeLists.txt:39 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Shared libraries disabled
-- compiling zlib from sources
CMake Deprecation Warning at contrib/zlib/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Looking for sys/types.h
-- Looking for sys/types.h - found
-- Looking for stdint.h
-- Looking for stdint.h - found
-- Looking for stddef.h
-- Looking for stddef.h - found
-- Check size of off64_t
-- Check size of off64_t - done
-- Looking for fseeko
-- Looking for fseeko - found
-- Looking for unistd.h
-- Looking for unistd.h - found
-- Build an import-only version of Assimp.
CMake Deprecation Warning at code/CMakeLists.txt:46 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Warning (dev) at /usr/share/cmake-3.29/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
  The package name passed to `find_package_handle_standard_args` (rt) does
  not match the name of the calling package (RT).  This can lead to problems
  in calling code that expects `find_package` result variables (e.g.,
  `_FOUND`) to follow a certain pattern.
Call Stack (most recent call first):
  cmake-modules/FindRT.cmake:19 (find_package_handle_standard_args)
  code/CMakeLists.txt:1013 (FIND_PACKAGE)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found rt: /usr/lib/aarch64-linux-gnu/librt.so
-- Enabled importer formats: AMF 3DS AC ASE ASSBIN B3D BVH COLLADA DXF CSM HMP IRRMESH IRR LWO LWS MD2 MD3 MD5 MDC MDL NFF NDO OFF OBJ OGRE OPENGEX PLY MS3D COB BLEND IFC XGL FBX Q3D Q3BSP RAW SIB SMD STL TERRAGEN 3D X X3D GLTF 3MF MMD STEP
-- Disabled importer formats:
-- Enabled exporter formats:
-- Disabled exporter formats: 3DS ASSBIN ASSXML COLLADA OBJ OPENGEX PLY FBX STL X X3D GLTF 3MF ASSJSON STEP
-- Configuring done (1.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp-build
[ 22%] Performing build step for 'ext_assimp'
[  0%] Building CXX object contrib/irrXML/CMakeFiles/IrrXML.dir/irrXML.cpp.o
[  0%] Building C object contrib/zlib/CMakeFiles/zlib.dir/adler32.o
[  0%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/compress.o
[  0%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/adler32.o
[  0%] Building C object contrib/zlib/CMakeFiles/zlib.dir/compress.o
[  1%] Building C object contrib/zlib/CMakeFiles/zlib.dir/crc32.o
[  1%] Building C object contrib/zlib/CMakeFiles/zlib.dir/deflate.o
[  2%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/crc32.o
[  3%] Building C object contrib/zlib/CMakeFiles/zlib.dir/gzclose.o
[  3%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/deflate.o
[  3%] Building C object contrib/zlib/CMakeFiles/zlib.dir/gzlib.o
[  4%] Building C object contrib/zlib/CMakeFiles/zlib.dir/gzread.o
[  5%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/gzclose.o
[  5%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/gzlib.o
[  6%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/gzread.o
[  6%] Building C object contrib/zlib/CMakeFiles/zlib.dir/gzwrite.o
[  6%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/gzwrite.o
[  7%] Building C object contrib/zlib/CMakeFiles/zlib.dir/infback.o
[  7%] Building C object contrib/zlib/CMakeFiles/zlib.dir/inflate.o
[  7%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/inflate.o
[  7%] Building C object contrib/zlib/CMakeFiles/zlib.dir/inftrees.o
[  8%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/infback.o
[  9%] Building C object contrib/zlib/CMakeFiles/zlib.dir/inffast.o
[  9%] Building C object contrib/zlib/CMakeFiles/zlib.dir/trees.o
[  9%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/inftrees.o
[ 10%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/inffast.o
[ 10%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/trees.o
[ 11%] Building C object contrib/zlib/CMakeFiles/zlib.dir/uncompr.o
[ 12%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/uncompr.o
[ 12%] Building C object contrib/zlib/CMakeFiles/zlibstatic.dir/zutil.o
[ 12%] Building C object contrib/zlib/CMakeFiles/zlib.dir/zutil.o
[ 13%] Linking C shared library libzlib.so
[ 13%] Built target zlib
[ 14%] Linking C static library libzlibstatic.a
[ 14%] Built target zlibstatic
[ 14%] Linking CXX static library libIrrXML.a
[ 14%] Built target IrrXML
[ 16%] Building CXX object code/CMakeFiles/assimp.dir/Common/BaseProcess.cpp.o
[ 16%] Building CXX object code/CMakeFiles/assimp.dir/CApi/CInterfaceIOWrapper.cpp.o
[ 17%] Building CXX object code/CMakeFiles/assimp.dir/Common/PostStepRegistry.cpp.o
[ 17%] Building CXX object code/CMakeFiles/assimp.dir/Common/ImporterRegistry.cpp.o
[ 17%] Building CXX object code/CMakeFiles/assimp.dir/Common/BaseImporter.cpp.o
[ 17%] Building CXX object code/CMakeFiles/assimp.dir/Common/Assimp.cpp.o
[ 17%] Building CXX object code/CMakeFiles/assimp.dir/Common/DefaultIOStream.cpp.o
[ 18%] Building CXX object code/CMakeFiles/assimp.dir/Common/DefaultIOSystem.cpp.o
[ 18%] Building CXX object code/CMakeFiles/assimp.dir/Common/ZipArchiveIOSystem.cpp.o
[ 19%] Building CXX object code/CMakeFiles/assimp.dir/Common/Importer.cpp.o
[ 19%] Building CXX object code/CMakeFiles/assimp.dir/Common/SGSpatialSort.cpp.o
[ 20%] Building CXX object code/CMakeFiles/assimp.dir/Common/VertexTriangleAdjacency.cpp.o
[ 20%] Building CXX object code/CMakeFiles/assimp.dir/Common/SpatialSort.cpp.o
[ 20%] Building CXX object code/CMakeFiles/assimp.dir/Common/SceneCombiner.cpp.o
[ 21%] Building CXX object code/CMakeFiles/assimp.dir/Common/ScenePreprocessor.cpp.o
[ 21%] Building CXX object code/CMakeFiles/assimp.dir/Common/SkeletonMeshBuilder.cpp.o
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SpatialSort.cpp: In function ‘{anonymous}::BinFloat {anonymous}::ToBinary(const ai_real&)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SpatialSort.cpp:229:23: warning: bitwise comparison always evaluates to false [-Wtautological-compare]
  229 |         else if( (-42 == (42 | (-0))) && (binValue & 0x80000000)) // -0 = 1000... binary
      |                   ~~~ ^~ ~~~~~~~~~~~
[ 22%] Building CXX object code/CMakeFiles/assimp.dir/Common/SplitByBoneCountProcess.cpp.o
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::CopySceneFlat(aiScene**, const aiScene*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:998:40: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘struct aiScene’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
  998 |     ::memcpy(*_dest,src,sizeof(aiScene));
      |                                        ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:242:8: note: ‘struct aiScene’ declared here
  242 | struct aiScene
      |        ^~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiMesh**, const aiMesh*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1069:37: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiMesh’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1069 |     ::memcpy(dest,src,sizeof(aiMesh));
      |                                     ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:53,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/mesh.h:571:8: note: ‘struct aiMesh’ declared here
  571 | struct aiMesh
      |        ^~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiAnimMesh**, const aiAnimMesh*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1108:43: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiAnimMesh’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1108 |     ::memcpy(dest, src, sizeof(aiAnimMesh));
      |                                           ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:53,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/mesh.h:416:8: note: ‘struct aiAnimMesh’ declared here
  416 | struct aiAnimMesh
      |        ^~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiTexture**, const aiTexture*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1165:40: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiTexture’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1165 |     ::memcpy(dest,src,sizeof(aiTexture));
      |                                        ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:52,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/texture.h:136:8: note: ‘struct aiTexture’ declared here
  136 | struct aiTexture {
      |        ^~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiAnimation**, const aiAnimation*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1195:42: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiAnimation’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1195 |     ::memcpy(dest,src,sizeof(aiAnimation));
      |                                          ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:57,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/anim.h:416:8: note: ‘struct aiAnimation’ declared here
  416 | struct aiAnimation {
      |        ^~~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiNodeAnim**, const aiNodeAnim*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1210:41: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiNodeAnim’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1210 |     ::memcpy(dest,src,sizeof(aiNodeAnim));
      |                                         ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:57,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/anim.h:276:8: note: ‘struct aiNodeAnim’ declared here
  276 | struct aiNodeAnim {
      |        ^~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiCamera**, const aiCamera*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1227:39: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiCamera’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1227 |     ::memcpy(dest,src,sizeof(aiCamera));
      |                                       ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:55,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/camera.h:99:8: note: ‘struct aiCamera’ declared here
   99 | struct aiCamera
      |        ^~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiLight**, const aiLight*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1239:38: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiLight’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1239 |     ::memcpy(dest,src,sizeof(aiLight));
      |                                      ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:54,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/light.h:114:8: note: ‘struct aiLight’ declared here
  114 | struct aiLight
      |        ^~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiBone**, const aiBone*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1251:37: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiBone’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1251 |     ::memcpy(dest,src,sizeof(aiBone));
      |                                     ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:53,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/mesh.h:259:8: note: ‘struct aiBone’ declared here
  259 | struct aiBone {
      |        ^~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In static member function ‘static void Assimp::SceneCombiner::Copy(aiNode**, const aiNode*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1265:37: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiNode’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 1265 |     ::memcpy(dest,src,sizeof(aiNode));
      |                                     ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:80:19: note: ‘struct aiNode’ declared here
   80 | struct ASSIMP_API aiNode
      |                   ^~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp: In instantiation of ‘void Assimp::GetArrayCopy(Type*&, ai_uint) [with Type = aiFace; ai_uint = unsigned int]’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:1089:46:   required from here
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:981:13: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘struct aiFace’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
  981 |     ::memcpy(dest, old, sizeof(Type) * num);
      |     ~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/scene.h:53,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Common/SceneCombiner.cpp:62:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/mesh.h:128:8: note: ‘struct aiFace’ declared here
  128 | struct aiFace
      |        ^~~~~~
[ 22%] Building CXX object code/CMakeFiles/assimp.dir/Common/StandardShapes.cpp.o
[ 23%] Building CXX object code/CMakeFiles/assimp.dir/Common/TargetAnimation.cpp.o
[ 23%] Building CXX object code/CMakeFiles/assimp.dir/Common/RemoveComments.cpp.o
[ 24%] Building CXX object code/CMakeFiles/assimp.dir/Common/Subdivision.cpp.o
[ 24%] Building CXX object code/CMakeFiles/assimp.dir/Common/scene.cpp.o
[ 24%] Building CXX object code/CMakeFiles/assimp.dir/Common/Bitmap.cpp.o
[ 25%] Building CXX object code/CMakeFiles/assimp.dir/Common/Version.cpp.o
[ 25%] Building CXX object code/CMakeFiles/assimp.dir/Common/CreateAnimMesh.cpp.o
[ 26%] Building CXX object code/CMakeFiles/assimp.dir/Common/simd.cpp.o
[ 26%] Building CXX object code/CMakeFiles/assimp.dir/Common/DefaultLogger.cpp.o
[ 27%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/CalcTangentsProcess.cpp.o
[ 27%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ComputeUVMappingProcess.cpp.o
[ 27%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ConvertToLHProcess.cpp.o
[ 28%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/EmbedTexturesProcess.cpp.o
[ 28%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/FindDegenerates.cpp.o
[ 29%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/FindInstancesProcess.cpp.o
[ 29%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/FindInvalidDataProcess.cpp.o
[ 30%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/FixNormalsStep.cpp.o
[ 30%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/DropFaceNormalsProcess.cpp.o
Switched to a new branch 'open3d_patch'
Branch 'open3d_patch' set up to track remote branch 'open3d_patch' from 'origin'.
[ 22%] No update step for 'ext_faiss'
[ 22%] No patch step for 'ext_faiss'
[ 22%] Performing configure step for 'ext_faiss'
loading initial cache file /home/nvidia/driveHalo-py/Open3D/build/faiss/tmp/ext_faiss-cache-Release.cmake
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
[ 30%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/GenFaceNormalsProcess.cpp.o
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
[ 31%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/GenVertexNormalsProcess.cpp.o
[ 31%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/PretransformVertices.cpp.o
[ 32%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ImproveCacheLocality.cpp.o
[ 32%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/JoinVerticesProcess.cpp.o
[ 33%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/LimitBoneWeightsProcess.cpp.o
-- The CUDA compiler identification is NVIDIA 11.4.315
-- Detecting CUDA compiler ABI info
[ 33%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/RemoveRedundantMaterials.cpp.o
[ 34%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/RemoveVCProcess.cpp.o
[ 34%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/SortByPTypeProcess.cpp.o
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Building with OpenMP
-- Building faiss with BLAS from source
-- FAISS_BLAS_TARGET: 
ERRORInvalid BLAS Target
-- Found CUDAToolkit: /usr/local/cuda/targets/aarch64-linux/include (found version "11.4.315")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
[ 34%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/SplitLargeMeshes.cpp.o
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
[ 35%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/TextureTransform.cpp.o
[ 35%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/TriangulateProcess.cpp.o
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
[ 36%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ValidateDataStructure.cpp.o
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring done (4.1s)
-- Generating done (0.0s)
-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/faiss/src/ext_faiss-build
[ 22%] Performing build step for 'ext_faiss'
[  0%] Building CXX object faiss/CMakeFiles/faiss.dir/Clustering.cpp.o
[  1%] Building CXX object faiss/CMakeFiles/faiss.dir/IVFlib.cpp.o
[ 36%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/OptimizeGraph.cpp.o
[ 37%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/OptimizeMeshes.cpp.o
[  2%] Building CXX object faiss/CMakeFiles/faiss.dir/Index.cpp.o
[ 37%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/DeboneProcess.cpp.o
[  2%] Building CXX object faiss/CMakeFiles/faiss.dir/Index2Layer.cpp.o
[ 38%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ProcessHelper.cpp.o
[ 38%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/MakeVerboseFormat.cpp.o
[  3%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexBinary.cpp.o
[ 38%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/ScaleProcess.cpp.o
[  4%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexBinaryFlat.cpp.o
[  4%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexBinaryFromFloat.cpp.o
[ 39%] Building CXX object code/CMakeFiles/assimp.dir/PostProcessing/GenBoundingBoxesProcess.cpp.o
[ 39%] Building CXX object code/CMakeFiles/assimp.dir/Material/MaterialSystem.cpp.o
[ 40%] Building CXX object code/CMakeFiles/assimp.dir/Importer/STEPParser/STEPFileReader.cpp.o
[  5%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexBinaryHash.cpp.o
[  6%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexBinaryIVF.cpp.o
[ 40%] Building CXX object code/CMakeFiles/assimp.dir/Importer/STEPParser/STEPFileEncoding.cpp.o
[  6%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexFlat.cpp.o
[ 40%] Building CXX object code/CMakeFiles/assimp.dir/AMF/AMFImporter.cpp.o
[ 41%] Building CXX object code/CMakeFiles/assimp.dir/AMF/AMFImporter_Geometry.cpp.o
[ 41%] Building CXX object code/CMakeFiles/assimp.dir/AMF/AMFImporter_Material.cpp.o
[  7%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVF.cpp.o
[  8%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVFFlat.cpp.o
[  9%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVFPQ.cpp.o
[  9%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVFPQR.cpp.o
[ 10%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVFPQFastScan.cpp.o
[ 42%] Building CXX object code/CMakeFiles/assimp.dir/AMF/AMFImporter_Postprocess.cpp.o
[ 11%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexPQFastScan.cpp.o
[ 11%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexIVFSpectralHash.cpp.o
[ 42%] Building CXX object code/CMakeFiles/assimp.dir/3DS/3DSConverter.cpp.o
[ 43%] Building CXX object code/CMakeFiles/assimp.dir/3DS/3DSLoader.cpp.o
[ 43%] Building CXX object code/CMakeFiles/assimp.dir/AC/ACLoader.cpp.o
[ 12%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexLSH.cpp.o
[ 13%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexLattice.cpp.o
[ 13%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexPQ.cpp.o
[ 14%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexPreTransform.cpp.o
[ 15%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexReplicas.cpp.o
[ 15%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexShards.cpp.o
[ 16%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexScalarQuantizer.cpp.o
[ 17%] Building CXX object faiss/CMakeFiles/faiss.dir/IndexRefine.cpp.o
[ 18%] Building CXX object faiss/CMakeFiles/faiss.dir/MatrixStats.cpp.o
[ 18%] Building CXX object faiss/CMakeFiles/faiss.dir/MetaIndexes.cpp.o
[ 19%] Building CXX object faiss/CMakeFiles/faiss.dir/VectorTransform.cpp.o
[ 20%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/AuxIndexStructures.cpp.o
[ 44%] Building CXX object code/CMakeFiles/assimp.dir/ASE/ASELoader.cpp.o
[ 20%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/FaissException.cpp.o
[ 44%] Building CXX object code/CMakeFiles/assimp.dir/ASE/ASEParser.cpp.o
[ 44%] Building CXX object code/CMakeFiles/assimp.dir/Assbin/AssbinLoader.cpp.o
[ 21%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/PolysemousTraining.cpp.o
[ 45%] Building CXX object code/CMakeFiles/assimp.dir/B3D/B3DImporter.cpp.o
[ 22%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/ProductQuantizer.cpp.o
[ 23%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/ScalarQuantizer.cpp.o
[ 23%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/pq4_fast_scan.cpp.o
[ 24%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/pq4_fast_scan_search_1.cpp.o
[ 45%] Building CXX object code/CMakeFiles/assimp.dir/BVH/BVHLoader.cpp.o
[ 46%] Building CXX object code/CMakeFiles/assimp.dir/Collada/ColladaLoader.cpp.o
[ 25%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/pq4_fast_scan_search_qbs.cpp.o
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/ASE/ASEParser.cpp: In member function ‘void Assimp::ASE::Parser::ParseLV1SoftSkinBlock()’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/ASE/ASEParser.cpp:149:39: warning: ‘%s’ directive output may be truncated writing up to 1023 bytes into a region of size between 1007 and 1016 [-Wformat-truncation=]
  149 |     ai_snprintf(szTemp,1024,"Line %u: %s",iLineNumber,szWarn);
      |                                       ^~
......
  840 |         LogWarning(szBuffer);
      |                    ~~~~~~~~            
In file included from /usr/include/stdio.h:867,
                 from /usr/include/c++/9/cstdio:42,
                 from /usr/include/c++/9/ext/string_conversions.h:43,
                 from /usr/include/c++/9/bits/basic_string.h:6496,
                 from /usr/include/c++/9/string:55,
                 from /usr/include/c++/9/stdexcept:39,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/Exceptional.h:44,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/BaseImporter.h:47,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/PostProcessing/TextureTransform.h:47,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/ASE/ASEParser.cpp:52:
/usr/include/aarch64-linux-gnu/bits/stdio2.h:67:35: note: ‘__builtin___snprintf_chk’ output between 9 and 1041 bytes into a destination of size 1024
   67 |   return __builtin___snprintf_chk (__s, __n, __USE_FORTIFY_LEVEL - 1,
      |          ~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   68 |        __bos (__s), __fmt, __va_arg_pack ());
      |        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[ 46%] Building CXX object code/CMakeFiles/assimp.dir/Collada/ColladaParser.cpp.o
[ 47%] Building CXX object code/CMakeFiles/assimp.dir/DXF/DXFLoader.cpp.o
[ 47%] Building CXX object code/CMakeFiles/assimp.dir/CSM/CSMLoader.cpp.o
[ 48%] Building CXX object code/CMakeFiles/assimp.dir/HMP/HMPLoader.cpp.o
[ 48%] Building CXX object code/CMakeFiles/assimp.dir/Irr/IRRMeshLoader.cpp.o
[ 48%] Building CXX object code/CMakeFiles/assimp.dir/Irr/IRRShared.cpp.o
[ 49%] Building CXX object code/CMakeFiles/assimp.dir/Irr/IRRLoader.cpp.o
[ 25%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/io.cpp.o
[ 49%] Building CXX object code/CMakeFiles/assimp.dir/LWO/LWOAnimation.cpp.o
[ 50%] Building CXX object code/CMakeFiles/assimp.dir/LWO/LWOBLoader.cpp.o
[ 50%] Building CXX object code/CMakeFiles/assimp.dir/LWO/LWOLoader.cpp.o
[ 26%] Building CXX object faiss/CMakeFiles/faiss.dir/impl/lattice_Zn.cpp.o
[ 50%] Building CXX object code/CMakeFiles/assimp.dir/LWO/LWOMaterial.cpp.o
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/LWO/LWOMaterial.cpp: In member function ‘bool Assimp::LWOImporter::HandleTextures(aiMaterial*, const TextureList&, aiTextureType)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/LWO/LWOMaterial.cpp:141:9: warning: ‘mapping’ may be used uninitialized in this function [-Wmaybe-uninitialized]
  141 |         if (mapping != aiTextureMapping_UV) {
      |         ^~
[ 51%] Building CXX object code/CMakeFiles/assimp.dir/LWS/LWSLoader.cpp.o
[ 27%] Building CXX object faiss/CMakeFiles/faiss.dir/invlists/DirectMap.cpp.o
[ 51%] Building CXX object code/CMakeFiles/assimp.dir/MD2/MD2Loader.cpp.o
[ 52%] Building CXX object code/CMakeFiles/assimp.dir/MD3/MD3Loader.cpp.o
[ 27%] Building CXX object faiss/CMakeFiles/faiss.dir/invlists/InvertedLists.cpp.o
[ 28%] Building CXX object faiss/CMakeFiles/faiss.dir/invlists/BlockInvertedLists.cpp.o
[ 29%] Building CXX object faiss/CMakeFiles/faiss.dir/invlists/InvertedListsIOHook.cpp.o
[ 30%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/Heap.cpp.o
[ 30%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/WorkerThread.cpp.o
[ 31%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/distances.cpp.o
[ 32%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/distances_simd.cpp.o
[ 52%] Building CXX object code/CMakeFiles/assimp.dir/MD5/MD5Loader.cpp.o
[ 32%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/extra_distances.cpp.o
[ 53%] Building CXX object code/CMakeFiles/assimp.dir/MD5/MD5Parser.cpp.o
[ 33%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/hamming.cpp.o
[ 53%] Building CXX object code/CMakeFiles/assimp.dir/MDC/MDCLoader.cpp.o
[ 34%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/partitioning.cpp.o
[ 34%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/quantize_lut.cpp.o
[ 54%] Building CXX object code/CMakeFiles/assimp.dir/MDL/MDLLoader.cpp.o
[ 54%] Building CXX object code/CMakeFiles/assimp.dir/MDL/MDLMaterialLoader.cpp.o
[ 35%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/random.cpp.o
[ 36%] Building CXX object faiss/CMakeFiles/faiss.dir/utils/utils.cpp.o
[ 37%] Building CXX object faiss/CMakeFiles/faiss.dir/invlists/OnDiskInvertedLists.cpp.o
[ 37%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuDistance.cu.o
[ 54%] Building CXX object code/CMakeFiles/assimp.dir/NFF/NFFLoader.cpp.o
[ 55%] Building CXX object code/CMakeFiles/assimp.dir/NDO/NDOLoader.cpp.o
[ 38%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndex.cu.o
[ 39%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexBinaryFlat.cu.o
[ 55%] Building CXX object code/CMakeFiles/assimp.dir/OFF/OFFLoader.cpp.o
[ 39%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexFlat.cu.o
[ 56%] Building CXX object code/CMakeFiles/assimp.dir/Obj/ObjFileImporter.cpp.o
[ 40%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexIVF.cu.o
[ 41%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexIVFFlat.cu.o
[ 56%] Building CXX object code/CMakeFiles/assimp.dir/Obj/ObjFileMtlImporter.cpp.o
[ 57%] Building CXX object code/CMakeFiles/assimp.dir/Obj/ObjFileParser.cpp.o
[ 57%] Building CXX object code/CMakeFiles/assimp.dir/Ogre/OgreImporter.cpp.o
[ 57%] Building CXX object code/CMakeFiles/assimp.dir/Ogre/OgreStructs.cpp.o
[ 41%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexIVFPQ.cu.o
[ 42%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/GpuIndexIVFScalarQuantizer.cu.o
[ 58%] Building CXX object code/CMakeFiles/assimp.dir/Ogre/OgreBinarySerializer.cpp.o
[ 58%] Building CXX object code/CMakeFiles/assimp.dir/Ogre/OgreXmlSerializer.cpp.o
[ 43%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/GpuResources.cpp.o
[ 44%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/StandardGpuResources.cpp.o
[ 44%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/BinaryDistance.cu.o
[ 59%] Building CXX object code/CMakeFiles/assimp.dir/Ogre/OgreMaterial.cpp.o
[ 45%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/BinaryFlatIndex.cu.o
[ 46%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/BroadcastSum.cu.o
[ 46%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/Distance.cu.o
[ 59%] Building CXX object code/CMakeFiles/assimp.dir/OpenGEX/OpenGEXImporter.cpp.o
[ 60%] Building CXX object code/CMakeFiles/assimp.dir/Ply/PlyLoader.cpp.o
[ 47%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/FlatIndex.cu.o
[ 60%] Building CXX object code/CMakeFiles/assimp.dir/Ply/PlyParser.cpp.o
[ 60%] Building CXX object code/CMakeFiles/assimp.dir/MS3D/MS3DLoader.cpp.o
[ 48%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/impl/InterleavedCodes.cpp.o
[ 61%] Building CXX object code/CMakeFiles/assimp.dir/COB/COBLoader.cpp.o
[ 48%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFAppend.cu.o
[ 49%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFBase.cu.o
[ 50%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFFlat.cu.o
[ 61%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderLoader.cpp.o
[ 62%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderDNA.cpp.o
[ 51%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFFlatScan.cu.o
[ 51%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFInterleaved.cu.o
[ 52%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFPQ.cu.o
[ 62%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderScene.cpp.o
[ 53%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFUtils.cu.o
[ 53%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFUtilsSelect1.cu.o
[ 54%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/IVFUtilsSelect2.cu.o
[ 55%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/L2Norm.cu.o
[ 63%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderModifier.cpp.o
[ 63%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderBMesh.cpp.o
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Blender/BlenderBMesh.cpp:49:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Blender/BlenderScene.h: In member function ‘void Assimp::BlenderBMeshConverter::AddFace(int, int, int, int)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Blender/BlenderScene.h:217:8: warning: ‘face.Assimp::Blender::MFace::flag’ may be used uninitialized in this function [-Wmaybe-uninitialized]
  217 | struct MFace : ElemBase {
      |        ^~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/Blender/BlenderBMesh.cpp:179:11: note: ‘face.Assimp::Blender::MFace::flag’ was declared here
  179 |     MFace face;
      |           ^~~~
[ 64%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderTessellator.cpp.o
[ 64%] Building CXX object code/CMakeFiles/assimp.dir/Blender/BlenderCustomData.cpp.o
[ 55%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/L2Select.cu.o
[ 64%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCLoader.cpp.o
[ 56%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/PQScanMultiPassPrecomputed.cu.o
[ 65%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCReaderGen1_2x3.cpp.o
[ 57%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/impl/RemapIndices.cpp.o
[ 58%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/VectorResidual.cu.o
[ 58%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved1.cu.o
HEAD is now at d1d873d27 Merge branch 'release' at v1.9.19 of https://github.com/google/filament
[ 22%] No update step for 'ext_filament'
[ 22%] No patch step for 'ext_filament'
[ 22%] Performing configure step for 'ext_filament'
-- The C compiler identification is Clang 7.0.1
-- The CXX compiler identification is Clang 7.0.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/clang-7 - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/clang++-7 - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE
-- The ASM compiler identification is Clang with GNU-like command-line
-- Found assembler: /usr/bin/clang-7
-- DFG LUT size set to 128x128
CMake Deprecation Warning at third_party/spirv-tools/CMakeLists.txt:15 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Found Python3: /usr/bin/python3.9 (found version "3.9.5") found components: Interpreter
CMake Deprecation Warning at third_party/spirv-tools/external/spirv-headers/CMakeLists.txt:31 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Warning (dev) at third_party/glslang/tnt/CMakeLists.txt:61 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: /usr/bin/python3 (found suitable version "3.8.10", minimum required is "3")
-- Configuring done (6.7s)
[ 59%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved32.cu.o
-- Generating done (0.8s)
CMake Warning:
  Manually-specified variables were not used by the project:

    USE_STATIC_CRT


-- Build files have been written to: /home/nvidia/driveHalo-py/Open3D/build/filament/src/ext_filament-build
[ 22%] Performing build step for 'ext_filament'
[  0%] Building CXX object third_party/libgtest/tnt/CMakeFiles/gtest.dir/__/googletest/src/gtest-all.cc.o
[  1%] Linking CXX static library libgtest.a
[  1%] Built target gtest
[  1%] Building CXX object libs/math/CMakeFiles/math.dir/src/dummy.cpp.o
[  1%] Linking CXX static library libmath.a
[  1%] Built target math
[  1%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/api_level.cpp.o
[  1%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/ashmem.cpp.o
[ 60%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved64.cu.o
[  1%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/debug.cpp.o
[  1%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Allocator.cpp.o
[  1%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/CallStack.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/CString.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/CountDownLatch.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/CyclicBarrier.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/EntityManager.cpp.o
[ 60%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved128.cu.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/JobSystem.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Log.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/NameComponentManager.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/ostream.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Panic.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Path.cpp.o
[  2%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Profiler.cpp.o
[  3%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/sstream.cpp.o
[  3%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/Systrace.cpp.o
[  3%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/linux/Condition.cpp.o
[  3%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/linux/Mutex.cpp.o
[  3%] Building CXX object libs/utils/CMakeFiles/utils.dir/src/linux/Path.cpp.o
[  3%] Linking CXX static library libutils.a
[  3%] Built target utils
[  3%] Building C object third_party/getopt/CMakeFiles/getopt.dir/src/getopt.c.o
[  3%] Building C object third_party/getopt/CMakeFiles/getopt.dir/src/getopt_long.c.o
[  3%] Linking C static library libgetopt.a
[  3%] Built target getopt
[  3%] Building CXX object libs/geometry/CMakeFiles/geometry.dir/src/SurfaceOrientation.cpp.o
[  3%] Linking CXX static library libgeometry.a
[  3%] Built target geometry
[  3%] Building CXX object libs/ibl/CMakeFiles/ibl-lite.dir/src/Cubemap.cpp.o
[  3%] Building CXX object libs/ibl/CMakeFiles/ibl-lite.dir/src/CubemapIBL.cpp.o
[  3%] Building CXX object libs/ibl/CMakeFiles/ibl-lite.dir/src/CubemapSH.cpp.o
[  3%] Building CXX object libs/ibl/CMakeFiles/ibl-lite.dir/src/CubemapUtils.cpp.o
[  3%] Building CXX object libs/ibl/CMakeFiles/ibl-lite.dir/src/Image.cpp.o
[ 61%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved512.cu.o
[ 62%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved256.cu.o
[  4%] Linking CXX static library libibl-lite.a
[  4%] Built target ibl-lite
[  4%] Building CXX object libs/bluegl/CMakeFiles/bluegl.dir/src/BlueGL.cpp.o
[  4%] Building CXX object libs/bluegl/CMakeFiles/bluegl.dir/src/BlueGLLinux.cpp.o
[  4%] Building ASM object libs/bluegl/CMakeFiles/bluegl.dir/src/BlueGLCoreLinuxUniversalImpl.S.o
[  4%] Linking CXX static library libbluegl.a
[  4%] Built target bluegl
[  4%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/adler32.c.o
[  4%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/compress.c.o
[  4%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/crc32.c.o
[  4%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/deflate.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/gzclose.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/gzlib.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/gzread.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/gzwrite.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/inflate.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/infback.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/inftrees.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/inffast.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/trees.c.o
[ 65%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCReaderGen2_2x3.cpp.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/uncompr.c.o
[  5%] Building C object third_party/libz/tnt/CMakeFiles/z.dir/__/zutil.c.o
[  6%] Linking C static library libz.a
[  6%] Built target z
[  6%] Building CXX object libs/ibl/CMakeFiles/ibl.dir/src/Cubemap.cpp.o
[  6%] Building CXX object libs/ibl/CMakeFiles/ibl.dir/src/CubemapIBL.cpp.o
[  6%] Building CXX object libs/ibl/CMakeFiles/ibl.dir/src/CubemapSH.cpp.o
[  6%] Building CXX object libs/ibl/CMakeFiles/ibl.dir/src/CubemapUtils.cpp.o
[  6%] Building CXX object libs/image/CMakeFiles/image.dir/src/ImageOps.cpp.o
[  6%] Building CXX object libs/image/CMakeFiles/image.dir/src/ImageSampler.cpp.o
[  6%] Building CXX object libs/ibl/CMakeFiles/ibl.dir/src/Image.cpp.o
[  6%] Linking CXX static library libibl.a
[  6%] Built target ibl
[ 66%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCUtil.cpp.o
[  6%] Building CXX object libs/image/CMakeFiles/image.dir/src/KtxBundle.cpp.o
[  6%] Building CXX object libs/image/CMakeFiles/image.dir/src/LinearImage.cpp.o
[  7%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_averages_and_directions.cpp.o
[ 66%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCGeometry.cpp.o
[  7%] Linking CXX static library libimage.a
[  7%] Built target image
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/Etc/Etc.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/Etc/EtcFilter.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/Etc/EtcImage.cpp.o
[  7%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_block_sizes2.cpp.o
[ 62%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved1024.cu.o
[ 67%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCMaterial.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/Etc/EtcMath.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4.cpp.o
[  7%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_color_quantize.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_ETC1.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_R11.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_RG11.cpp.o
[  7%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_color_unquantize.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_RGB8.cpp.o
[  7%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_compress_symbolic.cpp.o
[  7%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_RGB8A1.cpp.o
[  8%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcBlock4x4Encoding_RGBA8.cpp.o
[  8%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcDifferentialTrys.cpp.o
[  8%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcIndividualTrys.cpp.o
[  8%] Building CXX object third_party/etc2comp/EtcLib/CMakeFiles/EtcLib.dir/EtcCodec/EtcSortedBlockList.cpp.o
[  8%] Linking CXX static library libEtcLib.a
[  8%] Built target EtcLib
[  8%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_decompress_symbolic.cpp.o
[  8%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/png.c.o
[  8%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngerror.c.o
[ 67%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCProfile.cpp.o
[  8%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngget.c.o
[  8%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngmem.c.o
[  8%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngpread.c.o
[  8%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_encoding_choice_error.cpp.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngread.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngrio.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngrtran.c.o
[  9%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_find_best_partitioning.cpp.o
[  9%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_ideal_endpoints_and_weights.cpp.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngrutil.c.o
[  9%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_image_load_store.cpp.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngset.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngtrans.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngwio.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngwrite.c.o
[  9%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_integer_sequence.cpp.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngwtran.c.o
[  9%] Building C object third_party/libpng/tnt/CMakeFiles/png.dir/__/pngwutil.c.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_ktx_dds.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_kmeans_partitioning.cpp.o
[ 67%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCCurve.cpp.o
[ 10%] Linking C static library libpng.a
[ 10%] Built target png
[ 68%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCBoolean.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_partition_tables.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_percentile_tables.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_pick_best_endpoint_format.cpp.o
[ 68%] Building CXX object code/CMakeFiles/assimp.dir/Importer/IFC/IFCOpenings.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_quantization.cpp.o
[ 10%] Building CXX object third_party/spirv-cross/tnt/CMakeFiles/spirv-cross-msl.dir/__/spirv_msl.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_toplevel.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_symbolic_physical.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_weight_align.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_weight_quant_xfer_tables.cpp.o
[ 10%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/astc_compute_variance.cpp.o
[ 11%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/mathlib.cpp.o
[ 11%] Building CXX object third_party/astcenc/tnt/CMakeFiles/astcenc.dir/__/Source/softfloat.cpp.o
[ 69%] Building CXX object code/CMakeFiles/assimp.dir/XGL/XGLLoader.cpp.o
[ 63%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved2048.cu.o
[ 11%] Linking CXX static library libastcenc.a
[ 11%] Built target astcenc
[ 11%] Building CXX object tools/glslminifier/CMakeFiles/glslminifier.dir/src/main.cpp.o
[ 11%] Building CXX object tools/glslminifier/CMakeFiles/glslminifier.dir/src/GlslMinify.cpp.o
[ 69%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXImporter.cpp.o
[ 11%] Linking CXX executable glslminifier
[ 11%] Built target glslminifier
[ 11%] Building CXX object third_party/smol-v/tnt/CMakeFiles/smol-v.dir/__/source/smolv.cpp.o
[ 70%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXParser.cpp.o
[ 70%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXTokenizer.cpp.o
[ 11%] Linking CXX static library libsmol-v.a
[ 11%] Built target smol-v
[ 64%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/BlockSelectFloat.cu.o
[ 70%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXConverter.cpp.o
[ 71%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXUtil.cpp.o
[ 71%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXDocument.cpp.o
[ 65%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/BlockSelectHalf.cu.o
[ 72%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXProperties.cpp.o
[ 72%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXMeshGeometry.cpp.o
[ 65%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/DeviceUtils.cu.o
[ 66%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/utils/StackDeviceMemory.cpp.o
[ 11%] Linking CXX static library libspirv-cross-msl.a
[ 11%] Built target spirv-cross-msl
[ 11%] Generate language specific header for OpenCLDebugInfo100.
[ 11%] Built target spirv-tools-header-OpenCLDebugInfo100
[ 11%] Generate extended instruction tables for spv-amd-shader-explicit-vertex-parameter.
[ 11%] Built target spv-tools-spv-amd-sevp
[ 73%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXMaterial.cpp.o
[ 11%] Generate extended instruction tables for spv-amd-shader-trinary-minmax.
[ 11%] Built target spv-tools-spv-amd-stm
[ 11%] Generate extended instruction tables for spv-amd-gcn-shader.
[ 11%] Built target spv-tools-spv-amd-gs
[ 11%] Generate extended instruction tables for spv-amd-shader-ballot.
[ 11%] Built target spv-tools-spv-amd-sb
[ 11%] Generate extended instruction tables for debuginfo.
[ 11%] Built target spv-tools-debuginfo
[ 11%] Generate extended instruction tables for opencl.debuginfo.100.
[ 11%] Built target spv-tools-cldi100
[ 11%] Generate extended instruction tables for nonsemantic.clspvreflection.
[ 11%] Built target spv-tools-clspvreflection
[ 12%] Generate language specific header for DebugInfo.
[ 67%] Building CXX object faiss/CMakeFiles/faiss.dir/gpu/utils/Timer.cpp.o
[ 12%] Built target spirv-tools-header-DebugInfo
[ 12%] Generate info tables for SPIR-V vunified1 core instructions and operands.
[ 12%] Generate tables based on the SPIR-V XML registry.
[ 73%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXModel.cpp.o
[ 12%] Built target core_tables
[ 12%] Generate enum-string mapping for SPIR-V vunified1.
[ 12%] Built target enum_string_mapping
[ 12%] Building CXX object third_party/glslang/tnt/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/__/InitializeDll.cpp.o
[ 12%] Generating /home/nvidia/driveHalo-py/Open3D/build/filament/src/ext_filament-build/include/glslang/build_info.h
[ 12%] Built target glslang-build-info
[ 12%] Building CXX object third_party/glslang/tnt/glslang/OSDependent/Unix/CMakeFiles/OSDependent.dir/ossource.cpp.o
[ 12%] Linking CXX static library libOSDependent.a
[ 12%] Built target OSDependent
[ 12%] Building CXX object third_party/glslang/tnt/SPIRV/CMakeFiles/SPVRemapper.dir/__/SPVRemapper.cpp.o
[ 12%] Linking CXX static library libOGLCompiler.a
[ 12%] Built target OGLCompiler
[ 12%] Building CXX object third_party/spirv-cross/tnt/CMakeFiles/spirv-cross-core.dir/__/spirv_cross.cpp.o
[ 12%] Building CXX object third_party/glslang/tnt/SPIRV/CMakeFiles/SPVRemapper.dir/__/doc.cpp.o
[ 74%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXAnimation.cpp.o
[ 74%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXNodeAttribute.cpp.o
[ 67%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/WarpSelectFloat.cu.o
[ 74%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXDeformer.cpp.o
[ 75%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXBinaryTokenizer.cpp.o
[ 75%] Building CXX object code/CMakeFiles/assimp.dir/FBX/FBXDocumentUtil.cpp.o
[ 12%] Linking CXX static library libSPVRemapper.a
[ 12%] Built target SPVRemapper
[ 68%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/WarpSelectHalf.cu.o
[ 12%] Building CXX object third_party/spirv-cross/tnt/CMakeFiles/spirv-cross-core.dir/__/spirv_parser.cpp.o
[ 76%] Building CXX object code/CMakeFiles/assimp.dir/Q3D/Q3DLoader.cpp.o
[ 69%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat1.cu.o
[ 12%] Building CXX object third_party/spirv-cross/tnt/CMakeFiles/spirv-cross-core.dir/__/spirv_cross_parsed_ir.cpp.o
[ 69%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat128.cu.o
[ 76%] Building CXX object code/CMakeFiles/assimp.dir/Q3BSP/Q3BSPFileParser.cpp.o
[ 77%] Building CXX object code/CMakeFiles/assimp.dir/Q3BSP/Q3BSPFileImporter.cpp.o
[ 77%] Building CXX object code/CMakeFiles/assimp.dir/Raw/RawLoader.cpp.o
[ 12%] Building CXX object third_party/spirv-cross/tnt/CMakeFiles/spirv-cross-core.dir/__/spirv_cfg.cpp.o
[ 12%] Linking CXX static library libspirv-cross-core.a
[ 77%] Building CXX object code/CMakeFiles/assimp.dir/SIB/SIBImporter.cpp.o
[ 12%] Built target spirv-cross-core
[ 13%] Building CXX object third_party/imgui/tnt/CMakeFiles/imgui.dir/__/imgui.cpp.o
[ 78%] Building CXX object code/CMakeFiles/assimp.dir/SMD/SMDLoader.cpp.o
[ 13%] Building CXX object third_party/imgui/tnt/CMakeFiles/imgui.dir/__/imgui_demo.cpp.o
[ 78%] Building CXX object code/CMakeFiles/assimp.dir/STL/STLLoader.cpp.o
[ 79%] Building CXX object code/CMakeFiles/assimp.dir/Terragen/TerragenLoader.cpp.o
[ 70%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat256.cu.o
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.cpp: In member function ‘void Assimp::STLImporter::LoadASCIIFile(aiNode*)’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.cpp:379:100: warning: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘aiVector3D’ {aka ‘class aiVector3t<float>’} from an array of ‘float’ [-Wclass-memaccess]
  379 |             memcpy(pMesh->mVertices, &positionBuffer[0].x, pMesh->mNumVertices * sizeof(aiVector3D));
      |                                                                                                    ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/types.h:62,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/IOStream.hpp:51,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/DefaultIOStream.h:48,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/Exceptional.h:45,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/BaseImporter.h:47,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.h:49,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.cpp:50:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/vector3.h:66:7: note: ‘aiVector3D’ {aka ‘class aiVector3t<float>’} declared here
   66 | class aiVector3t
      |       ^~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.cpp:385:97: warning: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘aiVector3D’ {aka ‘class aiVector3t<float>’} from an array of ‘float’ [-Wclass-memaccess]
  385 |             memcpy(pMesh->mNormals, &normalBuffer[0].x, pMesh->mNumVertices * sizeof(aiVector3D));
      |                                                                                                 ^
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/types.h:62,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/IOStream.hpp:51,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/DefaultIOStream.h:48,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/Exceptional.h:45,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/BaseImporter.h:47,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.h:49,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/STL/STLLoader.cpp:50:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/include/assimp/vector3.h:66:7: note: ‘aiVector3D’ {aka ‘class aiVector3t<float>’} declared here
   66 | class aiVector3t
      |       ^~~~~~~~~~
[ 13%] Building CXX object third_party/imgui/tnt/CMakeFiles/imgui.dir/__/imgui_draw.cpp.o
[ 13%] Building CXX object third_party/imgui/tnt/CMakeFiles/imgui.dir/__/imgui_widgets.cpp.o
[ 79%] Building CXX object code/CMakeFiles/assimp.dir/Unreal/UnrealLoader.cpp.o
[ 80%] Building CXX object code/CMakeFiles/assimp.dir/X/XFileImporter.cpp.o
[ 13%] Linking CXX static library libimgui.a
[ 13%] Built target imgui
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/clusterizer.cpp.o
[ 71%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat32.cu.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/indexcodec.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/indexgenerator.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/overdrawanalyzer.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/overdrawoptimizer.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/simplifier.cpp.o
[ 80%] Building CXX object code/CMakeFiles/assimp.dir/X/XFileParser.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/stripifier.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/vcacheanalyzer.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/vcacheoptimizer.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/vertexcodec.cpp.o
[ 14%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/vfetchanalyzer.cpp.o
[ 15%] Building CXX object third_party/meshoptimizer/CMakeFiles/meshoptimizer.dir/src/vfetchoptimizer.cpp.o
[ 15%] Linking CXX static library libmeshoptimizer.a
[ 15%] Built target meshoptimizer
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_points_dec.dir/__/src/draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_decoder.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_points_dec.dir/__/src/draco/compression/point_cloud/algorithms/float_points_tree_decoder.cc.o
[ 80%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter.cpp.o
[ 16%] Built target draco_points_dec
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_attributes.dir/__/src/draco/attributes/attribute_octahedron_transform.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/attributes_decoder.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/kd_tree_attributes_decoder.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_attributes.dir/__/src/draco/attributes/attribute_quantization_transform.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/sequential_attribute_decoder.cc.o
[ 72%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat64.cu.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_attributes.dir/__/src/draco/attributes/attribute_transform.cc.o
[ 16%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_attributes.dir/__/src/draco/attributes/geometry_attribute.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/adaptive_rans_bit_decoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/adaptive_rans_bit_encoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_attributes.dir/__/src/draco/attributes/point_attribute.cc.o
[ 81%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Geometry2D.cpp.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/direct_bit_decoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/direct_bit_encoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/sequential_attribute_decoders_controller.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/rans_bit_decoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/rans_bit_encoder.cc.o
[ 81%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Geometry3D.cpp.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/sequential_integer_attribute_decoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/symbol_bit_decoder.cc.o
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_bit_coders.dir/__/src/draco/compression/bit_coders/symbol_bit_encoder.cc.o
[ 17%] Built target draco_attributes
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_dec_config.dir/draco_dec_config.cc.o
[ 17%] Built target draco_dec_config
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_decode.dir/__/src/draco/compression/decode.cc.o
[ 17%] Built target draco_compression_bit_coders
[ 17%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_entropy.dir/__/src/draco/compression/entropy/shannon_entropy.cc.o
[ 18%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_entropy.dir/__/src/draco/compression/entropy/symbol_decoding.cc.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/sequential_normal_attribute_decoder.cc.o
[ 19%] Built target draco_compression_decode
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_dec.dir/__/src/draco/compression/mesh/mesh_decoder.cc.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_dec.dir/__/src/draco/compression/mesh/mesh_edgebreaker_decoder.cc.o
[ 82%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Group.cpp.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_dec.dir/__/src/draco/compression/mesh/mesh_edgebreaker_decoder_impl.cc.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_dec.dir/__/src/draco/compression/mesh/mesh_sequential_decoder.cc.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_dec.dir/__/src/draco/compression/attributes/sequential_quantization_attribute_decoder.cc.o
[ 19%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_entropy.dir/__/src/draco/compression/entropy/symbol_encoding.cc.o
[ 82%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Light.cpp.o
[ 83%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Metadata.cpp.o
[ 19%] Built target draco_compression_attributes_dec
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_dec.dir/__/src/draco/compression/point_cloud/point_cloud_decoder.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_dec.dir/__/src/draco/compression/point_cloud/point_cloud_kd_tree_decoder.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_dec.dir/__/src/draco/compression/point_cloud/point_cloud_sequential_decoder.cc.o
[ 83%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Networking.cpp.o
[ 20%] Built target draco_compression_mesh_dec
[ 84%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Postprocess.cpp.o
[ 84%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Rendering.cpp.o
[ 20%] Built target draco_compression_point_cloud_dec
[ 84%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Shape.cpp.o
[ 20%] Built target draco_compression_entropy
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/bit_utils.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/bounding_box.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/cycle_timer.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/data_buffer.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/decoder_buffer.cc.o
[ 20%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/divide.cc.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/draco_types.cc.o
[ 85%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DImporter_Texturing.cpp.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/encoder_buffer.cc.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/hash_utils.cc.o
[ 85%] Building CXX object code/CMakeFiles/assimp.dir/X3D/FIReader.cpp.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/options.cc.o
[ 86%] Building CXX object code/CMakeFiles/assimp.dir/X3D/X3DVocabulary.cpp.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_core.dir/__/src/draco/core/quantization_utils.cc.o
[ 86%] Building CXX object code/CMakeFiles/assimp.dir/glTF/glTFCommon.cpp.o
[ 21%] Built target draco_core
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/file_reader_factory.cc.o
[ 87%] Building CXX object code/CMakeFiles/assimp.dir/glTF/glTFImporter.cpp.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/file_utils.cc.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/file_writer_factory.cc.o
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFAsset.h:66,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFImporter.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h: In instantiation of ‘void rapidjson::GenericValue<Encoding, Allocator>::SetObjectRaw(rapidjson::GenericValue<Encoding, Allocator>::Member*, rapidjson::SizeType, Allocator&) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; rapidjson::GenericValue<Encoding, Allocator>::Member = rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >; rapidjson::SizeType = unsigned int]’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2430:9:   required from ‘bool rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::EndObject(rapidjson::SizeType) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::SizeType = unsigned int]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2069:18:   required from ‘rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Transit(rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Token, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2173:35:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:560:58:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Parse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2226:22:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; SourceEncoding = rapidjson::UTF8<>; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2242:65:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2267:60:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with unsigned int parseFlags = 0; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2275:51:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFAsset.inl:1363:34:   required from here
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2015:24: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘rapidjson::GenericValue<rapidjson::UTF8<> >::Member’ {aka ‘struct rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >’} with no trivial copy-assignment; use copy-assignment instead [-Wclass-memaccess]
 2015 |             std::memcpy(m, members, count * sizeof(Member));
      |             ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFAsset.h:66,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFImporter.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:71:8: note: ‘rapidjson::GenericValue<rapidjson::UTF8<> >::Member’ {aka ‘struct rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >’} declared here
   71 | struct GenericMember {
      |        ^~~~~~~~~~~~~
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFAsset.h:66,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFImporter.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h: In instantiation of ‘void rapidjson::GenericValue<Encoding, Allocator>::SetArrayRaw(rapidjson::GenericValue<Encoding, Allocator>*, rapidjson::SizeType, Allocator&) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; rapidjson::SizeType = unsigned int]’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2438:9:   required from ‘bool rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::EndArray(rapidjson::SizeType) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::SizeType = unsigned int]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2099:18:   required from ‘rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Transit(rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Token, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2173:35:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:560:58:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Parse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2226:22:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; SourceEncoding = rapidjson::UTF8<>; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2242:65:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2267:60:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with unsigned int parseFlags = 0; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2275:51:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF/glTFAsset.inl:1363:34:   required from here
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2002:24: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘class rapidjson::GenericValue<rapidjson::UTF8<> >’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 2002 |             std::memcpy(e, values, count * sizeof(GenericValue));
      |             ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:551:7: note: ‘class rapidjson::GenericValue<rapidjson::UTF8<> >’ declared here
  551 | class GenericValue {
      |       ^~~~~~~~~~~~
[ 87%] Building CXX object code/CMakeFiles/assimp.dir/glTF2/glTF2Importer.cpp.o
[ 87%] Building CXX object code/CMakeFiles/assimp.dir/3MF/D3MFImporter.cpp.o
[ 21%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/mesh_io.cc.o
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/obj_decoder.cc.o
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Asset.h:67,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Importer.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h: In instantiation of ‘void rapidjson::GenericValue<Encoding, Allocator>::SetObjectRaw(rapidjson::GenericValue<Encoding, Allocator>::Member*, rapidjson::SizeType, Allocator&) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; rapidjson::GenericValue<Encoding, Allocator>::Member = rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >; rapidjson::SizeType = unsigned int]’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2430:9:   required from ‘bool rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::EndObject(rapidjson::SizeType) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::SizeType = unsigned int]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2069:18:   required from ‘rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Transit(rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Token, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2173:35:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:560:58:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Parse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2226:22:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; SourceEncoding = rapidjson::UTF8<>; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2242:65:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2267:60:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with unsigned int parseFlags = 0; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2275:51:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Asset.inl:1382:34:   required from here
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2015:24: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘rapidjson::GenericValue<rapidjson::UTF8<> >::Member’ {aka ‘struct rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >’} with no trivial copy-assignment; use copy-assignment instead [-Wclass-memaccess]
 2015 |             std::memcpy(m, members, count * sizeof(Member));
      |             ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Asset.h:67,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Importer.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:71:8: note: ‘rapidjson::GenericValue<rapidjson::UTF8<> >::Member’ {aka ‘struct rapidjson::GenericMember<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<> >’} declared here
   71 | struct GenericMember {
      |        ^~~~~~~~~~~~~
In file included from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Asset.h:67,
                 from /home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Importer.cpp:46:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h: In instantiation of ‘void rapidjson::GenericValue<Encoding, Allocator>::SetArrayRaw(rapidjson::GenericValue<Encoding, Allocator>*, rapidjson::SizeType, Allocator&) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; rapidjson::SizeType = unsigned int]’:
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2438:9:   required from ‘bool rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::EndArray(rapidjson::SizeType) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::SizeType = unsigned int]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2099:18:   required from ‘rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Transit(rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Token, rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParsingState, InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:2173:35:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::IterativeParse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/reader.h:560:58:   required from ‘rapidjson::ParseResult rapidjson::GenericReader<SourceEncoding, TargetEncoding, StackAllocator>::Parse(InputStream&, Handler&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Handler = rapidjson::GenericDocument<rapidjson::UTF8<> >; SourceEncoding = rapidjson::UTF8<>; TargetEncoding = rapidjson::UTF8<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2226:22:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; SourceEncoding = rapidjson::UTF8<>; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2242:65:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseStream(InputStream&) [with unsigned int parseFlags = 1; InputStream = rapidjson::GenericInsituStringStream<rapidjson::UTF8<> >; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2267:60:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with unsigned int parseFlags = 0; Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2275:51:   required from ‘rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>& rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::ParseInsitu(rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch*) [with Encoding = rapidjson::UTF8<>; Allocator = rapidjson::MemoryPoolAllocator<>; StackAllocator = rapidjson::CrtAllocator; rapidjson::GenericDocument<Encoding, Allocator, StackAllocator>::Ch = char]’
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/glTF2/glTF2Asset.inl:1382:34:   required from here
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:2002:24: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘class rapidjson::GenericValue<rapidjson::UTF8<> >’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
 2002 |             std::memcpy(e, values, count * sizeof(GenericValue));
      |             ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/nvidia/driveHalo-py/Open3D/build/assimp/src/ext_assimp/code/../contrib/rapidjson/include/rapidjson/document.h:551:7: note: ‘class rapidjson::GenericValue<rapidjson::UTF8<> >’ declared here
  551 | class GenericValue {
      |       ^~~~~~~~~~~~
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/obj_encoder.cc.o
[ 88%] Building CXX object code/CMakeFiles/assimp.dir/3MF/D3MFOpcPackage.cpp.o
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/parser_utils.cc.o
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/corner_table.cc.o
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/ply_decoder.cc.o
[ 22%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh.cc.o
[ 88%] Building CXX object code/CMakeFiles/assimp.dir/MMD/MMDImporter.cpp.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh_are_equivalent.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh_attribute_corner_table.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_metadata_dec.dir/__/src/draco/metadata/metadata_decoder.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh_cleanup.cc.o
[ 23%] Built target draco_metadata_dec
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh_misc_functions.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/ply_encoder.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/ply_reader.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/mesh_stripifier.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/point_cloud_io.cc.o
[ 89%] Building CXX object code/CMakeFiles/assimp.dir/MMD/MMDPmxParser.cpp.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_mesh.dir/__/src/draco/mesh/triangle_soup_mesh_builder.cc.o
[ 23%] Built target draco_mesh
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/stdio_file_reader.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_metadata.dir/__/src/draco/metadata/geometry_metadata.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_metadata.dir/__/src/draco/metadata/metadata.cc.o
[ 23%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_io.dir/__/src/draco/io/stdio_file_writer.cc.o
[ 24%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_animation_dec.dir/__/src/draco/animation/keyframe_animation_decoder.cc.o
[ 24%] Built target draco_io
[ 24%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_animation.dir/__/src/draco/animation/keyframe_animation.cc.o
[ 24%] Built target draco_metadata
[ 89%] Building CXX object code/CMakeFiles/assimp.dir/Importer/StepFile/StepFileImporter.cpp.o
[ 24%] Built target draco_animation
[ 24%] Built target draco_animation_dec
[ 90%] Building CXX object code/CMakeFiles/assimp.dir/Importer/StepFile/StepFileGen1.cpp.o
[ 24%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_point_cloud.dir/__/src/draco/point_cloud/point_cloud.cc.o
[ 90%] Building CXX object code/CMakeFiles/assimp.dir/Importer/StepFile/StepFileGen2.cpp.o
[ 90%] Building CXX object code/CMakeFiles/assimp.dir/Importer/StepFile/StepFileGen3.cpp.o
[ 91%] Building C object code/CMakeFiles/assimp.dir/__/contrib/unzip/ioapi.c.o
[ 91%] Building C object code/CMakeFiles/assimp.dir/__/contrib/unzip/unzip.c.o
[ 24%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_point_cloud.dir/__/src/draco/point_cloud/point_cloud_builder.cc.o
[ 92%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/common/shapes.cc.o
[ 24%] Building CXX object libs/math/CMakeFiles/test_math.dir/tests/test_fast.cpp.o
[ 24%] Built target draco_point_cloud
[ 72%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF1024.cu.o
[ 24%] Building CXX object libs/math/CMakeFiles/test_math.dir/tests/test_half.cpp.o
[ 24%] Building CXX object libs/math/CMakeFiles/test_math.dir/tests/test_mat.cpp.o
[ 24%] Building CXX object libs/math/CMakeFiles/test_math.dir/tests/test_vec.cpp.o
[ 24%] Building CXX object libs/math/CMakeFiles/test_math.dir/tests/test_quat.cpp.o
[ 24%] Linking CXX executable test_math
[ 24%] Built target test_math
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/benchmark.cc.o
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/benchmark_api_internal.cc.o
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/benchmark_main.cc.o
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/benchmark_register.cc.o
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/benchmark_runner.cc.o
[ 92%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/advancing_front.cc.o
[ 24%] Building CXX object libs/mathio/CMakeFiles/mathio.dir/src/ostream.cpp.o
[ 24%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/colorprint.cc.o
[ 73%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF2048.cu.o
[ 25%] Linking CXX static library libmathio.a
[ 25%] Built target mathio
[ 25%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_algorithm.cpp.o
[ 25%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/commandlineflags.cc.o
[ 25%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/complexity.cc.o
[ 25%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_Allocators.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/console_reporter.cc.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/counter.cc.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/csv_reporter.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_bitset.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/json_reporter.cc.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/reporter.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_CountDownLatch.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/sleep.cc.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/statistics.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_CString.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/string_util.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_CyclicBarrier.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/sysinfo.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_Entity.cpp.o
[ 26%] Building CXX object third_party/benchmark/tnt/CMakeFiles/benchmark.dir/__/src/timers.cc.o
[ 26%] Linking CXX static library libbenchmark.a
[ 26%] Built target benchmark
[ 26%] Building CXX object libs/utils/CMakeFiles/benchmark_utils_callee.dir/benchmark/benchmark_callee.cpp.o
[ 26%] Linking CXX shared library libbenchmark_utils_callee.so
[ 26%] Built target benchmark_utils_callee
[ 74%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF512.cu.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_JobSystem.cpp.o
[ 93%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/cdt.cc.o
[ 93%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/sweep.cc.o
[ 94%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/sweep_context.cc.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_StructureOfArrays.cpp.o
[ 26%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_sstream.cpp.o
[ 27%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_utils_main.cpp.o
[ 27%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_Zip2Iterator.cpp.o
[ 27%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_BinaryTreeArray.cpp.o
[ 27%] Building CXX object libs/utils/CMakeFiles/test_utils.dir/test/test_Path.cpp.o
[ 74%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT1024.cu.o
[ 27%] Linking CXX executable test_utils
[ 27%] Built target test_utils
[ 27%] Building C object third_party/civetweb/tnt/CMakeFiles/civetweb.dir/__/src/civetweb.c.o
[ 28%] Building CXX object third_party/civetweb/tnt/CMakeFiles/civetweb.dir/__/src/CivetServer.cpp.o
[ 28%] Linking CXX static library libcivetweb.a
[ 28%] Built target civetweb
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/attributes_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/kd_tree_attributes_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/sequential_attribute_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/sequential_attribute_encoders_controller.cc.o
[ 75%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT2048.cu.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/sequential_integer_attribute_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/sequential_normal_attribute_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_enc.dir/__/src/draco/compression/attributes/sequential_quantization_attribute_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_pred_schemes_dec.dir/draco_compression_attributes_pred_schemes_dec.cc.o
[ 28%] Built target draco_compression_attributes_pred_schemes_dec
[ 76%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT512.cu.o
[ 28%] Built target draco_compression_attributes_enc
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_attributes_pred_schemes_enc.dir/__/src/draco/compression/attributes/prediction_schemes/prediction_scheme_encoder_factory.cc.o
[ 28%] Built target draco_compression_attributes_pred_schemes_enc
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_enc_config.dir/draco_enc_config.cc.o
[ 28%] Built target draco_enc_config
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_encode.dir/__/src/draco/compression/encode.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_encode.dir/__/src/draco/compression/expert_encode.cc.o
[ 28%] Built target draco_compression_encode
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_traverser.dir/draco_compression_mesh_traverser.cc.o
[ 28%] Built target draco_compression_mesh_traverser
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_enc.dir/__/src/draco/compression/mesh/mesh_edgebreaker_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_enc.dir/__/src/draco/compression/mesh/mesh_edgebreaker_encoder_impl.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_enc.dir/__/src/draco/compression/mesh/mesh_encoder.cc.o
[ 76%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalf1.cu.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_mesh_enc.dir/__/src/draco/compression/mesh/mesh_sequential_encoder.cc.o
[ 28%] Built target draco_compression_mesh_enc
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_enc.dir/__/src/draco/compression/point_cloud/point_cloud_encoder.cc.o
[ 77%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalf128.cu.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_enc.dir/__/src/draco/compression/point_cloud/point_cloud_kd_tree_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_compression_point_cloud_enc.dir/__/src/draco/compression/point_cloud/point_cloud_sequential_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_metadata_enc.dir/__/src/draco/metadata/metadata_encoder.cc.o
[ 28%] Built target draco_compression_point_cloud_enc
[ 78%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalf256.cu.o
[ 28%] Built target draco_metadata_enc
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_animation_enc.dir/__/src/draco/animation/keyframe_animation_encoder.cc.o
[ 28%] Built target draco_animation_enc
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_points_enc.dir/__/src/draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_encoder.cc.o
[ 28%] Building CXX object third_party/draco/tnt/CMakeFiles/draco_points_enc.dir/__/src/draco/compression/point_cloud/algorithms/float_points_tree_encoder.cc.o
[ 28%] Built target draco_points_enc
[ 28%] Building CXX object libs/bluegl/CMakeFiles/test_bluegl.dir/tests/OpenGLSupport.cpp.o
[ 28%] Building CXX object libs/bluegl/CMakeFiles/test_bluegl.dir/tests/test_bluegl.cpp.o
[ 94%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/clipper/clipper.cpp.o
[ 29%] Linking CXX executable test_bluegl
[ 29%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Assimp.cpp.o
[ 29%] Built target test_bluegl
[ 94%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/OpenDDLParser.cpp.o
[ 29%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/BaseImporter.cpp.o
[ 95%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/DDLNode.cpp.o
[ 95%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/OpenDDLCommon.cpp.o
[ 96%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/OpenDDLExport.cpp.o
[ 96%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/Value.cpp.o
[ 29%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/BaseProcess.cpp.o
[ 29%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Bitmap.cpp.o
[ 29%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL.c.o
[ 29%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/CApi/CInterfaceIOWrapper.cpp.o
[ 29%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL_assert.c.o
[ 29%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL_dataqueue.c.o
[ 29%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL_error.c.o
[ 30%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/CalcTangentsProcess.cpp.o
[ 30%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL_hints.c.o
[ 30%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ComputeUVMappingProcess.cpp.o
[ 30%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/SDL_log.c.o
[ 30%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/atomic/SDL_atomic.c.o
[ 30%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/atomic/SDL_spinlock.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_audio.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_audiocvt.c.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ConvertToLHProcess.cpp.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_audiodev.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_audiotypecvt.c.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/CreateAnimMesh.cpp.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_mixer.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/SDL_wave.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/disk/SDL_diskaudio.c.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/DeboneProcess.cpp.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/dsp/SDL_dspaudio.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/audio/dummy/SDL_dummyaudio.c.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/DefaultIOStream.cpp.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/cpuinfo/SDL_cpuinfo.c.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/dynapi/SDL_dynapi.c.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/DefaultIOSystem.cpp.o
[ 31%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/DefaultLogger.cpp.o
[ 31%] Building CXX object tools/glslminifier/CMakeFiles/test_glslminifier.dir/src/GlslMinify.cpp.o
[ 31%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_clipboardevents.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_dropevents.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_events.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_gesture.c.o
[ 79%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalf32.cu.o
[ 32%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/DropFaceNormalsProcess.cpp.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_keyboard.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_mouse.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_quit.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_touch.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/events/SDL_windowevents.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/file/SDL_rwops.c.o
[ 32%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/EmbedTexturesProcess.cpp.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/haptic/SDL_haptic.c.o
[ 32%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/joystick/SDL_gamecontroller.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/joystick/SDL_joystick.c.o
[ 33%] Building CXX object tools/glslminifier/CMakeFiles/test_glslminifier.dir/tests/test_glslminifier.cpp.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/joystick/steam/SDL_steamcontroller.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_atan2.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_fmod.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_log.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_log10.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_pow.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_rem_pio2.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/e_sqrt.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/k_cos.c.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/k_rem_pio2.c.o
[ 33%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Exporter.cpp.o
[ 33%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/k_sin.c.o
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/k_tan.c.o
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_atan.c.o
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_copysign.c.o
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_cos.c.o
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_fabs.c.o
[ 34%] Linking CXX executable test_glslminifier
[ 34%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_floor.c.o
[ 34%] Built target test_glslminifier
[ 35%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXAnimation.cpp.o
[ 35%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_scalbn.c.o
[ 35%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_sin.c.o
[ 35%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/libm/s_tan.c.o
[ 35%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/power/SDL_power.c.o
[ 35%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/SDL_d3dmath.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/SDL_render.c.o
[ 97%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/openddlparser/code/OpenDDLStream.cpp.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/SDL_yuv_sw.c.o
[ 97%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/Open3DGC/o3dgcArithmeticCodec.cpp.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/direct3d/SDL_render_d3d.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/direct3d/SDL_shaders_d3d.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/direct3d11/SDL_render_d3d11.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/direct3d11/SDL_shaders_d3d11.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/opengl/SDL_render_gl.c.o
[ 97%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/Open3DGC/o3dgcDynamicVectorDecoder.cpp.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/opengl/SDL_shaders_gl.c.o
[ 98%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/Open3DGC/o3dgcDynamicVectorEncoder.cpp.o
[ 36%] Building CXX object tools/specular-color/CMakeFiles/specular-color.dir/src/main.cpp.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/opengles/SDL_render_gles.c.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/opengles2/SDL_render_gles2.c.o
[ 36%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXBinaryTokenizer.cpp.o
[ 36%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/opengles2/SDL_shaders_gles2.c.o
[ 37%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/psp/SDL_render_psp.c.o
[ 37%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_blendfillrect.c.o
[ 98%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/Open3DGC/o3dgcTools.cpp.o
[ 99%] Building CXX object code/CMakeFiles/assimp.dir/__/contrib/Open3DGC/o3dgcTriangleFans.cpp.o
[ 37%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXConverter.cpp.o
[ 37%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_blendline.c.o
[ 99%] Building C object code/CMakeFiles/assimp.dir/__/contrib/zip/src/zip.c.o
[ 38%] Linking CXX executable specular-color
[ 38%] Built target specular-color
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_blendpoint.c.o
[ 38%] Building CXX object libs/camutils/CMakeFiles/camutils.dir/src/Bookmark.cpp.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_drawline.c.o
[ 38%] Building CXX object libs/filabridge/CMakeFiles/filabridge.dir/src/SamplerBindingMap.cpp.o
[ 38%] Building CXX object libs/camutils/CMakeFiles/camutils.dir/src/Manipulator.cpp.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_drawpoint.c.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_render_sw.c.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/render/software/SDL_rotate.c.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_getenv.c.o
[ 38%] Linking CXX static library libcamutils.a
[ 38%] Built target camutils
[ 38%] Building CXX object libs/filabridge/CMakeFiles/filabridge.dir/src/SamplerInterfaceBlock.cpp.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_iconv.c.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_malloc.c.o
[ 38%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_qsort.c.o
[ 38%] Building CXX object tools/resgen/CMakeFiles/resgen.dir/src/main.cpp.o
[ 39%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_stdlib.c.o
[ 39%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/stdlib/SDL_string.c.o
[ 39%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/SDL_thread.c.o
[ 39%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/timer/SDL_timer.c.o
[100%] Linking CXX static library libassimp.a
[ 39%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_RLEaccel.c.o
[ 40%] Building CXX object libs/filabridge/CMakeFiles/filabridge.dir/src/UniformInterfaceBlock.cpp.o
[ 40%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit.c.o
[ 40%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_0.c.o
[ 40%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_1.c.o
[ 41%] Linking CXX executable resgen
[ 41%] Built target resgen
[ 41%] Building CXX object libs/filabridge/CMakeFiles/filabridge.dir/src/UibGenerator.cpp.o
[ 41%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_A.c.o
[ 41%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_N.c.o
[ 41%] Building CXX object libs/filabridge/CMakeFiles/filabridge.dir/src/SibGenerator.cpp.o
[ 41%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_auto.c.o
[ 41%] Linking CXX static library libfilabridge.a
[ 41%] Built target filabridge
[ 41%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/BackendUtils.cpp.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_copy.c.o
[ 42%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/Callable.cpp.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_blit_slow.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_bmp.c.o
[ 42%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/CircularBuffer.cpp.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_clipboard.c.o
[ 42%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/CommandBufferQueue.cpp.o
[ 42%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/CommandStream.cpp.o
[ 42%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXDeformer.cpp.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_egl.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_fillrect.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_pixels.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_rect.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_shape.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_stretch.c.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_surface.c.o
[ 42%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/Driver.cpp.o
[ 42%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_video.c.o
[ 42%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXDocument.cpp.o
[ 42%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXDocumentUtil.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_vulkan_utils.c.o
[ 79%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalf64.cu.o
[ 43%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXImporter.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/SDL_yuv.c.o
[ 43%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/Handle.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/dummy/SDL_nullevents.c.o
[ 43%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/noop/NoopDriver.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/dummy/SDL_nullframebuffer.c.o
[ 43%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXMaterial.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/dummy/SDL_nullvideo.c.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/yuv2rgb/yuv_rgb.c.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/core/unix/SDL_poll.c.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/filesystem/unix/SDL_sysfilesystem.c.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/haptic/linux/SDL_syshaptic.c.o
[ 43%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXMeshGeometry.cpp.o
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/joystick/linux/SDL_sysjoystick.c.o
[100%] Built target assimp
[ 22%] Performing install step for 'ext_assimp'
[  7%] Built target zlib
[ 14%] Built target zlibstatic
[ 14%] Built target IrrXML
[ 43%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/loadso/dlopen/SDL_sysloadso.c.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/power/linux/SDL_syspower.c.o
[100%] Built target assimp
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/cmake/assimp-5.0/assimp-config.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/cmake/assimp-5.0/assimp-config-version.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/cmake/assimp-5.0/assimpTargets.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/cmake/assimp-5.0/assimpTargets-release.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/pkgconfig/assimp.pc
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/libzlibstatic.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/libIrrXML.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/lib/libassimp.a
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/pthread/SDL_syscond.c.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/pthread/SDL_sysmutex.c.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/pthread/SDL_syssem.c.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/pthread/SDL_systhread.c.o
[ 44%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/noop/PlatformNoop.cpp.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/thread/pthread/SDL_systls.c.o
[ 44%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/timer/unix/SDL_systimer.c.o
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/anim.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/aabb.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/ai_assert.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/camera.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/color4.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/color4.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/config.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/defs.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Defines.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/cfileio.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/light.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/material.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/material.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/matrix3x3.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/matrix3x3.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/matrix4x4.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/matrix4x4.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/mesh.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/pbrmaterial.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/postprocess.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/quaternion.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/quaternion.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/scene.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/metadata.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/texture.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/types.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/vector2.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/vector2.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/vector3.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/vector3.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/version.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/cimport.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/importerdesc.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Importer.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/DefaultLogger.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/ProgressHandler.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/IOStream.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/IOSystem.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Logger.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/LogStream.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/NullLogger.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/cexport.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Exporter.hpp
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/DefaultIOStream.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/DefaultIOSystem.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/ZipArchiveIOSystem.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SceneCombiner.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/fast_atof.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/qnan.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/BaseImporter.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Hash.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/MemoryIOWrapper.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/ParsingUtils.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/StreamReader.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/StreamWriter.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/StringComparison.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/StringUtils.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SGSpatialSort.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/GenericProperty.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SpatialSort.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SkeletonMeshBuilder.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SmoothingGroups.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/SmoothingGroups.inl
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/StandardShapes.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/RemoveComments.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Subdivision.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Vertex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/LineSplitter.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/TinyFormatter.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Profiler.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/LogAux.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Bitmap.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/XMLTools.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/IOStreamBuffer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/CreateAnimMesh.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/irrXMLWrapper.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/BlobIOSystem.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/MathFunctions.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Macros.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Exceptional.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/ByteSwapper.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Compiler/pushpack1.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Compiler/poppack1.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/assimp/include/assimp/Compiler/pstdint.h
[ 22%] Completed 'ext_assimp'
[ 22%] Built target ext_assimp
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/Platform.cpp.o
[ 45%] Building C object third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/x11/SDL_x11clipboard.c.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/Program.cpp.o
In file included from /home/nvidia/driveHalo-py/Open3D/build/filament/src/ext_filament/third_party/libsdl2/src/video/x11/SDL_x11clipboard.c:28:
/home/nvidia/driveHalo-py/Open3D/build/filament/src/ext_filament/third_party/libsdl2/src/video/x11/SDL_x11video.h:56:10: fatal error: 'X11/extensions/xf86vmode.h' file not found
#include <X11/extensions/xf86vmode.h>
         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated.
make[5]: *** [third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/build.make:1728: third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/__/src/video/x11/SDL_x11clipboard.c.o] Error 1
make[4]: *** [CMakeFiles/Makefile2:5404: third_party/libsdl2/tnt/CMakeFiles/sdl2.dir/all] Error 2
make[4]: *** Waiting for unfinished jobs....
[ 45%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXModel.cpp.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/SamplerGroup.cpp.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/TextureReshaper.cpp.o
[ 45%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXNodeAttribute.cpp.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/gl_headers.cpp.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/GLUtils.cpp.o
[ 45%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/OpenGLBlitter.cpp.o
[ 46%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXParser.cpp.o
[ 46%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXProperties.cpp.o
[ 46%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/OpenGLContext.cpp.o
[ 46%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXTokenizer.cpp.o
[ 46%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/OpenGLDriver.cpp.o
[ 46%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/FBX/FBXUtil.cpp.o
[ 46%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/OpenGLProgram.cpp.o
[ 46%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/OpenGLPlatform.cpp.o
[ 46%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/X3D/FIReader.cpp.o
[ 47%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/TimerQuery.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/FindDegenerates.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/FindInstancesProcess.cpp.o
[ 47%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/PlatformGLX.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/FindInvalidDataProcess.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/FixNormalsStep.cpp.o
[ 47%] Building CXX object filament/backend/CMakeFiles/backend.dir/src/opengl/PlatformEGLHeadless.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/GenFaceNormalsProcess.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/GenBoundingBoxesProcess.cpp.o
[ 47%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/GenVertexNormalsProcess.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Importer.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/ImporterRegistry.cpp.o
[ 80%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfF1024.cu.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ImproveCacheLocality.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/JoinVerticesProcess.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/LimitBoneWeightsProcess.cpp.o
[ 48%] Linking CXX static library libbackend.a
[ 48%] Built target backend
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/MakeVerboseFormat.cpp.o
[ 81%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfF2048.cu.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Material/MaterialSystem.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Obj/ObjFileImporter.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Obj/ObjFileMtlImporter.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Obj/ObjFileParser.cpp.o
[ 48%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/OptimizeGraph.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/OptimizeMeshes.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/PostStepRegistry.cpp.o
[ 81%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfF512.cu.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/PretransformVertices.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ProcessHelper.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/RemoveComments.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/RemoveRedundantMaterials.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/RemoveVCProcess.cpp.o
[ 82%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfT1024.cu.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/SGSpatialSort.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ScaleProcess.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/SceneCombiner.cpp.o
[ 83%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfT2048.cu.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/ScenePreprocessor.cpp.o
[ 49%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/SkeletonMeshBuilder.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/SortByPTypeProcess.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/SpatialSort.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/SplitByBoneCountProcess.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/SplitLargeMeshes.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/StandardShapes.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Subdivision.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/TargetAnimation.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/TextureTransform.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/TriangulateProcess.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/PostProcessing/ValidateDataStructure.cpp.o
[ 50%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/Version.cpp.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/VertexTriangleAdjacency.cpp.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/code/Common/scene.cpp.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/clipper/clipper.cpp.o
[ 83%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectHalfT512.cu.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/irrXML/irrXML.cpp.o
[ 84%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat1.cu.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/common/shapes.cc.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/advancing_front.cc.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/cdt.cc.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/sweep_context.cc.o
[ 51%] Building CXX object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/poly2tri/poly2tri/sweep/sweep.cc.o
[ 51%] Building C object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/unzip/ioapi.c.o
[ 85%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat128.cu.o
[ 51%] Building C object third_party/libassimp/tnt/CMakeFiles/assimp.dir/__/contrib/unzip/unzip.c.o
[ 51%] Linking CXX static library libassimp.a
[ 86%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat256.cu.o
[ 51%] Built target assimp
make[3]: *** [Makefile:136: all] Error 2
make[2]: *** [CMakeFiles/ext_filament.dir/build.make:86: filament/src/ext_filament-stamp/ext_filament-build] Error 2
make[1]: *** [CMakeFiles/Makefile2:976: CMakeFiles/ext_filament.dir/all] Error 2
make[1]: *** Waiting for unfinished jobs....
[ 86%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat32.cu.o
[ 87%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat64.cu.o
[ 88%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF1024.cu.o
[ 88%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF2048.cu.o
[ 89%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF512.cu.o
[ 90%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT1024.cu.o
[ 90%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT2048.cu.o
[ 91%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT512.cu.o
[ 92%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalf1.cu.o
[ 93%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalf128.cu.o
[ 93%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalf256.cu.o
[ 94%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalf32.cu.o
[ 95%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalf64.cu.o
[ 95%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfF1024.cu.o
[ 96%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfF2048.cu.o
[ 97%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfF512.cu.o
[ 97%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfT1024.cu.o
[ 98%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfT2048.cu.o
[ 99%] Building CUDA object faiss/CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectHalfT512.cu.o
[100%] Linking CXX static library libfaiss.a
[100%] Built target faiss
[ 22%] Performing install step for 'ext_faiss'
[100%] Built target faiss
Install the project...
-- Install configuration: "Release"
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/lib/libfaiss.a
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/Clustering.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IVFlib.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/Index.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/Index2Layer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexBinary.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexBinaryFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexBinaryFromFloat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexBinaryHash.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexBinaryIVF.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVF.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVFFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVFPQ.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVFPQR.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVFPQFastScan.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexPQFastScan.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexIVFSpectralHash.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexLSH.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexLattice.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexPQ.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexPreTransform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexReplicas.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexScalarQuantizer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexShards.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/IndexRefine.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/MatrixStats.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/MetaIndexes.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/MetricType.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/VectorTransform.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/index_io.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/ParallelUtil.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/AuxIndexStructures.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/FaissAssert.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/FaissException.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/PolysemousTraining.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/ProductQuantizer-inl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/ProductQuantizer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/ScalarQuantizer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/ThreadedIndex-inl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/ThreadedIndex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/io.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/io_macros.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/lattice_Zn.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/pq4_fast_scan.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/simd_result_handlers.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/impl/platform_macros.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/invlists/InvertedLists.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/invlists/BlockInvertedLists.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/invlists/DirectMap.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/invlists/InvertedListsIOHook.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/Heap.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/AlignedTable.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/WorkerThread.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/distances.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/extra_distances.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/hamming-inl.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/hamming.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/ordered_key_value.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/partitioning.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/quantize_lut.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/random.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/simdlib.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/simdlib_emulated.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/simdlib_avx2.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/utils/utils.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/invlists/OnDiskInvertedLists.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/share/faiss/faiss-config.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/share/faiss/faiss-config-version.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/share/faiss/faiss-targets.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/share/faiss/faiss-targets-release.cmake
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuDistance.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuFaissAssert.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexBinaryFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndex.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexIVFFlat.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexIVF.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexIVFPQ.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndexIVFScalarQuantizer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuIndicesOptions.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/GpuResources.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/StandardGpuResources.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/BinaryDistance.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/BinaryFlatIndex.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/BroadcastSum.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/Distance.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/DistanceUtils.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/FlatIndex.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/GeneralDistance.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/GpuScalarQuantizer.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFAppend.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFBase.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFFlat.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFFlatScan.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFPQ.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/IVFUtils.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/L2Norm.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/L2Select.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQCodeDistances.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQCodeDistances-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQCodeLoad.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/RemapIndices.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/impl/VectorResidual.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/blockselect/BlockSelectImpl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/BlockSelectKernel.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Comparators.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/ConversionOperators.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/CopyUtils.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/DeviceDefs.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/DeviceTensor.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/DeviceTensor-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/DeviceUtils.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/DeviceVector.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Float16.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/HostTensor.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/HostTensor-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Limits.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/LoadStoreOperators.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MathOperators.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MatrixMult.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MatrixMult-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MergeNetworkBlock.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MergeNetworkUtils.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/MergeNetworkWarp.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/NoTypeTensor.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Pair.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/PtxUtils.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/ReductionOperators.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Reductions.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Select.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/StackDeviceMemory.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/StaticUtils.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Tensor.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Tensor-inl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/ThrustAllocator.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Timer.h
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/Transpose.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/WarpSelectKernel.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/warpselect/WarpSelectImpl.cuh
-- Installing: /home/nvidia/driveHalo-py/Open3D/build/mkl_install/include/faiss/gpu/utils/WarpShuffles.cuh
[ 22%] Completed 'ext_faiss'
[ 22%] Built target ext_faiss
make: *** [Makefile:136: all] Error 2
nvidia@tegra-ubuntu:~/driveHalo-py/Open3D/build$ 
