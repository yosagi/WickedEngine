# Builds the NVIDIA PhysX 5 SDK as part of this build tree.
#
#	include(.../FetchPhysX.cmake)          # defines the PhysX targets
#	target_link_libraries(foo PRIVATE physx_lib)
#
# Everything lives under the build directory (${FETCHCONTENT_BASE_DIR}/physx-src and -build): there is no external
# checkout to keep in sync and no separate install step. Deleting the build directory removes the SDK with it.
#
# Notes on the choices here:
#	- The release tarball is used instead of a git clone. The repository also carries omni/, blast/ and flow/, and a
#	  shallow clone pulls 309 MB through git-lfs; the tarball keeps the lfs files as pointers and is 11 MB, and its
#	  physx/ tree is byte identical. It also avoids git entirely, which matters because a `url.<base>.insteadOf`
#	  rewrite in the user's ~/.gitconfig can redirect https://github.com/ to ssh and break an anonymous clone.
#	- `SOURCE_SUBDIR physx` points at the SDK's own add_subdirectory/FetchContent entry point (new in 5.9). It
#	  declares the `physx_lib` INTERFACE target that carries the include paths and the eight static libraries.
#	- CPU only. The GPU pipeline needs the CUDA toolkit and buys nothing for a single robot.
include_guard(GLOBAL)
include(FetchContent)

set(PHYSX_TAG "110.1-omni-and-physx-5.9.0" CACHE STRING "PhysX SDK release tag to build against (SDK 5.9.0)")
set(PHYSX_URL_HASH "SHA256=c924aa80e73562730d6b3335a31843fe1acbe2af9f5a56563e0953ecbe3ff77d"
	CACHE STRING "Checksum of the PhysX release tarball named by PHYSX_TAG")
mark_as_advanced(PHYSX_TAG PHYSX_URL_HASH)

# The SDK reads these as option()s in its own CMakeLists, so they have to exist in the cache before it is added
set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "PhysX: build the CUDA pipeline" FORCE)
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "PhysX: build static libraries" FORCE)
set(PX_BUILDSNIPPETS OFF CACHE BOOL "PhysX: build the SDK samples" FORCE)
set(PX_BUILDPVDRUNTIME OFF CACHE BOOL "PhysX: build the OmniPVD runtime" FORCE)

set(_physx_args)
if(NOT CMAKE_VERSION VERSION_LESS 3.24)
	list(APPEND _physx_args DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()
FetchContent_Declare(physx
	URL "https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/${PHYSX_TAG}.tar.gz"
	URL_HASH "${PHYSX_URL_HASH}"
	SOURCE_SUBDIR physx
	${_physx_args}
)
message(STATUS "PhysX: fetching ${PHYSX_TAG} (set FETCHCONTENT_SOURCE_DIR_PHYSX to use a local checkout instead)")

# The SDK compiles its own sources with -Werror and its own warning selection. A consumer that has called
#	add_compile_options() at directory scope (the engine adds -Wall) leaks those into the fetched subdirectory,
#	and the extra warnings then become errors inside SDK code we do not maintain: GCC 11 reports
#	maybe-uninitialized in DyFeatherstoneArticulation.cpp, for one. Hand the subdirectory an empty option list and
#	put the consumer's back afterwards. Compile *definitions* are deliberately left alone, since
#	_GLIBCXX_USE_CXX11_ABI has to match between the SDK and the code that links it.
get_directory_property(_physx_saved_compile_options COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")
FetchContent_MakeAvailable(physx)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_physx_saved_compile_options}")

# The SDK only knows the four configuration names debug / checked / profile / release and selects its preprocessor
#	defines with $<$<CONFIG:release>:...> generator expressions. Under any other CMAKE_BUILD_TYPE (the engine builds
#	RelWithDebInfo) none of them match and the libraries end up without NDEBUG / PX_SUPPORT_PVD, which makes
#	PxPreprocessor.h derive a different PX_DEBUG inside the SDK than in the code that includes its headers.
#	Give those configurations the release set explicitly. This is a no-op when one of the four names is used.
set(_physx_foreign_config "$<NOT:$<CONFIG:debug,checked,profile,release>>")
foreach(_target PhysX PhysXCommon PhysXFoundation PhysXExtensions PhysXPvdSDK PhysXCooking PhysXCharacterKinematic PhysXVehicle)
	if(TARGET ${_target})
		target_compile_definitions(${_target} PRIVATE
			"$<${_physx_foreign_config}:NDEBUG;PX_SUPPORT_PVD=0;PX_SUPPORT_OMNI_PVD=0>")
		# The SDK compiles with -Werror and a -Wno- list written for the compilers of its release date. A newer
		#	compiler adds warnings that list does not know (GCC 15: -Wdangling-pointer in CmRadixSort.cpp) and the
		#	build of code we do not maintain stops. Keep the warnings, drop the error.
		if(NOT MSVC)
			target_compile_options(${_target} PRIVATE -Wno-error)
		endif()
		set_target_properties(${_target} PROPERTIES FOLDER PhysX)
	endif()
endforeach()
# The same choice has to reach the consumers, since PxPreprocessor.h is a public header
target_compile_definitions(physx_lib INTERFACE
	"$<${_physx_foreign_config}:NDEBUG;PX_SUPPORT_PVD=0;PX_SUPPORT_OMNI_PVD=0>")
