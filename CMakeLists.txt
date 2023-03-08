cmake_minimum_required(VERSION 3.5)

# ##############################################################################
# CMake Project ##
# ##############################################################################

# include useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake)

# read metadata from package.xml and setup project
zetton_cc_metadata()

# set project name and version
project(
  ${extracted_name}
  VERSION ${extracted_version}
  LANGUAGES C CXX)

# =========#
# Settings #
# =========#

# introduce default building settings
zetton_cc_settings()

# =============#
# Dependencies #
# =============#

# ----------------------#
# Third-party libraries #
# ----------------------#

find_package(Threads)

# -----------------#
# Zetton libraries #
# -----------------#

find_package(zetton_common REQUIRED)
find_package(zetton_inference REQUIRED)

# ##############################################################################
# Build #
# ##############################################################################

# ==============#
# Build targets #
# ==============#

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ---------------#
# Library target #
# ---------------#

# find all source files
file(GLOB_RECURSE src_files ${PROJECT_SOURCE_DIR}/src/*.cc
     ${PROJECT_SOURCE_DIR}/src/*/*.cc)

# common library
zetton_cc_library(
  NAME
  ${PROJECT_NAME}
  SRCS
  ${src_files}
  DEPS
  nvinfer
  nvinfer_plugin
  nvparsers
  nvonnxparser
  Threads::Threads
  absl::flags
  absl::flags_parse
  zetton_common::zetton_common
  zetton_inference::zetton_inference)

# ------------------#
# Executable targets #
# ------------------#

zetton_cc_apps()
zetton_cc_examples()

# -------------#
# Test targets #
# -------------#

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)
zetton_cc_tests("inference")
