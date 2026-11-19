# ==========================================
# TITAN PROJECT MAKEFILE (Cross-platform)
# Windows: Clang (C++23) | Linux: GCC/Clang (C++23)
# Tích hợp Auto Dependency Tracking (.d)
# ==========================================

# Source export of TITAN commit d859c145c36b for HoloLift 1.2.1: the build stamp names that commit,
# and the portable profile (AVX2/FMA) is the default.
export TITAN_GIT_COMMIT = d859c145c36b290611bfbcd27877862c0fd79a87
export TITAN_BUILD_NUMBER = 464
export TITAN_GIT_DIRTY = false
export TITAN_RELEASE_BUILD = false
export SOURCE_DATE_EPOCH = 1795098715
PROFILE ?= portable
TOOLCHAIN ?= auto
TITAN_CXX23_FLOOR ?= clang20-gxx13
LLVM_VERSION ?= 22
LLVM_PREFIX ?= /usr/lib/llvm-$(LLVM_VERSION)
BUILD_ROOT = build
BUILD_DIR = $(BUILD_ROOT)/$(PROFILE)
SRC_ROOT = src
GENERATED_DIR = $(BUILD_DIR)/generated
VERSION_FILE = VERSION
BUILD_INFO_GENERATOR = tools/generate_build_info.py
BUILD_INFO_TEST = tools/test_build_metadata.py
BUILD_INFO_HEADER = $(GENERATED_DIR)/titan_version_generated.h
BUILD_INFO_MANIFEST = $(GENERATED_DIR)/titan_build_manifest.json
BUILD_INFO_RC = $(GENERATED_DIR)/titan_version.rc
CLI_REGISTRY_GENERATOR = tools/generate_cli_registry.py
CLI_PARSER_SOURCE = $(SRC_ROOT)/cli/parse_arguments.cpp $(SRC_ROOT)/cli/parse_helpers.cpp \
                    $(sort $(wildcard $(SRC_ROOT)/cli/parse_options_*.cpp))
CLI_REGISTRY_HEADER = $(GENERATED_DIR)/titan_cli_registry_generated.h
CLI_VALIDATION_TEST = tools/test_cli_validation.py
INCLUDE_DIRS = -I$(GENERATED_DIR) -I$(SRC_ROOT) -I$(SRC_ROOT)/gmxtraj/include -I$(SRC_ROOT)/eigen

COMMON_CXX_DEFINES = -DTITAN_CXX23_FLOOR_CLANG20_GXX13=1 -DTITAN_RUNTIME_DISPATCH_HOTSPOTS=1
ifeq ($(PROFILE),native)
    COMMON_CXX_DEFINES += -DTITAN_BUILD_PROFILE_TOKEN=native
else ifeq ($(PROFILE),portable)
    COMMON_CXX_DEFINES += -DTITAN_BUILD_PROFILE_TOKEN=portable
else
    COMMON_CXX_DEFINES += -DTITAN_BUILD_PROFILE_TOKEN=unknown
endif

ifeq ($(PROFILE),native)
    ARCH_FLAGS = -march=native
else ifeq ($(PROFILE),portable)
    ARCH_FLAGS = -mavx2 -mfma -mtune=generic
else
    $(error Unsupported PROFILE '$(PROFILE)'. Use PROFILE=native or PROFILE=portable)
endif

ifeq ($(OS),Windows_NT)
    # ─── CẤU HÌNH CHO WINDOWS (CLANG) ───
    MSYS2_UCRT64_BIN ?= C:/msys64/ucrt64/bin
    CXX  = $(MSYS2_UCRT64_BIN)/clang++.exe
    PYTHON ?= $(MSYS2_UCRT64_BIN)/python.exe
    WINDRES ?= $(MSYS2_UCRT64_BIN)/windres.exe
    TARGET = titan.exe
    PIPE_FLAG = -pipe
    STACK_NOTE_FLAG =
    
    # Clang Flags: Thêm cờ chặn cảnh báo rác của Clang
    # Force Eigen to use its own aligned allocator on Windows.
    # This avoids AVX-aligned loads/stores from touching only 16-byte-aligned buffers.
    CXXFLAGS = $(INCLUDE_DIRS) -O3 $(ARCH_FLAGS) -fopenmp $(PIPE_FLAG) $(STACK_NOTE_FLAG) -std=c++23 -Wall -Wno-unused-command-line-argument -MMD -MP $(COMMON_CXX_DEFINES)
    
    # Ép dùng LLD linker siêu tốc của LLVM
    LDFLAGS = -fuse-ld=lld -static -static-libgcc -static-libstdc++ -fopenmp
    
    # Clang dùng ThinLTO đa luồng
    LTO_FLAG = -flto=thin
    
    # Lệnh dọn dẹp trên Windows CMD/MSYS2
    # Keep recipe syntax aligned with the shell that invokes mingw32-make.
    # GitHub Actions uses an MSYS2 shell, while local PowerShell/CMD builds do not.
    ifneq ($(strip $(MSYSTEM)),)
        RM = rm -f
        MKDIR_CMD = mkdir -p "$(@D)"
        CLEAN_CMD = rm -rf $(BUILD_DIR) $(TARGET)
        CLEAN_ALL_CMD = rm -rf $(BUILD_ROOT) $(TARGET)
        CLEAN_GENERATED_CMD = $(RM) network_edges.csv network_nodes.csv network_communities.csv network_community_summary.csv network_supergraph.csv pca_fel.csv tica_projections.csv extract_basins.sh tmp_test_dham*.in 2>/dev/null || true
    else
        RM = del /Q /F
        MKDIR_CMD = if not exist "$(subst /,\,$(@D))" mkdir "$(subst /,\,$(@D))"
        CLEAN_CMD = if exist "$(subst /,\,$(BUILD_DIR))" rmdir /S /Q "$(subst /,\,$(BUILD_DIR))" & if exist "$(TARGET)" del /Q /F "$(TARGET)"
        CLEAN_ALL_CMD = if exist "$(subst /,\,$(BUILD_ROOT))" rmdir /S /Q "$(subst /,\,$(BUILD_ROOT))" & if exist "$(TARGET)" del /Q /F "$(TARGET)"
        CLEAN_GENERATED_CMD = $(RM) network_edges.csv network_nodes.csv network_communities.csv network_community_summary.csv network_supergraph.csv pca_fel.csv tica_projections.csv extract_basins.sh tmp_test_dham*.in 2>nul || exit 0
    endif
else
    # ─── CẤU HÌNH CHO LINUX (GCC 13) ───
    TOOLCHAIN_NAME := $(TOOLCHAIN)
    ifeq ($(TOOLCHAIN_NAME),auto)
        ifneq ($(findstring clang,$(notdir $(CXX))),)
            TOOLCHAIN_NAME := clang
        else
            TOOLCHAIN_NAME := gcc
        endif
    endif
    ifeq ($(TOOLCHAIN_NAME),clang)
        ifeq ($(origin CXX),default)
            CXX = clang++-22
        endif
        ifeq ($(origin CC),default)
        endif
        OPENMP_FLAG = -fopenmp=libomp
        CXX_STDLIB_FLAG = -stdlib=libc++
        CLANG_STDLIB_INCLUDE_FLAGS = -nostdinc++ -isystem $(LLVM_PREFIX)/include/c++/v1 -isystem $(LLVM_PREFIX)/include/x86_64-pc-linux-gnu/c++/v1
        CLANG_OMP_INCLUDE_FLAGS = -isystem $(LLVM_PREFIX)/lib/clang/$(LLVM_VERSION)/include
        CLANG_STDLIB_LINK_FLAGS = -L$(LLVM_PREFIX)/lib -Wl,-rpath,$(LLVM_PREFIX)/lib
        CLANG_STDLIB_LIBS = -lc++abi
        STDLIB_LINK_FLAGS =
        LTO_FLAG = -flto=thin
    else ifeq ($(TOOLCHAIN_NAME),gcc)
        ifeq ($(origin CXX),default)
            CXX = g++
        endif
        ifeq ($(origin CC),default)
        endif
        OPENMP_FLAG = -fopenmp
        CXX_STDLIB_FLAG =
        CLANG_STDLIB_INCLUDE_FLAGS =
        CLANG_OMP_INCLUDE_FLAGS =
        CLANG_STDLIB_LINK_FLAGS =
        CLANG_STDLIB_LIBS =
        STDLIB_LINK_FLAGS = -static-libgcc -static-libstdc++
        LTO_FLAG = -flto=auto
    else
        $(error Unsupported TOOLCHAIN '$(TOOLCHAIN)'. Use TOOLCHAIN=auto, gcc, or clang)
    endif
    RM   = rm -f
    PYTHON ?= python3
    TARGET = titan
    PIPE_FLAG =
    STACK_NOTE_FLAG = -Wa,--noexecstack
    
    # GCC Flags: Chuẩn C++23, loại bỏ các cờ đặc thù của Clang
    # Keep Eigen on its own aligned-allocation path across Linux builds too.
    CXXFLAGS = $(INCLUDE_DIRS) -O3 $(ARCH_FLAGS) $(OPENMP_FLAG) $(CXX_STDLIB_FLAG) $(CLANG_STDLIB_INCLUDE_FLAGS) $(CLANG_OMP_INCLUDE_FLAGS) $(PIPE_FLAG) $(STACK_NOTE_FLAG) -std=c++23 -Wall -MMD -MP $(COMMON_CXX_DEFINES)
    
    # Linker mặc định của GCC (ld)
    LDFLAGS = $(STDLIB_LINK_FLAGS) $(CLANG_STDLIB_LINK_FLAGS) $(OPENMP_FLAG) $(CXX_STDLIB_FLAG) $(CLANG_STDLIB_LIBS)
    
    # GCC dùng LTO tự động nhận diện số nhân CPU
    
    # Lệnh dọn dẹp trên Linux bash
    CLEAN_CMD = rm -rf $(BUILD_DIR) $(TARGET)
    CLEAN_ALL_CMD = rm -rf $(BUILD_ROOT) $(TARGET)
    CLEAN_GENERATED_CMD = $(RM) network_edges.csv network_nodes.csv network_communities.csv network_community_summary.csv network_supergraph.csv pca_fel.csv tica_projections.csv extract_basins.sh tmp_test_dham*.in 2>/dev/null || true
endif

# ─── Tệp nguồn (Sources) ───
# Opt-in full translation-unit instrumentation. Use a separate BUILD_ROOT so
# sanitized and ordinary objects are never mixed. No default build is changed.
SANITIZERS ?=
ifneq ($(strip $(SANITIZERS)),)
ifeq ($(OS),Windows_NT)
    $(error SANITIZERS is currently supported by the Linux toolchain only)
endif
    CXXFLAGS := $(filter-out -O3 -O2,$(CXXFLAGS)) -O1 -g1 -fno-omit-frame-pointer -fsanitize=$(SANITIZERS)
    LDFLAGS += -fsanitize=$(SANITIZERS)
endif

# KIN-10: the bundled random-forest affinity model is closed pending a model
# card, and the engine no longer emits a pKd. Link the closed stub instead of
# the 199,809-line generated source, which costs ~6 s and ~843 MB of peak
# compiler memory on GCC 13.3 and adds a 3.3-4.9 MB object to the binary.
#
# TITAN_ENABLE_RF_MODEL=1 restores the generated model. Run a clean when you
# change this value: make is timestamp driven and cannot see that the
# configuration changed, so an existing binary would otherwise be left in place
# with the other RF object still linked in.
#   make clean && make TITAN_ENABLE_RF_MODEL=1
TITAN_ENABLE_RF_MODEL ?= 0
ifeq ($(TITAN_ENABLE_RF_MODEL),1)
    RF_MODEL_EXCLUDED_SRC = $(SRC_ROOT)/titan_rf_model_closed.cpp
else
    RF_MODEL_EXCLUDED_SRC = $(SRC_ROOT)/titan_rf_model.cpp
endif

ROOT_SRCS = $(filter-out $(SRC_ROOT)/parse_arguments.cpp $(RF_MODEL_EXCLUDED_SRC),$(wildcard $(SRC_ROOT)/*.cpp))
CLI_SRCS = $(wildcard $(SRC_ROOT)/cli/*.cpp)
PBE_GRID_SRCS = $(wildcard $(SRC_ROOT)/pbe/grid/*.cpp)
BIE_SRCS = $(wildcard $(SRC_ROOT)/bie/*.cpp) \
           $(wildcard $(SRC_ROOT)/bie/fmm/cartesian/*.cpp)
PBE_SRCS  = $(wildcard $(SRC_ROOT)/tabi/*.cpp) $(BIE_SRCS) $(PBE_GRID_SRCS)
PBCTOPO_SRCS = $(wildcard $(SRC_ROOT)/pbctopo/*.cpp) \
               $(wildcard $(SRC_ROOT)/pbctopo/solve/*.cpp)
# HOLOLIFT=0 builds without HoloLift-MD (src/hololift is neither compiled nor
# needed); -pbctopo-temporal=hololift is then refused.
HOLOLIFT ?= 1
ifeq ($(HOLOLIFT),0)
HOLOLIFT_SRCS =
COMMON_CXX_DEFINES += -DTITAN_WITHOUT_HOLOLIFT=1
else
HOLOLIFT_SRCS = $(wildcard $(SRC_ROOT)/hololift/*.cpp)
endif
TITAN_PATH_SRCS = $(wildcard $(SRC_ROOT)/titan_path/*.cpp)
GMXTRAJ_SRCS = $(wildcard $(SRC_ROOT)/gmxtraj/*.cpp)
POSTPROC_SRCS = $(wildcard $(SRC_ROOT)/postproc/*.cpp)
ENGINE_SRCS = $(wildcard $(SRC_ROOT)/engine/*.cpp)
NOCAL_SRCS = $(wildcard $(SRC_ROOT)/nocal/*.cpp)
PTOPO_SRCS = $(wildcard $(SRC_ROOT)/ptopo/*.cpp)
TOPOLOGY_SRCS = $(wildcard $(SRC_ROOT)/topology/*.cpp)
FREE_ENERGY_SRCS = $(wildcard $(SRC_ROOT)/free_energy/*.cpp)
GBSA_SRCS = $(wildcard $(SRC_ROOT)/gbsa/*.cpp)
ENERGY_SRCS = $(wildcard $(SRC_ROOT)/energy/*.cpp)
SRCS  = $(ROOT_SRCS) $(CLI_SRCS) $(PBE_SRCS) $(PBCTOPO_SRCS) $(HOLOLIFT_SRCS) $(TITAN_PATH_SRCS) $(GMXTRAJ_SRCS) $(PTOPO_SRCS) \
        $(POSTPROC_SRCS) $(ENGINE_SRCS) $(NOCAL_SRCS) $(TOPOLOGY_SRCS) $(FREE_ENERGY_SRCS) $(GBSA_SRCS) $(ENERGY_SRCS)
OBJS  = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS  = $(OBJS:.o=.d)

ifeq ($(OS),Windows_NT)
    RESOURCE_OBJ = $(BUILD_DIR)/titan_version_resource.o
    SCHEMA_SMOKE_TARGET = $(BUILD_DIR)/titan_build_schema_smoke.exe
    CLI_VALIDATION_SMOKE_TARGET = $(BUILD_DIR)/titan_cli_validation_smoke.exe
else
    RESOURCE_OBJ =
    SCHEMA_SMOKE_TARGET = $(BUILD_DIR)/titan_build_schema_smoke
    CLI_VALIDATION_SMOKE_TARGET = $(BUILD_DIR)/titan_cli_validation_smoke
endif
LINK_INPUTS = $(OBJS) $(RESOURCE_OBJ)
SCHEMA_SMOKE_CXXFLAGS = $(filter-out -MMD -MP,$(CXXFLAGS))

# ==========================================
# BUILD RULES
# ==========================================

.DELETE_ON_ERROR:
.PHONY: all build-info cli-registry build-metadata-test tracked-includes-test version-smoke cli-validation-test clean clean-all clean-generated release native portable release-native release-portable print-config FORCE

FORCE:

$(BUILD_INFO_HEADER): FORCE $(VERSION_FILE) $(BUILD_INFO_GENERATOR)
	@$(PYTHON) $(BUILD_INFO_GENERATOR) --repo-root . --version-file $(VERSION_FILE) --header-output $(BUILD_INFO_HEADER) --manifest-output $(BUILD_INFO_MANIFEST) --rc-output $(BUILD_INFO_RC) --profile $(PROFILE) --executable-name $(notdir $(TARGET))

build-info: $(BUILD_INFO_HEADER)

$(CLI_REGISTRY_HEADER): $(CLI_PARSER_SOURCE) $(CLI_REGISTRY_GENERATOR)
	@$(PYTHON) $(CLI_REGISTRY_GENERATOR) $(addprefix --parser ,$(CLI_PARSER_SOURCE)) --output $(CLI_REGISTRY_HEADER)

cli-registry: $(CLI_REGISTRY_HEADER)

tracked-includes-test:
	$(PYTHON) tools/check_tracked_includes.py

build-metadata-test:
	@$(PYTHON) $(BUILD_INFO_TEST)

$(SCHEMA_SMOKE_TARGET): tools/titan_build_schema_smoke.cpp $(SRC_ROOT)/titan_config_schema.cpp $(BUILD_INFO_HEADER)
	@echo Compiling build-schema smoke...
	@$(CXX) $(SCHEMA_SMOKE_CXXFLAGS) tools/titan_build_schema_smoke.cpp $(SRC_ROOT)/titan_config_schema.cpp -o $@ $(LDFLAGS)

$(CLI_VALIDATION_SMOKE_TARGET): tools/titan_cli_validation_smoke.cpp $(CLI_REGISTRY_HEADER)
	@echo Compiling CLI validation smoke...
	@$(CXX) $(SCHEMA_SMOKE_CXXFLAGS) tools/titan_cli_validation_smoke.cpp -o $@ $(LDFLAGS)

all: $(TARGET)

$(OBJS): | $(BUILD_INFO_HEADER) $(CLI_REGISTRY_HEADER)

ifeq ($(OS),Windows_NT)
$(RESOURCE_OBJ): $(BUILD_INFO_HEADER) $(BUILD_INFO_GENERATOR)
	@echo Compiling Windows version resource...
	@$(WINDRES) --input-format=rc --output-format=coff -i $(BUILD_INFO_RC) -o $@
endif

$(TARGET): $(LINK_INPUTS)
	@echo Linking $(TARGET) [$(PROFILE)]...
	@$(CXX) -o $@ $^ $(LDFLAGS)
	@echo Build successful!

version-smoke: all $(SCHEMA_SMOKE_TARGET)
	@$(PYTHON) $(BUILD_INFO_TEST) --binary $(TARGET) --manifest $(BUILD_INFO_MANIFEST) --schema-driver $(SCHEMA_SMOKE_TARGET)

cli-validation-test: all $(CLI_VALIDATION_SMOKE_TARGET)
	@$(CLI_VALIDATION_SMOKE_TARGET)
	@$(PYTHON) $(CLI_VALIDATION_TEST) --binary $(TARGET)

$(BUILD_DIR)/$(SRC_ROOT)/titan_rf_model.o: CXXFLAGS := $(filter-out -O3,$(CXXFLAGS)) -O0

ifneq ($(OS),Windows_NT)
$(BUILD_DIR)/$(SRC_ROOT)/titan_rf_model.o: CXXFLAGS := $(filter-out -O3 -pipe,$(CXXFLAGS)) -O0
endif

$(BUILD_DIR)/%.o: %.cpp
	@echo Compiling $<...
ifeq ($(OS),Windows_NT)
	@$(MKDIR_CMD)
else
	@mkdir -p $(@D)
endif
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo Cleaning up...
	@-$(CLEAN_CMD)
	@echo Clean done!

clean-all:
	@echo Cleaning all profile build directories...
	@-$(CLEAN_ALL_CMD)
	@echo Clean all done!

clean-generated:
	@echo Cleaning generated analysis outputs...
	@-$(CLEAN_GENERATED_CMD)
	@echo Clean generated done!

# Rule 'release' tự động gắn cờ LTO phù hợp với hệ điều hành
release: CXXFLAGS += $(LTO_FLAG)
release: LDFLAGS += $(LTO_FLAG)
release: clean all

native: PROFILE := native
native: clean all

portable: PROFILE := portable
portable: clean all

release-native: PROFILE := native
release-native: release

release-portable: PROFILE := portable
release-portable: release

print-config:
	@echo PROFILE=$(PROFILE)
	@echo TOOLCHAIN=$(TOOLCHAIN)
	@echo LLVM_VERSION=$(LLVM_VERSION)
	@echo LLVM_PREFIX=$(LLVM_PREFIX)
	@echo CXX=$(CXX)
	@echo TITAN_CXX23_FLOOR=$(TITAN_CXX23_FLOOR)
	@echo BUILD_DIR=$(BUILD_DIR)
	@echo GENERATED_DIR=$(GENERATED_DIR)
	@echo VERSION_FILE=$(VERSION_FILE)
	@echo BUILD_INFO_MANIFEST=$(BUILD_INFO_MANIFEST)
	@echo CLI_REGISTRY_HEADER=$(CLI_REGISTRY_HEADER)
	@echo PYTHON=$(PYTHON)
	@echo ARCH_FLAGS=$(ARCH_FLAGS)
	@echo COMMON_CXX_DEFINES=$(COMMON_CXX_DEFINES)
	@echo OPENMP_FLAG=$(OPENMP_FLAG)
	@echo CXX_STDLIB_FLAG=$(CXX_STDLIB_FLAG)
	@echo CLANG_STDLIB_INCLUDE_FLAGS=$(CLANG_STDLIB_INCLUDE_FLAGS)
	@echo CLANG_OMP_INCLUDE_FLAGS=$(CLANG_OMP_INCLUDE_FLAGS)
	@echo CLANG_STDLIB_LINK_FLAGS=$(CLANG_STDLIB_LINK_FLAGS)
	@echo CLANG_STDLIB_LIBS=$(CLANG_STDLIB_LIBS)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo LDFLAGS=$(LDFLAGS)

# Nạp cây phụ thuộc (Dependency Tracking)
-include $(DEPS)
