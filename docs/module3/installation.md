# Module 3: 依赖与安装

本模块依赖 acados 优化框架（C 库 + Python 绑定），需从源码编译。以下为经过验证的完整安装流程。

## 11.1 系统要求

- macOS 或 Linux（以下以 macOS 为例）
- C 编译器（Xcode Command Line Tools: `xcode-select --install`）
- CMake >= 3.17
- Python >= 3.8

## 11.2 Python 依赖

```bash
pip install casadi>=3.6.0 numpy scipy matplotlib
```

经验证的版本组合：

| 包 | 版本 |
|---|---|
| casadi | 3.7.2 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| matplotlib | 3.9.2 |

## 11.3 acados 源码编译

acados 安装在项目的 `external/acados/` 目录下，与项目代码一起管理（但通过 `.gitignore` 排除）。

### 步骤 1: 获取源码

```bash
cd airforce/external
git clone https://github.com/acados/acados.git
cd acados
git submodule update --init --recursive
```

### 步骤 2: CMake 构建

```bash
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$(cd .. && pwd)" \
    -DACADOS_WITH_QPOASES=ON \
    -DACADOS_WITH_OSQP=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release
make install -j$(sysctl -n hw.ncpu)
```

关键 CMake 选项说明：
- `CMAKE_INSTALL_PREFIX`: 设为 acados 源码根目录，使 `lib/`、`include/` 直接位于 `external/acados/` 下
- `ACADOS_WITH_QPOASES`: 启用 qpOASES QP 求解器（备选）
- `ACADOS_WITH_OSQP`: 启用 OSQP QP 求解器（备选）
- `BUILD_SHARED_LIBS`: 构建动态链接库（`.dylib` / `.so`）

构建完成后 `external/acados/lib/` 下应有：
```
libacados.dylib
libhpipm.dylib
libblasfeo.dylib
libqpOASES_e.dylib
libosqp.dylib
```

### 步骤 3: 下载 tera_renderer

acados 代码生成依赖 `tera_renderer` 模板引擎二进制。需手动下载放入 `bin/` 目录。

**macOS x86_64:**
```bash
cd external/acados
mkdir -p bin
# 从 GitHub releases 下载 v0.2.0
curl -L https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-osx-x86_64.tar.gz \
    -o bin/t_renderer.tar.gz
cd bin && tar xzf t_renderer.tar.gz && rm t_renderer.tar.gz
chmod +x t_renderer
```

**macOS ARM64 (Apple Silicon):**
```bash
# ARM64 二进制可能不可用，使用 x86_64 版本通过 Rosetta 运行：
curl -L https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-osx-x86_64.tar.gz \
    -o bin/t_renderer.tar.gz
cd bin && tar xzf t_renderer.tar.gz && rm t_renderer.tar.gz
chmod +x t_renderer
```

**Linux x86_64:**
```bash
curl -L https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-x86_64.tar.gz \
    -o bin/t_renderer.tar.gz
cd bin && tar xzf t_renderer.tar.gz && rm t_renderer.tar.gz
chmod +x t_renderer
```

验证：`./external/acados/bin/t_renderer --version`

### 步骤 4: 安装 Python 绑定

```bash
pip install -e external/acados/interfaces/acados_template
```

这将以开发模式安装 `acados_template` 包，自动关联到编译好的 C 库。

### 步骤 5: macOS rpath 修复

macOS 上 acados 生成的求解器 `.dylib` 使用 `@rpath` 引用底层库。需将 acados 的 `lib/` 目录注册到 rpath：

```bash
ACADOS_LIB="$(cd external/acados/lib && pwd)"

# 对所有 acados 库添加 rpath
for lib in $ACADOS_LIB/libacados.dylib $ACADOS_LIB/libhpipm.dylib $ACADOS_LIB/libblasfeo.dylib; do
    install_name_tool -add_rpath "$ACADOS_LIB" "$lib" 2>/dev/null || true
done
```

这确保运行时动态链接器能找到 `libhpipm.dylib`、`libblasfeo.dylib` 等依赖。

## 11.4 环境变量

Module 3 的代码在 `module3/mhe_solver.py` 中自动设置环境变量，无需手动 export。但如果在其他脚本中直接使用 acados，需要：

```bash
export ACADOS_SOURCE_DIR="/path/to/airforce/external/acados"

# macOS
export DYLD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$DYLD_LIBRARY_PATH"

# Linux
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
```

`mhe_solver.py` 中的自动设置逻辑：
```python
_ACADOS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'external', 'acados')
os.environ.setdefault('ACADOS_SOURCE_DIR', os.path.abspath(_ACADOS_ROOT))
_lib_path = os.path.join(_ACADOS_ROOT, 'lib')
os.environ['DYLD_LIBRARY_PATH'] = _lib_path + ':' + os.environ.get('DYLD_LIBRARY_PATH', '')
```

## 11.5 验证安装

```bash
cd airforce

# 1. 验证 Python 导入
python -c "from acados_template import AcadosOcp, AcadosOcpSolver; print('acados_template OK')"
python -c "import casadi; print('casadi', casadi.__version__)"

# 2. 运行 Module 3 单元测试（不需要 acados solver，仅测试 ODE 和投影模型）
python -c "
from module3.shuttlecock_model import integrate_trajectory
from module3.measurement_model import project_world_to_pixel
import numpy as np
traj = integrate_trajectory(np.array([0,2,20,10]), 0.217, 1/30, 30)
print(f'ODE integration OK: {traj.shape}')
"

# 3. 运行完整测试套件（含 acados solver 编译和求解）
python -m tests.test_module3.run_tests
```

首次运行 `run_tests` 时，acados 会为每种 N 值生成 C 代码并编译（约 5 秒/种），后续运行使用缓存。

## 11.6 .gitignore 配置

以下目录应在 `.gitignore` 中：
```
/external              # acados 源码和编译产物
/c_generated_code/     # acados 运行时生成的 C 代码
__pycache__/
```

## 11.7 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Library not loaded: @rpath/libhpipm.dylib` | macOS rpath 未设置 | 执行 §11.3 步骤 5 的 rpath 修复 |
| `t_renderer not found` | tera_renderer 未下载 | 执行 §11.3 步骤 3 |
| `ImportError: No module named 'acados_template'` | Python 绑定未安装 | `pip install -e external/acados/interfaces/acados_template` |
| SQP status=1, iter=0 | QP 求解失败（初始化太差或权重不合理） | 检查初始猜测质量；增大 `levenberg_marquardt`；启用 `MERIT_BACKTRACKING` 全局化 |
| `acados was compiled without OpenMP` | 编译时未启用 OpenMP | 可忽略，不影响功能（仅影响并行性能，MHE 问题规模小无需并行） |
| CasADi `reshape` 与 numpy `flatten` 不匹配 | CasADi reshape 为列优先（Fortran order），numpy flatten 为行优先（C order） | 使用 `horzcat/vertcat` 显式索引构建矩阵，避免 `ca.reshape` |
