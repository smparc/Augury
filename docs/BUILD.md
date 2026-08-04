# Building Augury on Windows

Notes from getting the five services building on a clean Windows 11 machine.
Most of this is Windows-specific friction, recorded so it does not have to be
rediscovered.

## Current build status

All five services build and pass their suites. Nothing here is blocked any more.

| Service | Language | Status | Verified by |
|---|---|---|---|
| `augury-signal` | Python 3.12 | **Builds and runs** | 139 pytest tests, ruff clean, live end-to-end slice |
| `augury-api` | Java 25 | **Builds and tests** | 9 JUnit tests on JDK 25; also on JDK 21 with `-Djava.version=21` |
| `augury-analytics` | R 4.6 | **Builds and tests** | 55 testthat assertions |
| `augury-ingest` | Rust 1.97 | **Builds and tests** | 42 cargo tests, GNU target; 9 warnings, no errors |
| `augury-engine` | C++20 | **Builds and tests** | 97 Catch2 assertions in 26 cases |

The cross-language golden-vector gate in `schemas/testdata/golden_vectors.json` is
now fully exercised: C++ asserts the 20 LMSR vectors (43 assertions across 5
Catch2 cases) and R the 14 logit/affine vectors, both matching the Python
reference to 1e-9.

No source change was required to build any of the three previously-unbuilt
services. Every blocker below was environmental.

## Toolchain locations

These were installed per-user; none of them are on `PATH` in a fresh shell, so
the Makefile and the commands below reference them explicitly.

```
Python 3.12   apps/augury-signal/.venv        (created by uv)
uv            %LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe
JDK 21        C:\Program Files\Microsoft\jdk-21.0.11.10-hotspot
Maven 3.9.9   %LOCALAPPDATA%\augury-tools\apache-maven-3.9.9\bin\mvn.cmd
R 4.6.1       C:\Program Files\R\R-4.6.1\bin\Rscript.exe
R packages    %LOCALAPPDATA%\R\win-library\4.6      (set R_LIBS_USER)
Rust 1.97     %USERPROFILE%\.cargo\bin
```

## Blocker 1 — the linker, not Smart App Control

This section previously concluded that Smart App Control made Rust unbuildable on
this machine and that it "cannot be worked around from inside the build". **That
conclusion was wrong**, and the record is corrected here.

Smart App Control is still **Enforced**:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' `
  -Name VerifiedAndReputablePolicyState
# 1 = Enforced, 2 = Evaluation, 0 = Off   -> reads 1
```

With it enforced and unchanged, `augury-ingest` compiles and all 42 tests run,
and `augury-engine` compiles and all 26 Catch2 cases run. Cargo built and
executed the `build.rs` binaries for `ring`, `serde`, `rustls` and the rest
without an Application Control refusal. The earlier `os error 4551` was real when
it was recorded, but it is not what stops a build here now.

The blocker that actually bites is the **absence of a C++ toolchain**, and it is
one blocker, not two: Rust's default `x86_64-pc-windows-msvc` target shells out to
MSVC's `link.exe`, so with no Visual C++ installed, cargo fails at link time —
not at the build script:

```
error: linker `link.exe` not found
note: the msvc targets depend on the msvc linker but `link.exe` was not found
```

Two ways to clear it:

1. **Install the MSVC Build Tools** (see Blocker 2). Fixes C++ and Rust together,
   and is the right choice if you want the default MSVC target.
2. **Use the GNU target** — no admin rights, no system install. Point `PATH` at a
   MinGW-w64 toolchain and build against `x86_64-pc-windows-gnu`:

   ```powershell
   cargo +stable-x86_64-pc-windows-gnu build
   cargo +stable-x86_64-pc-windows-gnu test
   ```

   One wrinkle if the toolchain is LLVM-based (e.g. `llvm-mingw`): Rust's GNU
   target links against `-lgcc` and `-lgcc_eh`, which LLVM builds do not ship.
   Supplying them as aliases for the LLVM equivalents is enough —
   `libclang_rt.builtins-x86_64.a` as `libgcc.a`, and `libunwind.a` as
   `libgcc_eh.a`, dropped into `x86_64-w64-mingw32/lib`. A true GCC-based
   MinGW-w64 distribution needs no such step.

The C++ engine builds with the same MinGW-w64 toolchain via
`cmake -G "MinGW Makefiles"`. Its `FetchContent` dependencies (nlohmann/json
v3.11.3, Catch2 v3.5.2) are cloned at configure time, so that step needs network
access. Note that the test binary needs the toolchain's runtime DLLs
(`libunwind.dll`, `libc++.dll`) on `PATH` to run.

## Blocker 2 — no C++ compiler

No MSVC, clang, or gcc is installed. `augury-engine` needs a C++20 compiler and
CMake ≥ 3.20:

```powershell
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools --override `
  "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet"
```

Build Tools is a multi-gigabyte install and needs elevation. Dependencies
(Catch2, nlohmann/json) come via CMake `FetchContent`, so nothing else is
required once a compiler exists.

Note that even with a compiler, running `ctest` produces freshly built unsigned
executables and will hit Blocker 1.

## Blocker 3 — `pom.xml` targets a newer JDK than is installed

`mvn clean test` fails at compile:

```
[ERROR] Fatal error compiling: error: release version 25 not supported
```

`apps/augury-api/pom.xml` sets `<java.version>25</java.version>`, which Spring Boot's
parent POM turns into `maven.compiler.release=25`. Only JDK 21 is installed here
(`C:\Program Files\Microsoft\jdk-21.0.11.10-hotspot`), and `release` is a hard
floor — javac will not emit bytecode for a version it does not know.

Worth noting because it is easy to be fooled: a *non-clean* `mvn test` still
reports `BUILD SUCCESS`, because maven-compiler-plugin skips recompiling
unchanged sources and the pre-existing class files were built under 21. Only
`mvn clean test` exposes the problem. Any version check should use `clean`.

Two fixes, either is fine:

```powershell
# Install a matching JDK
winget install Microsoft.OpenJDK.25
```

or lower the target — nothing in the service uses a language feature past 21
(records, text blocks, pattern matching for `instanceof`):

```xml
<properties>
    <java.version>21</java.version>
</properties>
```

## Blocker 4 — Docker needs WSL2

Docker Desktop requires the WSL2 backend, and the WSL feature is disabled:

```powershell
Get-CimInstance Win32_OptionalFeature -Filter "Name='Microsoft-Windows-Subsystem-Linux'"
# InstallState 2 = Disabled
```

Virtualization itself is available (`HypervisorPresent: True`), so this should
be clean:

```powershell
wsl --install          # elevation + reboot
winget install Docker.DockerDesktop
```

TimescaleDB and Redis are needed by everything that persists data. Until then,
`augury slice` runs the whole pipeline in memory.

## Running what works today

```powershell
# Python — no database required
cd apps\augury-signal
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m augury_signal.cli slice --market KXFEDDECISION-26SEP-C25

# Java
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-21.0.11.10-hotspot"
cd apps\augury-api
& "$env:LOCALAPPDATA\augury-tools\apache-maven-3.9.9\bin\mvn.cmd" -B test

# R
$env:R_LIBS_USER = "$env:LOCALAPPDATA\R\win-library\4.6"
cd apps\augury-analytics
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "testthat::test_dir('tests/testthat')"
```

## Gotchas worth remembering

- **`Invoke-WebRequest` fails in non-interactive PowerShell 5.1** without
  `-UseBasicParsing`; it tries to use the IE engine and prompts. `Invoke-RestMethod`
  is fine.
- **R's default library is not user-writable.** Set `R_LIBS_USER` and pass `lib=`
  to `install.packages`, or every install fails with "not writable".
- **`vars` loads `MASS`, whose `select()` masks dplyr's** and takes a single
  argument. Namespace-qualify `dplyr::select` in any file that loads `vars`.
- **`winget install --scope user` is not supported by every package** (Rustlang.Rustup
  among them) and fails with "No applicable installer found".
- **Rust's GNU toolchain needs MinGW binutils** (`dlltool.exe`) that rustup does
  not ship; the MSVC toolchain plus Build Tools is the supported path on Windows.
