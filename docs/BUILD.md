# Building Augury on Windows

Notes from getting the five services building on a clean Windows 11 machine.
Most of this is Windows-specific friction, recorded so it does not have to be
rediscovered.

## Current build status

| Service | Language | Status | Verified by |
|---|---|---|---|
| `augury-signal` | Python 3.12 | **Builds and runs** | 139 pytest tests, ruff clean, live end-to-end slice |
| `augury-api` | Java 25 | **Blocked — JDK version** | 9 JUnit tests passed under JDK 21, before `pom.xml` was raised to 25 |
| `augury-analytics` | R 4.6 | **Builds and tests** | 55 testthat tests |
| `augury-ingest` | Rust 1.97 | **Blocked — see Smart App Control** | not compiled |
| `augury-engine` | C++20 | **Blocked — no compiler installed** | not compiled |

Python, Java, and R all agree on the shared golden vectors in
`schemas/testdata/golden_vectors.json`, which is the cross-language correctness
gate. The C++ suite asserts the same vectors and will exercise that gate once it
can be compiled.

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

## Blocker 1 — Smart App Control blocks Rust builds

`cargo build` fails with:

```
error: failed to run custom build command for `generic-array v0.14.7`
Caused by: An Application Control policy has blocked this file. (os error 4551)
```

Smart App Control is **Enforced** on this machine:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' `
  -Name VerifiedAndReputablePolicyState
# 1 = Enforced, 2 = Evaluation, 0 = Off
```

Cargo compiles each dependency's `build.rs` into a fresh, unsigned executable
and runs it. Smart App Control blocks unsigned binaries without established
reputation, so every crate with a build script fails. This affects any toolchain
that compiles-then-executes locally, which includes the C++ tests and benchmarks.

**This cannot be worked around from inside the build.** Turning Smart App Control
off is a deliberate, effectively irreversible decision — Windows cannot re-enable
it without resetting the machine — so it is the owner's call, not a build step.

Options, roughly in order of preference:

1. **Build Rust and C++ in CI or a container**, leaving this machine for the
   Python/Java/R work that already runs. Nothing about the architecture requires
   all five services to build in one place.
2. **Build inside WSL2**, which is a separate Linux userland and unaffected.
   Needed for Docker anyway.
3. **Turn Smart App Control off** (Windows Security → App & browser control →
   Smart App Control → Off). Irreversible without a Windows reset.

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
