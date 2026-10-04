#!/usr/bin/env python3
"""Generate deterministic TITAN build metadata for C++ and release tooling."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping


SEMVER_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got {value!r}")


def parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def read_product_version(version_file: Path, env: Mapping[str, str]) -> tuple[str, tuple[int, int, int]]:
    version = env.get("TITAN_VERSION_OVERRIDE", "").strip()
    if not version:
        try:
            version = version_file.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError as exc:
            raise ValueError(f"version file not found: {version_file}") from exc
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise ValueError(
            f"TITAN version must be a SemVer MAJOR.MINOR.PATCH triplet, got {version!r}"
        )
    return version, tuple(int(part) for part in match.groups())


def source_date(repo_root: Path, env: Mapping[str, str]) -> tuple[int, str]:
    raw_epoch = env.get("SOURCE_DATE_EPOCH", "").strip()
    if not raw_epoch:
        raw_epoch = run_git(repo_root, "show", "-s", "--format=%ct", "HEAD") or "0"
    epoch = parse_nonnegative_int(raw_epoch, "SOURCE_DATE_EPOCH")
    if epoch == 0:
        return 0, "unknown"
    utc = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
    return epoch, utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def compose_full_version(
    version: str,
    build_number: int,
    short_commit: str,
    dirty: bool,
    release_build: bool,
) -> str:
    if release_build:
        return version
    metadata = f"g{short_commit}" if short_commit != "unknown" else "source.unknown"
    if dirty:
        metadata += ".dirty"
    return f"{version}-dev.{build_number}+{metadata}"


def collect_build_info(
    repo_root: Path,
    version_file: Path,
    profile: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    values = os.environ if env is None else env
    if not SAFE_TOKEN_RE.fullmatch(profile):
        raise ValueError(f"invalid build profile {profile!r}")

    version, numeric_version = read_product_version(version_file, values)

    commit = values.get("TITAN_GIT_COMMIT", "").strip()
    if not commit:
        commit = run_git(repo_root, "rev-parse", "HEAD") or "unknown"
    if commit != "unknown" and not GIT_SHA_RE.fullmatch(commit):
        raise ValueError("TITAN_GIT_COMMIT must be a hexadecimal Git object ID")
    commit = commit.lower()
    short_commit = commit[:12] if commit != "unknown" else "unknown"

    build_override = values.get("TITAN_BUILD_NUMBER", "").strip()
    if build_override:
        build_number = parse_nonnegative_int(build_override, "TITAN_BUILD_NUMBER")
    else:
        commit_count = run_git(repo_root, "rev-list", "--count", "HEAD")
        build_number = parse_nonnegative_int(commit_count or "0", "Git commit count")

    dirty_override = values.get("TITAN_GIT_DIRTY", "").strip()
    if dirty_override:
        dirty = parse_bool(dirty_override, "TITAN_GIT_DIRTY")
    else:
        status = run_git(
            repo_root, "status", "--porcelain=v1", "--untracked-files=normal"
        )
        dirty = bool(status) if status is not None else False

    expected_tag = f"titan-v{version}"
    tag_override = values.get("TITAN_GIT_TAG", "").strip()
    if tag_override:
        exact_tag = tag_override
    else:
        exact_tag = (
            run_git(repo_root, "tag", "--points-at", "HEAD", "--list", expected_tag)
            or ""
        )

    release_override = values.get("TITAN_RELEASE_BUILD", "").strip()
    if release_override:
        release_build = parse_bool(release_override, "TITAN_RELEASE_BUILD")
    else:
        release_build = exact_tag == expected_tag and not dirty
    if release_build and exact_tag != expected_tag:
        raise ValueError(
            f"release build requires exact tag {expected_tag!r}, got {exact_tag!r}"
        )
    if release_build and dirty:
        raise ValueError("a release build cannot be marked dirty")
    if release_build and commit == "unknown":
        raise ValueError("a release build requires a known Git commit")

    ci_run_number = (
        values.get("TITAN_CI_BUILD_NUMBER", "").strip()
        or values.get("GITHUB_RUN_NUMBER", "").strip()
    )
    if ci_run_number:
        parse_nonnegative_int(ci_run_number, "CI build number")
    ci_run_id = (
        values.get("TITAN_CI_RUN_ID", "").strip()
        or values.get("GITHUB_RUN_ID", "").strip()
    )
    if ci_run_id and not ci_run_id.isdigit():
        raise ValueError("CI run ID must contain digits only")
    if values.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        ci_provider = "github-actions"
    elif values.get("CI", "").strip().lower() in {"1", "true", "yes"}:
        ci_provider = "ci"
    else:
        ci_provider = "local"
    embed_ci_raw = values.get("TITAN_EMBED_CI_METADATA", "").strip()
    embed_ci_metadata = (
        parse_bool(embed_ci_raw, "TITAN_EMBED_CI_METADATA")
        if embed_ci_raw
        else False
    )

    epoch, source_date_utc = source_date(repo_root, values)
    full_version = compose_full_version(
        version, build_number, short_commit, dirty, release_build
    )

    return {
        "schema_version": 1,
        "product_name": "TITAN",
        "version": version,
        "version_major": numeric_version[0],
        "version_minor": numeric_version[1],
        "version_patch": numeric_version[2],
        "full_version": full_version,
        "build_number": build_number,
        "git_commit": commit,
        "git_commit_short": short_commit,
        "git_dirty": dirty,
        "git_tag": exact_tag,
        "release_build": release_build,
        "source_date_epoch": epoch,
        "source_date_utc": source_date_utc,
        "profile": profile,
        "ci_provider": ci_provider,
        "ci_build_number": ci_run_number,
        "ci_run_id": ci_run_id,
        "ci_metadata_embedded": embed_ci_metadata,
    }


def cpp_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def render_header(info: Mapping[str, object]) -> str:
    embed_ci = bool(info["ci_metadata_embedded"])
    embedded_provider = info["ci_provider"] if embed_ci else "not-embedded"
    embedded_build_number = info["ci_build_number"] if embed_ci else ""
    embedded_run_id = info["ci_run_id"] if embed_ci else ""
    return f"""// Generated by tools/generate_build_info.py. Do not edit.
#ifndef TITAN_VERSION_GENERATED_H
#define TITAN_VERSION_GENERATED_H

#define TITAN_BUILD_METADATA_GENERATED 1
#define TITAN_PRODUCT_VERSION {cpp_string(info['version'])}
#define TITAN_FULL_VERSION {cpp_string(info['full_version'])}
#define TITAN_BUILD_NUMBER {info['build_number']}ULL
#define TITAN_BUILD_NUMBER_STRING {cpp_string(info['build_number'])}
#define TITAN_VERSION_MAJOR {info['version_major']}
#define TITAN_VERSION_MINOR {info['version_minor']}
#define TITAN_VERSION_PATCH {info['version_patch']}
#define TITAN_GIT_COMMIT {cpp_string(info['git_commit'])}
#define TITAN_GIT_COMMIT_SHORT {cpp_string(info['git_commit_short'])}
#define TITAN_GIT_DIRTY {1 if info['git_dirty'] else 0}
#define TITAN_GIT_TAG {cpp_string(info['git_tag'])}
#define TITAN_RELEASE_BUILD {1 if info['release_build'] else 0}
#define TITAN_SOURCE_DATE_EPOCH {info['source_date_epoch']}ULL
#define TITAN_SOURCE_DATE_UTC {cpp_string(info['source_date_utc'])}
#define TITAN_CI_PROVIDER {cpp_string(embedded_provider)}
#define TITAN_CI_BUILD_NUMBER {cpp_string(embedded_build_number)}
#define TITAN_CI_RUN_ID {cpp_string(embedded_run_id)}
#define TITAN_GENERATED_BUILD_PROFILE {cpp_string(info['profile'])}

#endif
"""


def rc_string(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def render_windows_resource(info: Mapping[str, object], executable_name: str) -> str:
    numeric_build = min(int(info["build_number"]), 65535)
    flags = 0
    if not bool(info["release_build"]):
        flags |= 0x2  # VS_FF_PRERELEASE
    if bool(info["git_dirty"]):
        flags |= 0x8  # VS_FF_PRIVATEBUILD
    embedded_ci_build = (
        info["ci_build_number"] if bool(info.get("ci_metadata_embedded", False)) else ""
    )
    return f"""// Generated by tools/generate_build_info.py. Do not edit.
#include <winver.h>

1 VERSIONINFO
 FILEVERSION {info['version_major']},{info['version_minor']},{info['version_patch']},{numeric_build}
 PRODUCTVERSION {info['version_major']},{info['version_minor']},{info['version_patch']},0
 FILEFLAGSMASK VS_FFI_FILEFLAGSMASK
 FILEFLAGS 0x{flags:02x}L
 FILEOS VOS_NT_WINDOWS32
 FILETYPE VFT_APP
 FILESUBTYPE VFT2_UNKNOWN
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "FileDescription", "TITAN scientific computing engine\\0"
            VALUE "FileVersion", "{rc_string(info['full_version'])}\\0"
            VALUE "InternalName", "TITAN\\0"
            VALUE "OriginalFilename", "{rc_string(executable_name)}\\0"
            VALUE "ProductName", "TITAN\\0"
            VALUE "ProductVersion", "{rc_string(info['version'])}\\0"
            VALUE "PrivateBuild", "{rc_string(info['git_commit_short'])}\\0"
            VALUE "SpecialBuild", "{rc_string(embedded_ci_build)}\\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"""


def write_if_changed(path: Path, content: str) -> bool:
    normalized = content.replace("\r\n", "\n")
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(normalized)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--version-file", type=Path, required=True)
    parser.add_argument("--header-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--rc-output", type=Path)
    parser.add_argument("--profile", default="unknown")
    parser.add_argument("--executable-name", default="titan.exe")
    parser.add_argument("--print-full-version", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    version_file = (
        args.version_file
        if args.version_file.is_absolute()
        else repo_root / args.version_file
    )
    info = collect_build_info(repo_root, version_file, args.profile)

    write_if_changed(args.header_output, render_header(info))
    manifest = json.dumps(info, indent=2, sort_keys=True) + "\n"
    write_if_changed(args.manifest_output, manifest)
    if args.rc_output is not None:
        write_if_changed(
            args.rc_output,
            render_windows_resource(info, args.executable_name),
        )
    if args.print_full_version:
        print(info["full_version"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"generate_build_info: {error}") from error
