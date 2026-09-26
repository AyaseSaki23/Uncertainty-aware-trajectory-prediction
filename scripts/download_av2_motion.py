"""Download AV2 Motion Forecasting splits from public S3 with resumable sync.

The script is intended for AutoDL/Linux. It stores raw scenarios below the
project's ``data/raw/av2`` directory, clears proxy variables that interfere
with AWS S3, and uses ``s5cmd sync --size-only`` so interrupted downloads can
be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REMOTE_BASE = "s3://argoverse/datasets/av2/motion-forecasting"
VALID_SPLITS = ("train", "val", "test")
PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def parse_splits(values: Sequence[str]) -> list[str]:
    """Normalize comma- or space-separated split arguments while preserving order."""

    splits: list[str] = []
    for value in values:
        for split in value.split(","):
            normalized = split.strip().lower()
            if not normalized:
                continue
            if normalized not in VALID_SPLITS:
                raise ValueError(
                    f"Unknown split {normalized!r}; choose from {', '.join(VALID_SPLITS)}."
                )
            if normalized not in splits:
                splits.append(normalized)
    if not splits:
        raise ValueError("At least one AV2 split is required.")
    return splits


def without_proxy_variables(environment: dict[str, str]) -> dict[str, str]:
    """Return a child-process environment without HTTP proxy overrides."""

    cleaned = environment.copy()
    for variable in PROXY_VARIABLES:
        cleaned.pop(variable, None)
    return cleaned


def build_sync_command(
    s5cmd: str,
    *,
    split: str,
    destination: Path,
    workers: int,
    dry_run: bool,
    log_level: str,
) -> list[str]:
    """Build one resumable split download command."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported AV2 split: {split!r}")
    if workers <= 0:
        raise ValueError(f"workers must be positive; received {workers}.")
    command = [
        s5cmd,
        "--no-sign-request",
        "--numworkers",
        str(workers),
        "--log",
        log_level,
        "--stat",
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "sync",
            "--size-only",
            f"{REMOTE_BASE}/{split}/*",
            f"{destination.as_posix().rstrip('/')}/",
        ]
    )
    return command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_pid_file(path: Path) -> None:
    if path.is_file():
        try:
            existing_pid = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"Another AV2 download appears to be running with PID {existing_pid}. "
                f"Check {path} before starting another process."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("av2_motion_download")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _write_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_s5cmd(requested: str | None) -> str:
    candidate = requested or shutil.which("s5cmd")
    if candidate is None:
        raise FileNotFoundError(
            "s5cmd was not found in PATH. Activate the trajpred environment or pass "
            "--s5cmd /root/miniconda3/envs/trajpred/bin/s5cmd."
        )
    path = Path(candidate).expanduser()
    if path.parent != Path(".") and not path.is_file():
        raise FileNotFoundError(f"s5cmd executable does not exist: {path}")
    return str(path)


def _check_free_space(data_root: Path, minimum_free_gb: float) -> float:
    free_gb = shutil.disk_usage(data_root).free / (1024**3)
    if minimum_free_gb > 0 and free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB are free at {data_root}; at least "
            f"{minimum_free_gb:.1f} GiB were requested. Expand the AutoDL data disk or "
            "explicitly lower --min-free-gb after checking the remaining download size."
        )
    return free_gb


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to download; accepts 'train val' or 'train,val' (default: train val).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "raw" / "av2",
        help="Destination root containing train/val/test directories.",
    )
    parser.add_argument("--workers", type=int, default=32, help="s5cmd worker count.")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=70.0,
        help="Abort before download when available space is below this value; 0 disables.",
    )
    parser.add_argument("--s5cmd", help="Explicit s5cmd executable path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask s5cmd to list planned transfers without downloading files.",
    )
    parser.add_argument(
        "--s5cmd-log-level",
        choices=("trace", "debug", "info", "error"),
        default="error",
        help="s5cmd output verbosity stored in the download log.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=project_root / "logs" / "download_av2_motion.log",
        help="Append Python and s5cmd output to this file.",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=project_root / "logs" / "download_av2_motion.pid",
        help="PID guard used to prevent duplicate download processes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        splits = parse_splits(args.splits)
        if args.workers <= 0:
            raise ValueError("--workers must be positive.")
        if args.min_free_gb < 0:
            raise ValueError("--min-free-gb cannot be negative.")
        s5cmd = _resolve_s5cmd(args.s5cmd)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    data_root = args.data_root.expanduser().resolve()
    log_path = args.log_file.expanduser().resolve()
    pid_path = args.pid_file.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(log_path)

    try:
        _acquire_pid_file(pid_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    status_path = data_root / "download_status.json"
    status: dict[str, object] = {
        "dataset": "argoverse2_motion_forecasting",
        "remote_base": REMOTE_BASE,
        "splits": splits,
        "data_root": str(data_root),
        "workers": args.workers,
        "dry_run": args.dry_run,
        "started_at_utc": _utc_now(),
        "state": "running",
        "completed_splits": [],
    }
    _write_status(status_path, status)

    try:
        free_gb = _check_free_space(data_root, args.min_free_gb)
        logger.info("AV2 download root: %s", data_root)
        logger.info("Selected splits: %s", ", ".join(splits))
        logger.info("Available space before download: %.1f GiB", free_gb)
        logger.info("Proxy variables are removed for s5cmd child processes.")
        child_environment = without_proxy_variables(dict(os.environ))

        for split in splits:
            destination = data_root / split
            destination.mkdir(parents=True, exist_ok=True)
            command = build_sync_command(
                s5cmd,
                split=split,
                destination=destination,
                workers=args.workers,
                dry_run=args.dry_run,
                log_level=args.s5cmd_log_level,
            )
            logger.info("Starting %s: %s", split, shlex.join(command))
            with log_path.open("a", encoding="utf-8") as output:
                result = subprocess.run(
                    command,
                    env=child_environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"s5cmd failed for split {split!r} with exit code {result.returncode}. "
                    f"Inspect {log_path}; rerun the same command to resume safely."
                )
            completed = status["completed_splits"]
            assert isinstance(completed, list)
            completed.append(split)
            _write_status(status_path, status)
            logger.info("Completed split: %s", split)

        status["state"] = "dry_run_complete" if args.dry_run else "complete"
        status["completed_at_utc"] = _utc_now()
        status["free_gib_after"] = round(
            shutil.disk_usage(data_root).free / (1024**3), 3
        )
        _write_status(status_path, status)
        logger.info("AV2 download finished with state=%s", status["state"])
        logger.info("Status metadata: %s", status_path)
        return 0
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        status["finished_at_utc"] = _utc_now()
        _write_status(status_path, status)
        logger.warning("Download interrupted. Run the same command again to resume.")
        return 130
    except Exception as exc:
        status["state"] = "failed"
        status["finished_at_utc"] = _utc_now()
        status["error"] = str(exc)
        _write_status(status_path, status)
        logger.error("%s", exc)
        return 1
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
