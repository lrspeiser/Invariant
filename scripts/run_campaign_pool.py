from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a bounded pool of task-type-filtered Sigma campaign workers."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--task-types", required=True)
    parser.add_argument("--duration", default="6h")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--log-directory", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and the measured useful maximum of 20")
    task_types = sorted({item.strip() for item in args.task_types.split(",") if item.strip()})
    if not task_types:
        raise ValueError("at least one task type is required for pooled execution")
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_root = args.log_directory.resolve() / stamp
    log_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[subprocess.Popen[bytes], BinaryIO, BinaryIO]] = []
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for index in range(args.workers):
            worker_id = f"pool-{stamp}-{index:02d}"
            stdout_handle = (log_root / f"{worker_id}.stdout.log").open("wb")
            stderr_handle = (log_root / f"{worker_id}.stderr.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "sigma_theory_compiler.campaign_cli",
                "run",
                "--database",
                str(database),
                "--worker-id",
                worker_id,
                "--duration",
                args.duration,
                "--follow",
                "--poll-seconds",
                str(args.poll_seconds),
                "--task-types",
                ",".join(task_types),
            ]
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creation_flags,
            )
            processes.append((process, stdout_handle, stderr_handle))
        print(f"workers={len(processes)}")
        print(f"task_types={','.join(task_types)}")
        print(f"logs={log_root}")
        return_codes = [process.wait() for process, _, _ in processes]
        failed = sum(code != 0 for code in return_codes)
        print(f"failed_workers={failed}")
        return 0 if failed == 0 else 1
    except KeyboardInterrupt:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return 130
    finally:
        for _, stdout_handle, stderr_handle in processes:
            stdout_handle.close()
            stderr_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
