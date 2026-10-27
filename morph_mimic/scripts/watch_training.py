"""Follow a background training job across changing child log files.

Uses only the standard library and does not attach to, stop, or change training.
Its stdout may be redirected to a stable progress.log for ``tail -F``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


TERMINAL_STATES = {"validated", "budget_exhausted", "failed"}
PHASES = {
    "setup": "准备中", "waiting_for_existing_training": "首段训练中",
    "train": "训练中", "evaluate": "整段评估中",
    "checkpoint_metadata": "读取检查点", "finished": "已停止", "stopped": "已停止",
}


def read_tail(path: Path, size: int = 16384) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def snapshot(run_dir: Path) -> dict:
    # The supervisor publishes by atomic replacement: reopen on every poll.
    status = json.loads((run_dir / "supervisor_status.json").read_text())
    children = status.get("children", [])
    training = next((item for item in reversed(children) if item.get("phase") == "train"), None)
    training_log = Path(training["log"]) if training else run_dir / "training.log"
    lines = read_tail(training_log).splitlines()
    iteration_lines = [line for line in lines if re.match(r"^iter\s+\d+\s*\|", line)]
    last_line = iteration_lines[-1] if iteration_lines else None
    logged_iteration = int(re.match(r"^iter\s+(\d+)", last_line)[1]) if last_line else None
    completed = status.get("current_iteration")
    # A completed checkpoint can be newer than the last console log line.
    visible = max(value for value in (logged_iteration, completed) if value is not None) if (
        logged_iteration is not None or completed is not None) else None
    active_log = Path(children[-1]["log"]) if children else training_log
    evaluations = status.get("evaluations", [])
    return dict(
        state=status["state"], phase=status.get("phase"),
        latest_logged_iteration=logged_iteration, last_training_line=last_line,
        visible_iteration=visible, completed_checkpoint_iteration=completed,
        maximum_iterations=status["maximum_iterations"],
        chunk_iterations=status.get("chunk_iterations"),
        target_iteration=status.get("target_iteration", status.get("chunk_iterations")),
        consecutive_successes=status.get("consecutive_successes", 0),
        required_successes=status["required_successes"], evaluation_count=len(evaluations),
        last_evaluation=evaluations[-1] if evaluations else None,
        training_log=str(training_log), active_log=str(active_log),
        error=status.get("error"), criteria=status.get("criteria", {}),
    )


def render(item: dict) -> list[str]:
    state = item["state"]
    iteration = item["visible_iteration"]
    progress = "尚无轮次日志" if iteration is None else f"轮次≈{iteration}/{item['maximum_iterations']}"
    lines = [
        f"{state.upper()} | {PHASES.get(item['phase'], item['phase'])} | {progress} | "
        f"本段目标={item['target_iteration']} | "
        f"连续达标={item['consecutive_successes']}/{item['required_successes']} | "
        f"已完成整段评估={item['evaluation_count']}",
    ]
    if state == "validated":
        lines.append("STOP validated：连续多个不同检查点通过完整起点评估，训练已自动停止。")
    elif state == "budget_exhausted":
        lines.append("STOP budget_exhausted：达到训练上限，验收尚未通过，训练已自动停止。")
    elif state == "failed":
        lines.append(f"STOP failed：自动任务发生异常并停止。原因：{item.get('error')}")
    else:
        lines.append(f"当前子进程日志：{item['active_log']}")
    evaluation = item["last_evaluation"]
    if evaluation:
        verdict = "通过" if evaluation["passed"] else "未通过，继续训练"
        lines.append(f"最近完整评估：第 {evaluation['iteration']} 轮，{verdict}；"
                     f"连续达标 {evaluation['consecutive_successes']}/{item['required_successes']}")
        if evaluation.get("reasons"):
            lines.append("未通过项：" + "; ".join(evaluation["reasons"]))
    if item["last_training_line"]:
        lines.append(item["last_training_line"])
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    args.run_dir = args.run_dir.resolve()
    print("此查看器只读取日志；退出查看器或tail不会停止训练。", flush=True)
    previous = None
    printed_at = 0.0
    rules_printed = False
    while True:
        try:
            item = snapshot(args.run_dir)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            if not args.follow:
                parser.error(f"training status unavailable: {error}")
            item = None
            signature = str(error)
        else:
            signature = json.dumps(item, sort_keys=True)
        now = time.monotonic()
        if signature != previous or now - printed_at >= 60:
            if item is not None and not rules_printed:
                print(f"自动停止规则：每{item['chunk_iterations']}轮做完整起点评估；"
                      f"连续{item['required_successes']}个不同检查点达标，或总轮次达到"
                      f"{item['maximum_iterations']}轮上限。", flush=True)
                print("工程误差门限：" + "; ".join(
                    f"{name}≤{limit}" for name, limit in item["criteria"].items()), flush=True)
                rules_printed = True
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{stamp} Asia/Shanghai]", flush=True)
            for line in render(item) if item is not None else ["等待训练状态文件生成。"]:
                print(line, flush=True)
            previous, printed_at = signature, now
        if not args.follow or (item is not None and item["state"] in TERMINAL_STATES):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
