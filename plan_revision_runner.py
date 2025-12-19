"""
Batch validation runner for plan layout revision.

Reads validation_set.csv with columns: data, task, context.
For each row, runs up to MAX_ITERS iterations of:
  plan(LLM) -> execute(rule-based) -> review(LLM supervisor)

Outputs:
  results/<basename>_<task>_<context>/
    input.png
    candidate_iter{i}.json
    candidate_iter{i}.png
  results/validation_result.csv
"""

from __future__ import annotations

import base64
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from utils import render_floorplan_png_from_json


# Ensure UTF-8 stdout on Windows consoles
if sys.stdout.encoding != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# Load environment and configure MLflow
load_dotenv(find_dotenv(), override=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.openai.autolog()

# OpenAI client
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
client = OpenAI()


# ----------------------------
# JSON Schemas (model outputs)
# ----------------------------
planning_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Plan",
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "integer"},
                    "action": {"type": "string"},
                    "new_start_point": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "new_end_point": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "displacement": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "connected_adjustments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element_id": {"type": "integer"},
                                "endpoint": {"type": "string"},
                                "new_point": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "required": ["element_id", "endpoint", "new_point"],
                            "additionalProperties": True,
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["element_id", "action"],
                "additionalProperties": True,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["steps"],
    "additionalProperties": True,
}

apply_review_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ApplyReview",
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["pass"],
    "additionalProperties": True,
}

# OpenAI response_format wrappers require a name + schema
planning_schema_spec = {"name": "planning_schema", "schema": planning_schema}
apply_review_schema_spec = {"name": "apply_review_schema", "schema": apply_review_schema}


# ----------------------------
# Helpers
# ----------------------------
TASK1_TEXT = "How can I expand the living room without affecting other spaces?"
TASK2_TEXT = "How can I expand the living room while maintaining the building boundary?"
MAX_ITERS = 10

Point = Tuple[float, float]


def slugify_task(task: str) -> str:
    if task.strip() == TASK1_TEXT:
        return "task1"
    if task.strip() == TASK2_TEXT:
        return "task2"
    return task.strip().replace(" ", "_")


def encode_image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def load_floorplan_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_element_ids(floorplan: dict) -> dict:
    """Keep ids as numeric ints; also sanitize '/123' style ids."""
    if "Elements" in floorplan:
        for elem in floorplan["Elements"]:
            if "element_id" in elem:
                try:
                    eid = elem["element_id"]
                    if isinstance(eid, str):
                        eid = eid.lstrip("/")
                        elem["element_id"] = int(eid)
                    elif not isinstance(eid, int):
                        elem["element_id"] = int(eid)
                except (ValueError, TypeError):
                    print(f"[WARN] Could not parse element_id: {elem.get('element_id')}")

    if "Rooms" in floorplan:
        for room in floorplan["Rooms"]:
            if "room_id" in room:
                try:
                    rid = room["room_id"]
                    if isinstance(rid, str):
                        rid = rid.lstrip("/")
                        room["room_id"] = int(rid)
                    elif not isinstance(rid, int):
                        room["room_id"] = int(rid)
                except (ValueError, TypeError):
                    print(f"[WARN] Could not parse room_id: {room.get('room_id')}")

            if "bounding_elements_ids" in room:
                normalized_ids = []
                for eid in room["bounding_elements_ids"]:
                    try:
                        if isinstance(eid, str):
                            eid = eid.lstrip("/")
                            normalized_ids.append(int(eid))
                        elif isinstance(eid, int):
                            normalized_ids.append(eid)
                        else:
                            normalized_ids.append(int(eid))
                    except (ValueError, TypeError):
                        print(f"[WARN] Could not parse bounding element_id: {eid}")
                room["bounding_elements_ids"] = normalized_ids
    return floorplan


def _pt_key(p: Point, tol: float = 1e-6) -> Tuple[int, int]:
    # quantize to stable key
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


def _to_point(v: Any) -> Point:
    return (float(v[0]), float(v[1]))


def apply_plan_rule_based(floorplan: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based relocation execution.

    - Update moved walls (start/end) using:
        (a) new_start_point / new_end_point if provided
        (b) else displacement if provided (translate both endpoints)
    - Then propagate connectivity by rewriting endpoints of ALL other walls that used to
      touch old endpoints -> to the new endpoints (point-mapping).
    """
    fp = deepcopy(floorplan)
    fp = normalize_element_ids(fp)

    elements: List[Dict[str, Any]] = fp.get("Elements", [])
    elements_by_id: Dict[int, Dict[str, Any]] = {int(e["element_id"]): e for e in elements}

    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not isinstance(steps, list) or not steps:
        return fp

    moved_ids: List[int] = []
    point_map: Dict[Tuple[int, int], Point] = {}

    # 1) build point mappings from moved walls
    for s in steps:
        if not isinstance(s, dict):
            continue
        if "element_id" not in s:
            continue

        eid = int(s["element_id"])
        e = elements_by_id.get(eid)
        if not e:
            continue

        old_s = _to_point(e["start_point"])
        old_t = _to_point(e["end_point"])

        new_s: Optional[Point] = None
        new_t: Optional[Point] = None

        if "new_start_point" in s and isinstance(s["new_start_point"], list) and len(s["new_start_point"]) == 2:
            new_s = _to_point(s["new_start_point"])
        if "new_end_point" in s and isinstance(s["new_end_point"], list) and len(s["new_end_point"]) == 2:
            new_t = _to_point(s["new_end_point"])

        if new_s is None and new_t is None:
            if "displacement" in s and isinstance(s["displacement"], list) and len(s["displacement"]) == 2:
                dx, dy = float(s["displacement"][0]), float(s["displacement"][1])
                new_s = (old_s[0] + dx, old_s[1] + dy)
                new_t = (old_t[0] + dx, old_t[1] + dy)
            else:
                # nothing actionable
                continue
        else:
            # partial update allowed
            if new_s is None:
                new_s = old_s
            if new_t is None:
                new_t = old_t

        moved_ids.append(eid)

        # mapping: old endpoints -> new endpoints
        point_map[_pt_key(old_s)] = new_s
        point_map[_pt_key(old_t)] = new_t

        # also update the moved wall itself
        e["start_point"] = [new_s[0], new_s[1]]
        e["end_point"] = [new_t[0], new_t[1]]

    moved_set = set(moved_ids)

    # 2) propagate mapping to all other walls (connectivity fix)
    if point_map:
        for e in elements:
            eid = int(e["element_id"])
            if eid in moved_set:
                continue

            s_pt = _to_point(e["start_point"])
            t_pt = _to_point(e["end_point"])

            ks = _pt_key(s_pt)
            kt = _pt_key(t_pt)

            if ks in point_map:
                ns = point_map[ks]
                e["start_point"] = [ns[0], ns[1]]
            if kt in point_map:
                nt = point_map[kt]
                e["end_point"] = [nt[0], nt[1]]

    return normalize_element_ids(fp)


def run_single_case(data_file: str, task_text: str, context_mode: str) -> tuple[int, bool]:
    """Run a single validation case and return (iterations_used, passed)."""

    basename = data_file.replace(".json", "")
    task_slug = slugify_task(task_text)
    context_slug = context_mode.strip().lower()
    run_tag = f"{basename}_{task_slug}_{context_slug}"

    json_path = Path("dataset/json") / data_file
    if not json_path.exists():
        raise FileNotFoundError(f"Floorplan JSON not found: {json_path}")

    outputs_dir = Path("results") / run_tag
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Render input
    input_png_path = outputs_dir / "input.png"
    try:
        render_floorplan_png_from_json(json_path=json_path, out_png_path=input_png_path, scale=0.001)
    except Exception as e:
        print(f"[WARN] Input rendering failed for {json_path}: {e}")

    image_data_url = (
        encode_image_to_data_url(input_png_path)
        if context_slug == "multimodal" and input_png_path.exists()
        else None
    )

    floorplan_json = load_floorplan_json(json_path)
    floorplan_json = normalize_element_ids(floorplan_json)

    requirements_text = "Expand the living room using wall moves only (shift/extend wall segments)."
    constraints_text = (
        "Do not reduce areas of other rooms. Only move/reshape existing walls; do not add openings, finishes, or virtual elements. "
        "Keep identifiers stable and use only existing element_ids (numeric)."
    )

    # Per-case MLflow experiment
    mlflow.set_experiment(run_tag)

    feedback_text: Optional[str] = None
    passed = False
    iterations_used = MAX_ITERS

    with mlflow.start_run(run_name=run_tag):
        for i in range(1, MAX_ITERS + 1):
            print(f"\n=== Iteration {i} | {run_tag} ===")

            # -------------
            # Plan (LLM)
            # -------------
            planning_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an architectural plan assistant. Produce a concise, actionable plan using the schema. "
                        f"Primary goal: {task_text}. "
                        "Work ONLY with existing element_ids and rooms. Focus on living_room (room_id=2) boundaries facing exterior/recess. "
                        "Limit changes to wall moves/resizing (shift/extend endpoints); no openings/finishes/furniture. "
                        "For each step, include either new_start_point/new_end_point OR displacement. "
                        "Do NOT invent element_ids."
                    ),
                },
            ]

            user_blocks = [
                {"type": "text", "text": task_text},
                {"type": "text", "text": "Requirements:"},
                {"type": "text", "text": requirements_text},
                {"type": "text", "text": "Constraints:"},
                {"type": "text", "text": constraints_text},
                {"type": "text", "text": "Here is the current floorplan JSON."},
                {"type": "text", "text": json.dumps(floorplan_json, ensure_ascii=False)},
            ]
            if context_slug == "multimodal" and image_data_url:
                user_blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})

            planning_messages.append({"role": "user", "content": user_blocks})

            if feedback_text:
                planning_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Supervisor feedback from previous attempt:\n{feedback_text}"}],
                    }
                )

            plan_completion = client.chat.completions.create(
                model="gpt-5.1",
                temperature=0.4,
                top_p=0.9,
                response_format={"type": "json_schema", "json_schema": planning_schema_spec},
                messages=planning_messages,
            )

            plan_content = plan_completion.choices[0].message.content
            plan_text = plan_content[0].text if isinstance(plan_content, list) else plan_content

            try:
                plan_parsed = json.loads(plan_text)
                print("[PLAN]\n" + json.dumps(plan_parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print("[WARN] Plan JSON decode failed, raw output:\n", plan_text)
                feedback_text = "Plan output was not valid JSON. Output must conform to the plan schema."
                continue

            # ---------------------------
            # Execute relocation (RULE)
            # ---------------------------
            try:
                updated_json = apply_plan_rule_based(floorplan_json, plan_parsed)
                print("[EXECUTE/RULE]\n" + json.dumps(updated_json, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"[WARN] Rule-based execute failed: {e}")
                feedback_text = f"Rule-based execute failed: {str(e)[:200]}"
                continue

            out_json_path = outputs_dir / f"candidate_iter{i}.json"
            out_png_path = outputs_dir / f"candidate_iter{i}.png"

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(updated_json, f, ensure_ascii=False, indent=2)

            try:
                render_floorplan_png_from_json(out_json_path, out_png_path, scale=0.001)
            except Exception as e:
                print(f"[WARN] Rendering failed: {e}")

            # -------------
            # Review (LLM)
            # -------------
            apply_review_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict supervisor. Compare the updated floorplan to the plan and requirements. "
                        "Return only JSON using the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Task:"},
                        {"type": "text", "text": task_text},
                        {"type": "text", "text": "Requirements:"},
                        {"type": "text", "text": requirements_text},
                        {"type": "text", "text": "Constraints:"},
                        {"type": "text", "text": constraints_text},
                        {"type": "text", "text": "Plan:"},
                        {"type": "text", "text": json.dumps(plan_parsed, ensure_ascii=False)},
                        {"type": "text", "text": "Original JSON:"},
                        {"type": "text", "text": json.dumps(floorplan_json, ensure_ascii=False)},
                        {"type": "text", "text": "Updated JSON:"},
                        {"type": "text", "text": json.dumps(updated_json, ensure_ascii=False)},
                    ],
                },
            ]

            if context_slug == "multimodal" and out_png_path.exists():
                apply_review_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": encode_image_to_data_url(out_png_path)}}],
                    }
                )

            apply_review = client.chat.completions.create(
                model="gpt-5.1",
                temperature=0.0,
                response_format={"type": "json_schema", "json_schema": apply_review_schema_spec},
                messages=apply_review_messages,
            )

            ar_content = apply_review.choices[0].message.content
            ar_text = ar_content[0].text if isinstance(ar_content, list) else ar_content

            try:
                ar_parsed = json.loads(ar_text)
                apply_ok = bool(ar_parsed.get("pass"))
                apply_issues = ar_parsed.get("issues", [])
                apply_suggestions = ar_parsed.get("suggestions", [])
                print("[REVIEW]", json.dumps(ar_parsed, ensure_ascii=False))
            except json.JSONDecodeError:
                print("[WARN] Apply review JSON decode failed, raw output:\n", ar_text)
                feedback_text = "Supervisor review output was not valid JSON."
                floorplan_json = updated_json
                if context_slug == "multimodal" and out_png_path.exists():
                    image_data_url = encode_image_to_data_url(out_png_path)
                continue

            if apply_ok:
                print(f"\n[SUCCESS] Supervisor passed at iteration {i}.")
                passed = True
                iterations_used = i
                break

            feedback_lines = []
            feedback_lines.append("Apply issues: " + "; ".join(apply_issues) if apply_issues else "Apply issues: (none listed)")
            if apply_suggestions:
                feedback_lines.append("Apply suggestions: " + "; ".join(apply_suggestions))
            feedback_text = "\n".join(feedback_lines)

            # update state for next iter
            floorplan_json = updated_json
            if context_slug == "multimodal" and out_png_path.exists():
                image_data_url = encode_image_to_data_url(out_png_path)
        else:
            print("\n[INCOMPLETE] Supervisor did not pass within the max iterations.")

    return iterations_used, passed


def main():
    validation_path = Path("validation_set.csv")
    if not validation_path.exists():
        raise FileNotFoundError("validation_set.csv not found.")

    results_rows: List[Dict[str, str]] = []

    # ✅ 핵심 수정: utf-8-sig 로 BOM 제거
    with open(validation_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    start_time = time.time()

    for row in rows:
        data_file = (row.get("data") or "").strip()
        task_text = (row.get("task") or "").strip()
        context_mode = (row.get("context") or "").strip()

        if not data_file or not task_text or not context_mode:
            print(f"[WARN] Skipping malformed row: {row}")
            continue

        print(f"\n=== Running case: data={data_file}, task={task_text}, context={context_mode} ===")
        iterations_used, passed = run_single_case(data_file, task_text, context_mode)

        results_rows.append(
            {
                "data": data_file,
                "task": task_text,
                "context": context_mode,
                "iteration": str(iterations_used),
                "pass": str(passed).lower(),
            }
        )

    elapsed = time.time() - start_time
    print(f"\nAll validation cases completed in {elapsed:.1f}s")

    results_path = Path("results") / "validation_result.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["data", "task", "context", "iteration", "pass"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)


if __name__ == "__main__":
    main()
