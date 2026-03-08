from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    images_dirname: str = "images"
    labels_dirname: str = "labels"
    manifest_name: str = "manifest.jsonl"


def _ensure_dirs(layout: DatasetLayout) -> Dict[str, Path]:
    img_dir = layout.root / layout.images_dirname
    lab_dir = layout.root / layout.labels_dirname
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    return {"img": img_dir, "lab": lab_dir, "manifest": layout.root / layout.manifest_name}


def save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    im.save(path)


def jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, set):
        return [jsonable(v) for v in sorted(list(x), key=lambda t: str(t))]
    if hasattr(x, "model_dump"):
        return jsonable(x.model_dump())
    try:
        import dataclasses
        if dataclasses.is_dataclass(x):
            return jsonable(dataclasses.asdict(x))
    except Exception:
        pass
    if hasattr(x, "to_dict"):
        try:
            return jsonable(x.to_dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        return jsonable({k: v for k, v in x.__dict__.items() if not k.startswith("_")})
    return str(x)


def save_sample(
    layout: DatasetLayout,
    sample_id: str,
    topdown_rgb: np.ndarray,
    radar_rgb: np.ndarray,
    label_json: Dict[str, Any],
    spec_json: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = _ensure_dirs(layout)

    top_path = paths["img"] / f"{sample_id}_topdown.png"
    rad_path = paths["img"] / f"{sample_id}_radar.png"
    lab_path = paths["lab"] / f"{sample_id}.json"

    save_rgb_image(top_path, topdown_rgb)
    save_rgb_image(rad_path, radar_rgb)

    meta = extra_meta or {}

    # IMPORTANT: also embed meta into labels.json for full reproducibility
    label_payload = dict(label_json)
    label_payload["meta"] = meta

    payload: Dict[str, Any] = {
        "sample_id": sample_id,
        "images": {
            "topdown": str(top_path.relative_to(layout.root)),
            "radar": str(rad_path.relative_to(layout.root)),
        },
        "label": {"path": str(lab_path.relative_to(layout.root))},
        "spec": spec_json,
        "meta": meta,
    }

    payload["pattern"] = meta.get("pattern")
    payload["difficulty"] = meta.get("difficulty")

    with lab_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(label_payload), f, ensure_ascii=False, indent=2)

    with paths["manifest"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")

    return payload