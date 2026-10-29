"""Implementation helpers for `run_s0_blinded_review` — extracted verbatim from `run_s0_blinded_review.py`.

Holds the helper functions and module constants so `run_s0_blinded_review.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from scratch.run_refiner_pq_allfold import (  # noqa: E402
    imread_unicode,
)
from scratch.run_s0_baseline_rescore import (  # noqa: E402
    annotation_mask,
    load_prediction_cache,
    prediction_cache_path,
    resolve_preprocessed_image_path,
    resize_mask,
)
from scratch.run_v42_verifier_official_eval import load_masks  # noqa: E402


PACKAGE_ID = "s0-blinded-review-2026-08-24-v1"


PROTOCOL_VERSION = "s0-visual-review-v1"


DEFAULT_SEED = 20260824


DISPLAY_SIZE = 512


SCIENCE_CACHE = ROOT / "output/s0/prediction_cache_512"


VERIFIER_CACHE = ROOT / "output/s0/v42_verifier_official_cache"


MODEL_VERIFIER = "v42_nested_verifier"


COMPARISONS = {
    "matched_spine_tradeoff": ("raw", "v35"),
    "stratified_rescue": ("v35", "v42"),
    "late_time_rejection": ("v42", MODEL_VERIFIER),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def balanced_assignments(count: int, rng: np.random.Generator) -> np.ndarray:
    """Return a shuffled, position-balanced assignment for output A."""
    values = np.zeros(count, dtype=bool)
    values[: count // 2] = True
    if count % 2:
        values[count // 2] = bool(rng.integers(0, 2))
    rng.shuffle(values)
    return values


def top_unique(
    candidates: pd.DataFrame,
    count: int,
    excluded: set[str],
) -> pd.DataFrame:
    """Select deterministic top-scoring rows without reusing observations."""
    available = candidates.loc[~candidates.file_name.isin(excluded)].copy()
    available = available.sort_values(
        ["selection_score", "file_name"], ascending=[False, True]
    )
    selected = available.drop_duplicates("file_name").head(count)
    if len(selected) != count:
        raise ValueError(f"needed {count} unique cases, found {len(selected)}")
    excluded.update(selected.file_name.astype(str))
    return selected


def _mean_table(science: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spine_coverage_matched",
        "cldice",
        "filament_f1",
        "spine_coverage",
        "filament_recall",
        "filament_precision",
        "prediction_count",
    ]
    return science.groupby(["file_name", "model"])[metrics].mean().unstack("model")


def _metadata_table(science: pd.DataFrame) -> pd.DataFrame:
    reference = science.loc[science.model == "v35"]
    return reference.groupby("file_name").agg(
        fold=("fold", "first"),
        site_code=("site_code", "first"),
        year=("year", "first"),
        time_block_role=("time_block_role", "first"),
        area_groups=("area_group", lambda values: "|".join(sorted(set(values)))),
        density_groups=("density_group", lambda values: "|".join(sorted(set(values)))),
        radial_groups=("radial_group", lambda values: "|".join(sorted(set(values)))),
    )


def select_cases(
    science: pd.DataFrame,
    verifier_official: pd.DataFrame,
    per_comparison: int,
) -> pd.DataFrame:
    """Apply the predeclared S0-4 case-selection rules."""
    means = _mean_table(science)
    metadata = _metadata_table(science)
    excluded: set[str] = set()
    selected_parts = []

    tradeoff = metadata.copy()
    tradeoff["raw_matched_advantage"] = (
        means[("spine_coverage_matched", "raw")]
        - means[("spine_coverage_matched", "v35")]
        + means[("cldice", "raw")]
        - means[("cldice", "v35")]
    )
    tradeoff["v35_catalog_advantage"] = (
        means[("filament_f1", "v35")] - means[("filament_f1", "raw")]
    )
    tradeoff["selection_score"] = (
        tradeoff.raw_matched_advantage + 0.5 * tradeoff.v35_catalog_advantage
    )
    tradeoff = tradeoff.loc[
        (tradeoff.raw_matched_advantage > 0)
        & (tradeoff.v35_catalog_advantage >= 0)
    ].reset_index()
    tradeoff["comparison"] = "matched_spine_tradeoff"
    tradeoff["selection_stratum"] = "tradeoff"
    selected_parts.append(top_unique(tradeoff, per_comparison, excluded))

    rescue = metadata.copy()
    rescue["spine_gain"] = (
        means[("spine_coverage", "v42")] - means[("spine_coverage", "v35")]
    )
    rescue["recall_gain"] = (
        means[("filament_recall", "v42")]
        - means[("filament_recall", "v35")]
    )
    rescue["precision_change"] = (
        means[("filament_precision", "v42")]
        - means[("filament_precision", "v35")]
    )
    rescue["selection_score"] = rescue.spine_gain + rescue.recall_gain
    rescue = rescue.loc[
        (rescue.spine_gain > 0) & (rescue.recall_gain > 0)
    ].reset_index()
    rescue["comparison"] = "stratified_rescue"
    stratum_specs = [
        ("small", "area_groups", "small_le_1000"),
        ("dense", "density_groups", "dense_gt_10"),
        ("limb", "radial_groups", "limb_proxy_gt_0.66"),
    ]
    base = per_comparison // len(stratum_specs)
    counts = [base] * len(stratum_specs)
    for index in range(per_comparison - sum(counts)):
        counts[index] += 1
    for (label, column, token), count in zip(stratum_specs, counts):
        candidates = rescue.loc[rescue[column].str.contains(token, regex=False)].copy()
        candidates["selection_stratum"] = label
        selected_parts.append(top_unique(candidates, count, excluded))

    official = verifier_official.set_index("file_name")
    rejection = metadata.join(
        means[("prediction_count", "v42")].rename("v42_prediction_count")
    ).join(official[["output_count", "fn", "accepted_count", "fused_count"]])
    rejection["dropped_count"] = (
        rejection.v42_prediction_count - rejection.output_count
    )
    rejection["v42_recall_gain"] = (
        means[("filament_recall", "v42")]
        - means[("filament_recall", "v35")]
    )
    rejection["selection_score"] = (
        rejection.dropped_count
        + 0.25 * rejection.fn
        + rejection.v42_recall_gain.clip(lower=0)
    )
    rejection = rejection.loc[
        (rejection.time_block_role == "locked_evaluation")
        & (rejection.dropped_count > 0)
    ].reset_index()
    rejection["comparison"] = "late_time_rejection"
    rejection["selection_stratum"] = "2017_2022"
    selected_parts.append(top_unique(rejection, per_comparison, excluded))

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    if selected.file_name.duplicated().any():
        raise AssertionError("selection contains duplicate observations")
    return selected


def apply_blinding(
    selected: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Balance A/B positions within each comparison and randomize case order."""
    rng = np.random.default_rng(seed)
    blinded_parts = []
    for comparison, group in selected.groupby("comparison", sort=True):
        model_1, model_2 = COMPARISONS[comparison]
        ordered = group.sort_values(
            ["selection_score", "file_name"], ascending=[False, True]
        ).copy()
        model_1_is_a = balanced_assignments(len(ordered), rng)
        ordered["output_a"] = np.where(model_1_is_a, model_1, model_2)
        ordered["output_b"] = np.where(model_1_is_a, model_2, model_1)
        blinded_parts.append(ordered)
    blinded = pd.concat(blinded_parts, ignore_index=True, sort=False)
    order = rng.permutation(len(blinded))
    blinded = blinded.iloc[order].reset_index(drop=True)
    blinded["case_id"] = [f"case_{index:03d}" for index in range(1, len(blinded) + 1)]
    key = []
    for row in blinded.itertuples(index=False):
        key.append({
            "case_id": row.case_id,
            "file_name": row.file_name,
            "comparison": row.comparison,
            "selection_stratum": row.selection_stratum,
            "selection_score": float(row.selection_score),
            "output_a": row.output_a,
            "output_b": row.output_b,
        })
    return blinded, key


def outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return dilated & ~eroded


def focus_box(
    masks_a: list[np.ndarray],
    masks_b: list[np.ndarray],
    reference_masks: list[np.ndarray],
    size: int = DISPLAY_SIZE,
) -> tuple[int, int, int, int]:
    """Focus on the largest A/B disagreement while retaining useful context."""
    union_a = np.logical_or.reduce(masks_a) if masks_a else np.zeros((size, size), bool)
    union_b = np.logical_or.reduce(masks_b) if masks_b else np.zeros((size, size), bool)
    interest = np.logical_xor(union_a, union_b)
    if interest.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            interest.astype(np.uint8), 8
        )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            interest = labels == largest
    elif union_a.any() or union_b.any():
        interest = union_a | union_b
    elif reference_masks:
        interest = np.logical_or.reduce(reference_masks)
    else:
        return 0, 0, size, size
    ys, xs = np.where(interest)
    cx = (int(xs.min()) + int(xs.max()) + 1) / 2
    cy = (int(ys.min()) + int(ys.max()) + 1) / 2
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    side = min(size, max(160, int(np.ceil(span * 2.2))))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(0, x0), size - side)
    y0 = min(max(0, y0), size - side)
    return x0, y0, x0 + side, y0 + side


def load_prediction(model: str, name: str, fold: int) -> list[np.ndarray]:
    if model == MODEL_VERIFIER:
        path = VERIFIER_CACHE / f"fold_{fold}" / f"{Path(name).stem}.npz"
        loaded = load_masks(path, name, fold)
        if loaded is None:
            raise FileNotFoundError(f"verifier prediction missing or invalid: {path}")
        masks, _ = loaded
        return [resize_mask(mask, DISPLAY_SIZE) for mask in masks]
    path = prediction_cache_path(SCIENCE_CACHE, model, fold, name)
    masks = load_prediction_cache(path, model, DISPLAY_SIZE)
    if masks is None:
        raise FileNotFoundError(f"science prediction missing or invalid: {path}")
    return masks


def _draw_panel(
    axis,
    base: np.ndarray,
    predictions: list[np.ndarray],
    references: list[tuple[np.ndarray, int]],
    spines: list[tuple[np.ndarray, int]],
    title: str,
    box: tuple[int, int, int, int] | None = None,
) -> None:
    if box is None:
        box = (0, 0, base.shape[1], base.shape[0])
    x0, y0, x1, y1 = box
    axis.imshow(base[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    canvas = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    prediction_palette = [
        (0.20, 1.00, 0.35),
        (1.00, 0.45, 0.15),
        (0.35, 0.70, 1.00),
        (1.00, 0.25, 0.80),
        (0.80, 0.90, 0.20),
    ]
    for index, mask in enumerate(predictions):
        local = mask[y0:y1, x0:x1]
        color = prediction_palette[index % len(prediction_palette)]
        canvas[local, :3] = color
        canvas[local, 3] = np.maximum(canvas[local, 3], 0.18)
        edge = outline(mask, 1)[y0:y1, x0:x1]
        canvas[edge, :3] = color
        canvas[edge, 3] = 0.95
    reference_palette = [
        (0.00, 1.00, 1.00),
        (1.00, 0.30, 1.00),
        (1.00, 0.95, 0.15),
    ]
    for mask, variant in references:
        edge = outline(mask, 1)[y0:y1, x0:x1]
        color = reference_palette[variant % len(reference_palette)]
        canvas[edge, :3] = color
        canvas[edge, 3] = 0.80
    axis.imshow(canvas)
    for points, variant in spines:
        color = reference_palette[variant % len(reference_palette)]
        axis.plot(
            points[:, 0] - x0,
            points[:, 1] - y0,
            color=color,
            linewidth=1.4,
            alpha=0.95,
        )
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def load_references(coco: dict) -> tuple[dict, dict, dict]:
    images = {str(item["id"]): item for item in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    file_to_ids = defaultdict(list)
    for image_id, image in images.items():
        file_to_ids[image["file_name"]].append(image_id)
    return images, annotations, file_to_ids


def render_case_panel(
    row,
    images: dict,
    annotations: dict,
    file_to_ids: dict,
    image_root: Path,
    output_path: Path,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = row.file_name
    fold = int(row.fold)
    image_path = resolve_preprocessed_image_path(image_root, name)
    image = imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
    masks_a = load_prediction(row.output_a, name, fold)
    masks_b = load_prediction(row.output_b, name, fold)
    references: list[tuple[np.ndarray, int]] = []
    spines: list[tuple[np.ndarray, int]] = []
    for variant, image_id in enumerate(file_to_ids[name]):
        info = images[image_id]
        for annotation in annotations[image_id]:
            mask = annotation_mask(annotation, int(info["height"]), int(info["width"]))
            references.append((resize_mask(mask, DISPLAY_SIZE), variant))
            values = np.asarray(annotation.get("spine", []), dtype=np.float64)
            if values.size >= 4 and values.size % 2 == 0:
                points = values.reshape(-1, 2)
                points[:, 0] *= DISPLAY_SIZE / float(info["width"])
                points[:, 1] *= DISPLAY_SIZE / float(info["height"])
                spines.append((points, variant))
    box = focus_box(masks_a, masks_b, [item[0] for item in references])
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    _draw_panel(axes[0, 0], gray, [], references, spines, "Input + references")
    _draw_panel(axes[0, 1], gray, masks_a, references, spines, "Output A")
    _draw_panel(axes[0, 2], gray, masks_b, references, spines, "Output B")
    _draw_panel(axes[1, 0], gray, [], references, spines, "Zoom: references", box)
    _draw_panel(axes[1, 1], gray, masks_a, references, spines, "Zoom: Output A", box)
    _draw_panel(axes[1, 2], gray, masks_b, references, spines, "Zoom: Output B", box)
    figure.suptitle(
        f"{row.case_id}   observation {name}   cyan/magenta/yellow = annotation variants",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110, facecolor="#111827")
    plt.close(figure)
    return {
        "case_id": row.case_id,
        "file_name": name,
        "fold": fold,
        "image": output_path.name,
        "focus_box_512": list(box),
        "output_a_instances": len(masks_a),
        "output_b_instances": len(masks_b),
        "reference_variants": len(file_to_ids[name]),
    }


def build_review_html(cases: list[dict], package_id: str = PACKAGE_ID) -> str:
    cards = []
    def preference_options(case_id: str, field: str) -> str:
        group = f"{case_id}-{field}"
        return (
            f'<label><input type="radio" name="{group}" value="A">A</label>'
            f'<label><input type="radio" name="{group}" value="B">B</label>'
            f'<label><input type="radio" name="{group}" value="tie">同等</label>'
            f'<label><input type="radio" name="{group}" value="neither">どちらも不可</label>'
        )
    tags = [
        ("spine_missing", "spine欠落"),
        ("fragmentation", "切断/split"),
        ("merge", "merge"),
        ("false_positive", "偽陽性"),
        ("limb_artifact", "太陽縁artifact"),
        ("unclear", "判定困難"),
    ]
    for case in cases:
        case_id = html.escape(case["case_id"])
        morphology_options = preference_options(case_id, "morphology")
        catalog_options = preference_options(case_id, "catalog")
        tag_a = "".join(
            f'<label><input type="checkbox" value="{value}">{label}</label>'
            for value, label in tags
        )
        tag_b = tag_a
        cards.append(f"""
        <article class="case-card" data-case="{case_id}">
          <header><h2>{case_id}</h2><span class="status">未回答</span></header>
          <img src="assets/{html.escape(case['image'])}" alt="{case_id} blinded comparison">
          <fieldset data-field="morphology"><legend>1. reference spine・輪郭をより保存するのは？</legend>{morphology_options}</fieldset>
          <fieldset data-field="catalog"><legend>2. カタログ用途で有害なFP / split / mergeが少ないのは？</legend>{catalog_options}</fieldset>
          <div class="tag-grid">
            <fieldset data-field="tags_a"><legend>Output Aの問題</legend>{tag_a}</fieldset>
            <fieldset data-field="tags_b"><legend>Output Bの問題</legend>{tag_b}</fieldset>
          </div>
          <label class="confidence">確信度
            <select data-field="confidence"><option value="">未選択</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
          </label>
          <label class="notes">メモ<textarea data-field="notes" rows="2"></textarea></label>
        </article>""")
    case_ids = [case["case_id"] for case in cases]
    template = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>S0-4 Blinded Visual Review</title>
<style>
:root{color-scheme:dark;--bg:#08111f;--panel:#111c2e;--line:#2c3e57;--text:#e8eef8;--muted:#9eacc2;--accent:#4dd4ac}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}
.top{position:sticky;top:0;z-index:5;background:#08111ff2;border-bottom:1px solid var(--line);padding:14px 24px;backdrop-filter:blur(8px)}
.top h1{margin:0 0 4px;font-size:21px}.top p{margin:3px 0;color:var(--muted)}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
button,.file-button{border:1px solid var(--line);background:#17263c;color:var(--text);padding:8px 12px;border-radius:7px;cursor:pointer}button.primary{background:#12674f;border-color:#279a78}.file-button input{display:none}
main{max-width:1500px;margin:auto;padding:20px}.case-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px;margin:0 0 22px}.case-card header{display:flex;justify-content:space-between;align-items:center}.case-card h2{margin:0;font-size:18px}.status{color:#ffca6b}.complete .status{color:var(--accent)}
img{display:block;width:100%;height:auto;border-radius:6px;background:#050a12;margin:12px 0}fieldset{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:10px}legend{color:#c7d6eb;padding:0 5px}fieldset label{display:inline-flex;gap:5px;align-items:center;margin:3px 14px 3px 0}.tag-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.confidence,.notes{display:block;margin-top:10px}.notes textarea{display:block;width:100%;background:#091322;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px}select,input{accent-color:var(--accent)}
@media(max-width:800px){.tag-grid{grid-template-columns:1fr}.top{position:static}main{padding:10px}}
</style></head><body>
<section class="top"><h1>S0-4 Blinded Visual Review</h1>
<p>モデル対応表はこのページに含まれません。reference outline / spineを基準にA/Bを独立判定してください。</p>
<p><strong id="progress">0 / __TOTAL__ 完了</strong>　回答はこのブラウザ内へ自動保存されます。</p>
<div class="controls"><label>Reviewer ID <input id="reviewer" autocomplete="off"></label><button class="primary" id="export">判定JSONを書き出す</button><label class="file-button">JSONを読み込む<input id="import" type="file" accept="application/json"></label><button id="clear">このブラウザの回答を消去</button></div></section>
<main>__CARDS__</main>
<script>
const packageId=__PACKAGE_JSON__,caseIds=__CASE_IDS__,storageKey=packageId+':responses';
let state={package_id:packageId,protocol_version:'s0-visual-review-v1',reviewer_id:'',responses:{}};
const load=()=>{try{const saved=JSON.parse(localStorage.getItem(storageKey));if(saved&&saved.package_id===packageId)state=saved}catch(e){}};
const valueFor=(card,field)=>{const root=card.querySelector(`[data-field="${field}"]`);if(!root)return null;if(root.matches('select,textarea'))return root.value;const checked=root.querySelector('input[type=radio]:checked');return checked?checked.value:null};
const tagsFor=(card,field)=>[...card.querySelectorAll(`[data-field="${field}"] input:checked`)].map(x=>x.value);
function collect(){state.reviewer_id=document.getElementById('reviewer').value.trim();document.querySelectorAll('.case-card').forEach(card=>{const id=card.dataset.case;state.responses[id]={morphology:valueFor(card,'morphology'),catalog:valueFor(card,'catalog'),tags_a:tagsFor(card,'tags_a'),tags_b:tagsFor(card,'tags_b'),confidence:valueFor(card,'confidence'),notes:valueFor(card,'notes')||''}});state.saved_at=new Date().toISOString();localStorage.setItem(storageKey,JSON.stringify(state));update()}
function hydrate(){document.getElementById('reviewer').value=state.reviewer_id||'';document.querySelectorAll('.case-card').forEach(card=>{const r=state.responses[card.dataset.case]||{};for(const field of ['morphology','catalog']){const el=card.querySelector(`[data-field="${field}"] input[value="${r[field]||''}"]`);if(el)el.checked=true}for(const field of ['tags_a','tags_b'])for(const value of r[field]||[]){const el=card.querySelector(`[data-field="${field}"] input[value="${value}"]`);if(el)el.checked=true}for(const field of ['confidence','notes']){const el=card.querySelector(`[data-field="${field}"]`);if(el)el.value=r[field]||''}});update()}
function update(){let done=0;document.querySelectorAll('.case-card').forEach(card=>{const r=state.responses[card.dataset.case]||{};const ok=!!r.morphology&&!!r.catalog&&!!r.confidence;card.classList.toggle('complete',ok);card.querySelector('.status').textContent=ok?'完了':'未回答';if(ok)done++});document.getElementById('progress').textContent=`${done} / ${caseIds.length} 完了`}
document.addEventListener('change',collect);document.addEventListener('input',e=>{if(e.target.matches('textarea,#reviewer'))collect()});
document.getElementById('export').onclick=()=>{collect();state.exported_at=new Date().toISOString();state.case_order=caseIds;const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=packageId+'-responses.json';a.click();URL.revokeObjectURL(a.href)};
document.getElementById('import').onchange=async e=>{const parsed=JSON.parse(await e.target.files[0].text());if(parsed.package_id!==packageId){alert('別packageのJSONです');return}state=parsed;localStorage.setItem(storageKey,JSON.stringify(state));hydrate()};
document.getElementById('clear').onclick=()=>{if(confirm('このブラウザに保存した回答を消去しますか？')){localStorage.removeItem(storageKey);state={package_id:packageId,protocol_version:'s0-visual-review-v1',reviewer_id:'',responses:{}};location.reload()}};
load();hydrate();
</script></body></html>"""
    return (
        template.replace("__TOTAL__", str(len(cases)))
        .replace("__CARDS__", "\n".join(cards))
        .replace("__PACKAGE_JSON__", json.dumps(package_id))
        .replace("__CASE_IDS__", json.dumps(case_ids, ensure_ascii=False))
    )


def protocol_markdown() -> str:
    return """# S0-4 blinded visual review protocol

1. `review.html`だけを開く。回答JSONを書き出すまで`private/`を開かない。
2. 各caseでfull diskとzoomを確認する。cyan / magenta / yellowはannotator variantの輪郭とspine。
3. 形態保存、カタログ上の有害error、A/Bそれぞれのerror tag、確信度を独立に回答する。
4. modelを推測して回答しない。reference間不一致が大きければ「判定困難」を付ける。
5. 24件すべて完了後、Reviewer IDを入れて「判定JSONを書き出す」を押す。
6. JSONを保存してからのみ`private/blind_key.json`でunblindする。

ブラウザ内回答はlocalStorageへ自動保存される。学習labelへ自動転用しない。
"""
