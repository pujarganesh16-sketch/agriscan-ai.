"""
AgriScan AI — Advanced Crop Health Screening
-------------------------------------------
A Gradio application for leaf-image screening using an existing Keras classifier.

Implemented:
- Persistent SQLite scan history and My Garden records
- Single-image and batch-image analysis
- Image quality/context pre-check (heuristic; not a trained edge detector)
- Top predictions with confidence bars
- Optional Grad-CAM-style visualization when the saved model architecture supports it
- CSV and PDF report export
- Crop-specific cautious guidance cards
- Searchable scan history and garden tracker
- Configurable model paths and graceful diagnostics

Important: This is a research prototype. It is not a confirmed diagnosis or pesticide prescription.
"""

from __future__ import annotations

import os
import math
import logging
import io
import json
import csv
import sqlite3
import html
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageStat, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

APP_NAME = "AgriScan AI"
BASE_DIR = Path(os.environ.get("AGRISCAN_HOME", str(Path(__file__).resolve().parent))).resolve()
MODEL_PATH = Path(os.environ.get("AGRISCAN_MODEL", str(BASE_DIR / "agriscan_model.keras"))).expanduser()
LABELS_PATH = Path(os.environ.get("AGRISCAN_LABELS", str(BASE_DIR / "class_labels.json"))).expanduser()
DB_PATH = Path(os.environ.get("AGRISCAN_DB", str(BASE_DIR / "agriscan_app.db"))).expanduser()
EXPORT_DIR = BASE_DIR / "exports"
IMG_SIZE = int(os.environ.get("AGRISCAN_IMAGE_SIZE", "160"))
TOP_K = 5

BASE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------- Model loading ----------------------------

def load_model_and_labels():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. Train the model first, "
            "or set AGRISCAN_MODEL to the correct path."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Class-label file was not found at {LABELS_PATH}. "
            "Expected a JSON list saved by the training notebook."
        )
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    if not isinstance(labels, list) or not labels:
        raise ValueError("class_labels.json must contain a non-empty JSON list.")
    if int(loaded_model.output_shape[-1]) != len(labels):
        raise ValueError(
            f"Model output has {loaded_model.output_shape[-1]} classes but "
            f"the label file has {len(labels)} labels. Use matching files."
        )
    return loaded_model, [str(x) for x in labels]


try:
    MODEL, CLASS_NAMES = load_model_and_labels()
    MODEL_STATUS = f"Model loaded · {len(CLASS_NAMES)} classes"
except Exception as exc:
    MODEL, CLASS_NAMES = None, []
    MODEL_STATUS = f"Model not ready: {exc}"


# ---------------------------- Database ----------------------------

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                plant_name TEXT,
                original_filename TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                healthy_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                all_predictions_json TEXT NOT NULL,
                quality_note TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS garden (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_name TEXT NOT NULL,
                crop_type TEXT,
                location_note TEXT,
                created_at TEXT NOT NULL,
                last_scan_id INTEGER,
                notes TEXT DEFAULT '',
                FOREIGN KEY(last_scan_id) REFERENCES scans(id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                garden_id INTEGER NOT NULL,
                reminder_date TEXT NOT NULL,
                message TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(garden_id) REFERENCES garden(id)
            )
        """)

init_db()


# ---------------------------- Image preparation and pre-check ----------------------------

def pretty_label(label: str) -> str:
    return str(label).replace("___", " — ").replace("_", " ").strip()


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def image_precheck(image: Image.Image | None):
    if image is None:
        return False, "No image selected.", "Upload a leaf photo."
    img = image.convert("RGB")
    if min(img.size) < 120:
        return False, "The image resolution is too small.", "Upload a larger, clearer photo."
    thumb = img.resize((128, 128))
    arr = np.asarray(thumb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green = (g > r * 1.04) & (g > b * 0.96) & (g > 40)
    green_ratio = float(green.mean())
    gray_img = img.convert("L").resize((128, 128))
    gray = np.asarray(gray_img, dtype=np.uint8)
    contrast = float(ImageStat.Stat(gray_img).stddev[0])
    edges = np.asarray(gray_img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_strength = float(edges.mean())

    if contrast < 10 or edge_strength < 1.8:
        return False, "The photo appears blurry, too dark, or low-contrast.", (
            "We couldn't detect a clear leaf photo. Use natural light, hold the camera steady, "
            "and make the leaf fill more of the frame."
        )
    if green_ratio < 0.025:
        return False, "Leaf context is uncertain; little green plant-like area was found.", (
            "We couldn't confidently recognize a leaf-like image. Please upload a close-up leaf photo."
        )
    if green_ratio < 0.08:
        return True, "Leaf context uncertain; image passed only a basic quality check.", (
            "Photo quality is acceptable, but this quick check cannot confirm that the image is a leaf."
        )
    return True, "Basic photo check passed.", (
        "Photo quality check passed. This is a heuristic, not an object-recognition model."
    )


# ---------------------------- Guidance ----------------------------

def guidance_for(label: str) -> dict[str, str]:
    low = label.lower()
    if "healthy" in low:
        return {
            "summary": "Likely healthy class",
            "organic": "Continue observation and maintain healthy soil using crop-appropriate compost or organic matter.",
            "chemical": "This prediction alone does not indicate a need for chemical treatment. Avoid unnecessary spraying.",
            "prevention": "Water at soil level where practical, provide airflow, clean tools, and inspect new growth weekly.",
            "urgency": "Routine monitoring",
        }
    if "late blight" in low:
        return {
            "summary": "Possible late-blight pattern",
            "organic": "Inspect several plants. Where local guidance recommends it, remove heavily affected material and dispose of it safely; do not compost suspected infected tissue.",
            "chemical": "Confirm the disease and check current local crop guidance before using any product. Only use products registered for this crop and disease, strictly according to the label. Copper products are not universally suitable.",
            "prevention": "Reduce leaf wetness, avoid overhead irrigation where practical, improve spacing and airflow, and inspect after cool/wet weather.",
            "urgency": "Prompt local confirmation recommended",
        }
    if "powdery mildew" in low:
        return {
            "summary": "Possible powdery-mildew pattern",
            "organic": "Improve airflow and remove affected material only when appropriate for the crop and local advice.",
            "chemical": "If confirmed, ask an agricultural extension officer about locally registered, crop-approved options. Follow the label.",
            "prevention": "Avoid overcrowding, monitor new growth, and use crop-appropriate watering and sanitation.",
            "urgency": "Confirm before treatment",
        }
    if "bacterial" in low:
        return {
            "summary": "Possible bacterial-disease pattern",
            "organic": "Clean tools between plants and remove affected material only where recommended by local crop guidance.",
            "chemical": "Many fungicides do not cure established bacterial disease. Confirm the cause before selecting a product; use only registered products as labelled.",
            "prevention": "Reduce leaf wetness, avoid handling wet plants, clean tools, and consider crop rotation where appropriate.",
            "urgency": "Confirm before treatment",
        }
    if "mosaic" in low or "yellow leaf curl" in low or "virus" in low:
        return {
            "summary": "Possible viral-symptom pattern",
            "organic": "Check nearby plants and insect activity. Ask a local extension officer before removing plants or applying treatments.",
            "chemical": "There is no general chemical cure for an established plant virus. Do not apply pesticides based on this image alone; local advice can help manage vectors where appropriate.",
            "prevention": "Use healthy planting material, monitor insect vectors, sanitize tools, and follow crop-specific local guidance.",
            "urgency": "Prompt confirmation recommended",
        }
    return {
        "summary": "Possible disease pattern",
        "organic": "Compare symptoms on several leaves with a trusted local crop guide before taking action.",
        "chemical": "Do not select a pesticide from this prediction alone. Confirm the disease and use only products registered for the crop and region, according to the label.",
        "prevention": "Provide good airflow, avoid unnecessary leaf wetness, clean tools, and inspect nearby plants.",
        "urgency": "Uncertain — confirm locally",
    }


# ---------------------------- Optional visual explanation ----------------------------

def try_gradcam(image: Image.Image, class_index: int):
    """
    Best-effort Grad-CAM for models containing a MobileNetV2 base named
    'mobilenetv2_1.00_160'. Returns None if the architecture is incompatible.
    This heatmap is a model-attention visualization, not a disease segmentation mask.
    """
    if MODEL is None:
        return None
    try:
        base = MODEL.get_layer("mobilenetv2_1.00_160")
        last_conv = base.get_layer("out_relu")
        grad_model = tf.keras.Model(
            inputs=MODEL.inputs,
            outputs=[last_conv.output, MODEL.output]
        )
        batch = tf.convert_to_tensor(preprocess_image(image))
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(batch, training=False)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)[0]
        cam = tf.maximum(cam, 0)
        maximum = tf.reduce_max(cam)
        cam = cam / (maximum + 1e-8)
        heat = np.asarray(tf.image.resize(cam[..., None], image.size[::-1]).numpy().squeeze())
        heat = np.uint8(255 * heat)
        # Create a red-yellow heat layer without matplotlib dependency.
        heat_rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)
        heat_rgb[:, :, 0] = heat
        heat_rgb[:, :, 1] = np.clip(heat.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        overlay = Image.blend(image.convert("RGB").resize((heat.shape[1], heat.shape[0])), Image.fromarray(heat_rgb), 0.38)
        return overlay
    except Exception:
        return None


# ---------------------------- Analysis and persistence ----------------------------

def analyze_one(image: Image.Image, filename: str = "", plant_name: str = "") -> dict[str, Any]:
    if MODEL is None:
        raise RuntimeError(MODEL_STATUS)
    ok, quality_note, user_message = image_precheck(image)
    if not ok:
        return {
            "accepted": False,
            "quality_note": quality_note,
            "message": user_message,
            "prediction": "",
            "confidence": 0.0,
            "healthy_probability": 0.0,
            "predictions": [],
            "guidance": {},
            "overlay": None,
            "scan_id": None,
        }

    probs = np.asarray(MODEL.predict(preprocess_image(image), verbose=0)[0], dtype=float)
    order = np.argsort(probs)[::-1][:TOP_K]
    top = [{"label": str(CLASS_NAMES[int(i)]), "display": pretty_label(CLASS_NAMES[int(i)]),
            "probability": float(probs[int(i)])} for i in order]
    best_idx = int(order[0])
    label = str(CLASS_NAMES[best_idx])
    confidence = float(probs[best_idx])
    healthy_prob = float(sum(probs[i] for i, name in enumerate(CLASS_NAMES) if "healthy" in name.lower()))
    risk = "Lower apparent risk" if "healthy" in label.lower() else (
        "Higher concern; confirm locally" if confidence >= 0.70 else "Uncertain; confirm locally"
    )
    guidance = guidance_for(label)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with db_connect() as con:
        cur = con.execute("""
            INSERT INTO scans(created_at, plant_name, original_filename, prediction,
                confidence, healthy_probability, risk_level, all_predictions_json, quality_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, plant_name.strip(), filename, label, confidence, healthy_prob, risk,
              json.dumps(top), quality_note))
        scan_id = int(cur.lastrowid)
        if plant_name.strip():
            con.execute("""
                UPDATE garden SET last_scan_id=?
                WHERE id=(SELECT id FROM garden WHERE plant_name=? ORDER BY id DESC LIMIT 1)
            """, (scan_id, plant_name.strip()))
    overlay = try_gradcam(image.convert("RGB"), best_idx)
    return {
        "accepted": True, "quality_note": quality_note, "message": user_message,
        "prediction": label, "display_prediction": pretty_label(label),
        "confidence": confidence, "healthy_probability": healthy_prob, "risk": risk,
        "predictions": top, "guidance": guidance, "overlay": overlay, "scan_id": scan_id,
        "created_at": now, "filename": filename, "plant_name": plant_name.strip()
    }


def render_prediction(result: dict[str, Any]) -> str:
    if not result.get("accepted"):
        return f"""
        <div class="card warning">
          <div class="eyebrow">PHOTO PRE-CHECK</div><h2>Let's improve the photo</h2>
          <p>{html.escape(result.get('message', 'Please upload a leaf image.'))}</p>
          <small>{html.escape(result.get('quality_note', ''))}</small>
        </div>"""
    bars = []
    for item in result["predictions"]:
        p = item["probability"]
        if "healthy" in item["label"].lower():
            color = "#27b875"
        elif p >= 0.70:
            color = "#d84b5b"
        else:
            color = "#e2a53b"
        bars.append(f"""
        <div class="prob-row">
          <div class="prob-label"><span>{html.escape(item['display'])}</span><b>{p:.1%}</b></div>
          <div class="track"><div class="fill" style="width:{p*100:.2f}%;background:{color}"></div></div>
        </div>""")
    return f"""
    <div class="card">
      <div class="eyebrow">SCAN #{result['scan_id']} · {html.escape(result['created_at'])}</div>
      <h2>{html.escape(result['display_prediction'])}</h2>
      <div class="metric-grid">
        <div class="metric"><span>Top-class confidence</span><strong>{result['confidence']:.1%}</strong></div>
        <div class="metric"><span>Healthy-class probability</span><strong>{result['healthy_probability']:.1%}</strong></div>
      </div>
      <p class="risk">{html.escape(result['risk'])}</p>
      <p class="muted">{html.escape(result['message'])}</p>
    </div>
    <div class="card"><h3>Prediction confidence</h3>{''.join(bars)}</div>
    """


def render_guidance(result: dict[str, Any]) -> str:
    if not result.get("accepted"):
        return ""
    g = result["guidance"]
    return f"""
    <div class="guidance-grid">
      <div class="guide organic"><div class="eyebrow">01 · LOW-IMPACT FIRST</div><h3>Organic & cultural</h3><p>{html.escape(g['organic'])}</p></div>
      <div class="guide chemical"><div class="eyebrow">02 · CONFIRM BEFORE USE</div><h3>Chemical options</h3><p>{html.escape(g['chemical'])}</p></div>
      <div class="guide prevention"><div class="eyebrow">03 · LONG-TERM CARE</div><h3>Prevention protocol</h3><p>{html.escape(g['prevention'])}</p></div>
    </div>
    <div class="disclaimer">AI screening is not a confirmed diagnosis. Do not apply pesticides based only on this result.</div>
    """


def scan_ui(image, plant_name):
    if image is None:
        blank = {"accepted": False, "message": "Upload a leaf image to begin.", "quality_note": ""}
        return render_prediction(blank), "", None, None, "Please upload a leaf image.", None, None
    try:
        result = analyze_one(image, plant_name=plant_name or "")
        if not result.get("accepted"):
            return render_prediction(result), "", None, None, result["message"], None, result
        csv_path = make_scan_csv([result])
        return (
            render_prediction(result), render_guidance(result), result["overlay"],
            csv_path, result["message"], result["scan_id"], result
        )
    except Exception as exc:
        msg = f"Analysis failed: {exc}"
        return f"<div class='card warning'><h3>Could not analyze image</h3><p>{html.escape(msg)}</p></div>", "", None, None, msg, None, None


def batch_scan(files, plant_name):
    if not files:
        return pd.DataFrame(), "Upload one or more images.", None
    rows = []
    for file in files:
        path = getattr(file, "name", str(file))
        try:
            with Image.open(path) as im:
                result = analyze_one(im.copy(), filename=Path(path).name, plant_name=plant_name or "")
            if result.get("accepted"):
                rows.append({
                    "Scan ID": result["scan_id"],
                    "Filename": Path(path).name,
                    "Plant": plant_name or "",
                    "Prediction": result["display_prediction"],
                    "Confidence": round(result["confidence"], 4),
                    "Healthy probability": round(result["healthy_probability"], 4),
                    "Risk note": result["risk"],
                })
            else:
                rows.append({
                    "Scan ID": None, "Filename": Path(path).name, "Plant": plant_name or "",
                    "Prediction": "Photo check failed", "Confidence": 0.0,
                    "Healthy probability": 0.0, "Risk note": result["message"]
                })
        except Exception as exc:
            rows.append({
                "Scan ID": None, "Filename": Path(path).name, "Plant": plant_name or "",
                                "Prediction": "Error", "Confidence": 0.0, "Healthy probability": 0.0,
                "Risk note": str(exc)
            })
    df = pd.DataFrame(rows)
    path = EXPORT_DIR / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return df, f"Processed {len(rows)} image(s). Review low-confidence and failed checks carefully.", str(path)


# ---------------------------- Exports and history ----------------------------

def make_scan_csv(results):
    path = EXPORT_DIR / f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    for r in results:
        rows.append({
            "scan_id": r.get("scan_id"),
            "created_at": r.get("created_at"),
            "plant_name": r.get("plant_name", ""),
            "filename": r.get("filename", ""),
            "prediction": r.get("display_prediction", r.get("prediction", "")),
            "confidence": r.get("confidence", 0),
            "healthy_probability": r.get("healthy_probability", 0),
            "risk": r.get("risk", ""),
            "quality_note": r.get("quality_note", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def export_history():
    with db_connect() as con:
        df = pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", con)
    if df.empty:
        return None, "No scans have been saved yet."
    path = EXPORT_DIR / f"scan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return str(path), f"Exported {len(df)} scan record(s)."


def get_history(search_text=""):
    with db_connect() as con:
        if search_text and search_text.strip():
            q = f"%{search_text.strip()}%"
            df = pd.read_sql_query("""
                SELECT id AS 'Scan ID', created_at AS 'Date', plant_name AS 'Plant',
                       original_filename AS 'File', prediction AS 'Prediction',
                       confidence AS 'Confidence', healthy_probability AS 'Healthy probability',
                       risk_level AS 'Risk'
                FROM scans
                WHERE plant_name LIKE ? OR prediction LIKE ? OR original_filename LIKE ?
                ORDER BY id DESC LIMIT 300
            """, con, params=(q, q, q))
        else:
            df = pd.read_sql_query("""
                SELECT id AS 'Scan ID', created_at AS 'Date', plant_name AS 'Plant',
                       original_filename AS 'File', prediction AS 'Prediction',
                       confidence AS 'Confidence', healthy_probability AS 'Healthy probability',
                       risk_level AS 'Risk'
                FROM scans ORDER BY id DESC LIMIT 300
            """, con)
    return df


# ---------------------------- My Garden and reminders ----------------------------

def garden_df():
    with db_connect() as con:
        return pd.read_sql_query("""
            SELECT g.id AS 'ID', g.plant_name AS 'Plant name', g.crop_type AS 'Crop',
                   g.location_note AS 'Location / plot', g.created_at AS 'Added',
                   s.prediction AS 'Latest model prediction',
                   s.confidence AS 'Confidence', g.notes AS 'Notes'
            FROM garden g LEFT JOIN scans s ON s.id=g.last_scan_id
            ORDER BY g.id DESC
        """, con)


def add_garden(plant_name, crop_type, location_note, notes):
    if not plant_name or not plant_name.strip():
        return garden_df(), "Please enter a plant name."
    with db_connect() as con:
        con.execute("""
            INSERT INTO garden(plant_name, crop_type, location_note, created_at, notes)
            VALUES(?,?,?,?,?)
        """, (plant_name.strip(), (crop_type or "").strip(), (location_note or "").strip(),
              datetime.now().astimezone().isoformat(timespec="seconds"), (notes or "").strip()))
    return garden_df(), f"Added '{plant_name.strip()}' to My Garden. Records persist in SQLite in this Colab runtime."


def add_reminder(garden_id, reminder_date, message):
    if not garden_id or not str(garden_id).strip().isdigit():
        return "Enter a valid garden record ID."
    if not reminder_date:
        return "Choose a reminder date."
    with db_connect() as con:
        exists = con.execute("SELECT id FROM garden WHERE id=?", (int(garden_id),)).fetchone()
        if not exists:
            return "Garden record ID was not found."
        con.execute("""
            INSERT INTO reminders(garden_id, reminder_date, message, created_at)
            VALUES(?,?,?,?)
        """, (int(garden_id), str(reminder_date), (message or "Recheck plant").strip(),
              datetime.now().astimezone().isoformat(timespec="seconds")))
    return "Reminder saved in the app database. This prototype does not send push notifications."


def reminders_df():
    with db_connect() as con:
        return pd.read_sql_query("""
            SELECT r.id AS 'Reminder ID', g.plant_name AS 'Plant',
                   r.reminder_date AS 'Date', r.message AS 'Reminder',
                   r.completed AS 'Completed'
            FROM reminders r JOIN garden g ON g.id=r.garden_id
            ORDER BY r.reminder_date
        """, con)


def complete_reminder(reminder_id):
    try:
        rid = int(reminder_id)
    except Exception:
        return reminders_df(), "Enter a valid reminder ID."
    with db_connect() as con:
        cur = con.execute("UPDATE reminders SET completed=1 WHERE id=?", (rid,))
    if cur.rowcount == 0:
        return reminders_df(), "Reminder not found."
    return reminders_df(), "Reminder marked complete."


def export_garden():
    df = garden_df()
    if df.empty:
        return None
    path = EXPORT_DIR / f"my_garden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return str(path)


def create_pdf_report(scan_id):
    try:
        sid = int(scan_id)
    except Exception:
        return None, "Enter a valid scan ID."
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
    except ImportError:
        return None, "PDF support is missing. Run: %pip install reportlab"
    with db_connect() as con:
        row = con.execute("SELECT * FROM scans WHERE id=?", (sid,)).fetchone()
    if not row:
        return None, "Scan ID not found."
    path = EXPORT_DIR / f"agriscan_scan_{sid}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
    story = [
        Paragraph("AgriScan AI — Crop Leaf Screening Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph("Research prototype. This report is not a confirmed diagnosis or pesticide recommendation.", styles["Italic"]),
        Spacer(1, 12),
    ]
    rows = [
        ["Field", "Result"],
        ["Scan ID", str(row["id"])],
        ["Date", str(row["created_at"])],
        ["Plant name", str(row["plant_name"] or "Not supplied")],
        ["Image", str(row["original_filename"] or "Uploaded image")],
        ["Top prediction", pretty_label(row["prediction"])],
        ["Top-class confidence", f"{row['confidence']:.1%}"],
        ["Healthy-class probability", f"{row['healthy_probability']:.1%}"],
        ["Risk note", str(row["risk_level"])],
        ["Photo quality note", str(row["quality_note"])],
    ]
    table = Table(rows, colWidths=[48*mm, 112*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17452e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#b8cbbd")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f7f2")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Review several leaves and consult a qualified local agricultural professional before treatment.", styles["BodyText"]))
    doc.build(story)
    return str(path), "PDF report created."


# ---------------------------- Interface ----------------------------



# =============================================================================
# PREMIUM ANALYTICS, NUMERICAL METHODS, QUALITY METRICS, AND RELIABILITY LAYER
# =============================================================================
# This section is intentionally made of reusable, testable Python functions.
# It adds transparent numerical tools rather than artificial line-count padding.
# Model scores are estimates, not confirmed plant diagnoses or treatment advice.

from dataclasses import dataclass, asdict
from statistics import median
from typing import NamedTuple


@dataclass(frozen=True)
class QualityMetric:
    """A named numerical metric with its unit and a human-readable explanation."""
    name: str
    value: float
    unit: str = ""
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize a metric for JSON and report output."""
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval with method metadata."""
    estimate: float
    lower: float
    upper: float
    confidence: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class ImageQualityReport:
    """Non-diagnostic image quality measurements for one uploaded photo."""
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    saturation: float
    green_fraction: float
    entropy: float
    quality_score: float
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Convert the report to a plain dictionary."""
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


class MetricRow(NamedTuple):
    """Compact row type for calibration and evaluation reports."""
    name: str
    value: float
    support: int


def _finite_array(values: Any, *, ndim: int | None = None) -> np.ndarray:
    """Convert values into a finite float array, replacing invalid values by zero."""
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {array.ndim}.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with a deterministic fallback for zero or invalid denominators."""
    denominator = float(denominator)
    numerator = float(numerator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return float(default)
    result = numerator / denominator
    return result if math.isfinite(result) else float(default)


def stable_softmax(logits: Any, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax probabilities.

    Subtracting the maximum logit prevents overflow. Temperature is restricted
    to a positive range; values above one soften the distribution, while values
    below one sharpen it. Temperature scaling does not create calibrated
    probabilities unless a suitable validation set is used to fit the value.
    """
    values = _finite_array(logits)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    temperature = float(np.clip(_safe_float(temperature, 1.0), 0.05, 20.0))
    shifted = values / temperature
    shifted = shifted - np.max(shifted)
    exponentials = np.exp(np.clip(shifted, -745.0, 80.0))
    total = float(np.sum(exponentials))
    if total <= 0 or not math.isfinite(total):
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return exponentials / total


def normalize_scores(scores: Any) -> np.ndarray:
    """Normalize a score vector to sum to one without allowing negative mass."""
    values = np.maximum(_finite_array(scores).reshape(-1), 0.0)
    total = float(values.sum())
    if values.size == 0:
        return values
    if total <= np.finfo(float).eps:
        return np.full(values.size, 1.0 / values.size, dtype=np.float64)
    return values / total


def shannon_entropy(probabilities: Any, base: float = 2.0) -> float:
    """Compute Shannon entropy, ignoring zero-probability terms."""
    p = normalize_scores(probabilities)
    if p.size <= 1:
        return 0.0
    positive = p[p > 0]
    logarithm_base = max(float(base), 1.000001)
    return float(-np.sum(positive * (np.log(positive) / np.log(logarithm_base))))


def normalized_entropy(probabilities: Any) -> float:
    """Return entropy normalized to [0, 1] for a multi-class distribution."""
    p = normalize_scores(probabilities)
    if p.size <= 1:
        return 0.0
    return float(np.clip(shannon_entropy(p) / np.log2(p.size), 0.0, 1.0))


def prediction_margin(probabilities: Any) -> float:
    """Difference between the largest and second-largest normalized scores."""
    p = np.sort(normalize_scores(probabilities))
    if p.size == 0:
        return 0.0
    if p.size == 1:
        return float(p[-1])
    return float(p[-1] - p[-2])


def top_k_mass(probabilities: Any, k: int = 3) -> float:
    """Sum the probability mass assigned to the k most likely classes."""
    p = np.sort(normalize_scores(probabilities))[::-1]
    if p.size == 0:
        return 0.0
    k = int(np.clip(k, 1, p.size))
    return float(np.sum(p[:k]))


def entropy_confidence(probabilities: Any) -> float:
    """Return an uncertainty-aware confidence proxy in [0, 1].

    This is a descriptive score based on distribution concentration. It is not
    the probability that the predicted plant disease is correct.
    """
    return float(1.0 - normalized_entropy(probabilities))


def brier_score(probabilities: Any, true_index: int) -> float:
    """Compute the multi-class Brier score for one labeled example."""
    p = normalize_scores(probabilities)
    if p.size == 0:
        return 0.0
    if int(true_index) < 0 or int(true_index) >= p.size:
        raise ValueError("true_index is outside the probability vector.")
    target = np.zeros_like(p)
    target[int(true_index)] = 1.0
    return float(np.mean(np.square(p - target)))


def multiclass_brier_score(probabilities: Any, true_indices: Any) -> float:
    """Compute mean multi-class Brier score over a labeled evaluation matrix."""
    p = _finite_array(probabilities, ndim=2)
    if p.shape[0] == 0 or p.shape[1] == 0:
        return 0.0
    p = np.apply_along_axis(normalize_scores, 1, p)
    labels = np.asarray(true_indices, dtype=int).reshape(-1)
    if labels.size != p.shape[0]:
        raise ValueError("One true label is required for each probability row.")
    if np.any(labels < 0) or np.any(labels >= p.shape[1]):
        raise ValueError("A true label index is outside the class range.")
    targets = np.eye(p.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum(np.square(p - targets), axis=1)))


def log_loss_score(probabilities: Any, true_indices: Any, epsilon: float = 1e-12) -> float:
    """Compute multiclass logarithmic loss with clipped probabilities."""
    p = _finite_array(probabilities, ndim=2)
    labels = np.asarray(true_indices, dtype=int).reshape(-1)
    if p.shape[0] != labels.size:
        raise ValueError("Probability rows and labels must have equal length.")
    if p.shape[0] == 0:
        return 0.0
    if np.any(labels < 0) or np.any(labels >= p.shape[1]):
        raise ValueError("A true label index is outside the class range.")
    p = np.apply_along_axis(normalize_scores, 1, p)
    selected = np.clip(p[np.arange(labels.size), labels], epsilon, 1.0)
    return float(-np.mean(np.log(selected)))
    
def expected_calibration_error(
    probabilities: Any,
    true_indices: Any,
    bins: int = 15,
) -> tuple[float, list[dict[str, float | int]]]:
    """Estimate expected calibration error using equal-width confidence bins."""
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(true_indices, dtype=int).reshape(-1)

    if p.ndim != 2 or p.shape[0] != labels.size:
        raise ValueError("Probabilities and labels must have matching rows.")
    if p.shape[0] == 0:
        return 0.0, []
    if bins < 1:
        raise ValueError("bins must be at least 1.")
    if np.any(labels < 0) or np.any(labels >= p.shape[1]):
        raise ValueError("A true label index is outside the class range.")

    p = np.apply_along_axis(normalize_scores, 1, p)
    predictions = np.argmax(p, axis=1)
    confidences = np.max(p, axis=1)
    correct = (predictions == labels).astype(float)

    ece = 0.0
    details: list[dict[str, float | int]] = []

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(mask))
        if count == 0:
            continue

        mean_confidence = float(np.mean(confidences[mask]))
        accuracy = float(np.mean(correct[mask]))
        ece += (count / labels.size) * abs(accuracy - mean_confidence)

        details.append({
            "bin": index + 1,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": abs(accuracy - mean_confidence),
        })

    return float(ece), details



def reliability_diagram_frame(
    probabilities: Any,
    true_indices: Any,
    bins: int = 15,
) -> pd.DataFrame:
    """Return calibration-bin values suitable for a table or chart."""
    _, rows = expected_calibration_error(probabilities, true_indices, bins)
    return pd.DataFrame(rows, columns=[
        "bin", "lower", "upper", "count", "mean_confidence", "accuracy"
    ])


def confusion_matrix_counts(true_indices: Any, predicted_indices: Any, class_count: int) -> np.ndarray:
    """Build a confusion matrix without depending on scikit-learn."""
    true_values = np.asarray(true_indices, dtype=int).reshape(-1)
    pred_values = np.asarray(predicted_indices, dtype=int).reshape(-1)
    if true_values.size != pred_values.size:
        raise ValueError("True and predicted arrays must have equal length.")
    class_count = int(class_count)
    if class_count <= 0:
        raise ValueError("class_count must be positive.")
    if np.any(true_values < 0) or np.any(true_values >= class_count):
        raise ValueError("True class index out of range.")
    if np.any(pred_values < 0) or np.any(pred_values >= class_count):
        raise ValueError("Predicted class index out of range.")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (true_values, pred_values), 1)
    return matrix


def classification_report_rows(matrix: Any, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Compute per-class precision, recall, specificity, F1, and support."""
    cm = np.asarray(matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    class_count = cm.shape[0]
    if labels is None:
        labels = [f"Class {i}" for i in range(class_count)]
    if len(labels) != class_count:
        raise ValueError("Label count must match the confusion matrix.")
    total = float(cm.sum())
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = float(cm[index, index])
        fn = float(cm[index, :].sum() - tp)
        fp = float(cm[:, index].sum() - tp)
        tn = max(0.0, total - tp - fn - fp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        rows.append({
            "class": str(label),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(tp + fn),
        })
    return pd.DataFrame(rows)


def macro_average(metrics: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Calculate unweighted class-level averages while ignoring non-finite values."""
    result: dict[str, float] = {}
    for column in columns:
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        result[column] = float(values.mean()) if not values.empty else 0.0
    return result


def weighted_average(metrics: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Calculate support-weighted class metrics."""
    result: dict[str, float] = {}
    if "support" not in metrics:
        return {column: 0.0 for column in columns if column in metrics}
    weights = pd.to_numeric(metrics["support"], errors="coerce").fillna(0.0).clip(lower=0)
    denominator = float(weights.sum())
    for column in columns:
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").fillna(0.0)
        result[column] = _safe_divide(float(np.sum(values * weights)), denominator)
    return result


def wilson_score_interval(successes: int, trials: int, confidence_z: float = 1.959963984540054) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion.

    This interval is more stable than the basic normal interval when the sample
    is small or the measured proportion is close to zero or one.
    """
    n = int(trials)
    x = int(successes)
    if n < 0 or x < 0 or x > n:
        raise ValueError("Require 0 <= successes <= trials.")
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 1.0, 0.95, "Wilson score")
    z = max(0.0, float(confidence_z))
    p_hat = x / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + z2 / (4.0 * n * n)) / denominator
    return ConfidenceInterval(
        estimate=p_hat,
        lower=max(0.0, center - radius),
        upper=min(1.0, center + radius),
        confidence=0.95,
        method="Wilson score",
    )


def bootstrap_mean_interval(
    values: Any,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Estimate a mean and percentile bootstrap interval with reproducible sampling."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, "Percentile bootstrap")
    confidence = float(np.clip(confidence, 0.5, 0.999))
    repetitions = int(np.clip(repetitions, 100, 50000))
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.choice(data, size=data.size, replace=True)
        means[index] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return ConfidenceInterval(
        estimate=float(data.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        method="Percentile bootstrap",
    )


def median_absolute_deviation(values: Any) -> float:
    """Calculate the median absolute deviation, a robust spread estimate."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return 0.0
    center = float(np.median(data))
    return float(np.median(np.abs(data - center)))


def robust_z_scores(values: Any, epsilon: float = 1e-9) -> np.ndarray:
    """Calculate robust z-scores based on median and MAD."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return data
    center = float(np.median(data))
    mad = median_absolute_deviation(data)
    if mad <= epsilon:
        return np.zeros_like(data)
    return 0.6744897501960817 * (data - center) / mad


def detect_outliers_mad(values: Any, threshold: float = 3.5) -> np.ndarray:
    """Return a boolean mask for robust median-absolute-deviation outliers."""
    return np.abs(robust_z_scores(values)) > max(0.1, float(threshold))


def moving_average(values: Any, window: int = 5) -> np.ndarray:
    """Compute a trailing moving average, retaining the original array length."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return data
    window = int(np.clip(window, 1, data.size))
    output = np.empty_like(data)
    cumulative = np.concatenate(([0.0], np.cumsum(data)))
    for index in range(data.size):
        start = max(0, index - window + 1)
        output[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return output


def exponential_moving_average(values: Any, alpha: float = 0.3) -> np.ndarray:
    """Compute an exponentially weighted series with a bounded smoothing factor."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return data
    alpha = float(np.clip(alpha, 0.001, 1.0))
    output = np.empty_like(data)
    output[0] = data[0]
    for index in range(1, data.size):
        output[index] = alpha * data[index] + (1.0 - alpha) * output[index - 1]
    return output


def linear_trend(values: Any) -> dict[str, float]:
    """Estimate a least-squares slope, intercept, and R-squared for a series."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
    x = np.arange(data.size, dtype=np.float64)
    if data.size == 1:
        return {"slope": 0.0, "intercept": float(data[0]), "r_squared": 0.0}
    slope, intercept = np.polyfit(x, data, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum(np.square(data - predicted)))
    total_sum = float(np.sum(np.square(data - data.mean())))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 1e-12 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(np.clip(r_squared, 0.0, 1.0)),
    }


def seasonal_difference(values: Any, period: int = 7) -> np.ndarray:
    """Difference a series against the same position in a previous period."""
    data = _finite_array(values).reshape(-1)
    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive.")
    if data.size <= period:
        return np.asarray([], dtype=np.float64)
    return data[period:] - data[:-period]


def correlation_matrix(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a numeric correlation matrix after dropping constant columns."""
    selected = frame if columns is None else frame[list(columns)]
    numeric = selected.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(method="pearson").fillna(0.0)


def safe_percentile(values: Any, percentile: float, default: float = 0.0) -> float:
    """Calculate a bounded percentile with a deterministic empty-input fallback."""
    data = _finite_array(values).reshape(-1)
    if data.size == 0:
        return float(default)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(data, percentile))


def normalize_minmax(values: Any, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Scale values to a fixed interval; constant arrays map to the midpoint."""
    data = _finite_array(values)
    if data.size == 0:
        return data
    source_min, source_max = float(data.min()), float(data.max())
    if source_max - source_min <= 1e-12:
        return np.full_like(data, (low + high) / 2.0)
    return low + (data - source_min) * (high - low) / (source_max - source_min)


def cosine_similarity(left: Any, right: Any) -> float:
    """Compute cosine similarity between vectors with zero-vector safety."""
    a = _finite_array(left).reshape(-1)
    b = _finite_array(right).reshape(-1)
    if a.size != b.size:
        raise ValueError("Vectors must have the same length.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def jensen_shannon_divergence(left: Any, right: Any) -> float:
    """Compute symmetric Jensen-Shannon divergence in bits."""
    p = normalize_scores(left)
    q = normalize_scores(right)
    if p.size != q.size:
        raise ValueError("Distributions must have equal length.")
    if p.size == 0:
        return 0.0
    middle = 0.5 * (p + q)

    def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-15))))

    return float(np.clip(0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle), 0.0, 1.0))


def histogram_intersection(left: Any, right: Any) -> float:
    """Compare non-negative histograms with an intersection score in [0, 1]."""
    a = normalize_scores(left)
    b = normalize_scores(right)
    if a.size != b.size:
        raise ValueError("Histograms must have equal length.")
    if a.size == 0:
        return 0.0
    return float(np.sum(np.minimum(a, b)))


def image_entropy(image: Image.Image) -> float:
    """Estimate grayscale image entropy from its 256-bin histogram."""
    grayscale = image.convert("L")
    histogram = np.asarray(grayscale.histogram(), dtype=np.float64)
    return shannon_entropy(histogram)


def image_sharpness(image: Image.Image) -> float:
    """Estimate sharpness using variance of a Laplacian-like filter response."""
    grayscale = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    padded = np.pad(grayscale, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def image_saturation(image: Image.Image) -> float:
    """Return mean HSV saturation normalized to [0, 1]."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    if hsv.size == 0:
        return 0.0
    return float(np.mean(hsv[:, :, 1]) / 255.0)


def image_green_fraction(image: Image.Image) -> float:
    """Estimate the fraction of pixels meeting a broad green-color heuristic.

    This is a photo descriptor, not a leaf detector. Green backgrounds can
    increase it and yellow, red, or brown leaves can decrease it.
    """
    rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (green > 45) & (green > red * 1.05) & (green > blue * 0.95)
    return float(np.mean(mask))


def image_color_histogram(image: Image.Image, bins: int = 16) -> np.ndarray:
    """Create a normalized RGB histogram vector for rough photo comparisons."""
    bins = int(np.clip(bins, 2, 64))
    array = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.uint8)
    pieces = []
    for channel in range(3):
        histogram, _ = np.histogram(array[:, :, channel], bins=bins, range=(0, 256))
        pieces.append(histogram.astype(np.float64))
    return normalize_scores(np.concatenate(pieces))


def image_quality_report(image: Image.Image) -> ImageQualityReport:
    """Measure visible image characteristics and return cautious quality notes."""
    valid, reason = _validate_image_object(image)
    if not valid:
        raise ValueError(reason)
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    brightness = float(np.mean(stat.mean) / 255.0)
    contrast = float(np.mean(stat.stddev) / 128.0)
    sharpness = image_sharpness(rgb)
    saturation = image_saturation(rgb)
    green_fraction = image_green_fraction(rgb)
    entropy = image_entropy(rgb)
    # Each feature is normalized independently before weighted aggregation.
    sharpness_component = float(np.clip(math.log1p(sharpness) / math.log1p(1500.0), 0, 1))
    brightness_component = float(np.clip(1.0 - abs(brightness - 0.52) / 0.52, 0, 1))
    contrast_component = float(np.clip(contrast / 0.45, 0, 1))
    entropy_component = float(np.clip(entropy / 8.0, 0, 1))
    quality_score = 100.0 * (
        0.40 * sharpness_component
        + 0.25 * brightness_component
        + 0.20 * contrast_component
        + 0.15 * entropy_component
    )
    notes: list[str] = []
    if brightness < 0.18:
        notes.append("Photo may be too dark; use diffuse daylight.")
    elif brightness > 0.88:
        notes.append("Photo may be overexposed; avoid direct glare.")
    if sharpness < 30:
        notes.append("Photo may be blurry; steady the camera and refocus.")
    if contrast < 0.08:
        notes.append("Low contrast may hide fine leaf detail.")
    if green_fraction < 0.03:
        notes.append("Little green color detected; this is not proof the image lacks a leaf.")
    if image.width < 320 or image.height < 320:
        notes.append("A higher-resolution image may preserve more detail.")
    if not notes:
        notes.append("No strong image-quality warning from these simple heuristics.")
    return ImageQualityReport(
        width=image.width,
        height=image.height,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,
        green_fraction=green_fraction,
        entropy=entropy,
        quality_score=float(np.clip(quality_score, 0, 100)),
        notes=tuple(notes),
    )


def image_quality_markdown(image: Image.Image | None) -> str:
    """Format image quality measurements for a readable Gradio panel."""
    if image is None:
        return "Upload a leaf photo to calculate image-quality measurements."
    try:
        report = image_quality_report(image)
    except Exception as exc:
        return f"**Image check:** {_safe_text(exc, 200)}"
    notes = "\n".join(f"- {html.escape(note)}" for note in report.notes)
    return (
        "### Photo quality diagnostics\n"
        f"- **Heuristic quality score:** {report.quality_score:.1f}/100\n"
        f"- **Resolution:** {report.width} × {report.height}\n"
        f"- **Brightness:** {report.brightness:.3f}\n"
        f"- **Contrast index:** {report.contrast:.3f}\n"
        f"- **Sharpness index:** {report.sharpness:.1f}\n"
        f"- **Color saturation:** {report.saturation:.3f}\n"
        f"- **Green-pixel fraction:** {report.green_fraction:.3f}\n"
        f"- **Image entropy:** {report.entropy:.2f} bits\n\n"
                "**Notes**\n" + notes
        + "\n\n*These are photo-quality heuristics, not disease severity measurements.*"
)
            def scan_probability_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract safe, concise uncertainty statistics from a scan result dictionary."""
    raw = result.get("probabilities", result.get("scores", []))
    probabilities = normalize_scores(raw)
    if probabilities.size == 0:
        return {
            "top_probability": 0.0,
            "margin": 0.0,
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "concentration_proxy": 0.0,
            "top_three_mass": 0.0,
        }
    sorted_values = np.sort(probabilities)[::-1]
    return {
        "top_probability": float(sorted_values[0]),
        "margin": prediction_margin(probabilities),
        "entropy_bits": shannon_entropy(probabilities),
        "normalized_entropy": normalized_entropy(probabilities),
        "concentration_proxy": entropy_confidence(probabilities),
        "top_three_mass": top_k_mass(probabilities, 3),
    }


def confidence_language(probability: float, margin: float) -> str:
    """Return cautious explanatory language for model-score concentration."""
    probability = float(np.clip(_safe_float(probability), 0.0, 1.0))
    margin = float(np.clip(_safe_float(margin), 0.0, 1.0))
    if probability >= 0.90 and margin >= 0.40:
        return "The model strongly favors one class among its available labels."
    if probability >= 0.70 and margin >= 0.15:
        return "The model favors one class, but independent confirmation is still important."
    if probability >= 0.45:
        return "The model is uncertain between several possible classes."
    return "The model has low class separation; treat this result as inconclusive."


def summarize_prediction_uncertainty(result: Mapping[str, Any]) -> str:
    """Create a human-readable uncertainty summary without overstating accuracy."""
    stats = scan_probability_summary(result)
    language = confidence_language(stats["top_probability"], stats["margin"])
    return (
        f"**Model uncertainty:** {html.escape(language)}\n\n"
        f"- Highest class score: {stats['top_probability']:.1%}\n"
        f"- Top-two margin: {stats['margin']:.1%}\n"
        f"- Normalized entropy: {stats['normalized_entropy']:.3f} (higher means more diffuse)\n"
        f"- Top-three probability mass: {stats['top_three_mass']:.1%}\n\n"
        "*These statistics describe the output distribution, not the real-world probability "
        "that a disease is present. Use a representative validation set to assess calibration.*"
    )


def healthy_probability(labels: Sequence[str], probabilities: Any) -> float:
    """Sum normalized model score for labels that contain the word 'healthy'."""
    p = normalize_scores(probabilities)
    if len(labels) != p.size:
        raise ValueError("The number of labels must match the score vector.")
    indices = [
        index for index, label in enumerate(labels)
        if "healthy" in str(label).casefold()
    ]
    return float(np.sum(p[indices])) if indices else 0.0


def aggregate_class_scores(
    score_rows: Sequence[Any],
    labels: Sequence[str],
    method: str = "mean",
) -> pd.DataFrame:
    """Aggregate repeated scores per class using mean, median, or maximum."""
    if not score_rows:
        return pd.DataFrame(columns=["class", "score", "support"])
    matrix = _finite_array(score_rows, ndim=2)
    if matrix.shape[1] != len(labels):
        raise ValueError("Score matrix width must match the label count.")
    matrix = np.apply_along_axis(normalize_scores, 1, matrix)
    if method == "median":
        values = np.median(matrix, axis=0)
    elif method == "max":
        values = np.max(matrix, axis=0)
    elif method == "mean":
        values = np.mean(matrix, axis=0)
    else:
        raise ValueError("method must be 'mean', 'median', or 'max'.")
    return pd.DataFrame({
        "class": list(labels),
        "score": values,
        "support": np.full(len(labels), matrix.shape[0], dtype=int),
    }).sort_values("score", ascending=False, ignore_index=True)


def distribution_shift_summary(reference: Any, current: Any) -> dict[str, float]:
    """Compare two class-score distributions using several descriptive distances."""
    p = normalize_scores(reference)
    q = normalize_scores(current)
    if p.size != q.size:
        raise ValueError("Reference and current distributions must have equal length.")
    return {
        "jensen_shannon_divergence": jensen_shannon_divergence(p, q),
        "histogram_intersection": histogram_intersection(p, q),
        "cosine_similarity": cosine_similarity(p, q),
        "total_variation_distance": float(0.5 * np.sum(np.abs(p - q))),
    }


def risk_index_from_conditions(
    temperature_c: float | None = None,
    relative_humidity: float | None = None,
    leaf_wetness_hours: float | None = None,
    crop_match: float = 0.5,
) -> dict[str, Any]:
    """Calculate a transparent, illustrative environment risk index.

    This is deliberately not an outbreak prediction. It uses smooth bounded
    transforms only to demonstrate how environmental signals can be combined.
    Real deployment requires a pathogen-specific, region-validated model and
    reliable weather data. Missing values receive neutral weights.
    """
    values = {
        "temperature_c": temperature_c,
        "relative_humidity": relative_humidity,
        "leaf_wetness_hours": leaf_wetness_hours,
    }
    components: dict[str, float] = {}
    if temperature_c is not None:
        temp = _safe_float(temperature_c, 20.0)
        # Smooth optimum centered at 24 C; generic illustrative curve only.
        components["temperature_suitability"] = math.exp(-0.5 * ((temp - 24.0) / 8.0) ** 2)
    if relative_humidity is not None:
        humidity = _clamp(relative_humidity, 0.0, 100.0) / 100.0
        components["humidity_suitability"] = float(np.clip((humidity - 0.35) / 0.55, 0.0, 1.0))
    if leaf_wetness_hours is not None:
        wetness = max(0.0, _safe_float(leaf_wetness_hours))
        components["wetness_suitability"] = float(1.0 - math.exp(-wetness / 8.0))
    components["crop_match"] = _clamp(crop_match, 0.0, 1.0)
    weights = {
        "temperature_suitability": 0.20,
        "humidity_suitability": 0.25,
        "wetness_suitability": 0.30,
        "crop_match": 0.25,
    }
    active_weight = sum(weights[key] for key in components)
    score = (
        sum(components[key] * weights[key] for key in components) / active_weight
        if active_weight else 0.0
    )
    missing = [key for key, value in values.items() if value is None]
    return {
        "illustrative_index": round(float(score * 100.0), 1),
        "components": components,
        "missing_inputs": missing,
        "interpretation": (
            "Illustrative screening index only—not a disease forecast or treatment recommendation."
        ),
    }


def score_threshold_sweep(
    probabilities: Any,
    true_indices: Any,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Evaluate coverage and accuracy for confidence-threshold abstention."""
    p = _finite_array(probabilities, ndim=2)
    labels = np.asarray(true_indices, dtype=int).reshape(-1)
    if p.shape[0] != labels.size:
        raise ValueError("Probability rows and labels must have equal length.")
    if thresholds is None:
        thresholds = np.linspace(0.25, 0.95, 15)
    if p.shape[0] == 0:
        return pd.DataFrame(columns=["threshold", "coverage", "selective_accuracy", "accepted"])
    p = np.apply_along_axis(normalize_scores, 1, p)
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = predicted == labels
    rows = []
    for threshold in thresholds:
        threshold = float(np.clip(threshold, 0.0, 1.0))
        accepted = confidence >= threshold
        count = int(accepted.sum())
        rows.append({
            "threshold": threshold,
            "coverage": count / len(labels),
            "selective_accuracy": float(correct[accepted].mean()) if count else 0.0,
            "accepted": count,
        })
    return pd.DataFrame(rows)


def _database_health_report() -> dict[str, Any]:
    """Inspect SQLite integrity and row counts without changing application data."""
    report: dict[str, Any] = {
        "database_path": str(DB_PATH),
        "exists": DB_PATH.is_file(),
        "size_bytes": DB_PATH.stat().st_size if DB_PATH.is_file() else 0,
        "integrity": "not checked",
        "tables": {},
        "error": "",
    }
    if not DB_PATH.is_file():
        report["error"] = "Database file has not been created yet."
        return report
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=3) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            report["integrity"] = integrity[0] if integrity else "unknown"
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for row in table_rows:
                table_name = str(row[0])
                # Names originate from sqlite_master; quote identifier defensively.
                quoted = '"' + table_name.replace('"', '""') + '"'
                try:
                    count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
                    report["tables"][table_name] = int(count)
                except sqlite3.Error:
                    report["tables"][table_name] = None
    except Exception as exc:
        report["error"] = _safe_text(exc, 300)
    return report


def _render_runtime_diagnostics() -> str:
    """Render model, filesystem, and SQLite checks as a Markdown status report."""
    try:
        runtime = _model_runtime_diagnostics()
    except Exception as exc:
        runtime = {"diagnostic_error": _safe_text(exc, 250)}
    database = _database_health_report()
    warnings = _check_runtime_configuration()
    lines = [
        "### Runtime diagnostics",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for key, value in runtime.items():
        lines.append(f"| {html.escape(str(key))} | {html.escape(str(value))} |")
    lines.extend([
        "",
        "### Database",
        "",
        f"- File exists: **{database['exists']}**",
        f"- Size: **{database['size_bytes']:,} bytes**",
        f"- Integrity check: **{html.escape(str(database['integrity']))}**",
    ])
    if database["error"]:
        lines.append(f"- Note: {html.escape(str(database['error']))}")
    if database["tables"]:
        lines.extend(["", "| Table | Rows |", "|---|---:|"])
        for table_name, count in sorted(database["tables"].items()):
            lines.append(f"| {html.escape(table_name)} | {count if count is not None else 'unavailable'} |")
    lines.extend(["", "### Configuration warnings"])
    lines.extend(f"- {html.escape(warning)}" for warning in warnings)
    if not warnings:
        lines.append("- No configuration warnings detected.")
    lines.extend([
        "",
        "*Diagnostics show process state only. They do not prove prediction accuracy or clinical/agronomic validity.*"
    ])
    return "\n".join(lines)


def _export_diagnostics_json() -> str:
    """Write a diagnostic snapshot and return its output path."""
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _model_runtime_diagnostics(),
        "database": _database_health_report(),
        "configuration_warnings": _check_runtime_configuration(),
    }
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / f"diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    _write_json_atomic(target, payload)
    return str(target)


def _history_analytics() -> pd.DataFrame:
    """Create simple analytics over stored scans without assuming labels are truth."""
    try:
        frame = get_history("")
    except Exception as exc:
        LOGGER.warning("History analytics unavailable: %s", _safe_text(exc, 200))
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame([{
            "metric": "Scans stored",
            "value": 0,
            "interpretation": "Run scans to populate this summary.",
        }])
    summary = _summarize_scan_frame(frame)
    rows = [
        {"metric": "Scans stored", "value": summary["total_scans"],
         "interpretation": "Count of recorded scans, not unique plants."},
        {"metric": "Mean reported confidence", "value": summary["mean_confidence"],
         "interpretation": "Mean model score; not measured accuracy."},
        {"metric": "Scans at or above 80% score", "value": summary["high_confidence_scans"],
         "interpretation": "Threshold is descriptive and not a correctness guarantee."},
        {"metric": "Distinct predicted labels", "value": summary["distinct_predictions"],
         "interpretation": "Variety of model outputs in stored history."},
    ]
    return pd.DataFrame(rows)


def _safe_csv_path(frame: pd.DataFrame, filename: str) -> str:
    """Write a CSV export with spreadsheet-formula mitigation."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / _safe_filename(filename, "export.csv")
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")
    _history_export_frame(frame).to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
    return str(target)


def _validated_reminder_date(value: Any) -> tuple[bool, str]:
    """Validate a date field and return a normalized ISO date when valid."""
    raw = _safe_text(value, 80)
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d").date()
        return True, parsed.isoformat()
    except ValueError:
        return False, "Use YYYY-MM-DD, for example 2026-12-31."


def _bounded_text_field(value: Any, maximum: int = 300) -> str:
    """Normalize free text for database storage and visible output."""
    return _safe_text(value, maximum)


def _validate_plant_record(name: Any, crop: Any, note: Any = "") -> tuple[bool, str, dict[str, str]]:
    """Validate a garden record before it is stored."""
    clean_name = _bounded_text_field(name, 100)
    clean_crop = _bounded_text_field(crop, 100)
    clean_note = _bounded_text_field(note, 1000)
    if not clean_name:
        return False, "Plant name is required.", {}
    if not clean_crop:
        return False, "Crop type is required.", {}
    return True, "Garden record looks valid.", {
        "plant_name": clean_name,
        "crop_type": clean_crop,
        "notes": clean_note,
    }


def _build_research_summary(result: Mapping[str, Any]) -> str:
    """Format a concise evidence-aware scan summary for a research notebook."""
    label = _safe_text(result.get("label", result.get("prediction", "Unknown")), 200)
    probability = _clamp(result.get("confidence", result.get("score", 0.0)), 0.0, 1.0)
    uncertainty = scan_probability_summary(result)
    return (
        f"Predicted class: {label}\n"
        f"Top score: {probability:.4f}\n"
        f"Top-two margin: {uncertainty['margin']:.4f}\n"
        f"Normalized entropy: {uncertainty['normalized_entropy']:.4f}\n"
        "Interpretation: model output only; independent crop-health confirmation is required."
    )


def _format_metric(value: Any, digits: int = 3) -> str:
    """Format finite metrics for dashboards and reports."""
    number = _safe_float(value, float("nan"))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{int(np.clip(digits, 0, 8))}f}"


def _time_bucket(iso_value: Any, bucket: str = "day") -> str:
    """Normalize an ISO timestamp into a day or month bucket."""
    raw = _safe_text(iso_value, 80)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if bucket == "month":
        return parsed.strftime("%Y-%m")
    if bucket == "hour":
        return parsed.strftime("%Y-%m-%d %H:00")
    return parsed.strftime("%Y-%m-%d")


def _group_history_by_period(frame: pd.DataFrame, timestamp_column: str = "created_at", bucket: str = "day") -> pd.DataFrame:
    """Aggregate history counts by a selected time bucket."""
    if frame is None or frame.empty or timestamp_column not in frame:
        return pd.DataFrame(columns=["period", "scan_count"])
    periods = frame[timestamp_column].map(lambda value: _time_bucket(value, bucket))
    result = periods.value_counts().rename_axis("period").reset_index(name="scan_count")
    return result.sort_values("period", ignore_index=True)


def _class_frequency_table(frame: pd.DataFrame, prediction_column: str = "prediction") -> pd.DataFrame:
    """Count predicted classes while safely handling missing columns and values."""
    if frame is None or frame.empty or prediction_column not in frame:
        return pd.DataFrame(columns=["prediction", "count", "share"])
    counts = frame[prediction_column].fillna("Unknown").astype(str).value_counts()
    total = int(counts.sum())
    return pd.DataFrame({
        "prediction": counts.index,
        "count": counts.values,
        "share": counts.values / total if total else np.zeros(len(counts)),
    }).reset_index(drop=True)


def _compare_confidence_groups(frame: pd.DataFrame, column: str = "confidence") -> dict[str, float]:
    """Summarize stored score dispersion without treating it as validation accuracy."""
    if frame is None or frame.empty or column not in frame:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0}
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
}
def _diagnostics_panel() -> tuple[str, pd.DataFrame]:
    """Return diagnostics and history analytics for the Advanced Lab tab."""
    return _render_runtime_diagnostics(), _history_analytics()


def _make_advanced_lab_tab() -> None:
    """Build the Advanced Lab tab; kept as a helper for future UI modularization."""
    # Gradio components must be instantiated inside the Blocks context.
    # The main layout below currently defines these components inline.
    return None


# ---------------------------- Advanced Lab UI ----------------------------
# This tab exposes useful system checks and stored-scan summaries. Statistical
# model-evaluation tools above accept labeled validation arrays; no ground-truth
# dataset is invented or inferred from the app's own predictions.


CSS = r"""
:root {
  color-scheme: dark;
  --agri-ink: #07111b;
  --agri-deep: #0b1724;
  --agri-panel: rgba(17, 34, 48, .88);
  --agri-line: rgba(130, 231, 191, .22);
  --agri-mint: #75f0b5;
  --agri-lime: #d6ff72;
  --agri-cyan: #65e4f3;
  --agri-violet: #b8a1ff;
  --agri-text: #f3fff9;
  --agri-muted: #a8c3bd;
}
html, body, .gradio-container {
  background:
    radial-gradient(ellipse at 0% 0%, rgba(26, 159, 112, .18), transparent 34%),
    radial-gradient(ellipse at 100% 8%, rgba(74, 112, 237, .14), transparent 32%),
    linear-gradient(145deg, #061018 0%, #0a1921 48%, #0a141d 100%) !important;
  color: var(--agri-text) !important;
}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 18px clamp(10px, 2.4vw, 30px) 34px !important;
}
#hero {
  position: relative;
  overflow: hidden;
  padding: clamp(22px, 4vw, 42px);
  border: 1px solid rgba(117, 240, 181, .36);
  border-radius: 28px;
  background:
    radial-gradient(circle at 88% 18%, rgba(101, 228, 243, .18), transparent 27%),
    radial-gradient(circle at 8% 110%, rgba(214, 255, 114, .13), transparent 34%),
    linear-gradient(125deg, rgba(16, 73, 59, .97), rgba(15, 35, 57, .97) 57%, rgba(35, 27, 68, .94));
  box-shadow: 0 24px 80px rgba(0, 0, 0, .22), inset 0 1px 0 rgba(255,255,255,.07);
  margin-bottom: 20px;
}
#hero:after {
  content: "✦";
  position: absolute;
  right: 6%;
  top: 7%;
  color: rgba(214,255,114,.75);
  font-size: clamp(44px, 7vw, 86px);
  filter: drop-shadow(0 0 22px rgba(214,255,114,.35));
  pointer-events: none;
}
#hero h1 {
  font-size: clamp(34px, 5vw, 58px);
  letter-spacing: -1.8px;
  line-height: 1.05;
  color: #f6fff8;
  margin: 12px 0;
  font-weight: 850;
  text-shadow: 0 4px 30px rgba(117,240,181,.12);
}
#hero p { color: #cce9e1; max-width: 760px; line-height: 1.7; }
#hero .eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(214,255,114,.32);
  border-radius: 99px;
  padding: 7px 12px;
  background: rgba(4, 18, 22, .35);
  color: var(--agri-lime);
  font-size: 11px;
  letter-spacing: 1.7px;
  font-weight: 900;
}
.card, .guide, .panel {
  background: linear-gradient(145deg, rgba(20, 42, 54, .92), rgba(12, 29, 39, .94)) !important;
  border: 1px solid var(--agri-line) !important;
  border-radius: 20px !important;
  padding: 19px !important;
  margin: 8px 0;
  box-shadow: 0 12px 32px rgba(0,0,0,.12), inset 0 1px 0 rgba(255,255,255,.025);
  transition: border-color .22s ease, transform .22s ease, box-shadow .22s ease;
}
.card:hover, .guide:hover, .panel:hover {
  border-color: rgba(117,240,181,.45) !important;
  box-shadow: 0 16px 38px rgba(0,0,0,.2);
}
.card h2 { font-size: clamp(23px, 3vw, 31px); color: var(--agri-text); margin: 10px 0; }
.card h3, .guide h3 { color: var(--agri-mint); }
.eyebrow { font-size: 11px; letter-spacing: 1.5px; font-weight: 850; color: var(--agri-lime); }
.metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.metric {
  background: linear-gradient(145deg, rgba(7, 23, 31, .9), rgba(13, 37, 44, .9));
  border: 1px solid rgba(101,228,243,.15);
  border-radius: 15px;
  padding: 14px;
}
.metric span { display:block; font-size:12px; color: var(--agri-muted); margin-bottom: 5px; }
.metric strong { font-size: clamp(20px, 3vw, 27px); color: var(--agri-text); letter-spacing: -.5px; }
.risk {
  display:inline-flex; align-items:center; gap:6px;
  border-radius:99px; border:1px solid rgba(214,255,114,.25);
  background:rgba(214,255,114,.09); padding:7px 12px; color:#eaffbc; font-size:13px;
}
.muted, .disclaimer { font-size:13px; color:var(--agri-muted); line-height:1.65; }
.prob-row { margin:15px 0; }
.prob-label { display:flex; justify-content:space-between; gap:10px; font-size:13px; color:#e7faf4; margin-bottom:7px; }
.prob-label span { overflow-wrap:anywhere; }
.track { height:10px; background:#1a3440; border-radius:99px; overflow:hidden; border:1px solid rgba(255,255,255,.035); }
.fill { height:100%; border-radius:99px; background:linear-gradient(90deg, var(--agri-mint), var(--agri-cyan), var(--agri-violet)); box-shadow:0 0 15px rgba(101,228,243,.18); transition:width .8s ease; }
.guidance-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:12px; }
.guide p { line-height:1.7; color:#c4dcd8; }
.organic { border-top:3px solid var(--agri-mint) !important; }
.chemical { border-top:3px solid #ffbd69 !important; }
.prevention { border-top:3px solid var(--agri-cyan) !important; }
.warning { border-color:rgba(255,189,105,.45) !important; background:linear-gradient(145deg,rgba(70,46,20,.55),rgba(30,28,26,.92)) !important; }
button, .gr-button {
  border-radius: 13px !important;
  font-weight: 750 !important;
  transition: transform .18s ease, filter .18s ease, box-shadow .18s ease !important;
}
button:hover, .gr-button:hover { transform:translateY(-1px) !important; filter:brightness(1.08); }
button.primary, .primary {
  background:linear-gradient(100deg,#13a66b,#55dfaa 58%,#8aeed0) !important;
  color:#04231b !important; border:0 !important;
  box-shadow:0 8px 24px rgba(32,191,129,.18) !important;
}
button.secondary { border:1px solid rgba(117,240,181,.3) !important; }
input, textarea, select, .wrap, .block, .form {
  border-radius:12px !important;
}
input, textarea, select {
  border-color:rgba(130,231,191,.19) !important;
}
input:focus, textarea:focus, select:focus {
  border-color:var(--agri-cyan) !important;
  box-shadow:0 0 0 2px rgba(101,228,243,.12) !important;
}
.tabs { border-bottom-color:rgba(130,231,191,.18) !important; }
.tab-nav button {
  border-radius:12px 12px 0 0 !important;
  font-weight:750 !important;
}
.tab-nav button.selected {
  color:var(--agri-lime) !important;
  border-color:rgba(117,240,181,.38) !important;
  background:rgba(117,240,181,.07) !important;
}
label, .label-wrap span { color:#d7eee7 !important; }
.prose, .markdown { color:#cce1dc; }
table { border-color:rgba(130,231,191,.18) !important; }
thead th { background:#153746 !important; color:#eafff5 !important; }
tbody tr:nth-child(even) { background:rgba(101,228,243,.035) !important; }
.file-preview, .upload-container {
  border-color:rgba(117,240,181,.25) !important;
  border-radius:16px !important;
}
.upload-container:hover { border-color:var(--agri-cyan) !important; }
footer { display:none !important; }
@media (prefers-reduced-motion: reduce) {
  *, *:before, *:after { transition:none !important; animation:none !important; scroll-behavior:auto !important; }
}
@media(max-width:720px) {
  .gradio-container { padding:10px 9px 24px !important; }
  #hero { padding:22px 18px; border-radius:20px; }
  #hero h1 { letter-spacing:-1px; }
  #hero:after { right:4%; top:3%; opacity:.45; }
  .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .card,.guide,.panel { padding:14px !important; border-radius:16px !important; }
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate"),
    css=CSS,
    title="AgriScan AI | Advanced Crop Health"
) as demo:
    latest_result = gr.State(None)
    gr.HTML(f"""
    <section id="hero">
      <div class="eyebrow">FIELD INTELLIGENCE · CROP CARE</div>
      <h1>🌿 AgriScan AI</h1>
      <p>Advanced leaf screening, evidence-aware guidance, and a persistent garden journal.</p>
      <p><small>{html.escape(MODEL_STATUS)}</small></p>
    </section>
    """)

    with gr.Tabs():
        with gr.Tab("Leaf scanner"):
            with gr.Row():
                with gr.Column(scale=5):
                    plant_for_scan = gr.Textbox(label="Plant name (optional)", placeholder="e.g. Tomato A / North plot")
                    image_input = gr.Image(type="pil", label="Drop a leaf photo or tap to upload", height=320)
                    scan_button = gr.Button("Analyze leaf →", variant="primary", size="lg")
                    quality_status = gr.Markdown("Photo quality checks run before model analysis.")
                    quality_details_button = gr.Button("Inspect photo quality")
                    quality_details_output = gr.Markdown()
                    quality_details_button.click(
                        image_quality_markdown,
                        inputs=[image_input],
                        outputs=[quality_details_output],
                    )
                with gr.Column(scale=6):
                    result_html = gr.HTML("<div class='card'><h3>Your scan will appear here</h3><p>Use a sharp, close-up photo in natural light.</p></div>")
                    guidance_html = gr.HTML()
            with gr.Row():
                heatmap_image = gr.Image(label="Optional model attention map (Grad-CAM)", interactive=False)
                scan_csv_file = gr.File(label="Download scan CSV report")
            scan_id_output = gr.Number(label="Latest scan ID", precision=0)
            with gr.Accordion("Create a PDF report for a scan", open=False):
                pdf_scan_id = gr.Number(label="Scan ID", precision=0)
                pdf_button = gr.Button("Create PDF report")
                pdf_file = gr.File(label="PDF report")
                pdf_status = gr.Markdown()

            scan_button.click(
                scan_ui,
                inputs=[image_input, plant_for_scan],
                outputs=[result_html, guidance_html, heatmap_image, scan_csv_file,
                         quality_status, scan_id_output, latest_result]
            )
            scan_id_output.change(lambda x: x, inputs=[scan_id_output], outputs=[pdf_scan_id])
            pdf_button.click(create_pdf_report, inputs=[pdf_scan_id], outputs=[pdf_file, pdf_status])

        with gr.Tab("Batch scan"):
            gr.Markdown("### Analyze multiple leaf photos\nBatch results are saved to scan history. The quality pre-check is heuristic and can reject valid leaves or accept non-leaf images.")
            batch_plant = gr.Textbox(label="Plant name (optional)")
            batch_files = gr.File(file_count="multiple", file_types=["image"], label="Upload several images")
            batch_button = gr.Button("Analyze batch", variant="primary")
            batch_table = gr.Dataframe(label="Batch results", interactive=False)
            batch_status = gr.Markdown()
            batch_csv = gr.File(label="Download batch CSV")
            batch_button.click(batch_scan, inputs=[batch_files, batch_plant], outputs=[batch_table, batch_status, batch_csv])

        with gr.Tab("My Garden"):
            gr.Markdown("### Persistent plant journal\nRecords are stored in SQLite in the current runtime. For long-term persistence across Colab resets, back up the database or deploy with persistent storage.")
            with gr.Row():
                garden_name = gr.Textbox(label="Plant name", placeholder="e.g. Tomato A")
                garden_crop = gr.Textbox(label="Crop type", placeholder="Tomato")
            garden_location = gr.Textbox(label="Plot / location note (optional)")
            garden_notes = gr.Textbox(label="Notes (optional)", lines=2)
            add_garden_button = gr.Button("Add plant", variant="primary")
            garden_status = gr.Markdown()
            garden_table = gr.Dataframe(value=garden_df, interactive=False, label="My Garden")
            refresh_garden = gr.Button("Refresh garden")
            garden_csv = gr.File(label="Export garden CSV")
            export_garden_button = gr.Button("Export garden")
            add_garden_button.click(add_garden, inputs=[garden_name, garden_crop, garden_location, garden_notes], outputs=[garden_table, garden_status])
            refresh_garden.click(garden_df, outputs=[garden_table])
            export_garden_button.click(export_garden, outputs=[garden_csv])

            gr.Markdown("### Follow-up reminders (saved locally; no push notifications)")
            with gr.Row():
                reminder_garden_id = gr.Number(label="Garden record ID", precision=0)
                reminder_date = gr.Textbox(label="Reminder date", placeholder="YYYY-MM-DD")
            reminder_message = gr.Textbox(label="Reminder message", value="Recheck the plant and take a new photo")
            reminder_add = gr.Button("Save reminder")
            reminder_status = gr.Markdown()
            reminder_table = gr.Dataframe(value=reminders_df, interactive=False, label="Reminders")
            reminder_refresh = gr.Button("Refresh reminders")
            reminder_complete_id = gr.Number(label="Reminder ID to mark complete", precision=0)
            reminder_complete = gr.Button("Mark reminder complete")
            reminder_add.click(add_reminder, inputs=[reminder_garden_id, reminder_date, reminder_message], outputs=[reminder_status])
            reminder_add.click(reminders_df, outputs=[reminder_table])
            reminder_refresh.click(reminders_df, outputs=[reminder_table])
            reminder_complete.click(complete_reminder, inputs=[reminder_complete_id], outputs=[reminder_table, reminder_status])

        with gr.Tab("Scan history"):
            search_history = gr.Textbox(label="Search by plant, prediction, or filename", placeholder="Type to filter")
            history_search_button = gr.Button("Search history")
            history_table = gr.Dataframe(value=get_history, interactive=False, label="Recent scans")
            history_refresh_button = gr.Button("Refresh")
            history_export_button = gr.Button("Export all history to CSV")
            history_export_file = gr.File(label="History CSV")
            history_export_status = gr.Markdown()
            history_search_button.click(get_history, inputs=[search_history], outputs=[history_table])
            history_refresh_button.click(lambda: get_history(""), outputs=[history_table])
            history_export_button.click(export_history, outputs=[history_export_file, history_export_status])

        with gr.Tab("Advanced Lab"):
            gr.Markdown("""
            ### Premium diagnostics and scan analytics
            This panel checks the runtime and summarizes recorded scans. It does not
            use predictions as ground truth and does not claim model accuracy.
            """)
            diagnostics_refresh = gr.Button("Run diagnostics", variant="primary")
            diagnostics_output = gr.Markdown()
            analytics_table = gr.Dataframe(label="Scan analytics", interactive=False)
            diagnostics_refresh.click(
                _diagnostics_panel,
                inputs=[],
                outputs=[diagnostics_output, analytics_table],
            )
            gr.Markdown("""
            **Numerical toolkit included:** stable softmax and temperature scaling, entropy and
            top-two margin, Brier score, log loss, expected calibration error, confusion matrix,
            per-class precision/recall/specificity/F1, Wilson intervals, bootstrap confidence
            intervals, robust outlier detection, moving averages, trend estimation,
            distribution-shift distances, and image-quality diagnostics.

            To evaluate accuracy or calibration, supply a representative labeled hold-out set.
            The app's own predictions are not ground-truth labels.
            """)

        with gr.Tab("Weather & safety"):
            gr.Markdown("""
            ### Local conditions and outbreak alerts
            Live weather and location access are **not enabled** in this version. This app does not collect your location.
            Check a trusted local forecast and agricultural extension advisory for your crop and district. Leaf wetness,
            humidity, temperature, crop variety, and local pathogen pressure all affect disease risk; a generic weather
            signal alone cannot confirm an outbreak.

            ### How to interpret this model
            - Confidence is the model's probability for a class, not a guarantee of correctness.
            - Healthy-class probability is the sum of probabilities for classes whose labels contain `healthy`.
            - The optional attention map is not a disease-spot bounding box or segmentation mask.
            - PlantVillage photos may differ from real farm conditions. Confirm important results locally.
            - Chemical product availability and legality vary by region. Follow the label and qualified local advice.
            """)

    gr.Markdown("""
    ---
    <small>AgriScan AI · Research prototype · Model and advice are not a substitute for a qualified agricultural professional.</small>
    """)

if MODEL is None:
    gr.Warning("Model is not loaded. Check the model path and label file shown in the status banner.")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        show_error=True
)

