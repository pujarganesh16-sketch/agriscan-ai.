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
BASE_DIR = Path(
    os.environ.get("AGRISCAN_HOME", str(Path(__file__).resolve().parent))
)
MODEL_PATH = Path(os.environ.get("AGRISCAN_MODEL", str(BASE_DIR / "agriscan_model.keras")))
LABELS_PATH = Path(os.environ.get("AGRISCAN_LABELS", str(BASE_DIR / "class_labels.json")))
DB_PATH = Path(os.environ.get("AGRISCAN_DB", str(BASE_DIR / "agriscan_app.db")))
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

CSS = r"""
.gradio-container{max-width:1250px!important;margin:auto!important;background:#07130d!important}
#hero{padding:28px 30px;border:1px solid #294d38;border-radius:24px;background:radial-gradient(ellipse at 12% 0%,#20583b 0%,#10281b 53%,#09150e 100%);margin-bottom:16px}
#hero h1{font-size:clamp(32px,5vw,49px);letter-spacing:-1.4px;color:#f3fff7;margin:5px 0}
#hero p{color:#c2dfcc}
.card,.guide,.panel{background:#102219;border:1px solid #2a4a38;border-radius:18px;padding:18px;margin:8px 0;transition:all .25s ease}
.card h2{font-size:26px;color:#f0fff5;margin:10px 0}
.eyebrow{font-size:11px;letter-spacing:1.5px;font-weight:800;color:#69d99a}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.metric{background:#091a11;border-radius:12px;padding:12px}
.metric span{display:block;font-size:12px;color:#a7c6b2}.metric strong{font-size:24px;color:#f0fff5}
.risk{display:inline-block;border-radius:99px;background:#1b3d2a;padding:7px 11px;color:#d7f8e2;font-size:13px}
.muted,.disclaimer{font-size:13px;color:#aac4b4}
.prob-row{margin:14px 0}.prob-label{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:#e1f1e6;margin-bottom:7px}.prob-label span{overflow-wrap:anywhere}
.track{height:9px;background:#263d2e;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;transition:width .8s ease}
.guidance-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.guide p{line-height:1.6;color:#c7dccd}.organic{border-top:3px solid #34bb79}.chemical{border-top:3px solid #e0a43d}.prevention{border-top:3px solid #60aee0}
.warning{border-color:#986b2d;background:#2b2111}
button.primary{background:linear-gradient(90deg,#15965b,#2cbd78)!important;border:0!important}
button{transition:transform .18s ease!important}button:hover{transform:translateY(-1px)!important}
footer{display:none!important}
@media(max-width:650px){#hero{padding:20px}.metric-grid{grid-template-columns:1fr 1fr}.card,.guide,.panel{padding:14px}}
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
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "true").lower() == "true",
        show_error=True
    )
