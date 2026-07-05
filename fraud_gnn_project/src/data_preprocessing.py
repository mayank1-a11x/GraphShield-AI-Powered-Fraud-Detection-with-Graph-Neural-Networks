"""Load raw IEEE-CIS CSVs, clean, and engineer features.

The IEEE-CIS dataset ships as two tables (transaction + identity) joined on
TransactionID. This module merges them, imputes missing values, encodes
categoricals, and returns a single clean DataFrame ready for graph
construction.
"""
import logging
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _cell_column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _coerce_excel_value(value: str):
    if value is None or value == "":
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _read_xlsx_sheet_stdlib(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Read a simple .xlsx sheet without requiring openpyxl.

    The provided synthetic dataset is a plain workbook with shared strings and
    numeric cells, so the standard library is enough. This fallback keeps the
    project runnable in lightweight environments.
    """
    with zipfile.ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", XLSX_NS):
                text = "".join(t.text or "" for t in item.findall(".//main:t", XLSX_NS))
                shared_strings.append(text)

        wb_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall("rel:Relationship", XLSX_NS)
        }

        sheets = wb_root.findall("main:sheets/main:sheet", XLSX_NS)
        selected = None
        for sheet in sheets:
            if sheet_name is None or sheet.attrib["name"] == sheet_name:
                selected = sheet
                break
        if selected is None:
            raise ValueError(f"Sheet not found in {path}: {sheet_name}")

        rel_id = selected.attrib[f"{{{XLSX_NS['office_rel']}}}id"]
        target = rel_targets[rel_id]
        sheet_path = f"xl/{target}" if not target.startswith("xl/") else target

        root = ET.fromstring(workbook.read(sheet_path))
        parsed_rows = []
        max_col = 0
        for row in root.findall("main:sheetData/main:row", XLSX_NS):
            values = {}
            for cell in row.findall("main:c", XLSX_NS):
                col_idx = _cell_column_index(cell.attrib["r"])
                max_col = max(max_col, col_idx)
                raw = cell.find("main:v", XLSX_NS)
                inline = cell.find("main:is/main:t", XLSX_NS)
                value = raw.text if raw is not None else (inline.text if inline is not None else "")
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
                elif cell.attrib.get("t") != "str":
                    value = _coerce_excel_value(value)
                values[col_idx] = value
            parsed_rows.append(values)

        matrix = []
        for row in parsed_rows:
            matrix.append([row.get(i, np.nan) for i in range(max_col + 1)])

    if not matrix:
        return pd.DataFrame()

    header = matrix[0]
    return pd.DataFrame(matrix[1:], columns=header)


def _read_excel_sheet(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError:
        logger.info("openpyxl is not installed; using stdlib .xlsx reader")
        return _read_xlsx_sheet_stdlib(path, sheet_name)


def _xlsx_sheet_names(path: str) -> list[str]:
    with zipfile.ZipFile(path) as workbook:
        root = ET.fromstring(workbook.read("xl/workbook.xml"))
        return [sheet.attrib["name"] for sheet in root.findall("main:sheets/main:sheet", XLSX_NS)]


def _read_data_frame(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel_sheet(path)
    raise ValueError(f"Unsupported data format: {path}")


def load_raw_data() -> pd.DataFrame:
    """Load and merge the transaction and identity tables."""
    if os.path.exists(config.WORKBOOK_PATH):
        logger.info("Loading workbook dataset from %s", config.WORKBOOK_PATH)
        sheet_names = _xlsx_sheet_names(config.WORKBOOK_PATH)
        if {"train_transaction", "train_identity"}.issubset(sheet_names):
            tx = _read_excel_sheet(config.WORKBOOK_PATH, "train_transaction")
            identity = _read_excel_sheet(config.WORKBOOK_PATH, "train_identity")
            df = tx.merge(identity, on="TransactionID", how="left")
        else:
            df = _read_data_frame(config.WORKBOOK_PATH)
        if "TransactionID" not in df.columns:
            raise KeyError("Workbook must contain a TransactionID column.")
        logger.info("Loaded workbook shape: %s (fraud rate: %.3f%%)",
                    df.shape, 100 * df["isFraud"].mean())
        return df

    logger.info("Loading transaction table from %s", config.TRANSACTION_CSV)
    tx = _read_data_frame(config.TRANSACTION_CSV)

    logger.info("Loading identity table from %s", config.IDENTITY_CSV)
    identity = _read_data_frame(config.IDENTITY_CSV)

    df = tx.merge(identity, on="TransactionID", how="left")
    logger.info("Merged shape: %s (fraud rate: %.3f%%)",
                df.shape, 100 * df["isFraud"].mean())
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Impute, encode, and add derived features.

    Keeps behavior simple and transparent on purpose — this is the part of the
    pipeline you'll likely want to iterate on most as you explore the data.
    """
    df = df.copy()

    # --- Numeric columns: impute with median, keep a missing-indicator ---
    numeric_cols = [c for c in config.NUMERIC_TRANSACTION_COLS if c in df.columns]
    for col in numeric_cols:
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)
        df[col] = df[col].fillna(df[col].median())

    # --- Entity columns: fill missing with an explicit "unknown" bucket ---
    entity_cols = [c for cols in config.ENTITY_COLUMNS.values() for c in cols]
    entity_cols = [c for c in entity_cols if c in df.columns]
    for col in entity_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    # --- Derived features ---
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    if "TransactionDT" in df.columns:
        # TransactionDT is seconds since a reference point, not a real
        # timestamp — but the delta itself is useful for temporal features.
        df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
        df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24

    # --- Categorical encoding for remaining low-cardinality columns ---
    low_card_cats = ["ProductCD", "card4", "card6", "DeviceType"]
    for col in low_card_cats:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])

    logger.info("Post-cleaning shape: %s", df.shape)
    return df


def scale_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features. Fit only on train rows in practice —
    here we fit on the full frame for simplicity; swap in a proper
    train-only fit before final experiments to avoid leakage."""
    scaler = StandardScaler()
    df = df.copy()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Assemble the final transaction node feature matrix."""
    numeric_cols = [c for c in config.NUMERIC_TRANSACTION_COLS if c in df.columns]
    missing_flag_cols = [f"{c}_missing" for c in numeric_cols if f"{c}_missing" in df.columns]
    derived_cols = [c for c in ["TransactionAmt_log", "day_of_week", "hour_of_day"] if c in df.columns]
    low_card_cats = [c for c in ["ProductCD", "card4", "card6", "DeviceType"] if c in df.columns]

    feature_cols = numeric_cols + missing_flag_cols + derived_cols + low_card_cats
    df_scaled, _ = scale_numeric_features(df, numeric_cols + derived_cols)

    X = df_scaled[feature_cols].fillna(0).values.astype(np.float32)
    return X, feature_cols


def run_preprocessing() -> pd.DataFrame:
    """Convenience entrypoint: load -> clean -> return."""
    df = load_raw_data()
    df = clean_and_engineer(df)
    return df


if __name__ == "__main__":
    data = run_preprocessing()
    print(data.shape)
    print(data["isFraud"].value_counts(normalize=True))
