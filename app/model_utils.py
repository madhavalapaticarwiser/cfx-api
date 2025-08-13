import warnings
from datetime import datetime

import pandas as pd
from fastai.tabular.all import load_learner
from rapidfuzz import process, fuzz

# Quiet fastai pickle warning
warnings.filterwarnings(
    'ignore',
    message="load_learner` uses Python's insecure pickle module"
)

# ---- Buckets ----
COND_BUCKETS = ["Excellent", "good", "average", "belowAverage", "rough", "great"]
LINE_BUCKETS = ["Mid", "Economy", "High", "Exotic"]
DRIVETRAIN_BUCKETS = [
    "FWD","AWD","RWD","front wheel drive","4x4","4x2","Front Wheel Drive",
    "four wheel drive","4WD","Four Wheel Drive","rear wheel drive", "all wheel drive",
    "2WD","Front"
]
TRANS_BUCKETS = {
    "AUTOMATIC", "Automatic", "A/T", "A",
    "8-Speed A/T", "10-Speed A/T", "6-Speed A/T", "5-Speed A/T", "4-Speed A/T", "1-Speed A/T",
    "8-speed automatic", "10-speed automatic", "6-speed automatic", "5-speed automatic", "4-speed automatic",
    "9-speed automatic", "7-speed automatic", "8-speed Tiptronic automatic", "8-speed Geartronic automatic",
    "ZF 8-speed automatic", "ZF 9-speed automatic", "ZF 6-speed automatic",
    "TorqueFlite 8-speed automatic", "TorqueFlite 9-speed automatic", "TorqueFlite 8 8-speed automatic",
    "Powertech 6-speed automatic", "TorqShift 5-speed automatic", "TorqShift 6-speed automatic", "TorqShift-G 6-speed automatic",
    "TorqShift 10-speed automatic", "Allison 10-speed automatic", "Allison 6-speed automatic",
    "Allison 5-speed automatic", "Allison 1000 5-speed automatic", "Getrag 6-speed automatic",
    "Getrag 6-speed multi-speed automatic", "SKYACTIV-Drive 6-speed automatic", "SKYACTIV-Drive 8-speed automatic",
    "CVT", "CVT Transmission", "1-speed CVT", "EFlite CVT", "EFlite 1-speed CVT", "E-Flite CVT",
    "2-speed CVT", "2-speed CVTi-S CVT", "Aisin CVT", "Aisin 2-speed CVT", "Jatco CVT", "Jatco 2-speed CVT",
    "5-speed CVT", "6-speed CVT", "8-speed CVT", "4-speed automatic/CVT", "Automatic/CVT",
    "2-speed automatic/CVT", "10-speed automatic/CVT", "10-speed Dynamic Shift automatic/CVT",
    "PowerSplit eCVT 2-speed CVT", "2-speed Intelligent Variable Transmission (IVT) CV",
    "2-speed Smartstream IVT CVT", "6-speed Xtronic CVT", "2-speed Xtronic CVT",
    "6-speed Lineartronic CVT", "8-speed Lineartronic CVT", "7-speed CVT", "7-speed Lineartronic CVT",
    "MANUAL", "Manual", "6-speed manual", "5-speed manual", "4-speed manual", "7-speed manual",
    "Tremec 6-speed manual", "TREMEC 6-speed manual", "TREMEC 10-speed manual", "Tremec 7-speed manual",
    "7-Speed M/T", "Aisin 5-speed manual", "ZF 6-speed manual", "SKYACTIV-MT 6-speed manual",
    "Getrag 5-speed manual",
    "8-speed auto-shift manual", "6-speed auto-shift manual", "7-speed auto-shift manual",
    "AUTOMATED_MANUAL", "PDK 7-speed auto-shift manual", "PDK 8-speed auto-shift manual",
    "TREMEC 7-speed auto-shift manual", "7-speed DSG auto-shift manual", "7-speed EcoShift DCT auto-shift manual",
    "6-speed EcoShift DCT auto-shift manual", "6-speed S tronic auto-shift manual", "7-speed S tronic auto-shift manual",
    "DIRECT_DRIVE", "8-speed multitronic CVT", "2-speed automatic"
}

def _fuzzy(val, choices, thresh=0.7, default=None):
    """Return best fuzzy match or default."""
    if val in choices:
        return val
    best = process.extractOne(val, choices, scorer=fuzz.token_sort_ratio)
    if best:
        match, score, _ = best
        if score / 100 >= thresh:
            return match
    return default

class CarPriceEnsemble:
    """Load 3 FastAI learners; expose one .predict_all that returns Retail/Private/Trade-In."""
    def __init__(self, model_paths: dict[str, str], data_path: str):
        self.learners = {name: load_learner(path) for name, path in model_paths.items()}
        df = pd.read_csv(data_path)

        self.valid_makes = df['make'].unique().tolist()
        self.models_by_make = {
            m: df.loc[df.make == m, 'model'].unique().tolist()
            for m in self.valid_makes
        }
        self.trims_by_m_m = {
            (m, md): df.loc[(df.make == m) & (df.model == md), 'trim'].unique().tolist()
            for m in self.models_by_make for md in self.models_by_make[m]
        }

    def _clean_row(self, raw: dict) -> tuple[pd.DataFrame, dict]:
        """Return (single-row DF, matched_info). Raises ValueError on unknown make/model."""
        # Compute age strictly from 'year'
        age = int(datetime.now().year) - int(raw["year"])

        make = _fuzzy(raw["make"], self.valid_makes, 0.7)
        if make is None:
            raise ValueError(f"Unknown make '{raw['make']}'")

        model = _fuzzy(raw["model"], self.models_by_make[make], 0.6)
        if model is None:
            raise ValueError(f"Unknown model '{raw['model']}' for make '{make}'")

        trim = _fuzzy(raw["trim"], self.trims_by_m_m.get((make, model), []), 0.5, "Other")

        cleaned = {
            "age": age,
            "mileage": raw["mileage"],
            "make": make,
            "model": model,
            "trim": trim,
            "interior":    _fuzzy(raw["interior"],    COND_BUCKETS,       0.7, "average"),
            "exterior":    _fuzzy(raw["exterior"],    COND_BUCKETS,       0.7, "average"),
            "mechanical":  _fuzzy(raw["mechanical"],  COND_BUCKETS,       0.7, "average"),
            "line":        _fuzzy(raw["line"],        LINE_BUCKETS,       0.7, "Economy"),
            "drivetrain":  _fuzzy(raw["drivetrain"],  DRIVETRAIN_BUCKETS, 0.7, "AWD"),
            "transmission":_fuzzy(raw["transmission"],TRANS_BUCKETS,      0.7, "Automatic"),
        }
        matched = {"make": make, "model": model, "trim": trim}
        return pd.DataFrame([cleaned]), matched

    @staticmethod
    def _enforce_gaps(r, p, t, gap=500):
        p = max(p, t + gap)
        r = max(r, p + gap)
        return r, p, t

    def predict_all(self, raw_payload: dict, enforce_gap=True) -> dict:
        row_df, matched = self._clean_row(raw_payload)
        preds = {}
        for label, learn in self.learners.items():
            dl = learn.dls.test_dl(row_df)
            pred, _ = learn.get_preds(dl=dl)
            preds[label] = float(pred[0])

        if enforce_gap:
            preds["Retail"], preds["Private"], preds["Trade-In"] = self._enforce_gaps(
                preds["Retail"], preds["Private"], preds["Trade-In"]
            )
        return {"predictions": preds, "matched_vehicle": matched}
