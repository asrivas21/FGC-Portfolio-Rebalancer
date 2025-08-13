import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# ------------------------- knobs you can tweak --------------------------
SHORTLIST_SIZE = 20          # candidates per replaceable slot
BUDGET_BAND = 0.02           # ±2% vs current market value
DUPLICATE_CUSIP_PENALTY = 1e6
INFEASIBLE = 1e12            # big number for "reject this candidate"

# Internal trade guards
MAX_QTY = 5500               # do not allow any adjustable bond to exceed this
SOFT_QTY_CAP = 3500          # prefer not to exceed this per bond
SOFT_PENALTY_WEIGHT = 6.0    # strength of soft-cap penalty
W_ORIG = 1.6                 # penalize original overshoot a bit more
W_CAND = 1.0                 # candidate overshoot penalty weight
CAP_HIT_PENALTY = 1e-3       # tiny tie-breaker penalty per capped bond
SWAP_REWARD = 0.05           # small reward per slot that actually uses a candidate

# Concentration controls (by market value share of the whole portfolio)
MAX_CONTRIB_PCT = 0.20       # e.g., 0.20 = 20% per CUSIP
CONTRIB_SOFT_WEIGHT = 200.0  # penalty weight per percent over cap (tune)
STRICT_CONTRIB_CAP = True   # True = hard constraint, False = soft penalty
CONTRIB_SLACK = 0.00         # allowed slack above cap before triggering
# -----------------------------------------------------------------------


# ---------- utilities ----------
def rating_to_score(r):
    if pd.isna(r): return np.nan
    r = str(r).upper().strip()
    if r.startswith("AAA"): return 1.0
    if r.startswith("AA"):  return 1.5
    if r.startswith("A"):   return 2.0
    if r.startswith("BBB"): return 3.0
    if r.startswith("BB"):  return 4.0
    if r.startswith("B"):   return 5.0
    return 6.0

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def mv_of(df, qty_col="Quantity"):
    return (df["Price"] * df[qty_col] / 100.0)

def weighted_avg(df, qty_col, value_col):
    mv = mv_of(df, qty_col)
    mask = df[value_col].notna()
    mv = mv[mask]
    if mv.sum() <= 0:
        return np.nan
    return (mv * df.loc[mask, value_col]).sum() / mv.sum()


def fill_and_audit_duration(ext):
    # Ensure the columns exist so .notna() never KeyErrors
    if "Duration" not in ext.columns:
        ext["Duration"] = np.nan
    if "DV01" not in ext.columns:
        ext["DV01"] = np.nan
    if "Price" not in ext.columns:
        ext["Price"] = np.nan

    # Tag source for debugging
    ext["DurSrc"] = np.where(ext["Duration"].notna(), "csv", "missing")

    n_total = len(ext)
    n_csv = ext["Duration"].notna().sum()
    print(f"[duration] start: {n_csv}/{n_total} rows with Duration in CSV")
    print(f"[duration] columns present: {sorted(list(ext.columns))[:12]} ...")

    # Try to infer a DV01 scale only if we actually have both DV01 and Price
    scale = 1.0
    mask_both = ext["Duration"].notna() & ext["DV01"].notna() & ext["Price"].notna()
    if mask_both.sum() >= 10:
        md_from_dv01 = (ext.loc[mask_both, "DV01"] * 10000.0) / ext.loc[mask_both, "Price"]
        ratio = (ext.loc[mask_both, "Duration"] / md_from_dv01).replace([np.inf, -np.inf], np.nan)
        med = np.nanmedian(ratio)
        if np.isfinite(med) and med > 0:
            scale = float(med)
        print(f"[duration] inferred DV01 scale median ≈ {scale:.3f}")

    # Fill missing Duration only when DV01 and Price are available
    mask_fill = ext["Duration"].isna() & ext["DV01"].notna() & ext["Price"].notna()
    if mask_fill.any():
        ext.loc[mask_fill, "Duration"] = scale * (ext.loc[mask_fill, "DV01"] * 10000.0) / ext.loc[mask_fill, "Price"]
        ext.loc[mask_fill, "DurSrc"] = "dv01"

    n_after = ext["Duration"].notna().sum()
    print(f"[duration] after dv01 fill: {n_after}/{n_total} have Duration (added {n_after - n_csv})")

    # What’s still missing?
    still_nan = ext["Duration"].isna()
    if still_nan.any():
        miss = ext.loc[still_nan, ["CUSIP", "Price", "Yield", "DV01"]]
        print(f"[duration] remaining NaN: {still_nan.sum()} rows "
              f"| missing DV01: {miss['DV01'].isna().sum()} | missing Price: {miss['Price'].isna().sum()}")
        print(miss.head(8).to_string(index=False))

    q = ext["Duration"].dropna()
    if len(q):
        q = q.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        print("[duration] quantiles (years):", {float(k): round(v, 3) for k, v in q.items()})
    else:
        print("[duration] quantiles (years): no non-NaN durations available yet")

    return ext



def fill_and_audit_dv01(ext):
    # make sure columns exist so we never KeyError
    for c in ["DV01", "Duration", "Price"]:
        if c not in ext.columns:
            ext[c] = np.nan

    n_total = len(ext)
    have0 = ext["DV01"].notna().sum()

    # 1) Direct fill: DV01 = ModDur * Price / 10000  (price per $100 par)
    m = ext["DV01"].isna() & ext["Duration"].notna() & ext["Price"].notna()
    if m.any():
        ext.loc[m, "DV01"] = ext.loc[m, "Duration"] * ext.loc[m, "Price"] / 10000.0

    have1 = ext["DV01"].notna().sum()
    print(f"[dv01] filled from Duration&Price: {have1 - have0} rows "
          f"({have1}/{n_total} now have DV01)")

    # 2) Optional: if still missing, estimate Duration from cash flows, then DV01
    # Works when we have coupon, maturity date, and yield (YTM). Frequency defaults to 2.
    need_dur = ext["DV01"].isna() & ext["Duration"].isna()
    if need_dur.any():
        # unify column names we might see
        coupon_col = next((c for c in ["Coupon", "CouponRate", "Coupon %"] if c in ext.columns), None)
        ytm_col    = "Yield" if "Yield" in ext.columns else ("YTM" if "YTM" in ext.columns else None)
        mat_col    = next((c for c in ["Maturity", "Maturity Date", "MaturityDate"] if c in ext.columns), None)
        freq_col   = next((c for c in ["Frequency", "Freq"] if c in ext.columns), None)

        if coupon_col and ytm_col and mat_col:
            def _est_mod_dur(row):
                try:
                    # inputs
                    price = row.get("Price", np.nan)
                    y = float(row[ytm_col]) / 100.0  # CSV is in %, convert to decimal
                    cpn = float(row[coupon_col]) / 100.0  # coupon % to decimal (annual)
                    f = int(row[freq_col]) if freq_col and pd.notna(row[freq_col]) else 2
                    mat = pd.to_datetime(row[mat_col], errors="coerce")
                    if not pd.notna(mat) or not np.isfinite(y) or not np.isfinite(cpn) or f <= 0:
                        return np.nan

                    # quick and dirty time to maturity in years
                    t_years = max(0.0, (mat - pd.Timestamp.today().normalize()).days / 365.0)
                    n = max(1, int(round(f * t_years)))

                    # cash flows per $100 par
                    cf = np.full(n, 100 * cpn / f, dtype=float)
                    cf[-1] += 100.0  # redemption
                    # discount factors
                    df = 1.0 / (1.0 + y / f) ** np.arange(1, n + 1)
                    pv = (cf * df).sum()
                    if pv <= 0:
                        return np.nan

                    # Macaulay duration (in years), then Modified duration
                    macaulay = (np.arange(1, n + 1) * cf * df).sum() / pv / f
                    mod_dur = macaulay / (1.0 + y / f)
                    return mod_dur
                except Exception:
                    return np.nan

            est_mask = need_dur.copy()
            est_vals = ext.loc[est_mask].apply(_est_mod_dur, axis=1)
            filled = est_vals.notna().sum()
            if filled:
                ext.loc[est_mask, "Duration"] = est_vals.values
                print(f"[dv01] estimated Duration from CF for {filled} rows")

                # backfill DV01 where we now have Duration & Price
                m2 = ext["DV01"].isna() & ext["Duration"].notna() & ext["Price"].notna()
                ext.loc[m2, "DV01"] = ext.loc[m2, "Duration"] * ext.loc[m2, "Price"] / 10000.0
                print(f"[dv01] backfilled DV01 after duration-estimate for {m2.sum()} rows")

    # final stats
    haveF = ext["DV01"].notna().sum()
    print(f"[dv01] final coverage: {haveF}/{n_total}")
    return ext


# ---------- 1) Load data ----------
def load_data(csv_path):
    # Current portfolio
    current = pd.DataFrame({
        "CUSIP": [
            "001055AQ5", "00206RDQ2", "00206RGQ9",
            "00206RJX1", "00206RJY9"
        ],
        "Price": [99.88, 97.84, 97.18, 95.23, 87.11],
        "Yield": [2.96, 5.47, 4.96, 4.67, 5.23],
        "Duration": [1.42, 0.50, 0.69, 0.84, 1.01],
        "Rating_Score": [2.5, 3.0, 3.0, 3.0, 3.0],
        "DV01": [0.0142, 0.0049, 0.0067, 0.0080, 0.0088],
        "Quantity": [2000, 2000, 2000, 2000, 2000]
    })

    ext = pd.read_csv(csv_path, low_memory=False)

    # normalize likely column names
    rename_map = {
        "YTM": "Yield",
        "ModDur": "Duration", "Mod Dur": "Duration",
        "ModifiedDuration": "Duration", "Modified Duration": "Duration",
        "Qty": "Quantity",
        "DV 01": "DV01", "Dv01": "DV01", "dv01": "DV01",
    }
    ext = ext.rename(columns=rename_map)

    # coerce numeric early
    ext = coerce_numeric(ext, ["Price", "Yield", "Duration", "DV01", "Quantity", "Min", "Max"])

    # robust Rating_Score
    if "Rating_Score" not in ext.columns:
        ext["Rating_Score"] = np.nan
    rating_cols = [
        "Rating_Score",
        "S&P Rating", "SP Rating", "S&P", "SP",
        "Moody's Rating", "Moodys Rating", "Moody Rating",
        "Fitch Rating", "Fitch",
        "Composite Rating", "BBG Composite Rating", "Bloomberg Composite Rating",
        "Composite", "Rating"
    ]
    for col in rating_cols:
        if col in ext.columns:
            ext["Rating_Score"] = ext["Rating_Score"].fillna(
                ext[col].astype(str).apply(rating_to_score)
            )
    ext["Rating_Score"] = ext["Rating_Score"].fillna(3.0)  # neutral fallback

    # build & audit Duration (incl. fill from DV01/Price)
    ext = fill_and_audit_duration(ext)
    ext = fill_and_audit_dv01(ext)

    # keep minimally required fields
    required_min = [c for c in ["CUSIP", "Price", "Yield"] if c in ext.columns]
    ext = ext.dropna(subset=required_min)

    # remove CUSIPs already held
    ext = ext[~ext["CUSIP"].isin(current["CUSIP"])].drop_duplicates(subset=["CUSIP"]).reset_index(drop=True)

    # final sanity
    ext = ext[(ext["Price"] > 0) & (ext["Yield"].notna())]

    print("\n[load_data] ext rows after cleaning:", len(ext))
    if len(ext) == 0:
        print("[load_data] WARNING: no candidates in universe after cleaning. Check CSV column names.")
    return current, ext


# ---------- 2) Get user input ----------
def get_user_inputs(current):
    print("Bonds in current portfolio:")
    for i, row in current.iterrows():
        print(f"{i+1}: {row['CUSIP']} (Qty={row['Quantity']})")

    goals = input("\nEnter parameters to optimize (e.g. 'yield duration' or 'yield'): ").strip().lower()
    selected_params = set([g for g in goals.replace(",", " ").split() if g])

    targets, tolerances = {}, {}
    for p in selected_params:
        targets[p] = float(input(f"Enter target for {p}: "))
        tolerances[p] = float(input(f"Enter tolerance for {p} (e.g., 0.1): "))

    min_qty = int(input("Enter minimum quantity per bond: "))
    lot_size = int(input("Enter lot size: "))

    keep_str = input("Which bonds do you want to KEEP? (e.g., '2 6'): ").strip()
    keep_indices = sorted(set(int(i) - 1 for i in keep_str.split() if i.isdigit()))

    # optional overrides
    bb = input(f"Budget band percent (default {int(BUDGET_BAND*100)}): ").strip()
    shortlist_sz = input(f"Shortlist size per slot (default {SHORTLIST_SIZE}): ").strip()
    band = BUDGET_BAND if not bb else float(bb)/100.0
    k_short = SHORTLIST_SIZE if not shortlist_sz else int(shortlist_sz)

    return selected_params, targets, tolerances, min_qty, lot_size, keep_indices, band, k_short


# ---------- 3) Build slots + shortlists ----------
def build_slots_and_shortlists(current, ext, selected_params, targets, tolerances, keep_indices, shortlist_k):
    # if duration is a target, do not consider NaN durations
    if "duration" in selected_params:
        ext = ext[ext["Duration"].notna()].copy()

    kept = current.iloc[keep_indices].copy().reset_index(drop=True)
    kept["Frozen"] = True

    repl_idx = [i for i in range(len(current)) if i not in keep_indices]
    slots = current.iloc[repl_idx].copy().reset_index(drop=True)
    slots["Frozen"] = False

    used_cusips = set(kept["CUSIP"])
    shortlists = []

    def feasible_pool(df):
        pool = df[~df["CUSIP"].isin(used_cusips)].copy()
        if "duration" in selected_params:
            max_dur = targets["duration"] + tolerances["duration"]
            feas = pool[pool["Duration"] <= max_dur]
            if len(feas) >= max(10, shortlist_k // 3):
                pool = feas
        return pool

    def proximity_score(row):
        score = 0.0
        if "yield" in selected_params and "yield" in targets:
            score += max(0.0, targets["yield"] - row["Yield"])  # penalize only below target
        if "duration" in selected_params and "duration" in targets:
            max_dur = targets["duration"] + tolerances["duration"]
            score += 0.1 * abs(row["Duration"] - targets["duration"])
            score += 10.0 * max(0.0, row["Duration"] - max_dur)
        return score

    k_prox  = max(5, shortlist_k // 2)
    k_yld   = max(5, shortlist_k // 3)
    k_short = max(0, shortlist_k - k_prox - k_yld)

    for _, _slot in slots.iterrows():
        pool = feasible_pool(ext)

        pool = pool.copy()
        pool["prox"] = pool.apply(proximity_score, axis=1)
        top_prox   = pool.sort_values("prox", ascending=True).head(k_prox)
        top_yield  = pool.nlargest(k_yld, "Yield")
        top_short  = pool.nsmallest(k_short, "Duration") if "duration" in selected_params else pool.head(0)

        merged = pd.concat([top_prox, top_yield, top_short], ignore_index=True)
        merged = merged.drop_duplicates(subset=["CUSIP"]).reset_index(drop=True)
        merged = merged.head(shortlist_k).reset_index(drop=True)
        shortlists.append(merged)

    # debug prints
    for j, sl in enumerate(shortlists, 1):
        max_y = sl["Yield"].max() if len(sl) else None
        print(f"Slot {j}: shortlist size={len(sl)} max_yield={max_y}")
        if len(sl):
            print(sl[["CUSIP","Yield","Duration","DurSrc","Rating_Score"]]
                  .nlargest(5, "Yield").to_string(index=False))

    return kept, slots, shortlists


# ---------- 4) Objective ----------
def build_objective(kept, slots, shortlists, selected_params, targets, tolerances, min_qty, lot_size, budget_band):
    current_total_mv = mv_of(pd.concat([kept, slots], ignore_index=True)).sum()
    budget_low = (1.0 - budget_band) * current_total_mv
    budget_high = (1.0 + budget_band) * current_total_mv

    base_avg_rating = weighted_avg(pd.concat([kept, slots], ignore_index=True), "Quantity", "Rating_Score")

    # decision layout: [sel, q_orig, q_cand] per slot
    layout = []
    for j, _ in enumerate(slots.itertuples(index=False)):
        S_j = len(shortlists[j])
        layout += [("sel", j, S_j), ("q_orig", j), ("q_cand", j)]

    kept_block = kept[["CUSIP","Price","Yield","Duration","Rating_Score","DV01","Quantity"]].copy()

    def objective(x):
        cap_hits_total = 0
        softcap_pen_units = 0.0
        swap_count = 0
        chosen_cusips = []
        cursor = 0
        built_rows = [kept_block.copy()]

        for idx_slot, slot in enumerate(slots.itertuples(index=False)):
            sel_val = x[cursor]; cursor += 1
            q_orig  = x[cursor]; cursor += 1
            q_cand  = x[cursor]; cursor += 1

            # soft-cap tracking
            excess_orig = max(0, q_orig - SOFT_QTY_CAP)
            excess_cand = max(0, q_cand - SOFT_QTY_CAP)
            softcap_pen_units += W_ORIG * excess_orig + W_CAND * excess_cand

            # map selector
            S_j = layout[3*idx_slot + 0][2]
            sel_idx = int(round(sel_val))
            sel_idx = max(0, min(sel_idx, S_j))

            # lot size & min qty
            q_orig = int(round(q_orig / lot_size)) * lot_size
            q_cand = int(round(q_cand / lot_size)) * lot_size
            if q_orig != 0 and q_orig < min_qty: return INFEASIBLE
            if q_cand != 0 and q_cand < min_qty: return INFEASIBLE

            # cap hits (tie-break)
            if q_orig >= MAX_QTY: cap_hits_total += 1
            if q_cand >= MAX_QTY: cap_hits_total += 1

            # original row
            built_rows.append(pd.DataFrame({
                "CUSIP":[slot.CUSIP],"Price":[slot.Price],"Yield":[slot.Yield],
                "Duration":[slot.Duration],"Rating_Score":[slot.Rating_Score],
                "DV01":[slot.DV01],"Quantity":[q_orig]
            }))

            # candidate row
            if sel_idx > 0:
                c = shortlists[idx_slot].iloc[sel_idx - 1]
                built_rows.append(pd.DataFrame({
                    "CUSIP":[c["CUSIP"]],"Price":[c["Price"]],"Yield":[c["Yield"]],
                    "Duration":[c["Duration"]],"Rating_Score":[c["Rating_Score"]],
                    "DV01":[c.get("DV01", np.nan)],"Quantity":[q_cand]
                }))
                if q_cand > 0:
                    chosen_cusips.append(c["CUSIP"])
                    swap_count += 1

        # combine and budget guard
        port = pd.concat(built_rows, ignore_index=True)
        total_mv = mv_of(port).sum()
        if total_mv < budget_low or total_mv > budget_high:
            return INFEASIBLE

        # drop zero qty
        port = port[port["Quantity"] > 0]
        if len(port) == 0:
            return INFEASIBLE

        # concentration cap (by MV share)
        mv_series = (port["Price"] * port["Quantity"] / 100.0)
        share_by_cusip = mv_series.groupby(port["CUSIP"]).sum() / max(1e-12, total_mv)
        over = (share_by_cusip - (MAX_CONTRIB_PCT + CONTRIB_SLACK)).clip(lower=0.0)
        if STRICT_CONTRIB_CAP and (over > 0).any():
            return INFEASIBLE
        contrib_penalty = over.sum() * CONTRIB_SOFT_WEIGHT

        # rating guard
        avg_rating = weighted_avg(port, "Quantity", "Rating_Score")
        if np.isnan(avg_rating) or avg_rating > base_avg_rating + 0.2:
            return INFEASIBLE

        # portfolio stats
        avg_yield = weighted_avg(port, "Quantity", "Yield")
        avg_dur   = weighted_avg(port, "Quantity", "Duration")
        tot_dv01  = (port["DV01"].fillna(0.0) * port["Quantity"]).sum()

        # hard constraints
        if "yield" in selected_params:
            if avg_yield + 1e-12 < targets["yield"] - tolerances["yield"]:
                return INFEASIBLE
        if "duration" in selected_params:
            if avg_dur - 1e-12 > targets["duration"] + tolerances["duration"]:
                return INFEASIBLE
        if "dv01" in selected_params:
            if abs(tot_dv01 - targets["dv01"]) > tolerances["dv01"]:
                return INFEASIBLE

        # no duplicate candidate CUSIPs
        if len(chosen_cusips) != len(set(chosen_cusips)):
            return DUPLICATE_CUSIP_PENALTY

        # objective (minimize)
        obj = -avg_yield if not np.isnan(avg_yield) else 0.0
        if "duration" in selected_params and not np.isnan(avg_dur):
            obj += 0.05 * avg_dur  # gentle nudge toward shorter subject to constraint

        # penalties & rewards (single application)
        obj += SOFT_PENALTY_WEIGHT * (softcap_pen_units / max(1.0, SOFT_QTY_CAP))
        obj += contrib_penalty
        obj -= SWAP_REWARD * swap_count
        obj += cap_hits_total * CAP_HIT_PENALTY

        return obj

    # bounds for DE: [sel, q_orig, q_cand] per slot
    bounds = []
    for j, _ in enumerate(slots.itertuples(index=False)):
        S_j = len(shortlists[j])
        bounds.append((0, S_j))       # selector
        bounds.append((0, MAX_QTY))   # original qty
        bounds.append((0, MAX_QTY))   # candidate qty

    return objective, bounds, current_total_mv, base_avg_rating


# ---------- 5) Run optimization ----------
def run_optimization(objective, bounds):
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=400,
        popsize=25,
        tol=1e-6,
        polish=False,
        disp=False,
        seed=42
    )
    return result


# ---------- 6) Reporting ----------
def report(current, kept, slots, shortlists, result, lot_size, selected_params, targets, tolerances, budget_band):
    x = result.x
    rows = []
    cursor = 0

    kept_out = kept.copy()
    kept_out["NewQty"] = kept_out["Quantity"]

    for j, slot in enumerate(slots.itertuples(index=False)):
        S_j = len(shortlists[j])
        sel = int(round(x[cursor])); cursor += 1
        q_orig = int(round(x[cursor] / lot_size))*lot_size; cursor += 1
        q_cand = int(round(x[cursor] / lot_size))*lot_size; cursor += 1

        rows.append({
            "CUSIP": slot.CUSIP, "Type": "Original",
            "Price": slot.Price, "Yield": slot.Yield, "Duration": slot.Duration,
            "Rating_Score": slot.Rating_Score, "DV01": slot.DV01,
            "OldQty": slot.Quantity, "NewQty": q_orig
        })

        if sel > 0 and sel <= S_j:
            c = shortlists[j].iloc[sel - 1]
            rows.append({
                "CUSIP": c["CUSIP"], "Type": "Candidate",
                "Price": c["Price"], "Yield": c["Yield"], "Duration": c["Duration"],
                "Rating_Score": c["Rating_Score"], "DV01": c.get("DV01", np.nan),
                "OldQty": 0, "NewQty": q_cand
            })

    out = pd.DataFrame(rows)
    full = pd.concat([
        kept_out.rename(columns={"Quantity":"OldQty"})[["CUSIP","Price","Yield","Duration","Rating_Score","DV01","OldQty","NewQty"]],
        out
    ], ignore_index=True)

    final = full.copy()
    final = final[final["NewQty"] > 0]

    total_mv = mv_of(final, "NewQty").sum()
    avg_yield = weighted_avg(final, "NewQty", "Yield")
    avg_dur   = weighted_avg(final, "NewQty", "Duration")
    tot_dv01  = (final["DV01"].fillna(0.0) * final["NewQty"]).sum()

        # ==== MATH AUDIT ====
    audit = final.copy()
    audit["MV"] = audit["Price"] * audit["NewQty"] / 100.0
    total_mv2 = audit["MV"].sum()

    # contribution (what each line adds to the numerator of the weighted average)
    audit["YieldContr"] = audit["MV"] * audit["Yield"]             # in %-MV units
    audit["DurContr"]   = audit["MV"] * audit["Duration"]          # in year-MV units

    # shares (to spot concentration)
    audit["MV_Share_%"] = 100.0 * audit["MV"] / max(1e-12, total_mv2)

    # show the biggest contributors
    print("\n--- Math audit (top 10 by MV) ---")
    cols = ["CUSIP","Type","NewQty","Price","Yield","Duration","MV","MV_Share_%","YieldContr","DurContr","DV01"]
    print(audit.sort_values("MV", ascending=False)[cols].head(10).to_string(index=False))

    # recompute portfolio stats from scratch
    avg_yield2 = audit["YieldContr"].sum() / max(1e-12, total_mv2)
    avg_dur2   = audit["DurContr"].sum() / max(1e-12, total_mv2)
    tot_dv012  = (audit["DV01"].fillna(0.0) * audit["NewQty"]).sum()

    print("\n--- Math audit totals ---")
    print(f"Total MV (recalc): {total_mv2:,.2f}")
    print(f"Yield (recalc):    {avg_yield2:.6f}%")
    print(f"Duration (recalc): {avg_dur2:.6f} years")
    print(f"DV01 (recalc):     {tot_dv012:.6f}")

    # consistency checks
    def _ok(a, b, tol):
        return abs(a - b) <= tol

    print("\n--- Consistency check (tolerances allow small rounding) ---")
    print("MV match:", _ok(total_mv, total_mv2, 1e-6))
    print("Yield match:", _ok(avg_yield, avg_yield2, 1e-9))
    print("Dur match:", _ok(avg_dur, avg_dur2, 1e-9))
    print("DV01 match:", _ok(tot_dv01, tot_dv012, 1e-9))

    # quick flags for suspicious rows
    bad_yield = audit.loc[audit["Yield"].abs() > 30, ["CUSIP","Yield","MV","MV_Share_%"]]
    bad_dur   = audit.loc[(audit["Duration"] < 0) | (audit["Duration"] > 20), ["CUSIP","Duration","MV","MV_Share_%"]]
    if len(bad_yield):
        print("\n⚠️  Suspicious yields (>30%):")
        print(bad_yield.to_string(index=False))
    if len(bad_dur):
        print("\n⚠️  Suspicious durations (<0 or >20y):")
        print(bad_dur.to_string(index=False))


    # Rating audit (portfolio-level)
    avg_rating_final = weighted_avg(final, "NewQty", "Rating_Score")

    # Baseline (same calc your objective used)
    baseline_df = pd.concat([kept, slots], ignore_index=True)
    base_avg_rating = weighted_avg(baseline_df, "Quantity", "Rating_Score")
    rating_cap = base_avg_rating + 0.2

    print("\n=== Rating audit ===")
    print(f"Baseline avg rating: {base_avg_rating:.3f}")
    print(f"Final    avg rating: {avg_rating_final:.3f}  (cap ≤ {rating_cap:.3f})")
    print("Rating constraint:", "OK" if avg_rating_final <= rating_cap else "VIOLATION")


    #=======================

    print("\n=== Rebalanced Portfolio (frozen keeps + per-slot decisions) ===")
    view = full.copy()
    view["Change"] = view["NewQty"] - view["OldQty"]
    print(view[["CUSIP","Type","OldQty","NewQty","Change","Price","Yield","Duration","Rating_Score"]].to_string(index=False))

    na_mv = mv_of(final[final["Duration"].isna()], "NewQty").sum()
    print(f"MV with Duration = NaN in final: {na_mv:,.2f}")

    print("\n=== Final Metrics ===")
    print(f"Total MV: {total_mv:,.2f}")
    print(f"Yield:    {avg_yield:.4f}%")
    print(f"Duration: {avg_dur:.4f} years")
    print(f"DV01:     {tot_dv01:.4f}")
    print(f"Budget band: ±{int(budget_band*100)}%")

    for p in selected_params:
        if p == "yield":
            ok = avg_yield >= targets[p] - tolerances[p]
            print(f"Yield constraint (≥ {targets[p]-tolerances[p]:.4f}): {'OK' if ok else 'VIOLATION'}")
        if p == "duration":
            ok = avg_dur <= targets[p] + tolerances[p]
            print(f"Duration constraint (≤ {targets[p]+tolerances[p]:.4f}): {'OK' if ok else 'VIOLATION'}")
        if p == "dv01":
            ok = abs(tot_dv01 - targets[p]) <= tolerances[p]
            print(f"DV01 constraint (±{tolerances[p]} around {targets[p]}): {'OK' if ok else 'VIOLATION'}")

    if not result.success:
        print("\n⚠️ Optimizer did not converge cleanly; result shown is best found.")


# ---------- main ----------
def main():
    # 1) data
    current, ext = load_data("ExchangeDataRequest_new(Corps).csv")
    print("\n[diag] Cleaned-universe top yields:")
    print(ext.sort_values("Yield", ascending=False)[["CUSIP","Yield","Rating_Score"]].head(10).to_string(index=False))
    print("[diag] Max surviving yield:", ext["Yield"].max())

    # Upper bound with rating cap + contrib cap only (greedy, approximate)
    base_avg_rating = weighted_avg(current, "Quantity", "Rating_Score")
    rating_cap = base_avg_rating + 0.2
    cap = MAX_CONTRIB_PCT

    pool = ext[ext["Rating_Score"] <= rating_cap].sort_values("Yield", ascending=False).copy()
    share = 0.0
    yb = 0.0
    for _, r in pool.iterrows():
        take = min(cap, 1.0 - share)
        yb += take * r["Yield"]
        share += take
        if share >= 0.999: break
    print(f"[diag] Greedy upper bound with rating≤cap & {int(cap*100)}% contrib cap: ~{yb:.3f}%")


    # --- original portfolio snapshot (before keeps/shortlists/opt) ---
    mv0 = mv_of(current).sum()
    y0  = weighted_avg(current, "Quantity", "Yield")
    d0  = weighted_avg(current, "Quantity", "Duration")
    print(f"[original] MV: {mv0:,.2f} | Yield: {y0:.4f}% | Duration: {d0:.4f} years")


    # 2) inputs
    selected_params, targets, tolerances, min_qty, lot_size, keep_indices, budget_band, shortlist_k = get_user_inputs(current)

    # 3) slots + shortlists
    kept, slots, shortlists = build_slots_and_shortlists(
        current, ext, selected_params, targets, tolerances, keep_indices, shortlist_k
    )

    # universe quick check (after kept/slots exist)
    baseline_df = pd.concat([kept, slots], ignore_index=True)
    base_avg_rating = weighted_avg(baseline_df, "Quantity", "Rating_Score")
    rating_cap = base_avg_rating + 0.2
    if "yield" in selected_params:
        y_floor = targets["yield"] - tolerances["yield"]
        print("\n=== Universe quick check ===")
        print("Universe size:", len(ext))
        print(f"Yield ≥ {y_floor:.3f}:", int((ext["Yield"] >= y_floor).sum()))
        print(f"Yield ≥ {y_floor:.3f} & rating ≤ {rating_cap:.2f}:",
              int(((ext["Yield"] >= y_floor) & (ext["Rating_Score"] <= rating_cap)).sum()))

    # 4) objective + bounds
    objective, bounds, _, _ = build_objective(
        kept, slots, shortlists, selected_params, targets, tolerances, min_qty, lot_size, budget_band
    )

    if len(slots) == 0:
        print("\nNothing to rebalance: all bonds are frozen.")
        return

    # 5) optimize
    result = run_optimization(objective, bounds)

    # 6) report
    report(current, kept, slots, shortlists, result, lot_size, selected_params, targets, tolerances, budget_band)


if __name__ == "__main__":
    main()
