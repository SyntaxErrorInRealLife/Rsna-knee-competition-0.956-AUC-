import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .config import TARGET_LABELS

# Bilingual (English + Spanish) anatomical & pathology concept rules
# Refined against the 58 gold-labeled studies to capture chondral/cartilage synonyms for OA,
# joint fluid/recess terminology for Effusion, and synovial/Hoffa terminology for Synovitis.
CONCEPT_RULES = {
    "ACL": {
        "anatomy": [
            r"\bacl\b", r"anterior cruciate", r"cruzado anterior", r"\blca\b",
            r"ligamentos? cruzados?", r"cruciate ligaments?", r"ambos cruzados"
        ],
        "positive": [
            r"tear", r"torn", r"ruptur\w*", r"sprain", r"avuls\w*", r"discontin\w*",
            r"non[- ]?visualiz\w*", r"absent", r"graft (?:tear|ruptur\w*|failure|impingement)",
            r"mucoid degenerat\w*", r"high[- ]grade", r"partial[- ]thickness", r"full[- ]thickness",
            r"complete", r"rotura", r"desgarro", r"esguince", r"lesi[oó]n", r"elongaci[oó]n",
            r"cambios? mucoide\w*", r"degeneraci[oó]n mucoide", r"plastia", r"reconstrucci[oó]n",
            r"injerto", r"graft", r"reconstructed", r"post[- ]?surgical", r"postquir[uú]rgic\w*"
        ],
    },
    "MCL": {
        "anatomy": [
            r"\bmcl\b", r"medial collateral", r"tibial collateral", r"superficial mcl", r"deep mcl",
            r"colateral medial", r"colateral tibial", r"\blcm\b",
            r"ligamentos? colaterales?", r"collateral ligaments?"
        ],
        "positive": [
            r"tear", r"torn", r"sprain", r"ruptur\w*", r"edema", r"thickening", r"thickened",
            r"grade [123i]+", r"grado [123i]+", r"high[- ]grade", r"low[- ]grade", r"partial",
            r"complete", r"meniscofemoral", r"meniscotibial", r"rotura", r"desgarro", r"esguince",
            r"distensi[oó]n", r"engrosamiento", r"lesi[oó]n", r"periligament\w*"
        ],
    },
    "Medial Meniscus": {
        "anatomy": [
            r"medial menisc\w*", r"menisco medial", r"menisco interno",
            r"both menisci", r"ambos meniscos", r"menisci\b", r"meniscos\b"
        ],
        "positive": [
            r"tear", r"torn", r"macerat\w*", r"extrus\w*", r"extruded", r"root", r"ra[ií]z",
            r"radial", r"horizontal", r"complex", r"vertical", r"longitudinal", r"flap",
            r"bucket[- ]?handle", r"asa de balde", r"cleavage", r"fraying", r"frayed",
            r"blunting", r"truncated", r"truncation", r"meniscectom\w*", r"meniscectom[ií]a",
            r"post[- ]?surgical", r"amputaci[oó]n", r"rotura", r"desgarro", r"luxaci[oó]n",
            r"subluxaci[oó]n", r"defecto", r"degenerat\w* signal", r"intrameniscal",
            r"grade 3", r"grado iii", r"grado 3", r"parameniscal cyst", r"quiste parameniscal"
        ],
    },
    "Lateral Meniscus": {
        "anatomy": [
            r"lateral menisc\w*", r"menisco lateral", r"menisco externo",
            r"both menisci", r"ambos meniscos", r"menisci\b", r"meniscos\b"
        ],
        "positive": [
            r"tear", r"torn", r"macerat\w*", r"extrus\w*", r"extruded", r"root", r"ra[ií]z",
            r"radial", r"horizontal", r"complex", r"vertical", r"longitudinal", r"flap",
            r"bucket[- ]?handle", r"asa de balde", r"cleavage", r"fraying", r"frayed",
            r"blunting", r"truncated", r"truncation", r"meniscectom\w*", r"meniscectom[ií]a",
            r"post[- ]?surgical", r"amputaci[oó]n", r"rotura", r"desgarro", r"luxaci[oó]n",
            r"subluxaci[oó]n", r"defecto", r"discoid", r"discoide", r"grade 3", r"grado iii",
            r"grado 3", r"parameniscal cyst", r"quiste parameniscal"
        ],
    },
    "Medial OA": {
        "anatomy": [
            r"medial (?:compartment|tibiofemoral|femoral condyle|tibial plateau|joint space|chondral|cartilage|osteoarthrit\w*|arthros\w*)",
            r"(?:medial|interno)[^.;]{0,50}(?:condyle|plateau|compartment|cartilage|chondral|chondromalacia|chondrosis|osteoarthrit\w*|arthros\w*|osteophyte\w*|narrowing|artrosis|condromalacia|cart[ií]lago|condral|osteofito\w*|pinzamiento|c[oó]ndilo|platillo)",
            r"(?:cartilage|chondral|chondromalacia|chondrosis|osteoarthrit\w*|arthros\w*|osteophyte\w*|narrowing|artrosis|condromalacia|cart[ií]lago|condral|osteofito\w*|pinzamiento|condropat[ií]a)[^.;]{0,50}(?:medial|interno|femorotibial medial|compartimento medial|c[oó]ndilo femoral medial|platillo tibial medial)",
            r"tricompartmental", r"tricompartimental", r"bicompartmental", r"osteoarthrit\w*", r"gonartrosis"
        ],
        "positive": [
            r"osteoarthrit\w*", r"arthros\w*", r"artrosis", r"gonartrosis", r"osteoartrosis",
            r"chondromalacia", r"condromalacia", r"chondrosis", r"condropat[ií]a",
            r"cartilage (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|denud\w*)",
            r"chondral (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|ulcer\w*|flap|lesion)",
            r"(?:full|partial)[- ]thickness (?:cartilage|chondral)",
            r"grade [234iiiv]+ (?:chondromalacia|chondrosis|chondral|cartilage)",
            r"grado [234iiiv]+", r"adelgazamiento del cart[ií]lago", r"lesi[oó]n (?:osteocondral|condral)",
            r"defecto condral", r"fisura\w* condral\w*", r"osteophyte\w*", r"osteofito\w*",
            r"joint space narrowing", r"pinzamiento", r"subchondral (?:cyst\w*|sclerosis|edema)"
        ],
    },
    "Lateral OA": {
        "anatomy": [
            r"lateral (?:compartment|tibiofemoral|femoral condyle|tibial plateau|joint space|chondral|cartilage|osteoarthrit\w*|arthros\w*)",
            r"(?:lateral|externo)[^.;]{0,50}(?:condyle|plateau|compartment|cartilage|chondral|chondromalacia|chondrosis|osteoarthrit\w*|arthros\w*|osteophyte\w*|narrowing|artrosis|condromalacia|cart[ií]lago|condral|osteofito\w*|pinzamiento|c[oó]ndilo|platillo)",
            r"(?:cartilage|chondral|chondromalacia|chondrosis|osteoarthrit\w*|arthros\w*|osteophyte\w*|narrowing|artrosis|condromalacia|cart[ií]lago|condral|osteofito\w*|pinzamiento|condropat[ií]a)[^.;]{0,50}(?:lateral|externo|femorotibial lateral|compartimento lateral|c[oó]ndilo femoral lateral|platillo tibial lateral)",
            r"tricompartmental", r"tricompartimental"
        ],
        "positive": [
            r"osteoarthrit\w*", r"arthros\w*", r"artrosis", r"gonartrosis", r"osteoartrosis",
            r"chondromalacia", r"condromalacia", r"chondrosis", r"condropat[ií]a",
            r"cartilage (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|denud\w*)",
            r"chondral (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|ulcer\w*|flap|lesion)",
            r"(?:full|partial)[- ]thickness (?:cartilage|chondral)",
            r"grade [234iiiv]+ (?:chondromalacia|chondrosis|chondral|cartilage)",
            r"grado [234iiiv]+", r"adelgazamiento del cart[ií]lago", r"lesi[oó]n (?:osteocondral|condral)",
            r"defecto condral", r"fisura\w* condral\w*", r"osteophyte\w*", r"osteofito\w*",
            r"joint space narrowing", r"pinzamiento", r"subchondral (?:cyst\w*|sclerosis|edema)"
        ],
    },
    "PF OA": {
        "anatomy": [
            r"patellofemoral", r"patellar", r"patella\b", r"trochlea\w*", r"retropatellar",
            r"femoropatelar", r"rotul\w*", r"r[oó]tula", r"tr[oó]clea\w*", r"patela\w*",
            r"tricompartmental", r"tricompartimental"
        ],
        "positive": [
            r"osteoarthrit\w*", r"arthros\w*", r"artrosis", r"osteoartrosis",
            r"chondromalacia", r"condromalacia", r"chondrosis", r"condropat[ií]a",
            r"cartilage (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|denud\w*|irregular\w*)",
            r"chondral (?:loss|defect|wear|thinning|fissur\w*|delaminat\w*|erosion|ulcer\w*|flap|lesion|irregular\w*)",
            r"(?:full|partial)[- ]thickness", r"fissur\w*", r"fisura\w*", r"delaminat\w*",
            r"grade [1234iv]+", r"grado [1234iv]+", r"adelgazamiento", r"defecto condral",
            r"lesi[oó]n condral", r"osteophyte\w*", r"osteofito\w*", r"subchondral",
            r"wear", r"thinning", r"loss", r"defect"
        ],
    },
    "Effusion": {
        "anatomy": [
            r"effusion", r"derrame", r"joint fluid", r"l[ií]quido (?:articular|intraarticular|sinovial|en receso|suprapatelar)",
            r"suprapatellar (?:fluid|recess|bursa)", r"hidrartrosis", r"intra[- ]?articular fluid"
        ],
        "positive": [
            r"small", r"mild", r"moderate", r"large", r"marked", r"significant",
            r"increased", r"present", r"seen", r"noted", r"demonstrated", r"there is",
            r"leve", r"moderad[oa]", r"sever[oa]", r"importante", r"abundante", r"discreto",
            r"escaso derrame", r"incremento", r"aumento", r"presencia de derrame", r"derrame articular"
        ],
    },
    "Synovitis": {
        "anatomy": [
            r"synovit\w*", r"sinovitis", r"synovial", r"sinovial",
            r"hoffa'?s? (?:fat pad|pad|disease|synovit\w*)", r"hoffitis", r"grasa de hoffa",
            r"infrapatellar fat pad", r"suprapatellar fat pad", r"plic[ae]"
        ],
        "positive": [
            r"synovit\w*", r"sinovitis", r"thickening", r"thickened", r"proliferat\w*",
            r"hypertroph\w*", r"inflammation", r"inflammatory", r"hyperintense", r"edema",
            r"impingement", r"reaction", r"frond\w*", r"pannus",
            r"engrosamiento", r"proliferaci[oó]n", r"hipertrofia", r"inflamaci[oó]n",
            r"cambios inflamatorios", r"edema en (?:la )?grasa de hoffa", r"hoffitis"
        ],
    },
    "Baker's": {
        "anatomy": [
            r"baker'?s?\b", r"popliteal cyst\w*", r"quiste de baker", r"quiste popl[ií]teo",
            r"gastrocnemius[- ]semimembranosus (?:bursa|cyst|fluid)"
        ],
        "positive": [
            r"baker'?s?", r"popliteal cyst\w*", r"quiste de baker", r"quiste popl[ií]teo",
            r"cyst", r"quiste", r"fluid collection", r"distension", r"distensi[oó]n", r"ruptured"
        ],
    },
    "Contusion": {
        "anatomy": [
            r"contusion\w*", r"contusi[oó]n", r"bone bruise\w*", r"marrow edema",
            r"edema [oó]seo", r"edema de (?:la )?m[eé]dula [oó]sea", r"edema trabecular",
            r"microtrabecular", r"subchondral edema", r"bone marrow (?:signal|edema)"
        ],
        "positive": [
            r"contusion\w*", r"contusi[oó]n", r"bruise\w*", r"edema", r"microtrabecular",
            r"hyperintens\w*", r"hiperintens\w*", r"post[- ]?traumatic"
        ],
    },
    "Fracture": {
        "anatomy": [
            r"fractur[ea]\w*", r"cortical (?:break|step[- ]?off|depression|impaction)",
            r"impaction injury", r"segond", r"osteochondral fracture", r"hundimiento cortical",
            r"trazo de fractura"
        ],
        "positive": [
            r"fractur[ea]\w*", r"cortical (?:break|depression|impaction|step)",
            r"impaction", r"avulsion fracture", r"segond", r"hundimiento", r"trazo"
        ],
    },
}

STRICT_NEGATION = re.compile(
    r"\b(?:no|not|without|neither|nor|never|none|"
    r"intact|preserved|unremarkable|normal|normals|within normal limits|negative for|free of|rule out|excluded|absent|"
    r"sin|no hay|no se observa\w*|no se identifica\w*|no se evidencia\w*|ausencia de|"
    r"conservad[oa]s?|[ií]ntegr[oa]s?|dentro de l[ií]mites normales|morfolog[ií]a y se[nñ]al normal\w*)\b",
    re.IGNORECASE,
)

UNCERTAINTY_CUE = re.compile(
    r"\b(?:possible|possibly|questionable|equivocal|cannot exclude|cannot rule out|suspected|indeterminate|"
    r"posible|dudoso|no se puede descartar)\b",
    re.IGNORECASE,
)


def parse_clinical_report(
    report_text: str, labels: List[str] = TARGET_LABELS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses an English or Spanish knee MRI radiology report into:
      - target_vec: float32 array of shape (12,) with values in {1.0, 0.0, np.nan}
      - weight_vec: float32 confidence weight in [0.0, 1.0]
    Omitted findings in brief reports remain NaN (masked). Uncertain findings remain NaN.
    """
    if not isinstance(report_text, str) or len(report_text.strip()) < 10:
        return np.full(len(labels), np.nan, dtype=np.float32), np.zeros(len(labels), dtype=np.float32)

    clean_text = report_text.replace("\r", "\n")
    clauses = [c.strip() for c in re.split(r"[.;\n]+", clean_text) if len(c.strip()) > 2]

    preds = np.full(len(labels), np.nan, dtype=np.float32)
    confs = np.zeros(len(labels), dtype=np.float32)

    is_full_structured_report = len(clean_text) >= 180

    for idx, label in enumerate(labels):
        rule = CONCEPT_RULES[label]
        anat_pats = [re.compile(p, re.IGNORECASE) for p in rule["anatomy"]]
        pos_pats = [re.compile(p, re.IGNORECASE) for p in rule["positive"]]

        matched_clauses = [c for c in clauses if any(p.search(c) for p in anat_pats)]

        if not matched_clauses:
            # Do NOT force 0.0 on short/incomplete reports; only assign weak negative if comprehensive report
            if is_full_structured_report:
                preds[idx] = 0.0
                confs[idx] = 0.55
            continue

        pos_score = 0.0
        neg_score = 0.0
        unc_flag = False

        for clause in matched_clauses:
            c_low = clause.lower()

            # Check if clause has a contrast conjunction ("ACL is intact, however medial meniscus is torn")
            sub_parts = re.split(r"\b(?:however|but|although|whereas|except|pero|aunque|mientras que|,)\b", c_low)
            relevant_parts = [sp for sp in sub_parts if any(p.search(sp) for p in anat_pats)]
            if not relevant_parts:
                relevant_parts = [c_low]

            for sp in relevant_parts:
                has_neg = bool(STRICT_NEGATION.search(sp))
                has_pos = any(p.search(sp) for p in pos_pats)
                has_unc = bool(UNCERTAINTY_CUE.search(sp))

                # Special physiological fluid rule for Effusion
                if label == "Effusion" and re.search(r"\b(?:trace|physiologic\w*|minimal|normal joint fluid|fisiol[oó]gic\w*)\b", sp):
                    if not re.search(r"\b(?:small[- ]to[- ]moderate|moderate|large|moderad\w*|sever\w*|abundante)\b", sp):
                        has_neg = True
                        has_pos = False

                # Check if negation explicitly negates the positive term ("no tear", "sin signos de rotura", "no fracture")
                explicit_neg_of_pos = bool(
                    re.search(
                        r"\b(?:no|without|not|neither|nor|negative for|free of|sin|no hay|no se observa\w*|ausencia de)\b"
                        r"[^.;]{0,40}\b(?:tear|torn|ruptur\w*|sprain|fractur\w*|contusion|edema|effusion|synovit\w*|"
                        r"cyst|osteoarthrit\w*|chondromalacia|defect|rotura|desgarro|lesi[oó]n|alteraci[oó]n|quiste|derrame)\b",
                        sp,
                    )
                )

                if explicit_neg_of_pos:
                    neg_score = max(neg_score, 0.90)
                elif has_unc and not (has_pos and not has_neg):
                    unc_flag = True
                elif has_pos and not has_neg:
                    pos_score = max(pos_score, 0.90)
                elif has_pos and has_neg:
                    # Both appear, e.g., "intact ACL with mucoid degeneration" -> positive abnormality
                    if re.search(r"\b(?:with|showing|demonstrating|con|presenta)\b", sp):
                        pos_score = max(pos_score, 0.85)
                    else:
                        neg_score = max(neg_score, 0.85)
                elif has_neg:
                    neg_score = max(neg_score, 0.90)

        if pos_score > 0:
            preds[idx] = 1.0
            confs[idx] = pos_score
        elif unc_flag:
            preds[idx] = np.nan
            confs[idx] = 0.0
        elif neg_score > 0:
            preds[idx] = 0.0
            confs[idx] = neg_score

    return preds, confs


def audit_report_labeler_on_gold(train_df: pd.DataFrame, labels: List[str] = TARGET_LABELS) -> dict:
    gold_mask = train_df[labels].notna().all(axis=1)
    gold_df = train_df[gold_mask].copy()
    parsed = [parse_clinical_report(r, labels) for r in gold_df["Report"]]
    pred_mat = np.vstack([p[0] for p in parsed])
    true_mat = gold_df[labels].values.astype(np.float32)

    metrics = {}
    f1s, accs = [], []
    for i, col in enumerate(labels):
        valid = ~np.isnan(pred_mat[:, i])
        yt = true_mat[valid, i]
        yp = pred_mat[valid, i]
        acc = float(accuracy_score(yt, yp)) if len(yt) > 0 else 0.0
        prec = float(precision_score(yt, yp, zero_division=0)) if len(yt) > 0 else 0.0
        rec = float(recall_score(yt, yp, zero_division=0)) if len(yt) > 0 else 0.0
        f1 = float(f1_score(yt, yp, zero_division=0)) if len(yt) > 0 else 0.0
        metrics[col] = {"cov": float(valid.mean()), "acc": acc, "prec": prec, "rec": rec, "f1": f1}
        f1s.append(f1)
        accs.append(acc)
    metrics["mean_acc"] = float(np.mean(accs))
    metrics["mean_f1"] = float(np.mean(f1s))
    return metrics
