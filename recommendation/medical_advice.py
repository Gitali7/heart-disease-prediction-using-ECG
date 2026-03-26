"""
recommendation/medical_advice.py
==================================
Rule-based medical guidance engine.

Why rule-based instead of another ML model?
  Medical recommendations must be:
  1. Explainable — a doctor must understand why a suggestion was made
  2. Auditable — easily updated when clinical guidelines change
  3. Safe — no hallucination risk (unlike LLM-based guidance)

  A lookup table of clinical guidelines is transparent and reliable.
  This maps predicted conditions → evidence-based recommendations
  sourced from ACC/AHA (American College of Cardiology / American Heart
  Association) guidelines.

Why include a disclaimer?
  AI diagnostic tools in healthcare require explicit disclaimers per
  FDA guidelines and medical ethics standards.
"""

import copy
from dataclasses import dataclass, field
from typing import List

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class MedicalReport:
    condition: str
    risk_level: str
    risk_color: str
    urgency: str
    symptoms_to_watch: List[str]
    lifestyle_advice: List[str]
    medical_actions: List[str]
    dietary_advice: List[str]
    precautions: List[str]
    follow_up: str
    disclaimer: str = (
        "⚠️ This AI analysis is for educational purposes only. "
        "It does NOT replace professional medical diagnosis. "
        "Please consult a qualified cardiologist or physician."
    )


# ─── Condition Database ───────────────────────────────────────────────────────
# Sources: ACC/AHA Heart Failure Guidelines 2022, ESC Guidelines 2021

CONDITION_ADVICE = {
    "Cardiomegaly": MedicalReport(
        condition="Cardiomegaly (Enlarged Heart)",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Consult a cardiologist within 1–2 weeks",
        symptoms_to_watch=[
            "Shortness of breath, especially when lying flat",
            "Swelling in legs, ankles, or feet",
            "Fatigue and reduced exercise tolerance",
            "Irregular heartbeat or palpitations",
            "Dizziness or fainting spells",
        ],
        lifestyle_advice=[
            "Limit sodium intake to <2,000mg/day",
            "Maintain a heart-healthy weight (BMI 18.5–24.9)",
            "Engage in light aerobic exercise (walking 30 min/day) as tolerated",
            "Quit smoking — nicotine worsens cardiac muscle stress",
            "Limit alcohol to ≤1 drink/day",
            "Manage stress with meditation or yoga",
        ],
        medical_actions=[
            "Schedule echocardiogram (ultrasound of heart)",
            "Get ECG/EKG to assess electrical activity",
            "Blood tests: BNP, troponin, CBC, metabolic panel",
            "Discuss medication options with cardiologist (ACE inhibitors, beta-blockers)",
            "Monitor blood pressure daily (target: <130/80 mmHg)",
        ],
        dietary_advice=[
            "Follow DASH diet: fruits, vegetables, whole grains, low-fat dairy",
            "Reduce saturated fats and trans fats",
            "Increase potassium-rich foods (bananas, sweet potatoes)",
            "Avoid processed foods and fast food",
            "Limit caffeine intake",
        ],
        precautions=[
            "Do not ignore chest pain — call emergency services immediately",
            "Avoid strenuous physical activity until cleared by doctor",
            "Take prescribed medications consistently",
            "Do not stop medications without physician approval",
            "Avoid NSAIDs (ibuprofen) — they worsen heart function",
        ],
        follow_up="Return for re-evaluation in 3–6 months or sooner if symptoms worsen",
    ),

    "Congestive Heart Failure": MedicalReport(
        condition="Congestive Heart Failure (CHF)",
        risk_level="CRITICAL",
        risk_color="#8B0000",
        urgency="Seek medical evaluation URGENTLY — same day or emergency department",
        symptoms_to_watch=[
            "Sudden weight gain (>2 kg in 2 days) — indicates fluid retention",
            "Severe shortness of breath at rest",
            "Pink or frothy sputum when coughing",
            "Confusion or difficulty concentrating",
            "Cold, clammy skin",
        ],
        lifestyle_advice=[
            "Strict fluid restriction: typically <1.5–2 liters/day",
            "Daily weight monitoring — weigh at same time each morning",
            "Elevate legs when sitting to reduce edema",
            "Sodium restriction: <1,500mg/day (stricter than cardiomegaly)",
            "Small, frequent meals (large meals increase cardiac workload)",
        ],
        medical_actions=[
            "Emergency echocardiogram to assess ejection fraction",
            "Chest X-ray to monitor fluid levels",
            "IV diuretics may be required for fluid overload",
            "Medication review: ACE inhibitors, ARBs, beta-blockers, diuretics, spironolactone",
            "Consider cardiac resynchronization therapy if ejection fraction <35%",
        ],
        dietary_advice=[
            "Strict sodium restriction (<1,500mg/day)",
            "Fluid restriction as directed by physician",
            "No alcohol — it directly depresses cardiac function",
            "Heart-healthy fats only (olive oil, avocado, fatty fish)",
        ],
        precautions=[
            "Any sudden worsening = call emergency services",
            "Have emergency contacts ready at all times",
            "Wear medical alert identification",
            "Do not fly or travel long distances without physician clearance",
        ],
        follow_up="Bi-weekly follow-up during stabilization, monthly when stable",
    ),

    "Coronary Artery Disease": MedicalReport(
        condition="Coronary Artery Disease (CAD)",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Cardiology referral within 1 week",
        symptoms_to_watch=[
            "Chest pressure or tightness (angina) during exertion",
            "Pain radiating to left arm, jaw, or back",
            "Shortness of breath with activity",
            "Nausea or sweating with chest discomfort",
            "Call 911 if chest pain lasts >15 minutes — possible heart attack",
        ],
        lifestyle_advice=[
            "Cardiac rehabilitation program is strongly recommended",
            "Aerobic exercise 150 min/week (walking, cycling, swimming)",
            "Complete smoking cessation — smoking accelerates plaque buildup",
            "Achieve and maintain healthy weight",
            "Manage diabetes strictly if applicable",
        ],
        medical_actions=[
            "Coronary CT Angiography (CCTA) or stress test",
            "Lipid panel — target LDL <70 mg/dL for high-risk patients",
            "Cardiac catheterization may be required",
            "Medication: aspirin, statins, beta-blockers, nitrates",
            "Monitor for silent ischemia with regular ECG",
        ],
        dietary_advice=[
            "Mediterranean diet — proven to reduce CAD events by 30%",
            "Reduce red meat to <2 servings/week",
            "Increase omega-3 rich fish (salmon, sardines) — 2 servings/week",
            "Eliminate trans fats completely",
            "Increase soluble fiber (oats, legumes, apples)",
        ],
        precautions=[
            "Nitroglycerin should be on hand if prescribed",
            "Know the signs of a heart attack — act immediately",
            "Avoid extreme cold/heat which stress the heart",
            "Limit intense emotional stress",
        ],
        follow_up="3-month follow-up after treatment initiation, 6-monthly when stable",
    ),

    "Cardiomyopathy": MedicalReport(
        condition="Cardiomyopathy (Weakened Heart Muscle)",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Cardiology evaluation within 1–2 weeks",
        symptoms_to_watch=[
            "Progressive breathlessness and fatigue",
            "Heart palpitations or racing heartbeat",
            "Swelling in legs and abdomen",
            "Chest pain or pressure",
            "Dizziness or fainting (syncope)",
        ],
        lifestyle_advice=[
            "Complete alcohol cessation (alcoholic cardiomyopathy can reverse with abstinence)",
            "Avoid competitive sports or strenuous exercise without cardiologist approval",
            "Manage underlying conditions (hypertension, diabetes, thyroid disease)",
            "Genetic counseling recommended for family members",
            "Moderate low-intensity exercise only",
        ],
        medical_actions=[
            "Echocardiogram to measure ejection fraction",
            "Cardiac MRI for detailed muscle tissue assessment",
            "Genetic testing if familial cardiomyopathy suspected",
            "Medications: ACE inhibitors, beta-blockers, diuretics, anticoagulants",
            "ICD (implantable cardioverter-defibrillator) if high arrhythmia risk",
        ],
        dietary_advice=[
            "Heart-healthy diet: low sodium, low saturated fat",
            "No alcohol under any circumstances",
            "Adequate protein intake to support muscle health",
            "Antioxidant-rich foods (berries, leafy greens)",
        ],
        precautions=[
            "Report any new palpitations or fainting immediately",
            "Avoid medications that worsen cardiomyopathy (certain antibiotics, chemotherapy drugs)",
            "Carry list of all medications when traveling",
            "Emergency contacts should be aware of condition",
        ],
        follow_up="Quarterly echocardiograms to track ejection fraction changes",
    ),

    "Pulmonary Edema": MedicalReport(
        condition="Pulmonary Edema (Fluid in Lungs)",
        risk_level="CRITICAL",
        risk_color="#8B0000",
        urgency="EMERGENCY — Call 999/112/911 immediately",
        symptoms_to_watch=[
            "Extreme difficulty breathing (cannot lie flat)",
            "Coughing up pink, frothy mucus",
            "Severe anxiety and feeling of suffocation",
            "Gurgling or wheezing sounds when breathing",
            "Blue-tinged lips or fingertips (cyanosis)",
        ],
        lifestyle_advice=[
            "Complete bed rest during acute phase",
            "Sit upright (legs dangling) to reduce preload",
            "After recovery: strict sodium restriction, fluid monitoring",
            "Cardiac rehabilitation after acute treatment",
        ],
        medical_actions=[
            "EMERGENCY: Supplemental oxygen / mechanical ventilation",
            "IV diuretics (furosemide) to rapidly remove fluid",
            "Morphine to reduce respiratory distress",
            "Treat underlying cause (heart failure, hypertension crisis)",
            "CPAP/BiPAP non-invasive ventilation if severe",
        ],
        dietary_advice=[
            "NPO (nothing by mouth) during acute phase",
            "After stabilization: strict fluid and sodium restriction",
        ],
        precautions=[
            "This is a life-threatening emergency — do not delay care",
            "Do not lie flat — keep head elevated at 45°",
            "Have emergency services on speed dial",
        ],
        follow_up="ICU monitoring during acute phase, intensive outpatient follow-up after discharge",
    ),

    "No Finding": MedicalReport(
        condition="No Significant Cardiac Abnormality Detected",
        risk_level="LOW",
        risk_color="#00C853",
        urgency="Routine medical check-up recommended",
        symptoms_to_watch=[
            "Chest pain or tightness (any new onset)",
            "Unexplained breathlessness",
            "Irregular heartbeat",
            "Swelling in extremities",
        ],
        lifestyle_advice=[
            "Continue regular aerobic exercise (150 min/week)",
            "Maintain healthy body weight",
            "No smoking",
            "Limit alcohol consumption",
            "Manage stress effectively",
        ],
        medical_actions=[
            "Annual physical examination",
            "Regular blood pressure monitoring",
            "Cholesterol check every 4–6 years (more often if at risk)",
            "Maintain diabetes control if applicable",
        ],
        dietary_advice=[
            "Heart-healthy balanced diet",
            "Limit processed foods and added sugars",
            "Adequate hydration (2–3 liters/day)",
            "Regular meals to maintain stable blood sugar",
        ],
        precautions=[
            "Know your family history of heart disease",
            "Seek medical attention promptly for any new cardiac symptoms",
            "Keep up with routine health screenings",
        ],
        follow_up="Annual routine check-up with GP",
    ),

    "Atrial Fibrillation": MedicalReport(
        condition="Atrial Fibrillation (Irregular Heartbeat)",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Cardiology evaluation within 1 week",
        symptoms_to_watch=["Heart palpitations", "Weakness or fatigue", "Shortness of breath"],
        lifestyle_advice=["Quit smoking", "Limit alcohol and caffeine", "Manage stress"],
        medical_actions=["ECG/EKG confirmation", "Discuss anticoagulants", "Consider rate/rhythm control meds"],
        dietary_advice=["Heart-healthy diet", "Consistent Vitamin K if on warfarin"],
        precautions=["High risk for stroke — adhere to blood thinners if prescribed"],
        follow_up="Regular monitoring by cardiologist",
    ),

    "Myocardial Infarction": MedicalReport(
        condition="Myocardial Infarction (Heart Attack indicators)",
        risk_level="CRITICAL",
        risk_color="#8B0000",
        urgency="EMERGENCY — Call 911 immediately if acute symptoms present",
        symptoms_to_watch=["Chest pain/pressure", "Shortness of breath", "Pain in arm/back/jaw"],
        lifestyle_advice=["Cardiac rehab", "Strict smoking cessation"],
        medical_actions=["Emergency evaluation", "Troponin blood tests", "Angiogram"],
        dietary_advice=["Strict low-sodium, low-fat diet"],
        precautions=["Do not delay seeking emergency care if symptomatic"],
        follow_up="Intensive cardiology follow-up",
    ),

    "Normal Sinus Rhythm": MedicalReport(
        condition="Normal Sinus Rhythm",
        risk_level="LOW",
        risk_color="#00C853",
        urgency="Routine medical check-up recommended",
        symptoms_to_watch=["Any new palpitations or chest pain"],
        lifestyle_advice=["Maintain a healthy lifestyle with regular exercise"],
        medical_actions=["Keep scheduled wellness visits"],
        dietary_advice=["Balanced diet"],
        precautions=["None at this time"],
        follow_up="Routine follow-up",
    ),

    "Attention Required Document": MedicalReport(
        condition="Attention Required Medical Record",
        risk_level="MODERATE",
        risk_color="#FF9800",
        urgency="Review with a medical professional soon",
        symptoms_to_watch=["Any symptoms related to the record context"],
        lifestyle_advice=["Follow specific advice from your care team"],
        medical_actions=["Schedule a consultation to discuss findings"],
        dietary_advice=["As recommended by your doctor"],
        precautions=["Keep original records safe for your appointment"],
        follow_up="Schedule appointment within 2-4 weeks",
    ),
    
    "Routine Medical Record": MedicalReport(
        condition="Routine Medical Record",
        risk_level="LOW",
        risk_color="#00C853",
        urgency="No immediate action needed based on AI scan",
        symptoms_to_watch=["None"],
        lifestyle_advice=["Maintain healthy habits"],
        medical_actions=["File record for future reference"],
        dietary_advice=["No specific restrictions"],
        precautions=["None"],
        follow_up="Routine check-ups",
    ),

    "Pulmonary Embolism": MedicalReport(
        condition="Pulmonary Embolism (Blood Clot in Lung)",
        risk_level="CRITICAL",
        risk_color="#8B0000",
        urgency="EMERGENCY — Seek immediate medical attention",
        symptoms_to_watch=["Sudden shortness of breath", "Chest pain that worsens with deep breath", "Coughing up blood"],
        lifestyle_advice=["Avoid prolonged immobility", "Use compression stockings if recommended"],
        medical_actions=["CT Pulmonary Angiography confirmation", "Anticoagulation therapy (blood thinners)"],
        dietary_advice=["Maintain consistent Vitamin K intake if on certain blood thinners"],
        precautions=["Do not ignore sudden breathing difficulties"],
        follow_up="Close monitoring by pulmonologist or hematologist",
    ),

    "Lung Nodule": MedicalReport(
        condition="Lung Nodule",
        risk_level="MODERATE",
        risk_color="#FF9800",
        urgency="Schedule a follow-up appointment within 1-2 weeks",
        symptoms_to_watch=["Persistent cough", "Unexplained weight loss", "Chest pain"],
        lifestyle_advice=["Quit smoking immediately", "Avoid exposure to environmental toxins"],
        medical_actions=["Compare with previous imaging", "Consider PET scan or biopsy if suspicious"],
        dietary_advice=["Antioxidant-rich diet (fruits and vegetables)"],
        precautions=["Adhere strictly to follow-up imaging schedules"],
        follow_up="Follow-up CT scan in 3-6 months depending on nodule size",
    ),

    "Aortic Aneurysm": MedicalReport(
        condition="Aortic Aneurysm",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Consult a vascular specialist promptly",
        symptoms_to_watch=["Sudden, severe pain in chest or upper back", "Trouble breathing/swallowing"],
        lifestyle_advice=["Strict blood pressure control", "Avoid heavy lifting or straining"],
        medical_actions=["Vascular surgery consultation", "Regular imaging to monitor size"],
        dietary_advice=["Low-sodium, heart-healthy diet to manage blood pressure"],
        precautions=["Seek emergency care for sudden tearing pain in back or chest"],
        follow_up="Routine imaging every 6-12 months",
    ),

    "Valve Regurgitation": MedicalReport(
        condition="Valve Regurgitation (Leaky Heart Valve)",
        risk_level="MODERATE",
        risk_color="#FF9800",
        urgency="Cardiology evaluation within 2-4 weeks",
        symptoms_to_watch=["Shortness of breath with activity", "Fatigue", "Heart murmur or palpitations"],
        lifestyle_advice=["Maintain healthy weight", "Moderate exercise as tolerated"],
        medical_actions=["Echocardiogram to quantify severity", "Monitor for signs of heart failure"],
        dietary_advice=["Low-sodium diet to prevent fluid retention"],
        precautions=["Take antibiotics before dental procedures if recommended by doctor"],
        follow_up="Annual or semi-annual cardiology review",
    ),

    "Reduced Ejection Fraction": MedicalReport(
        condition="Reduced Ejection Fraction (Heart Failure Risk)",
        risk_level="HIGH",
        risk_color="#FF4B4B",
        urgency="Cardiology consultation within 1-2 weeks",
        symptoms_to_watch=["Swelling in legs/ankles", "Rapid heartbeat", "Waking up breathless at night"],
        lifestyle_advice=["Daily weight checks", "Fluid restriction if prescribed"],
        medical_actions=["Optimization of heart failure medications (e.g., Beta-blockers, ACEi/ARBs)"],
        dietary_advice=["Strict sodium limits (<1500mg/day)"],
        precautions=["Monitor for sudden weight gain (>2 lbs in a day)"],
        follow_up="Regular monitoring of cardiac function",
    ),

    "Normal CT Scan": MedicalReport(
        condition="Normal CT Scan",
        risk_level="LOW",
        risk_color="#00C853",
        urgency="Routine medical check-up recommended",
        symptoms_to_watch=["Any new chest discomfort"],
        lifestyle_advice=["Maintain healthy habits"],
        medical_actions=["Keep routine wellness visits"],
        dietary_advice=["Balanced diet"],
        precautions=["None"],
        follow_up="Routine check-ups",
    ),

    "Normal Echocardiogram": MedicalReport(
        condition="Normal Echocardiogram",
        risk_level="LOW",
        risk_color="#00C853",
        urgency="Routine medical check-up recommended",
        symptoms_to_watch=["Any new palpitations or breathlessness"],
        lifestyle_advice=["Maintain a heart-healthy lifestyle"],
        medical_actions=["Routine wellness care"],
        dietary_advice=["Balanced, low-cholesterol diet"],
        precautions=["None"],
        follow_up="Routine check-ups",
    ),
}


# ─── Main API ─────────────────────────────────────────────────────────────────

def get_risk_level(confidence: float, is_positive: bool) -> dict:
    """
    Map prediction confidence to a clinical risk level.

    Risk thresholds are derived from clinical decision support literature:
    - <60% confidence → Low risk (borderline, monitor)
    - 60-75% → Moderate risk (follow up soon)
    - 75-90% → High risk (prompt cardiology referral)
    - >90% → Critical (urgent evaluation)

    For negative predictions, risk is uniformly Low regardless of confidence.

    Returns:
        dict with level, color, description
    """
    if not is_positive:
        return {"level": "LOW", "color": "#00C853",
                "description": "No significant cardiac abnormality detected"}

    if confidence >= 0.90:
        return {"level": "CRITICAL", "color": "#8B0000",
                "description": "Very high probability of significant cardiac condition — urgent evaluation required"}
    elif confidence >= 0.75:
        return {"level": "HIGH", "color": "#FF4B4B",
                "description": "High probability of cardiac abnormality — prompt cardiology referral recommended"}
    elif confidence >= 0.60:
        return {"level": "MODERATE", "color": "#FF9800",
                "description": "Moderate probability — follow-up imaging and clinical evaluation advised"}
    else:
        return {"level": "LOW-MODERATE", "color": "#FFC107",
                "description": "Borderline finding — clinical correlation and repeat imaging recommended"}


def get_condition_advice(condition_name: str) -> MedicalReport:
    """
    Retrieve the medical advice report for a given condition.

    Falls back to 'No Finding' report if condition not in database.

    Args:
        condition_name: One of the CONDITION_LABELS
    Returns:
        MedicalReport dataclass
    """
    # Try exact match, then partial match
    if condition_name in CONDITION_ADVICE:
        return CONDITION_ADVICE[condition_name]

    for key in CONDITION_ADVICE:
        if key.lower() in condition_name.lower() or condition_name.lower() in key.lower():
            return CONDITION_ADVICE[key]

    return CONDITION_ADVICE["No Finding"]


def select_likely_conditions(is_positive: bool, confidence: float, image_type: str = "X-Ray") -> List[str]:
    """
    Return a ranked list of likely conditions based on prediction and image type.
    """
    if not is_positive:
        if image_type == "ECG": return ["Normal Sinus Rhythm"]
        if image_type == "Record": return ["Routine Medical Record"]
        if image_type == "CT Scan": return ["Normal CT Scan"]
        if image_type == "Echocardiogram": return ["Normal Echocardiogram"]
        return ["No Finding"]

    if image_type == "ECG":
        if confidence >= 0.85: return ["Atrial Fibrillation", "Myocardial Infarction"]
        else: return ["Atrial Fibrillation"]

    if image_type == "Record":
        return ["Attention Required Document"]

    if image_type == "CT Scan":
        if confidence >= 0.85: return ["Pulmonary Embolism", "Aortic Aneurysm", "Lung Nodule"]
        else: return ["Lung Nodule", "Pulmonary Embolism"]

    if image_type == "Echocardiogram":
        if confidence >= 0.85: return ["Reduced Ejection Fraction", "Valve Regurgitation"]
        else: return ["Valve Regurgitation"]

    # X-ray fallback
    if confidence >= 0.85:
        return ["Cardiomegaly", "Congestive Heart Failure", "Cardiomyopathy"]
    elif confidence >= 0.70:
        return ["Cardiomegaly", "Coronary Artery Disease"]
    else:
        return ["Cardiomegaly"]


def format_full_report(prediction: dict, patient_data: dict = None) -> dict:
    """
    Aggregate prediction output + medical advice into a single report dict.

    This is the main function called by app.py to get everything needed
    for the results display in one structured object.

    Args:
        prediction: Output dict from predictor.predict() or demo_predict()
        patient_data: Optional dictionary containing age, BP, sugar, body_fat
    Returns:
        Complete report dict for UI rendering
    """
    is_positive = prediction["is_positive"]
    confidence = prediction["confidence"]

    risk = get_risk_level(confidence, is_positive)
    image_type = prediction.get("image_type", "X-Ray")
    conditions = select_likely_conditions(is_positive, confidence, image_type)
    primary_condition = conditions[0]
    
    # Deepcopy to avoid mutating the global database
    advice = copy.deepcopy(get_condition_advice(primary_condition))

    # Add dynamic advice based on patient_data physicals
    if patient_data:
        bp_sys = patient_data.get("bp_sys", 120)
        bp_dia = patient_data.get("bp_dia", 80)
        sugar = patient_data.get("sugar", "Unknown")
        age = patient_data.get("age")
        body_fat = patient_data.get("body_fat")

        if bp_sys >= 140 or bp_dia >= 90:
            advice.lifestyle_advice.insert(0, f"🩸 High Blood Pressure ({bp_sys}/{bp_dia}): Requires strict sodium limits (<1500mg) and continuous BP monitoring.")
            if risk["level"] == "LOW":
                risk["level"] = "MODERATE"
                risk["color"] = "#FF9800"
                risk["description"] = "No structural anomaly found, but elevated BP requires clinical attention."
        
        if sugar in ["Pre-diabetic", "Diabetic"]:
            advice.dietary_advice.insert(0, f"🍬 {sugar} Care: Strict glycemic control is critical to prevent vascular damage.")
            
        if body_fat:
            advice.lifestyle_advice.insert(0, f"⚖️ Body Fat ({body_fat}%): Weight reduction through caloric deficit and cardio will significantly lower cardiac workload.")

        if age and age > 65:
            advice.precautions.insert(0, f"🧓 Advanced Age ({age}): Age significantly multipliers cardiovascular risk. Do not skip follow-ups.")

    return {
        "prediction": prediction["prediction"],
        "is_positive": is_positive,
        "confidence_pct": prediction["confidence_pct"],
        "risk": risk,
        "primary_condition": primary_condition,
        "all_conditions": conditions,
        "advice": advice,
        "demo_mode": prediction.get("demo_mode", False),
        "image_type": image_type,
    }
