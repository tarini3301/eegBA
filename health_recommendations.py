"""
Health Recommendations Engine
==============================
Generates personalized health recommendations based on:
1. Brain age gap (predicted - chronological age)
2. SHAP feature contributions (which EEG frequency bands show accelerated aging)

Recommendations are evidence-based and sourced from neurophysiology literature.
"""

# ─────────────────────────────────────────────
# Feature-specific recommendations (by Frequency Band)
# ─────────────────────────────────────────────

BAND_RECOMMENDATIONS = {
    "Delta": {
        "icon": "🌊",
        "aging_recommendations": [
            "Excessive Delta waves while awake may indicate cognitive fatigue, slowing, or neurodegeneration.",
            "Prioritize deep, restorative sleep (7-9 hours). Delta waves are crucial during NREM sleep but should be minimal while awake.",
            "Consult a physician if you experience chronic daytime sleepiness or persistent brain fog.",
            "Regular aerobic exercise helps consolidate normal sleep architecture and limits awake slow-wave intrusions."
        ],
        "youthful_note": "Your Delta wave activity is at a healthy, minimal level for a resting wakeful state.",
    },
    "Theta": {
        "icon": "🧘",
        "aging_recommendations": [
            "Increased Theta power in resting wakefulness is strongly associated with cognitive decline, inattention, or early age-related slowing.",
            "Practice mindfulness meditation to improve focus and regulate Theta-to-Beta ratios.",
            "Ensure you are getting enough cardiovascular exercise, which improves overall cortical efficiency and metabolic health.",
            "Engage in cognitively demanding tasks (like learning a new language) to stimulate faster brain rhythms."
        ],
        "youthful_note": "Your Theta activity shows healthy cognitive control and an absence of drowsy-state intrusion.",
    },
    "Alpha": {
        "icon": "⚡",
        "aging_recommendations": [
            "Decreased Alpha power and slowing of the Alpha peak frequency (especially in the parieto-occipital regions) is a major hallmark of the aging brain.",
            "Engage in relaxation techniques; healthy Alpha waves are dominant when you are calm with your eyes closed.",
            "Activities that require visual processing and spatial navigation can help stimulate posterior Alpha networks.",
            "A Mediterranean diet rich in antioxidants helps preserve the structural integrity of networks generating Alpha Rhythms."
        ],
        "youthful_note": "Your Alpha power is strong and robust, typical of a healthy, youthful resting brain.",
    },
    "Beta": {
        "icon": "🎯",
        "aging_recommendations": [
            "Changes in Beta power can relate to stress, anxiety, or compensatory cognitive effort in the aging brain.",
            "Manage chronic stress and anxiety through structured relaxation—excess Frontal Beta can mean an 'overactive' or anxious mind.",
            "Focus on single-tasking rather than multi-tasking to improve cortical efficiency and reduce cognitive load.",
            "Reduce caffeine and stimulant intake if you feel chronically tense or suffer from insomnia."
        ],
        "youthful_note": "Your Beta wave distribution is optimal, suggesting an alert, attentive, but not overstimulated state.",
    },
    "Gamma": {
        "icon": "🧠",
        "aging_recommendations": [
            "Gamma waves are associated with high-level cognitive processing, sensory binding, and memory consolidation.",
            "Engage in complex, novel learning tasks (like playing a musical instrument) to boost Gamma synchronization.",
            "A diet rich in Omega-3 fatty acids (DHA, EPA) supports the parvalbumin interneurons that generate Gamma rhythms.",
            "Social interaction and active conversation significantly stimulate Gamma binding networks."
        ],
        "youthful_note": "Your Gamma activity indicates excellent neural network synchronization and profound cognitive readiness.",
    }
}


def get_overall_assessment(brain_age_gap):
    """
    Return an overall assessment based on the brain age gap.
    
    brain_age_gap = predicted_brain_age - chronological_age
    Positive = brain looks older than expected
    Negative = brain looks younger than expected
    """
    if brain_age_gap is None:
        return {
            "level": "info",
            "title": "Brain Age Prediction Complete",
            "summary": "Provide your chronological age for a personalized brain age gap assessment.",
            "color": "#60a5fa",
        }
    
    gap = brain_age_gap
    
    if gap <= -5:
        return {
            "level": "excellent",
            "title": "Exceptional Brain Health",
            "summary": f"Your brain's EEG profile appears {abs(gap):.1f} years YOUNGER than your chronological age. "
                       "This is an outstanding result indicating excellent neurophysiological health.",
            "color": "#22c55e",
        }
    elif gap <= -2:
        return {
            "level": "good",
            "title": "Good Brain Health",
            "summary": f"Your brain's EEG profile appears {abs(gap):.1f} years younger than expected. "
                       "Your brain is aging well — keep up your healthy habits.",
            "color": "#4ade80",
        }
    elif gap <= 2:
        return {
            "level": "normal",
            "title": "Normal Brain Aging",
            "summary": f"Your functional brain age is within {abs(gap):.1f} years of your chronological age. "
                       "This indicates typical, healthy brain wave patterns for your age group.",
            "color": "#facc15",
        }
    elif gap <= 5:
        return {
            "level": "attention",
            "title": "Mild Accelerated Aging",
            "summary": f"Your brain's EEG profile appears {gap:.1f} years OLDER than expected. "
                       "Consider lifestyle modifications to support neural synchronization and brain health. See recommendations below.",
            "color": "#fb923c",
        }
    else:
        return {
            "level": "concern",
            "title": "Significant Accelerated Aging",
            "summary": f"Your brain's EEG profile appears {gap:.1f} years OLDER than expected. "
                       "We recommend reviewing the specific spectral contributions and health recommendations below. "
                       "Consider consulting a healthcare provider if you have cognitive concerns.",
            "color": "#ef4444",
        }


def generate_recommendations(brain_age_gap, feature_contributions):
    """
    Generate personalized health recommendations based on the brain age gap
    and individual feature SHAP contributions.
    
    Args:
        brain_age_gap: float or None (predicted - chronological)
        feature_contributions: list of dicts from model.predict_with_explanation()
        
    Returns:
        dict with overall_assessment and per-feature recommendations
    """
    overall = get_overall_assessment(brain_age_gap)
    
    recommendations = []
    
    for contrib in feature_contributions:
        fname = contrib["feature"]
        raw_shap = contrib["shap_value"]
        adj_shap = contrib.get("adjusted_shap_value", raw_shap)
        
        # e.g., "Frontal_Alpha_Power" -> region="Frontal", band="Alpha"
        parts = fname.split('_')
        if len(parts) >= 2:
            region = parts[0]
            band = parts[1]
        else:
            continue
            
        rec_info = BAND_RECOMMENDATIONS.get(band, None)
        if rec_info is None:
            continue
        
        # Determine if this feature is pushing toward aging or youthful
        # We use the adjusted SHAP value to identify abnormal deviation from chronological age.
        is_aging = adj_shap > 0.4  # threshold for meaningful abnormal contribution
        is_youthful = adj_shap < -0.4
        
        region_display = f"{region} Lobe ({band} Band)"
        
        if is_aging:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "attention",
                "status_label": "Accelerated Aging Indicator",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": rec_info["aging_recommendations"],
            }
        elif is_youthful:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "positive",
                "status_label": "Youthful Indicator",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": [rec_info["youthful_note"]],
            }
        else:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "neutral",
                "status_label": "Normal for your age",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": ["This frequency band is consistent with normal aging for your specific chronological age."],
            }
        
        recommendations.append(rec)
    
    # General lifestyle recommendations for neural health (always shown)
    general_tips = [
        {
            "icon": "🏃",
            "title": "Regular Aerobic Exercise",
            "detail": "150+ minutes weekly. Aerobic exercise promotes neurogenesis and increases high-frequency neural synchronization.",
        },
        {
            "icon": "🐟",
            "title": "Omega-3 Rich Diet",
            "detail": "DHA and EPA fatty acids are critical for the formation and maintenance of synapses that generate fast brain rhythms.",
        },
        {
            "icon": "😴",
            "title": "Deep Sleep Architecture",
            "detail": "Slow-wave sleep clears amyloid beta. Poor sleep fragments daytime EEG and intrudes slow waves into wakefulness.",
        },
        {
            "icon": "🧘",
            "title": "Mindfulness & Relaxation",
            "detail": "Enhances resting-state Alpha power and reduces hyperactive Beta states associated with chronic stress and cortisol.",
        },
        {
            "icon": "🧠",
            "title": "Novel Cognitive Challenge",
            "detail": "Learning completely new skills (like an instrument) forces distinct neural ensembles to synchronize, boosting Gamma power.",
        },
    ]
    
    return {
        "overall_assessment": overall,
        "feature_recommendations": recommendations,
        "general_tips": general_tips,
    }
