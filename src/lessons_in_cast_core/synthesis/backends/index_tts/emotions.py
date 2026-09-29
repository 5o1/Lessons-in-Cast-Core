"""IndexTTS-only emotion-vector presets; initial artistic approximations.

Order: joy, anger, sadness, fear, disgust, depression, surprise, calm.
These are raw model control values before the adapter's bias/normalization and
profile emotion_alpha. Complex presets are not yet listening-calibrated.
"""

EMOTION_VECTORS = {
    "neutral":           (0, 0, 0, 0, 0, 0, 0, 0),
    "happy":             (.6, 0, 0, 0, 0, 0, 0, 0),
    "excited":           (.7, 0, 0, 0, 0, 0, .3, 0),
    "affectionate":      (.33, 0, 0, 0, 0, 0, 0, .27),
    "angry":             (0, .6, 0, 0, 0, 0, 0, 0),
    "sad":               (0, 0, .42, 0, 0, .18, 0, 0),
    "distressed":        (0, 0, .24, .21, 0, .15, 0, 0),
    "afraid":            (0, 0, 0, .6, 0, 0, 0, 0),
    "disgusted":         (0, 0, 0, 0, .6, 0, 0, 0),
    "embarrassed":       (0, 0, .15, .27, 0, 0, 0, .18),
    "surprised":         (0, 0, 0, 0, 0, 0, .6, 0),
    "confused":          (0, 0, 0, 0, 0, 0, .33, .27),
    "sarcastic":         (.12, 0, 0, 0, .3, 0, 0, .18),
    "calm":              (0, 0, 0, 0, 0, 0, 0, .2),
    "tearful_resolve":   (.08, 0, .28, .05, 0, .06, 0, .25),
    "restrained_grief":  (0, 0, .24, 0, 0, .16, 0, .22),
    "emotional_breakdown": (0, .08, .45, .4, 0, .16, .08, 0),
    "tearful_remorse":   (0, 0, .38, .16, 0, .15, 0, .04),
    "playful_affection": (.32, 0, 0, 0, 0, 0, .08, .14),
    "guarded_composure": (0, 0, .03, .03, 0, 0, 0, .28),
    "firm_boundary":     (0, .16, 0, 0, .05, 0, 0, .22),
    "cheerful_pressure": (.3, .06, 0, 0, 0, 0, .06, .12),
    "tentative_hope":    (.12, 0, .05, .09, 0, 0, .04, .18),
    "possessive_care":   (.14, 0, .04, .12, 0, 0, 0, .25),
}
