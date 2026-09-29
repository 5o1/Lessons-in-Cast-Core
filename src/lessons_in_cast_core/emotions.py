"""Backend-neutral acting labels. Model vectors and strengths do not belong here."""

EMOTION_LABELS = {
    "neutral": ("Ordinary unmarked speech; no deliberate emotional emphasis.", "The door is open."),
    "happy": ("Genuine, uncomplicated pleasure.", "I'm glad you came!"),
    "sad": ("Open sadness without a specific recovery or suppression beat.", "I miss you."),
    "angry": ("Direct anger or confrontation.", "Don't do that again!"),
    "afraid": ("Fear of an immediate or anticipated threat.", "What if they find us?"),
    "surprised": ("An unexpected realization or startled reaction.", "You're already here?"),
    "disgusted": ("Aversion or repulsion.", "Get that away from me."),
    "embarrassed": ("Self-conscious, awkward or bashful exposure.", "You weren't supposed to hear that."),
    "affectionate": ("Sincere warmth and closeness, without deliberate teasing.", "I'm happy here with you."),
    "excited": ("Eager, animated anticipation or delight.", "We actually did it!"),
    "calm": ("Audibly composed, settled delivery, not emotional concealment.", "Take your time. I'm listening."),
    "confused": ("Trying to understand an unclear situation.", "Wait, what do you mean?"),
    "sarcastic": ("An audible ironic or mocking edge.", "Well, that went wonderfully."),
    "distressed": ("Upset and struggling with pressure, but not a complete breakdown.", "I don't know what to do."),
    "tearful_resolve": ("Tears remain audible while purpose and agency begin to return; neither fresh collapse nor cheerful recovery.", "I'll be all right. I can try again."),
    "restrained_grief": ("Holding grief back while admitting pain; not an openly sobbing outburst.", "I still think about her sometimes."),
    "emotional_breakdown": ("Loss of emotional control in an acute crisis; not ordinary anger or merely loud speech.", "Please don't leave me!"),
    "tearful_remorse": ("Crying apology driven by guilt and regret, not angry protest.", "I'm sorry. I didn't mean it."),
    "playful_affection": ("Knowingly sweet or teasing affection used to invite attention; not generic excitement.", "You missed me, didn't you?"),
    "guarded_composure": ("Controlled, readable speech deliberately withholding emotional access; not robotic neutrality.", "There's nothing else to say."),
    "firm_boundary": ("Brief, decisive refusal with friendliness withdrawn; not shouting or raging.", "No. We're not doing that."),
    "cheerful_pressure": ("A bright friendly exterior carrying insistence or a pointed agenda; not uncomplicated happiness.", "You will come with us, won't you?"),
    "tentative_hope": ("Fragile willingness to believe reassurance, with hesitation still present.", "Do you really think I could?"),
    "possessive_care": ("Apparently warm, reasonable care colored by fear of losing closeness and a wish to control it.", "Wouldn't it be better if you stayed here?"),
}


def emotion_definitions(labels, catalog=None) -> dict[str, dict[str, str]]:
    """Provide semantic guidance to polish, never numerical backend controls."""
    catalog = EMOTION_LABELS if catalog is None else catalog
    unknown = set(labels) - catalog.keys()
    if unknown:
        raise ValueError(f"Undefined emotion labels: {sorted(unknown)}")
    return {label: {"description": catalog[label][0], "example": catalog[label][1]}
            for label in sorted(labels)}
