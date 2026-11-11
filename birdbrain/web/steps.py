"""The four components a photo passes through, spelled out the same way everywhere (False triggers intro, a visit's
"What this photo trains" box, the About page): Motion detector, Pi animal check, BioCLIP 2 species check, Claude.

Every number comes from the code that uses it, so the text can't drift from the behaviour. The wording names the
component every time (no "it" or "this" standing for a component): the owner asked for that.
"""
import json

from markupsafe import Markup, escape

from .. import classifier, config, db, identify, lessons, presence

MOTION, PI, SPECIES, CLAUDE = "Motion detector", "Pi animal check", "BioCLIP 2 species check", "Claude"
# What the tight look's square was drawn around, by the finder that pointed at it (identify.TIGHT_WHERE, naming
# the object detector's model here), and the same as "at ..." for "a tight look at the patch that moved".
FINDER_WHERE = {**identify.TIGHT_WHERE, "yolox": "around the box a small object detector (YOLOX-tiny) drew"}
TIGHT_AT = {name: phrase.replace("around ", "at ", 1) for name, phrase in FINDER_WHERE.items()}


def tight_at(finder: str | None) -> str:
    """Where the tight look looked, as "at ...", for the finder that pointed at the patch (default the Motion
    detector's: "at the patch that moved")."""
    return TIGHT_AT.get(finder or "motion", TIGHT_AT["motion"])


def _pct(x: float) -> str:
    """Whole percents, rounded the way the pills round them (bioclip_step's :.0%): 0.528 is 53%, 0.8 is 80%."""
    return f"{x:.0%}"


def _share(x: float) -> str:
    return "half" if x == 0.5 else _pct(x)


def components() -> list[dict]:
    """The four steps: n, name, question, pill (text, class), short (one or two sentences), does, train, control and
    the label on the arrow that leads to the next step. All HTML is written here, never taken from user input."""
    frames, weight, retrain = config.TRIGGER_FRAMES, f"{presence.CONFIRMED_WEIGHT:g}x", presence.RETRAIN_CHECK
    bowl, tight, share = _pct(classifier.GATE_MIN_SCORE_BOWL), _pct(classifier.GATE_MIN_SCORE_TIGHT), _share(classifier.GATE_MIN_LOCAL)
    settled, limit, every = db.CONFIRMED_MIN, db.LESSON_LIMIT, lessons.AUTO_EVERY
    minute = "about a minute" if retrain <= 60 else f"about {round(retrain / 60)} minutes"
    return [
        {"n": 1, "name": MOTION, "question": "Did something move in the zones?",
         "pill": ("can't be trained", "step-muted"), "next": "motion found",
         "short": f"The {MOTION}, a few plain Python modules, takes a burst of photos when enough pixels change inside the zones "
                  f"for {frames} frames in a row.",
         "does": f"<p>The {MOTION} watches the live video inside the zones (rim, rocks, water, ground). When enough pixels change "
                 f"for {frames} frames in a row, the {MOTION} takes a burst of photos. The open water and the ground have stricter "
                 f"rules of their own (see Zones and Detection rules below).</p>",
         "train": f"<p>The {MOTION} cannot be trained. It is plain code with OpenCV, not a trained model, and has no learning "
                  f"step. Its modules:</p><ul>"
                  f"<li><code>watch.py</code> watches the substream and takes the burst, using frame differencing per zone "
                  f"(rim and rocks, open water, ground).</li>"
                  f"<li><code>zones.py</code> and <code>autocal.py</code>: the zones in use, and their automatic recalibration.</li>"
                  f"<li><code>stay.py</code>: one visit per stay, holding new triggers while the visitor is still there.</li>"
                  f"<li><code>motion_circle.py</code>: the circle from the motion, when the Pi names a visit itself.</li></ul>",
         "control": "<p>The zones (Settings &gt; Zones) and the thresholds.</p>"},
        {"n": 2, "name": PI, "question": "Is an animal in the dashed box?",
         "pill": ("trains on your answers", "step-accent"), "next": "animal found",
         "short": f"The {PI} answers yes or no. On no, the photo goes to False triggers and {CLAUDE} is not asked. "
                  f"The {PI} learns from every label.",
         "does": f"<p>The {PI} asks: &ldquo;Is an animal in the dashed box?&rdquo; <b>No:</b> the photo goes to False triggers; "
                 f"nothing else runs and {CLAUDE} is not asked. <b>Yes:</b> the photo goes on to the {SPECIES}.</p>"
                 f"<p><b>Rain on the water:</b> by day, when the rim fired, the score is low and nothing new shows on land "
                 f"(rim, rocks or ground), the Pi takes the movement for rain or ripples: the photo goes to False triggers "
                 f"and {CLAUDE} is not asked.</p>",
         "train": f"<p>The {PI} <b>is</b> trainable:</p><ul>"
                  f"<li>BioCLIP 2 turns the dashed box into an embedding (like a fingerprint): a list of numbers that sums up what the box shows.</li>"
                  f"<li>The yes/no scorer of the {PI} learns from every labelled photo: pressing Confirm labels the photo "
                  f"&ldquo;empty&rdquo;, picking an animal labels the photo &ldquo;animal&rdquo;. Your labels count {weight} as much "
                  f"as {CLAUDE}&rsquo;s labels.</li>"
                  f"<li>{minute.capitalize()} after any label changes, the Pi retrains the yes/no scorer automatically.</li></ul>"
                  f"<p>Notes, names, circles on an empty photo and the Ask BioCLIP 2 button do not train the {PI}.</p>",
         "control": "<p>Your labels: Confirm, or the animal you pick.</p>"},
        {"n": 3, "name": SPECIES, "question": "Which species?",
         "pill": ("not trained, uses your checks", "step-medium"), "next": "species not settled",
         "short": f"The {SPECIES} names the animal on the Pi, without {CLAUDE}, only for familiar daytime regulars: a "
                  f"first look at the whole bowl, then, when that is not sure enough, a tight look at the patch that "
                  f"moved. Otherwise the photo goes to {CLAUDE}.",
         "does": f"<p>The {SPECIES} compares the embedding (like a fingerprint) with every bird and mammal name on the eBird and iNaturalist "
                 f"lists and picks the closest species.</p>"
                 f"<p>The {SPECIES} names the animal itself, without {CLAUDE}, only when <b>all</b> are true:</p><ul>"
                 f"<li>Daytime (never on night infrared photos).</li>"
                 f"<li>Very sure: top guess at least {bowl} on the first look, the {PI}&rsquo;s view of the whole bowl. When "
                 f"the first look falls short, the {SPECIES} takes one tight look at a square around the patch that moved "
                 f"(about 6 seconds more on the Pi), which must reach {tight}. The {MOTION} points at that patch; when "
                 f"the {MOTION} can't, the Pi tries the crowd counter's outline, then a small object detector (YOLOX-tiny) "
                 f"on the crop box. Either way, at least {share} of the belief on local species.</li>"
                 f"<li>The species is settled here: at least {settled} earlier visits answered with high confidence by {CLAUDE} "
                 f"or labelled by you. Visits the {SPECIES} named itself count only after you checked them.</li>"
                 f"<li>Only one animal in view.</li></ul>"
                 f"<p>If any condition fails, the photo goes to {CLAUDE}.</p>",
         "train": f"<p>The {SPECIES} is not trained, and BioCLIP 2 never changes. Confirming or correcting visits helps a "
                  f"species reach the {settled} settled visits, after which the {SPECIES} can name that species on the Pi for "
                  f"free. Pressing Looks right on a visit the {SPECIES} named counts toward that list.</p>",
         "control": "<p>Confirming or correcting visits.</p>"},
        {"n": 4, "name": CLAUDE, "question": "the easy button",
         "pill": ("reads your notes", "step-accent"), "next": "",
         "short": f"{CLAUDE} gets the photo only when the {SPECIES} did not settle the visit. {CLAUDE} reads all of "
                  f"{CLAUDE}&rsquo;s Notes fresh with every photo.",
         "does": f"<p>{CLAUDE} gets the photo only when the {SPECIES} did not settle the visit. Haiku answers first; Sonnet "
                 f"takes a second look when Haiku is unsure.</p>"
                 f"<p>Sent to {CLAUDE} every time: the photo plus all of {CLAUDE}&rsquo;s Notes (the Your yard boxes, the Lessons "
                 f"summary, and the newest {limit} lessons word for word; a lesson is one of your corrections or Notes for {CLAUDE}).</p>",
         "train": f"<p>{CLAUDE} is not trained; {CLAUDE} reads the notes fresh with every photo. A Note for {CLAUDE} "
                  f"describing each numbered circle (e.g. &ldquo;1: spiderweb moving in the wind&rdquo;) becomes a lesson; circles "
                  f"alone do not. After every {every} new lessons, {CLAUDE} drafts a new Lessons summary for you to accept.</p>",
         "control": f"<p>Your corrections and Notes for {CLAUDE}.</p>"},
    ]


def bottom_line() -> Markup:
    return Markup(f"<b>Bottom line:</b> the Pi handles empty photos (the {PI}) and familiar daytime regulars (the {SPECIES}) "
                  f"for free. {CLAUDE} sees only new, unsure, night or crowded photos.")


# --- one photo's real state -----------------------------------------------------------------------------------

def _calls(visit) -> dict:
    try:
        return (json.loads(visit["raw_json"] or "{}") or {}).get("calls") or {}
    except (ValueError, TypeError, AttributeError):
        return {}


def miss_text(code: str | None, guess: dict, settled: dict | None = None, view: str = "bowl") -> str:
    """Plain words for a classifier.gate_miss code (live numbers from the classifier's constants). `view` is which
    look the guess was ("bowl" or "tight"), since each has its own bar (classifier.gate_bar)."""
    top = (guess.get("top") or [{}])[0]
    if code == "night":
        return "night photos are infrared, which BioCLIP 2 was not trained on"
    if code == "none":
        return "no local species came up"
    if code == "score":
        return f"its top guess was {_pct(top.get('score') or 0)}, under the {_pct(classifier.gate_bar(view))} it needs"
    if code == "local":
        return f"less than {_share(classifier.GATE_MIN_LOCAL)} of its belief was on local species"
    if code == "unsettled":
        return f"{top.get('name')} is not settled here yet ({db.CONFIRMED_MIN} sure visits needed)"
    if code == "crowd":
        return "more than one animal was in view"
    if code == "changed":
        return "it would name this one today, but it was not trusted to when the photo was taken"
    return f"the {SPECIES} did not trust the guess"


def gate_reason(guess: dict, settled: dict, night: bool, crowd, mode: str, view: str = "bowl") -> str:
    """Why the BioCLIP 2 species check's guess did not name the visit, in the order classifier.gate_miss tests."""
    if mode != "gate":
        return f"the {SPECIES} is in {mode} mode, so it only keeps score"
    return miss_text(classifier.gate_miss(guess, settled, night, crowd, view), guess, view=view)


def for_visit(visit, decision: dict, teaches: dict | None, lesson: dict | None, settled: dict, current_visitor: str,
              mode: str) -> list[dict]:
    """The four components for one photo: n, name, pill (text, class) and body (HTML) saying what this photo does to
    each. `teaches` is presence.teaches, `lesson` the photo's row from db.all_lessons (or None), `settled` is
    db.settled_species, `mode` the classifier mode."""
    e = escape
    calls = _calls(visit)
    weight = f"{presence.CONFIRMED_WEIGHT:g}x"
    species = visit["corrected_species"] or visit["species"]
    steps = decision.get("steps") or []
    checked = bool(visit["confirmed_at"] or visit["corrected_visitor"] or visit["corrected_species"])
    skipped_empty = any(s.get("presence") and s.get("skipped_claude") for s in steps)
    claude_steps = [s for s in steps if s["model"] in ("Haiku", "Sonnet")]
    guess_step = next((s for s in steps if s.get("gated") or s.get("scorecard")), None)
    first_step = next((s for s in steps if s.get("first_look") and not s.get("gated")), None)
    tight_step = next((s for s in steps if s.get("tight") and not s.get("gated")), None)
    tight_finder = ((calls.get("classifier_tight") or {}).get("finder")
                    or ((calls.get("classifier") or {}).get("finder") if (calls.get("classifier") or {}).get("crop") == classifier.TIGHT_CROP else None))
    TIGHT = f"a tight look {tight_at(tight_finder)}"  # per finder: the patch that moved, the crowd counter's outline, ...

    # 1. Motion detector
    where = f" It spotted movement {decision['trigger']}." if decision.get("trigger") else ""
    rows = [{"n": 1, "name": MOTION, "pill": ("nothing", "step-muted"),
             "body": f"<b>Nothing.</b> The {MOTION} cannot be trained.{e(where)}"}]

    # 2. Pi animal check
    if teaches is None:
        pill = ("no example", "step-muted")
        body = (f"<b>Nothing.</b> The {PI} stored no embedding (like a fingerprint) for this photo, so the {PI} learns nothing from it.")
    elif not teaches["animal"]:
        if teaches["checked"]:
            pill, body = (f"counted {weight}", "step-accent"), f"<b>An empty frame, counted {weight}:</b> you checked it."
        else:
            pill, body = (f"counts {weight} on Confirm", "step-accent"), f"<b>An empty frame.</b> Counts {weight} once you Confirm."
        body += f" The {PI} reads only the answer you pick, never the Note for {CLAUDE} or the circles."
    elif teaches["used"]:
        pill = (f"counted {weight}", "step-accent") if teaches["checked"] else ("animal example", "step-accent")
        body = (f"<b>An animal example</b> from {e(teaches['why'])}" +
                (f", counted {weight}: you checked it." if teaches["checked"] else
                 f". Counts once; Looks right makes it count {weight}.") +
                f" The {PI} learns whether an animal is in the dashed box, not where, and no species names.")
    else:
        why = str(teaches["why"])
        pill, body = ("left out", "step-muted"), f"<b>Left out.</b> {e(why[:1].upper() + why[1:])}."
    rows.append({"n": 2, "name": PI, "pill": pill, "body": body})

    # 3. BioCLIP 2 species check
    label = e(species) if species else ""
    if guess_step and guess_step.get("gated"):
        pill = ("RPi", "pi-only")
        # Counts toward being settled as db.settled_species counts it: a species the owner named, or Looks right
        # with the type unchanged. "No animal" or a bare type (Squirrel) takes the visit out of that count.
        counts = bool(visit["corrected_species"]) or bool(visit["confirmed_at"] and not visit["corrected_visitor"])
        if counts:
            tail = f"You checked it, so the visit counts toward {label} being settled here."
        elif checked:
            said = "no animal" if current_visitor == "none" else e(current_visitor)
            tail = f"You said it was {said}, so the visit does not count toward {label} being settled here."
        else:
            tail = f"Press Looks right and the visit counts toward {label or 'the species'} being settled here."
        how = f", on {TIGHT} after a first look at the whole bowl fell short" if guess_step.get("tight") else ""
        body = f"<b>Named the animal on the Pi:</b> {e(', '.join(guess_step['guesses'][:1]))}{how}. {CLAUDE} was not asked. {tail}"
    elif skipped_empty:
        pill = ("not reached", "step-muted")
        body = f"<b>Not reached:</b> the photo was empty, so the {SPECIES} had nothing to name."
        if current_visitor != "none" and species:
            body += f" You corrected it to {label}; the correction counts toward {label} being settled here."
    elif guess_step or first_step:
        pill = ("guess only", "step-muted")
        if first_step:  # the first look's own reason, recorded when it was made
            shown = first_step
            why = miss_text(first_step.get("why"), calls.get("classifier_gate") or {})
        else:
            shown = guess_step
            raw_guess = calls.get("classifier") or {}
            why = gate_reason(raw_guess, settled, bool(raw_guess.get("night")), guess_step.get("crowd"), mode)
        body = f"<b>Guessed {e((shown['guesses'] or ['nothing local'])[0])}</b> but did not name the visit: {e(why)}."
        if tight_step:  # the tight look's own reason, recorded when it was made
            record = calls.get("classifier_tight") or {}
            why2 = miss_text(tight_step.get("why"), record, view="tight")
            name2 = ((record.get("top") or [{}])[0]).get("name") or "nothing local"
            body += f" Then {TIGHT} guessed {e(name2)}, but {e(why2)}."
        body += f" {CLAUDE} was asked."
    else:
        pill = ("no guess", "step-muted")
        why = ("night photos are infrared" if visit["is_night"] else
               f"the {SPECIES} is switched off" if mode == "off" else "no guess was recorded")
        body = f"<b>No guess:</b> {e(why)}."
    if species and current_visitor != "none" and not (guess_step and guess_step.get("gated")) and not skipped_empty:
        body += (f" {label} is {'settled' if species in settled else 'not settled yet'} here "
                 f"({db.CONFIRMED_MIN} sure visits needed).")
    rows.append({"n": 3, "name": SPECIES, "pill": pill, "body": body})

    # 4. Claude
    if claude_steps:
        last = claude_steps[-1]
        said = last.get("said") or "nothing (declined)"
        lead = (f"<b>Answered:</b> {e(said)} ({'Sonnet, after Haiku' if last['model'] == 'Sonnet' else 'Haiku'}).")
    elif skipped_empty:
        lead = f"<b>Not asked:</b> the {PI} called the photo empty."
    elif guess_step and guess_step.get("gated"):
        lead = f"<b>Not asked:</b> the {SPECIES} named the visit."
    else:
        lead = f"<b>Not asked.</b>"
    if lesson:
        where = ("sent word for word with every photo (among the newest " f"{db.LESSON_LIMIT})" if lesson["in_use"] else
                 "switched off, so not sent" if lesson["off"] else
                 f"too old to be sent word for word, but folded into the Lessons summary")
        pill, tail = ("lesson", "step-accent"), f" A lesson on Claude&rsquo;s Notes: {where}."
    else:
        pill = ("no lesson yet", "step-muted")
        tail = (f" Confirm alone is not a lesson, and neither are circles alone. A Note for {CLAUDE} describing each "
                f"numbered circle (e.g. &ldquo;1: spiderweb moving in the wind&rdquo;) becomes one.")
    rows.append({"n": 4, "name": CLAUDE, "pill": pill, "body": lead + tail})
    return rows

