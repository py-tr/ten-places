"""Language + vision task planner on OpenVINO GenAI (Qwen3-VL-4B INT4, Apache-2.0), with a symbolic verifier.

The VLM proposes, the verifier disposes:
  plan(command, image)   -> the VLM returns a JSON list of skills (schema-constrained, so it can only name
                            real skills); verify() then inserts physical prerequisites, drops duplicates
                            and skills already done, and records every correction it made.
  is_done(skill, image)  -> a yes/no visual check after each skill, used to retry or re-plan.
"""
import json
import re
import time
from pathlib import Path

import numpy as np

from .env_table import SKILLS

SKILL_NAMES = [s for s, _, _ in SKILLS]
SKILL_TEXT = {s: t for s, _, t in SKILLS}

# Physics of this table, not preferences: cutlery lies in the closed drawer; the plate starts on the fork's spot.
PREREQS = {"spoon": ["drawer"], "fork": ["drawer", "plate"], "plate": ["drawer"], "cup": [], "drawer": []}
PREREQ_WHY = {("spoon", "drawer"): "the spoon is inside the closed drawer",
              ("fork", "drawer"): "the fork is inside the closed drawer",
              ("fork", "plate"): "the plate starts on the fork's spot",
              ("plate", "drawer"): "the closed drawer's tray reaches the placemat's edge"}

DONE_QUESTION = {
    "drawer": "Is the wooden drawer on the left pulled open, so the cutlery inside is visible?",
    "spoon": "Is a spoon lying flat on the white strip on the near side of the green placemat?",
    "plate": "Is the white plate on the green placemat?",
    "fork": "Is a fork lying flat on the white strip on the far side of the green placemat?",
    "cup": "Is the blue cup standing on the white circle at the corner of the place setting?",
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string", "enum": SKILL_NAMES}, "maxItems": 8},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["steps"],
}
# What was asked that no skill can do ("pour water") is a separate question (cannot_do): folded into the plan
# answer it cost planning accuracy (10/10 -> 9/10) and turned "set the table and light a candle" into an empty
# plan. The plan prompt below is the one that scored 10/10.
UNSUPPORTED_SCHEMA = {
    "type": "object",
    "properties": {"unsupported": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3}},
    "required": ["unsupported"],
}
CHECK_SCHEMA = {"type": "object", "properties": {"done": {"type": "boolean"}}, "required": ["done"]}

PLAN_PROMPT = """You control two robot arms setting a dinner table (the image is a top view; arm A left, arm B right).
The robot can run these skills:
{skills}
Examples of commands and the right answer:
Command: "Lay a full place setting." -> {{"steps": ["drawer", "spoon", "plate", "fork", "cup"], "reason": "a full setting needs everything"}}
Command: "Only the cup, please." -> {{"steps": ["cup"], "reason": "just the cup"}}
Command: "Everything but the fork." -> {{"steps": ["drawer", "spoon", "plate", "cup"], "reason": "all skills except the fork"}}
Command: "I want to eat soup." -> {{"steps": ["drawer", "spoon"], "reason": "soup needs a spoon from the drawer"}}
Now the real command: "{command}"
Skills already completed: {done}. Leave those out.
Answer with JSON only: {{"steps": [...], "reason": "..."}}."""

CANNOT_PROMPT = """A robot setting a dinner table can only do these skills:
{skills}
A person said: "{command}"
List what they asked for that none of these skills does, as short phrases; an empty list if the skills cover
everything they asked for.
Examples:
"Set the table and pour some water." -> {{"unsupported": ["pour water"]}}
"Put the plate out." -> {{"unsupported": []}}
"Lay the cutlery and dim the lights." -> {{"unsupported": ["dim the lights"]}}
"I'd like a spoon for my soup." -> {{"unsupported": []}}
JSON only."""


_JOIN = r"(?:,|;|\band\b|\bthen\b|\balso\b|\bplus\b)"


def without_unsupported(command: str, unsupported) -> str:
    """The command with what no skill does cut out, and the joins it leaves dangling: "Set the table and light a
    candle." -> "Set the table." (the 4B model reads "Set the table and ." as incomplete and plans nothing)."""
    rest = command
    for u in unsupported:
        i = rest.lower().find(u.lower())
        if i >= 0:
            rest = rest[:i] + " " + rest[i + len(u):]
    rest = re.sub(rf"(?:\s*{_JOIN})+\s*([.!?]*)\s*$", r"\1", rest, flags=re.I)
    rest = re.sub(rf"^\s*(?:{_JOIN}\s*)+", "", rest, flags=re.I)
    return re.sub(r"\s+([,.!?])", r"\1", " ".join(rest.split()))


def verify(steps, done=()):
    """Make a proposed plan executable: prerequisites first, no duplicates, nothing already done."""
    done, out, notes = set(done), [], []

    def add(skill, because=None):
        if skill in done or skill in out:
            return
        for req in PREREQS.get(skill, []):
            if req not in done and req not in out:
                add(req, because=(skill, req))
        out.append(skill)
        if because:
            notes.append(f"added '{because[1]}' before '{because[0]}': {PREREQ_WHY.get(because, 'prerequisite')}")

    for s in steps:
        if s not in SKILL_NAMES:
            notes.append(f"dropped unknown skill '{s}'")
        elif s in done:
            notes.append(f"skipped '{s}': already done")
        elif s in out:
            notes.append(f"dropped duplicate '{s}'")
        else:
            add(s)
    # On this table only the canonical order is collision-free for every subset (measured with the scripted
    # controller: e.g. plate before spoon lets arm B's spoon placement graze the plate, cup before plate
    # leaves the plate landing against the cup). Keep the planner's choice of WHAT to do, fix the order.
    ordered = sorted(out, key=SKILL_NAMES.index)
    if ordered != out:
        notes.append(f"reordered to {' -> '.join(ordered)}: other orders collide on this table")
    # A change spoken mid-run can ask for a skill after one that normally comes later (spoon after the plate is
    # already down). It is still done — the person asked — but flagged: those orders were not collision-free.
    for s in ordered:
        later = [d for d in done if SKILL_NAMES.index(d) > SKILL_NAMES.index(s)]
        if later:
            notes.append(f"'{s}' after '{later[-1]}' is outside the order tested collision-free")
    return ordered, notes


class VLMPlanner:
    def __init__(self, model_dir: str | Path = "models/Qwen3-VL-4B-Instruct-int4-ov", device: str = "CPU",
                 warmup: bool = True, ov_config: dict | None = None, idle_config: dict | None = None):
        """ov_config: OpenVINO properties for the pipeline (e.g. the efficiency-core placement from
        tenplaces.cores), passed through unchanged; None keeps OpenVINO's defaults.
        idle_config: a second pipeline for calls made while the arms are still (idle=True: the first plan), which
        may use every core; {} means OpenVINO's defaults. On the i5-13600KF a plan takes ~14 s on the E-cores and
        6-8 s on all cores, so the arms start twice as soon. Costs a second copy of the model (~3 GB)."""
        import openvino_genai as og

        self.og = og

        def pipeline(cfg):
            return og.VLMPipeline(str(model_dir), device, **cfg) if cfg else og.VLMPipeline(str(model_dir), device)

        self.pipe = pipeline(ov_config)
        self.idle_pipe = pipeline(idle_config) if idle_config is not None else None
        self.device = device
        if warmup:  # the first generate compiles kernels (~15 s); pay that at load, not mid-demo
            self._ask("Warm-up.", np.zeros((336, 448, 3), np.uint8), CHECK_SCHEMA, max_new_tokens=4)
            if self.idle_pipe is not None:
                self._ask("Warm-up.", np.zeros((336, 448, 3), np.uint8), CHECK_SCHEMA, max_new_tokens=4, idle=True)

    def _ask(self, prompt: str, image: np.ndarray | None, schema: dict, max_new_tokens: int = 120, idle: bool = False):
        """One structured answer from the VLM; image=None asks text-only (no vision encoding). idle=True uses the
        all-core pipeline when there is one: only for calls made while nothing else needs the CPU."""
        import openvino as ov

        cfg = self.og.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.structured_output_config = self.og.StructuredOutputConfig(json_schema=json.dumps(schema))
        extra = {} if image is None else {"images": [ov.Tensor(np.ascontiguousarray(image[None]))]}
        pipe = self.idle_pipe if idle and self.idle_pipe is not None else self.pipe
        t = time.perf_counter()
        res = pipe.generate(prompt, generation_config=cfg, **extra)
        ms = 1000 * (time.perf_counter() - t)
        text = res.texts[0]
        try:
            return json.loads(text), text, ms
        except json.JSONDecodeError:
            return None, text, ms

    def plan(self, command: str, image: np.ndarray, done=(), idle: bool = False):
        skills = "\n".join(f"- {s}: {SKILL_TEXT[s]}" for s in SKILL_NAMES)
        prompt = PLAN_PROMPT.format(skills=skills, command=command, done=", ".join(done) or "none")
        parsed, raw, ms = self._ask(prompt, image, PLAN_SCHEMA, idle=idle)
        proposed = parsed.get("steps", []) if parsed else []
        steps, notes = verify(proposed, done)
        unsupported = [u.strip() for u in (parsed or {}).get("unsupported", []) if isinstance(u, str) and u.strip()]
        return {"command": command, "proposed": proposed, "steps": steps, "corrections": notes,
                "reason": (parsed or {}).get("reason", ""), "unsupported": unsupported, "raw": raw, "ms": ms}

    def cannot_do(self, command: str, image: np.ndarray | None = None, idle: bool = False):
        """What the command asks for that no skill does ("light a candle"), so the robot can say so instead of
        silently ignoring it. A separate question from plan(), so it cannot change the plan. Returns (list, ms).
        Text only: the question is about the command, not the scene (`image` is accepted and ignored), so it costs
        no vision encoding and can run while the arms already move."""
        skills = "\n".join(f"- {s}: {SKILL_TEXT[s]}" for s in SKILL_NAMES)
        parsed, raw, ms = self._ask(CANNOT_PROMPT.format(skills=skills, command=command), None, UNSUPPORTED_SCHEMA,
                                    max_new_tokens=60, idle=idle)
        items = [u.strip() for u in (parsed or {}).get("unsupported", []) if isinstance(u, str) and u.strip()]
        # Leaving one of the robot's own objects out ("skip the cup") is the plan's business, not something it
        # cannot do — it must not answer "Sorry, I can't skip the cup." ("a cup of coffee" stays: nothing is left out.)
        own = {"drawer", "spoon", "spoons", "plate", "plates", "fork", "forks", "cup", "cups", "cutlery"}
        omit = {"skip", "skipping", "leave", "without", "except", "no", "not", "omit", "don't"}

        def words(u):
            return set(u.lower().replace(",", " ").replace(".", " ").split())

        quantifiers = {"everything", "else", "the", "rest", "all", "of", "it", "and"}  # "everything else" asks nothing
        return [u for u in items if not (own & words(u) and omit & words(u)) and not words(u) <= quantifiers], ms

    def plan_intent(self, command: str, image: np.ndarray, done=(), idle: bool = False):
        """Like plan(), but the VLM only states the intent — the whole table or some skills, and what to leave out —
        and plain code assembles the steps (assemble_intent), as amend() does for spoken changes. Same dict as
        plan(), plus "intent"."""
        skills = "\n".join(f"- {s}: {SKILL_TEXT[s]}" for s in SKILL_NAMES)
        prompt = INTENT_PROMPT.format(skills=skills, command=command, done=", ".join(done) or "none")
        intent, raw, ms = self._ask(prompt, image, INTENT_SCHEMA, idle=idle)
        proposed = assemble_intent(intent) if intent else []
        steps, notes = verify(proposed, done)
        return {"command": command, "intent": intent, "proposed": proposed, "steps": steps, "corrections": notes,
                "reason": (intent or {}).get("reason", ""), "raw": raw, "ms": ms}

    def plan_checked(self, command: str, image: np.ndarray, done=(), use_intent: bool = False, idle: bool = False):
        """plan() (or plan_intent()) plus cannot_do(). If the plan came back empty although only part of the
        command is impossible ("set the table and light a candle" -> nothing at all), plan once more from the
        command without the impossible part. The plan dict gains "unsupported" (and "replanned_from")."""
        planner = self.plan_intent if use_intent else self.plan
        p = planner(command, image, done, idle=idle)
        unsupported, ms = self.cannot_do(command, idle=idle)  # text only
        p.update(unsupported=unsupported, ms=p["ms"] + ms)
        rest = without_unsupported(command, unsupported)
        if unsupported and not p["steps"] and rest != command and len(rest.strip(" ,.!?")) > 3:
            p2 = planner(rest, image, done, idle=idle)
            p2.update(command=command, unsupported=unsupported, ms=p["ms"] + p2["ms"], replanned_from=rest)
            return p2
        return p

    def is_done(self, skill: str, image: np.ndarray):
        parsed, raw, ms = self._ask(DONE_QUESTION[skill] + ' Answer as JSON: {"done": true or false}.', image,
                                    CHECK_SCHEMA, max_new_tokens=16)
        return bool(parsed and parsed.get("done")), ms

    def amend(self, command: str, heard: str, image: np.ndarray, done=(), current=None, remaining=()):
        """Something was said while the robot works: the new list of steps to run after the current one.

        The VLM only describes the change (remove / add / only / nothing more); apply_edit() applies it to what
        was planned and verify() makes the result executable. Asked to rewrite the whole list, the 4B model
        ignored removals and dropped steps it should have kept (4/10 on held-out sentences).
        """
        skills = "\n".join(f"- {s}: {SKILL_TEXT[s]}" for s in SKILL_NAMES)
        prompt = AMEND_PROMPT.format(skills=skills, command=command, done=", ".join(done) or "nothing",
                                     current=current or "nothing", planned=", ".join(remaining) or "nothing",
                                     heard=heard)
        edit, raw, ms = self._ask(prompt, image, EDIT_SCHEMA)
        proposed = apply_edit(edit, remaining) if edit else list(remaining)  # unreadable answer: keep the plan
        steps, notes = verify(proposed, list(done) + ([current] if current else []))
        return {"heard": heard, "edit": edit, "proposed": proposed, "steps": steps, "corrections": notes,
                "reason": (edit or {}).get("reason", ""), "raw": raw, "ms": ms}


_SKILL_LIST = {"type": "array", "items": {"type": "string", "enum": SKILL_NAMES}, "maxItems": 5}
EDIT_SCHEMA = {
    "type": "object",
    "properties": {"remove": _SKILL_LIST, "add": _SKILL_LIST, "only": _SKILL_LIST, "nothing_more": {"type": "boolean"},
                   "reason": {"type": "string", "maxLength": 200}},
    "required": ["remove", "add", "only", "nothing_more"],
}


def apply_edit(edit: dict, remaining) -> list:
    """The planned steps after a spoken change: nothing more, only these, or planned minus removed plus added."""
    if edit.get("nothing_more"):
        return []
    if edit.get("only"):
        return list(dict.fromkeys(edit["only"]))
    out = [s for s in remaining if s not in edit.get("remove", [])]
    return out + [s for s in dict.fromkeys(edit.get("add", [])) if s not in out]


AMEND_PROMPT = """You control two robot arms setting a dinner table (the image is a top view; arm A left, arm B right).
The robot can run these skills:
{skills}
The original command was: "{command}"
Already done: {done}. Being done right now: {current}. Still planned after that: {planned}.
While the robot works, the person just said: "{heard}"
Describe only what changes, as an edit of the planned steps:
- "remove": skills the person no longer wants
- "add": skills the person now wants as well
- "only": if the person now wants only certain skills, exactly those skills; otherwise []
- "nothing_more": true if the person wants nothing more after the current step
Examples:
Said "Please skip the plate." -> {{"remove": ["plate"], "add": [], "only": [], "nothing_more": false, "reason": "plate not wanted"}}
Said "Can you put the fork out as well?" -> {{"remove": [], "add": ["fork"], "only": [], "nothing_more": false, "reason": "fork wanted too"}}
Said "From now on only the plate." -> {{"remove": [], "add": [], "only": ["plate"], "nothing_more": false, "reason": "only the plate"}}
Said "That will do, thanks." -> {{"remove": [], "add": [], "only": [], "nothing_more": true, "reason": "finished"}}
Said "Looks good so far." -> {{"remove": [], "add": [], "only": [], "nothing_more": false, "reason": "no change"}}
JSON only."""

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"all": {"type": "boolean"}, "include": _SKILL_LIST, "exclude": _SKILL_LIST,
                   "reason": {"type": "string", "maxLength": 200}},
    "required": ["all", "include", "exclude"],
}


def assemble_intent(intent: dict) -> list:
    """Steps from a stated intent: every skill (the whole table) or the included ones, minus the excluded ones."""
    base = SKILL_NAMES if intent.get("all") else list(dict.fromkeys(intent.get("include", [])))
    return [s for s in base if s not in intent.get("exclude", [])]


INTENT_PROMPT = """You control two robot arms setting a dinner table (the image is a top view; arm A left, arm B right).
The robot can run these skills:
{skills}
The person said: "{command}"
Skills already completed: {done}.
State what the person wants, as an intent:
- "all": true if they want the whole table set (then put anything they leave out in "exclude")
- "include": if not the whole table, the skills they want (think about what their food or drink needs)
- "exclude": skills they do not want
Examples:
"Lay a full place setting." -> {{"all": true, "include": [], "exclude": [], "reason": "everything"}}
"Only the cup, please." -> {{"all": false, "include": ["cup"], "exclude": [], "reason": "just the cup"}}
"Everything but the fork." -> {{"all": true, "include": [], "exclude": ["fork"], "reason": "all except the fork"}}
"I want to eat soup." -> {{"all": false, "include": ["spoon"], "exclude": [], "reason": "soup needs a spoon"}}
JSON only."""
