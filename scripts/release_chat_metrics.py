"""Measurements for the ordinary Guffin and llama.cpp 0.5.0 chat CLIs.

Only frontend output is parsed; no sampler, execution, or warmup options are
changed. Call feed() for both streams, then end_turn() once complete timings
and the next input prompt have arrived. finish() rejects unfinished receipts.

The pinned llama.cpp prompt interval runs from post-tokenization prompt handling
through the first sampled token. Its generation rate counts N-1 steps; Guffin's
native displayed rate counts N selected tokens over the same first-to-last
selection interval. Both conventions are retained, with a separate common rate.
"""

import json
import re
from decimal import Decimal


class MetricsError(ValueError):
    """A CLI receipt is missing, malformed, or internally inconsistent."""


_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_RATE = rf"(?:{_NUMBER} t/s|n/a)"
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*?(?:\x07|\x1b\\))")
_GUFFIN_RATE = re.compile(
    rf"\[ Prompt: (?P<prompt>{_RATE}) \| Generation: (?P<generation>{_RATE})"
    rf" \| TTFT: (?P<ttft>{_NUMBER} s|n/a) \]"
)
_LLAMA_RATE = re.compile(
    rf"\[ Prompt: (?P<prompt>{_RATE}) \| Generation: (?P<generation>{_RATE}) \]"
)
_DETAIL = re.compile(
    rf"\[ Detail: Evaluated: (?P<prompt>[0-9]+) \| Cached: (?P<cached>[0-9]+)"
    rf" \| Decoded: (?P<selected>[0-9]+) \| Prompt: (?P<prompt_seconds>{_NUMBER}) s"
    rf" \| Generation: (?P<generation_seconds>{_NUMBER}) s \]"
)
_DECODE = re.compile(r"\[ Decode: (?P<json>\{.*\}) \]")
_SLOT = re.compile(
    r"slot\s+\S+:\s+id\s+(?P<slot>[0-9]+)\s+\|\s+task\s+(?P<task>[0-9]+)"
    r"\s+\|\s+(?P<body>.*)"
)
_PHASE = re.compile(
    rf"(?P<phase>prompt eval|eval) time =\s*(?P<ms>{_NUMBER}) ms /"
    rf"\s*(?P<count>[0-9]+) tokens \(\s*(?P<per_token>{_NUMBER}) ms per token,"
    rf"\s*(?P<rate>{_NUMBER}) tokens per second\)"
)
_TOTAL = re.compile(rf"total time =\s*(?P<ms>{_NUMBER}) ms /\s*(?P<count>[0-9]+) tokens")
_NEW_PROMPT = re.compile(
    r"new prompt, n_ctx_slot = (?P<context>[0-9]+), n_keep = -?[0-9]+,"
    r" task\.n_tokens = (?P<count>[0-9]+)"
)
_ACCEPTANCE = re.compile(
    rf"draft acceptance = (?P<ratio>{_NUMBER}) \(\s*(?P<accepted>[0-9]+) accepted /"
    rf"\s*(?P<proposed>[0-9]+) generated\), mean len =\s*(?P<mean>{_NUMBER})"
)


def clean_terminal_text(text):
    """Remove display escapes and ordinary overwrite/backspace echo artifacts.

    This is a text extraction helper, not a general terminal emulator. The CLIs'
    ordinary append-only input/output uses these controls; raw transcripts must
    also be saved by the caller.
    """
    text = _ANSI.sub("", text).replace("\r", "")
    visible = []
    for character in text:
        if character == "\b":
            if visible and visible[-1] != "\n":
                visible.pop()
        else:
            visible.append(character)
    return "".join(visible)


def is_prompt_ready(stdout_text):
    """Recognize the final input prompt; callers must also gate on full metrics."""
    return re.search(r"(?:^|\n)> \Z", clean_terminal_text(stdout_text)) is not None


def encode_prompt(engine, text):
    """Encode one exact user message for the default terminal input protocols.

    Guffin cannot submit a final literal backslash through its continuation
    protocol. llama.cpp's terminal '/' terminator can preserve final slashes or
    backslashes. No bracketed-paste sequences are supported by the pinned CLI.
    """
    if engine not in ("guffin", "llama"):
        raise ValueError("engine must be guffin or llama")
    if not isinstance(text, str) or not text:
        raise ValueError("prompt must be nonempty text")
    if any((ord(character) < 32 and character != "\n") or ord(character) == 127
           for character in text):
        raise ValueError("terminal prompts cannot contain control characters or tabs")
    if text.startswith("/"):
        raise ValueError("a leading slash would invoke a chat command")
    if engine == "guffin" and text.endswith("\\"):
        raise ValueError("Guffin cannot submit a final literal backslash exactly")
    lines = text.split("\n")
    encoded = "\\\n".join(lines)
    if engine == "llama" and text.endswith(("/", "\\")):
        encoded += "/"
    return (encoded + "\n").encode("utf-8")


def extract_answer(engine, clean_stdout_turn, prompt):
    """Extract the visible reply, retaining any final CLI display newline.

    The caller slices stdout immediately before prompt submission through the
    next input prompt. This returns displayed text, not raw decoded token bytes;
    token counts must come from metrics. Earlier metric lookalikes in the answer
    are preserved by using the last complete native rate line.
    """
    encode_prompt(engine, prompt)
    text = clean_terminal_text(clean_stdout_turn)
    rate_pattern = _GUFFIN_RATE if engine == "guffin" else _LLAMA_RATE
    matches = list(re.finditer(r"(?m)^" + rate_pattern.pattern + r"$", text))
    if not matches:
        raise MetricsError("missing final native rate line while extracting reply")
    reply = text[:matches[-1].start()]
    if engine == "llama":
        # show_info adds one separator newline before the rate line.
        if not reply.endswith("\n"):
            raise MetricsError("missing llama.cpp timing separator")
        reply = reply[:-1]
        echo = prompt if prompt.endswith(("/", "\\")) else prompt + "\n"
    else:
        # Guffin does not echo raw PTY input, but does print continuation prompts.
        echo = "... " * prompt.count("\n")
    if not reply.startswith(echo):
        raise MetricsError(f"{engine} input display does not match the submitted text")
    return reply[len(echo):]


def _duration(value, multiplier):
    value = Decimal(value)
    ns = value * multiplier
    if ns != ns.to_integral_value():
        raise MetricsError("timing has finer precision than nanoseconds")
    resolution = Decimal(1).scaleb(value.as_tuple().exponent) * multiplier
    return int(ns), resolution / 2


def _rate(value):
    return None if value == "n/a" else Decimal(value.removesuffix(" t/s"))


def _validate_rate(value, count, duration, half_resolution, label):
    if value is None:
        if count and duration:
            raise MetricsError(f"{label} omitted a defined rate")
        return
    rate_half = Decimal(1).scaleb(value.as_tuple().exponent) / 2
    if count == 0:
        if abs(value) > rate_half:
            raise MetricsError(f"{label} has a nonzero rate for zero tokens")
        return
    upper_duration = Decimal(duration) + half_resolution
    if upper_duration <= 0:
        raise MetricsError(f"{label} has tokens but no measurable duration")
    low = Decimal(count) * 1_000_000_000 / upper_duration
    lower_duration = Decimal(duration) - half_resolution
    high = (Decimal(count) * 1_000_000_000 / lower_duration
            if lower_duration > 0 else Decimal("Infinity"))
    if value + rate_half < low or value - rate_half > high:
        raise MetricsError(f"{label} count, duration and displayed rate disagree")


def _reject_nonfinite(value):
    raise ValueError("nonfinite JSON number")


def _integer(record, name):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricsError(f"invalid {name}")
    return value


def _common(prompt_count, cached, selected, prompt_ns, generation_ns, rates):
    steps = max(selected - 1, 0)
    return {
        "prompt_tokens": prompt_count,
        "cached_tokens": cached,
        "selected_tokens": selected,
        "generation_steps": steps,
        "prompt_ns": prompt_ns,
        "generation_ns": generation_ns,
        "internal_ttft_ns": prompt_ns,
        "engine_turn_ns": prompt_ns + generation_ns,
        "prompt_tps": prompt_count * 1e9 / prompt_ns if prompt_ns else None,
        "generation_tps": (steps * 1e9 / generation_ns if generation_ns
                           else 0.0 if steps == 0 else None),
        "native_prompt_tps": float(rates["prompt"]) if rates["prompt"] is not None else None,
        "native_generation_tps": (float(rates["generation"])
                                  if rates["generation"] is not None else None),
        "stop_reason": None,
    }


class _MetricsParser:
    required = ()

    def __init__(self):
        self._buffers = {"stdout": "", "stderr": ""}
        self._records = {}

    def feed(self, stream, text_chunk):
        if stream not in self._buffers:
            raise ValueError("stream must be stdout or stderr")
        if not isinstance(text_chunk, str):
            raise TypeError("feed expects decoded text chunks")
        self._buffers[stream] += text_chunk
        lines = self._buffers[stream].split("\n")
        self._buffers[stream] = lines.pop()
        for line in lines:
            self._line(stream, clean_terminal_text(line).strip())

    def _set(self, name, value):
        if name in self._records:
            raise MetricsError(f"duplicate {name} record before turn finalization")
        self._records[name] = value

    def _flush(self):
        for stream, buffer in self._buffers.items():
            if buffer:
                self._line(stream, clean_terminal_text(buffer).strip())
        self._buffers = dict.fromkeys(self._buffers, "")

    @property
    def can_finish_turn(self):
        return all(name in self._records for name in self.required)

    @property
    def pending(self):
        return bool(self._records)

    def _require_complete(self):
        self._flush()
        missing = [name for name in self.required if name not in self._records]
        if missing:
            raise MetricsError("incomplete turn metrics: " + ", ".join(missing))

    def finish(self):
        self._flush()
        if self.pending:
            raise MetricsError("unfinalized or incomplete turn metrics")


class GuffinMetricsParser(_MetricsParser):
    required = ("rate", "detail", "decode")

    def _line(self, stream, line):
        # These lines share stdout with answers in the default terminal UI.
        # Stage the last candidates, then validate them together at finalization;
        # quoted examples in the answer must not become receipts.
        for prefix, name in (("[ Prompt:", "rate"), ("[ Detail:", "detail"), ("[ Decode:", "decode")):
            if line.startswith(prefix):
                if name == "rate":
                    self._records.clear()
                self._records[name] = line
                return

    def end_turn(self):
        self._require_complete()
        rate_match = _GUFFIN_RATE.fullmatch(self._records["rate"])
        detail_match = _DETAIL.fullmatch(self._records["detail"])
        decode_match = _DECODE.fullmatch(self._records["decode"])
        if not (rate_match and detail_match and decode_match):
            raise MetricsError("malformed final Guffin timing block")
        rates = {
            "prompt": _rate(rate_match["prompt"]),
            "generation": _rate(rate_match["generation"]),
            "ttft": None if rate_match["ttft"] == "n/a"
            else _duration(rate_match["ttft"].removesuffix(" s"), 1_000_000_000),
        }
        detail = detail_match.groupdict()
        try:
            record = json.loads(decode_match["json"], parse_constant=_reject_nonfinite)
        except (ValueError, TypeError) as error:
            raise MetricsError("invalid Guffin decode JSON") from error
        if _integer(record, "schema_version") != 6:
            raise MetricsError("unsupported Guffin decode schema")
        prompt_count, cached, selected = (int(detail[name]) for name in ("prompt", "cached", "selected"))
        for field, expected in (
            ("evaluated_prompt_tokens", prompt_count), ("reused_cache_tokens", cached),
            ("selected_tokens", selected),
        ):
            if _integer(record, field) != expected:
                raise MetricsError(f"Guffin {field} disagrees with the detail line")
        native_decoded = _integer(record, "decoded_tokens")
        prompt_ns, prompt_half = _duration(detail["prompt_seconds"], 1_000_000_000)
        generation_ns, generation_half = _duration(detail["generation_seconds"], 1_000_000_000)
        _validate_rate(rates["prompt"], prompt_count, prompt_ns, prompt_half, "Guffin prompt")
        _validate_rate(rates["generation"], selected, generation_ns, generation_half, "Guffin generation")
        if rates["ttft"] is None:
            if prompt_ns:
                raise MetricsError("Guffin omitted a nonzero internal TTFT")
        elif abs(rates["ttft"][0] - prompt_ns) > rates["ttft"][1] + prompt_half:
            raise MetricsError("Guffin TTFT disagrees with the prompt interval")
        completion = _integer(record, "generation_completion_ns")
        full_turn = _integer(record, "full_turn_ns")
        first_text = _integer(record, "first_text_ns")
        if completion > full_turn or first_text > full_turn:
            raise MetricsError("Guffin complete-turn clocks are inconsistent")
        stop_reason = record.get("stop_reason")
        if stop_reason is not None and stop_reason not in (
            "token_limit", "stop_token", "callback_stop", "reverse_stop", "interrupted",
        ):
            raise MetricsError("invalid Guffin stop reason")
        result = _common(prompt_count, cached, selected, prompt_ns, generation_ns, rates)
        result.update(
            stop_reason=stop_reason,
            engine="guffin", native_generation_count_basis="selected_tokens",
            native_decoded_tokens=native_decoded,
            generation_completion_ns=completion, full_turn_ns=full_turn, first_text_ns=first_text,
            timing_resolution_ns={"prompt": int(prompt_half * 2), "generation": int(generation_half * 2)},
            raw_metrics=record,
        )
        self._records = {}
        return result


class LlamaMetricsParser(_MetricsParser):
    required = ("rate", "prompt", "generation", "total")

    def _line(self, stream, line):
        if stream == "stdout" and line.startswith("[ Prompt:"):
            self._records["rate"] = line
            return
        if stream != "stderr":
            return
        match = _SLOT.search(line)
        if not match:
            return
        body = match["body"]
        phase = _PHASE.fullmatch(body)
        total = _TOTAL.fullmatch(body)
        acceptance = _ACCEPTANCE.fullmatch(body)
        new_prompt = _NEW_PROMPT.fullmatch(body)
        if not (phase or total or acceptance or new_prompt):
            if body.startswith(("prompt eval time =", "eval time =", "total time =", "draft acceptance =", "new prompt,")):
                raise MetricsError("malformed llama.cpp timing summary")
            return
        key = (int(match["slot"]), int(match["task"]))
        previous = self._records.get("slot_task")
        if previous is not None and previous != key:
            raise MetricsError("llama.cpp interleaved different slot/task receipts")
        self._records["slot_task"] = key
        if phase:
            self._set("prompt" if phase["phase"] == "prompt eval" else "generation", phase.groupdict())
        elif total:
            self._set("total", total.groupdict())
        elif acceptance:
            self._set("acceptance", acceptance.groupdict())
        else:
            self._set("new_prompt", new_prompt.groupdict())

    def end_turn(self):
        self._require_complete()
        prompt, generation, total = (self._records[name]
                                      for name in ("prompt", "generation", "total"))
        match = _LLAMA_RATE.fullmatch(self._records["rate"])
        if not match:
            raise MetricsError("malformed final llama.cpp rate line")
        rates = {name: _rate(match[name]) for name in ("prompt", "generation")}
        prompt_count, selected = int(prompt["count"]), int(generation["count"])
        steps = max(selected - 1, 0)
        prompt_ns, prompt_half = _duration(prompt["ms"], 1_000_000)
        generation_ns, generation_half = _duration(generation["ms"], 1_000_000)
        total_ns, total_half = _duration(total["ms"], 1_000_000)
        if int(total["count"]) != prompt_count + selected:
            raise MetricsError("llama.cpp total token count disagrees with its phases")
        if abs(total_ns - prompt_ns - generation_ns) > total_half + prompt_half + generation_half:
            raise MetricsError("llama.cpp total duration disagrees with its phases")
        for name, count, ns, half in (
            ("prompt", prompt_count, prompt_ns, prompt_half),
            ("generation", steps, generation_ns, generation_half),
        ):
            phase = prompt if name == "prompt" else generation
            per_ns, per_half = _duration(phase["per_token"], 1_000_000)
            if ((count and abs(ns - per_ns * count) > half + per_half * count)
                    or (not count and per_ns != 0)):
                raise MetricsError(f"llama.cpp {name} per-token duration disagrees")
            _validate_rate(Decimal(phase["rate"]), count, ns, half, f"llama.cpp {name} summary")
            _validate_rate(rates[name], count, ns, half, f"llama.cpp {name} UI")
        full_prompt = self._records.get("new_prompt")
        total_prompt = int(full_prompt["count"]) if full_prompt else None
        if total_prompt is not None and total_prompt < prompt_count:
            raise MetricsError("llama.cpp processed more tokens than its full prompt")
        cached = total_prompt - prompt_count if total_prompt is not None else None
        result = _common(prompt_count, cached, selected, prompt_ns, generation_ns, rates)
        result.update(
            engine="llama", native_generation_count_basis="generation_steps",
            native_total_ns=total_ns, slot_id=self._records["slot_task"][0],
            task_id=self._records["slot_task"][1], generation_completion_ns=None,
            full_turn_ns=None, first_text_ns=None,
            timing_resolution_ns={"prompt": int(prompt_half * 2), "generation": int(generation_half * 2)},
            total_prompt_tokens=total_prompt,
            context_capacity=int(full_prompt["context"]) if full_prompt else None,
            raw_metrics={**self._records, "rate": {
                key: str(value) if value is not None else None for key, value in rates.items()
            }},
            draft_proposed_tokens=None, draft_accepted_tokens=None, draft_acceptance=None,
            native_mean_accepted_length=None,
        )
        if "acceptance" in self._records:
            acceptance = self._records["acceptance"]
            proposed, accepted = int(acceptance["proposed"]), int(acceptance["accepted"])
            ratio = Decimal(acceptance["ratio"])
            half = Decimal(1).scaleb(ratio.as_tuple().exponent) / 2
            if proposed == 0 or accepted > proposed or abs(ratio - Decimal(accepted) / proposed) > half:
                raise MetricsError("llama.cpp draft acceptance counts and ratio disagree")
            result.update(
                draft_proposed_tokens=proposed, draft_accepted_tokens=accepted,
                draft_acceptance=accepted / proposed,
                native_mean_accepted_length=float(acceptance["mean"]),
            )
        self._records = {}
        return result
