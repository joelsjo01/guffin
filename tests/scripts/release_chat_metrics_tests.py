import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from release_chat_metrics import (
    GuffinMetricsParser, LlamaMetricsParser, MetricsError,
    clean_terminal_text, encode_prompt, extract_answer, is_prompt_ready,
)


GUFFIN_RATE = "[ Prompt: 10.0 t/s | Generation: 11.0 t/s | TTFT: 1.000000 s ]"
LLAMA_RATE = "[ Prompt: 10.0 t/s | Generation: 10.0 t/s ]"


def guffin_receipt(**changes):
    record = {
        "schema_version": 6, "evaluated_prompt_tokens": 10,
        "reused_cache_tokens": 20, "selected_tokens": 11, "decoded_tokens": 11,
        "generation_completion_ns": 2_100_000_000,
        "full_turn_ns": 2_200_000_000, "first_text_ns": 1_200_000_000,
    }
    record.update(changes)
    return (
        GUFFIN_RATE + "\n"
        "[ Detail: Evaluated: 10 | Cached: 20 | Decoded: 11 | Prompt: 1.000000 s"
        " | Generation: 1.000000 s ]\n"
        "[ Decode: " + json.dumps(record) + " ]\n"
    )


def llama_receipt(slot=0, task=7):
    prefix = f"slot print_timing: id {slot} | task {task} | "
    return (
        prefix + "prompt eval time =    1000.00 ms /    10 tokens"
        " (  100.00 ms per token,    10.00 tokens per second)\n"
        + prefix + "       eval time =    1000.00 ms /    11 tokens"
        " (  100.00 ms per token,    10.00 tokens per second)\n"
        + prefix + "      total time =    2000.00 ms /    21 tokens\n"
        + prefix + "   graphs reused =          4\n"
    )


def feed_fragments(parser, stream, text, width=1):
    for offset in range(0, len(text), width):
        parser.feed(stream, text[offset:offset + width])


class GuffinMetricsTests(unittest.TestCase):
    def test_fragmented_diagnostics_preserve_native_and_common_counts(self):
        parser = GuffinMetricsParser()
        feed_fragments(parser, "stdout", "answer\r\n\x1b[32m" + guffin_receipt() + "\x1b[0m> ")
        self.assertTrue(parser.can_finish_turn)
        result = parser.end_turn()
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["cached_tokens"], 20)
        self.assertEqual(result["selected_tokens"], 11)
        self.assertEqual(result["generation_steps"], 10)
        self.assertEqual(result["native_generation_tps"], 11.0)
        self.assertEqual(result["generation_tps"], 10.0)
        self.assertEqual(result["internal_ttft_ns"], 1_000_000_000)
        self.assertEqual(result["engine_turn_ns"], 2_000_000_000)
        self.assertIsNone(result["stop_reason"])
        json.dumps(result)
        self.assertFalse(parser.pending)
        parser.finish()

    def test_two_turns_reset_without_losing_stderr_metrics(self):
        parser = GuffinMetricsParser()
        for cached in (20, 21):
            text = guffin_receipt(reused_cache_tokens=cached).replace(
                "Cached: 20", f"Cached: {cached}")
            parser.feed("stderr", text)
            self.assertEqual(parser.end_turn()["cached_tokens"], cached)
        parser.finish()

    def test_missing_duplicate_and_malformed_records_fail(self):
        for text in (
            GUFFIN_RATE + "\n",
            "[ Decode: {broken} ]\n", "[ Decode: {\"a\": NaN} ]\n",
            "[ Prompt: broken ]\n",
        ):
            with self.subTest(text=text), self.assertRaises(MetricsError):
                parser = GuffinMetricsParser()
                parser.feed("stdout", text)
                parser.end_turn()
        parser = GuffinMetricsParser()
        parser.feed("stdout", GUFFIN_RATE + "\n")
        with self.assertRaises(MetricsError):
            parser.finish()

    def test_attribution_counts_schema_and_turn_clocks_are_checked(self):
        for changes in (
            {"schema_version": 5}, {"schema_version": 6.0}, {"selected_tokens": True},
            {"selected_tokens": 12}, {"decoded_tokens": True},
            {"evaluated_prompt_tokens": -1}, {"reused_cache_tokens": 21},
            {"generation_completion_ns": 2_300_000_000},
            {"first_text_ns": 2_300_000_000}, {"full_turn_ns": None},
            {"stop_reason": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(MetricsError):
                parser = GuffinMetricsParser()
                parser.feed("stdout", guffin_receipt(**changes))
                parser.end_turn()

    def test_native_rate_basis_and_ttft_must_agree(self):
        for before, after in (("11.0 t/s", "10.0 t/s"),
                              ("TTFT: 1.000000 s", "TTFT: 0.999000 s")):
            parser = GuffinMetricsParser()
            parser.feed("stdout", guffin_receipt().replace(before, after))
            with self.assertRaises(MetricsError):
                parser.end_turn()

    def test_independent_native_decoded_count_is_preserved(self):
        parser = GuffinMetricsParser()
        parser.feed("stdout", guffin_receipt(decoded_tokens=10))
        result = parser.end_turn()
        self.assertEqual(result["native_decoded_tokens"], 10)
        self.assertEqual(result["selected_tokens"], 11)

    def test_answer_metric_lookalikes_do_not_replace_final_native_block(self):
        parser = GuffinMetricsParser()
        parser.feed("stdout", "[ Prompt: broken example ]\n[ Detail: example ]\n[ Decode: {broken} ]\n")
        parser.feed("stdout", guffin_receipt())
        self.assertEqual(parser.end_turn()["selected_tokens"], 11)
        parser = GuffinMetricsParser()
        parser.feed("stdout", guffin_receipt() + "[ Decode: {broken} ]\n")
        with self.assertRaises(MetricsError):
            parser.end_turn()

    def test_answer_detail_cannot_fill_a_missing_final_native_detail(self):
        parser = GuffinMetricsParser()
        prior_detail = guffin_receipt().splitlines()[1]
        parser.feed("stdout", prior_detail + "\n")
        final = "\n".join(line for line in guffin_receipt().splitlines()
                          if not line.startswith("[ Detail:")) + "\n"
        parser.feed("stdout", final)
        self.assertFalse(parser.can_finish_turn)
        with self.assertRaises(MetricsError):
            parser.end_turn()

    def test_additive_stop_reason_is_preserved(self):
        for reason in ("token_limit", "stop_token", "reverse_stop", "interrupted", "callback_stop"):
            parser = GuffinMetricsParser()
            parser.feed("stdout", guffin_receipt(stop_reason=reason))
            self.assertEqual(parser.end_turn()["stop_reason"], reason)

    def test_one_selected_token_has_no_common_generation_steps(self):
        text = guffin_receipt(selected_tokens=1, decoded_tokens=1)
        text = text.replace("Decoded: 11", "Decoded: 1").replace(
            "Generation: 1.000000 s", "Generation: 0.000001 s").replace(
            "11.0 t/s", "1000000.0 t/s")
        parser = GuffinMetricsParser()
        parser.feed("stdout", text)
        result = parser.end_turn()
        self.assertEqual(result["generation_steps"], 0)
        self.assertEqual(result["generation_tps"], 0.0)
        self.assertEqual(result["native_generation_tps"], 1_000_000.0)


class LlamaMetricsTests(unittest.TestCase):
    def parser_for(self, stderr=None, stdout=LLAMA_RATE + "\n"):
        parser = LlamaMetricsParser()
        parser.feed("stdout", stdout)
        parser.feed("stderr", llama_receipt() if stderr is None else stderr)
        return parser

    def test_fragmented_two_stream_timing_receipt(self):
        parser = LlamaMetricsParser()
        feed_fragments(parser, "stderr", "\x1b[0mI " + llama_receipt(), 3)
        self.assertFalse(parser.can_finish_turn)
        feed_fragments(parser, "stdout", "quoted > text\n" + LLAMA_RATE + "\r\n> ", 2)
        self.assertTrue(parser.can_finish_turn)
        result = parser.end_turn()
        self.assertEqual(result["selected_tokens"], 11)
        self.assertEqual(result["generation_steps"], 10)
        self.assertEqual(result["generation_tps"], 10.0)
        self.assertEqual(result["native_generation_tps"], 10.0)
        self.assertEqual(result["internal_ttft_ns"], 1_000_000_000)
        self.assertEqual(result["native_total_ns"], 2_000_000_000)
        self.assertIsNone(result["cached_tokens"])
        self.assertIsNone(result["stop_reason"])
        self.assertIsNone(result["draft_acceptance"])
        json.dumps(result)
        parser.finish()

    def test_answer_rate_lookalikes_are_staged_and_final_line_still_validated(self):
        parser = self.parser_for(stdout="[ Prompt: broken example ]\n" + LLAMA_RATE + "\n")
        self.assertEqual(parser.end_turn()["selected_tokens"], 11)
        with self.assertRaises(MetricsError):
            self.parser_for(stdout=LLAMA_RATE + "\n[ Prompt: broken final line ]\n").end_turn()

    def test_optional_trace_recovers_full_and_cached_prompt_counts(self):
        trace = "slot update_slots: id 0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = -1, task.n_tokens = 30\n"
        result = self.parser_for(trace + llama_receipt()).end_turn()
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["total_prompt_tokens"], 30)
        self.assertEqual(result["cached_tokens"], 20)
        self.assertEqual(result["context_capacity"], 4096)
        with self.assertRaises(MetricsError):
            self.parser_for(trace.replace("tokens = 30", "tokens = 9") + llama_receipt()).end_turn()

    def test_optional_acceptance_is_collected_at_end_not_first_completeness(self):
        parser = self.parser_for()
        self.assertTrue(parser.can_finish_turn)
        parser.feed("stderr", "slot print_timing: id 0 | task 7 | draft acceptance = 0.70000 (    7 accepted /    10 generated), mean len =  1.70\n")
        result = parser.end_turn()
        self.assertEqual(result["draft_proposed_tokens"], 10)
        self.assertEqual(result["draft_accepted_tokens"], 7)
        self.assertEqual(result["draft_acceptance"], 0.7)
        self.assertEqual(result["native_mean_accepted_length"], 1.7)
        for ratio, accepted, proposed in (("0.60000", 7, 10), ("1.10000", 11, 10), ("0.00000", 0, 0)):
            parser = self.parser_for()
            parser.feed("stderr", f"slot print_timing: id 0 | task 7 | draft acceptance = {ratio} ({accepted} accepted / {proposed} generated), mean len = 1.00\n")
            with self.assertRaises(MetricsError):
                parser.end_turn()

    def test_next_turn_can_use_a_different_task_after_reset(self):
        parser = self.parser_for()
        self.assertEqual(parser.end_turn()["task_id"], 7)
        parser.feed("stderr", llama_receipt(task=8))
        parser.feed("stdout", LLAMA_RATE + "\n")
        self.assertEqual(parser.end_turn()["task_id"], 8)

    def test_rounding_tolerance_is_derived_from_printed_precision(self):
        text = llama_receipt().replace("total time =    2000.00", "total time =    2000.01")
        self.assertEqual(self.parser_for(text).end_turn()["native_total_ns"], 2_000_010_000)
        with self.assertRaises(MetricsError):
            self.parser_for(text.replace("2000.01", "2000.02")).end_turn()
        # 0.001 ms is rounded to 0.00 ms by the native summary for a single token.
        text = llama_receipt().replace("eval time =    1000.00 ms /    11 tokens", "eval time =       0.00 ms /     1 tokens")
        lines = text.splitlines()
        lines[1] = lines[1].replace("100.00 ms per token", "0.00 ms per token").replace("10.00 tokens per second", "0.00 tokens per second")
        lines[2] = lines[2].replace("2000.00", "1000.00").replace("21 tokens", "11 tokens")
        result = self.parser_for("\n".join(lines) + "\n", LLAMA_RATE.replace("Generation: 10.0", "Generation: 0.0") + "\n").end_turn()
        self.assertEqual(result["generation_steps"], 0)
        self.assertEqual(result["generation_ns"], 0)
        self.assertEqual(result["generation_tps"], 0.0)

    def test_missing_duplicate_inconsistent_and_mixed_task_receipts_fail(self):
        variants = (
            llama_receipt().replace("total time =    2000.00 ms /    21 tokens\n", ""),
            llama_receipt() + llama_receipt(),
            llama_receipt().replace("21 tokens", "20 tokens"),
            llama_receipt().replace("100.00 ms per token", "90.00 ms per token"),
            llama_receipt().replace("10.00 tokens per second", "11.00 tokens per second"),
            llama_receipt().replace("eval time =    1000.00 ms /    11", "eval time = broken / 11"),
            llama_receipt().replace("task 7 |        eval", "task 8 |        eval"),
        )
        for text in variants:
            with self.subTest(text=text), self.assertRaises(MetricsError):
                self.parser_for(text).end_turn()
        parser = LlamaMetricsParser()
        parser.feed("stdout", LLAMA_RATE + "\n")
        with self.assertRaises(MetricsError):
            parser.finish()


class TerminalProtocolTests(unittest.TestCase):
    @staticmethod
    def submitted_text(engine, data):
        # Mirror the documented frontend line protocol, then llama's final LF removal.
        result = ""
        for line in data.decode("utf-8").splitlines():
            if engine == "llama" and line.endswith("/"):
                result += line[:-1]
                break
            if line.endswith("\\"):
                result += line[:-1] + "\n"
                continue
            result += line + ("\n" if engine == "llama" else "")
            break
        return result[:-1] if engine == "llama" and result.endswith("\n") else result

    def test_exact_unicode_multiline_and_trailing_newline_submission(self):
        prompts = ("Olá 世界", "first\nsecond", "\nfirst\n\nlast\n", "path/\ncode\\\nend", "x\n\n")
        for engine in ("guffin", "llama"):
            for prompt in prompts:
                with self.subTest(engine=engine, prompt=prompt):
                    self.assertEqual(self.submitted_text(engine, encode_prompt(engine, prompt)), prompt)
        for prompt in ("path/", "literal\\", "path//", " "):
            self.assertEqual(self.submitted_text("llama", encode_prompt("llama", prompt)), prompt)

    def test_unrepresentable_controls_commands_and_guffin_final_backslash_fail(self):
        for engine in ("guffin", "llama"):
            for prompt in ("", "/exit", "x\tx", "x\rx", "x\x00x", "x\x1bx", "x\x7fx"):
                with self.subTest(engine=engine, prompt=prompt), self.assertRaises(ValueError):
                    encode_prompt(engine, prompt)
        with self.assertRaises(ValueError):
            encode_prompt("guffin", "literal\\")

    def test_terminal_controls_and_final_prompt_detection(self):
        self.assertEqual(clean_terminal_text("a\\\b\\\b \b\r\nb\x1b[0m"), "a\nb")
        self.assertEqual(clean_terminal_text("\x1b]0;title\x07text"), "text")
        self.assertEqual(clean_terminal_text("\x1b]0;title\x1b\\text\x1b]0;next\x1b\\"), "text")
        self.assertTrue(is_prompt_ready("answer\r\n\x1b[32m> \x1b[0m"))
        for text in ("answer > ", "\n> quoted", "\n> \n", "\n> \nmore"):
            self.assertFalse(is_prompt_ready(text))

    def test_guffin_removes_only_the_known_continuation_prefix(self):
        answer = "... is also prose\n> quoted\n" + GUFFIN_RATE + "\nactual answer\n"
        stdout = "... ... " + answer + guffin_receipt() + "> "
        self.assertEqual(extract_answer("guffin", stdout, "one\ntwo\nthree"), answer)

    def test_llama_removes_exact_echo_controls_and_native_timing_separator(self):
        answer = "An answer\n> is text\n" + LLAMA_RATE + "\nmore\n"
        stdout = "first\\\b\\\b \b\r\nsecond\r\n" + answer + "\n" + LLAMA_RATE + "\n\n> "
        self.assertEqual(extract_answer("llama", stdout, "first\nsecond"), answer)
        stdout = "path//\b \b" + answer + "\n" + LLAMA_RATE + "\n\n> "
        self.assertEqual(extract_answer("llama", stdout, "path/"), answer)
        with self.assertRaises(MetricsError):
            extract_answer("llama", "wrong\nanswer\n\n" + LLAMA_RATE + "\n> ", "expected")
        with self.assertRaises(MetricsError):
            extract_answer("guffin", "no metadata", "prompt")


if __name__ == "__main__":
    unittest.main()
