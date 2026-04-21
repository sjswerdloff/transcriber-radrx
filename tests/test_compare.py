"""Tests for the compare subcommand and supporting functions.

Uses the sample data from docs/demo/ as realistic test fixtures:
- sample_gold_standard.txt: "Prescribed dose of 54 Gy in 30 fractions..."
- sample_dictation.txt: "prescribed dose of 54 Gy in 30 fractions..."
  (lowercase first word — the real ASR output from the GUI test)

No ASR models required. All tests are text-only.

Authors: silas-397300f6
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber_radrx.cli import _normalize_for_wer, _read_text_input

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEMO_DIR = _REPO_ROOT / "docs" / "demo"
_GOLD_PATH = _DEMO_DIR / "sample_gold_standard.txt"
_TRANSCRIPTION_PATH = _DEMO_DIR / "sample_dictation.txt"


# ---------------------------------------------------------------------------
# _normalize_for_wer
# ---------------------------------------------------------------------------


class TestNormalizeForWer:
    """Contract tests for whitespace normalization before WER."""

    def test_strips_trailing_newline(self) -> None:
        """Contract: trailing newline is removed."""
        assert _normalize_for_wer("hello world\n") == "hello world"

    def test_strips_leading_whitespace(self) -> None:
        """Contract: leading whitespace is removed."""
        assert _normalize_for_wer("  hello world") == "hello world"

    def test_collapses_internal_whitespace(self) -> None:
        """Contract: multiple internal spaces collapse to one."""
        assert _normalize_for_wer("hello   world") == "hello world"

    def test_preserves_case(self) -> None:
        """Contract: case is NOT changed (mGy vs MGy matters)."""
        assert _normalize_for_wer("50 mGy") == "50 mGy"
        assert _normalize_for_wer("50 MGy") == "50 MGy"
        assert _normalize_for_wer("50 mGy") != _normalize_for_wer("50 MGy")

    def test_empty_string(self) -> None:
        """Contract: empty string returns empty string."""
        assert _normalize_for_wer("") == ""

    def test_whitespace_only(self) -> None:
        """Contract: whitespace-only string returns empty string."""
        assert _normalize_for_wer("   \n  \t  ") == ""

    def test_tabs_and_newlines_collapse(self) -> None:
        """Contract: tabs and newlines treated as whitespace."""
        assert _normalize_for_wer("hello\tworld\nfoo") == "hello world foo"


# ---------------------------------------------------------------------------
# _read_text_input
# ---------------------------------------------------------------------------


class TestReadTextInput:
    """Contract tests for reading .txt and .docx files."""

    def test_reads_txt_file(self, tmp_path: Path) -> None:
        """Contract: reads plain text from .txt file."""
        txt = tmp_path / "test.txt"
        txt.write_text("Hello world.", encoding="utf-8")
        assert _read_text_input(txt) == "Hello world."

    def test_strips_whitespace_from_txt(self, tmp_path: Path) -> None:
        """Contract: strips leading/trailing whitespace from .txt."""
        txt = tmp_path / "test.txt"
        txt.write_text("  Hello world.  \n", encoding="utf-8")
        assert _read_text_input(txt) == "Hello world."

    def test_reads_docx_file(self, tmp_path: Path) -> None:
        """Contract: extracts paragraph text from .docx file."""
        pytest.importorskip("docx")
        from docx import Document

        docx_path = tmp_path / "test.docx"
        doc = Document()
        doc.add_paragraph("Hello world.")
        doc.save(str(docx_path))

        result = _read_text_input(docx_path)
        assert "Hello world." in result

    def test_raises_on_missing_file(self) -> None:
        """Contract: FileNotFoundError on nonexistent path."""
        with pytest.raises(FileNotFoundError):
            _read_text_input(Path("/nonexistent/file.txt"))

    def test_raises_on_corrupt_docx(self, tmp_path: Path) -> None:
        """Contract: ValueError on unparseable .docx."""
        bad_docx = tmp_path / "bad.docx"
        bad_docx.write_text("this is not a docx", encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot parse .docx"):
            _read_text_input(bad_docx)


# ---------------------------------------------------------------------------
# compare subcommand — end-to-end with sample data
# ---------------------------------------------------------------------------


class TestCompareWithSampleData:
    """End-to-end tests using the demo sample files.

    Requires docs/demo/sample_gold_standard.txt and
    docs/demo/sample_dictation.txt to exist.
    """

    @pytest.fixture(autouse=True)
    def _check_sample_files(self) -> None:
        if not _GOLD_PATH.exists() or not _TRANSCRIPTION_PATH.exists():
            pytest.skip("Sample demo files not present")

    def test_single_transcription_produces_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: compare with one transcription prints corrected text to stdout."""
        from transcriber_radrx.cli import main

        main(["compare", "--gold", str(_GOLD_PATH), "--transcription", str(_TRANSCRIPTION_PATH)])
        captured = capsys.readouterr()
        assert "54 Gy" in captured.out or "54 gy" in captured.out.lower()

    def test_single_transcription_reports_wer(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: compare prints WER to stderr."""
        from transcriber_radrx.cli import main

        main(["compare", "--gold", str(_GOLD_PATH), "--transcription", str(_TRANSCRIPTION_PATH)])
        captured = capsys.readouterr()
        assert "Raw WER" in captured.err
        assert "Corrected WER" in captured.err

    def test_wer_is_low_on_near_perfect_transcription(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: WER on the sample data is below 10% (only case difference)."""
        import re

        from transcriber_radrx.cli import main

        main(["compare", "--gold", str(_GOLD_PATH), "--transcription", str(_TRANSCRIPTION_PATH)])
        captured = capsys.readouterr()
        # Parse raw WER from stderr
        match = re.search(r"Raw WER\s*:\s*(\d+\.\d+)", captured.err)
        assert match is not None, f"Could not find Raw WER in stderr: {captured.err}"
        raw_wer = float(match.group(1))
        assert raw_wer < 0.10, f"WER {raw_wer} unexpectedly high for near-perfect transcription"

    def test_term_recall_reported(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: compare reports term recall."""
        from transcriber_radrx.cli import main

        main(["compare", "--gold", str(_GOLD_PATH), "--transcription", str(_TRANSCRIPTION_PATH)])
        captured = capsys.readouterr()
        assert "Term recall" in captured.err

    def test_two_identical_transcriptions_give_zero_uwr(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Contract: ensemble of two identical transcriptions has 0% UWR."""
        from transcriber_radrx.cli import main

        # Use gold standard as both transcriptions — perfect agreement
        main(
            [
                "compare",
                "--gold",
                str(_GOLD_PATH),
                "--transcription",
                str(_GOLD_PATH),
                "--transcription-b",
                str(_GOLD_PATH),
                "--output",
                str(tmp_path / "review.docx"),
            ]
        )
        captured = capsys.readouterr()
        assert "UWR" in captured.err
        assert "0.0000" in captured.err or "0.0%" in captured.err

    def test_two_transcriptions_produce_docx(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: ensemble compare creates a review .docx."""
        from transcriber_radrx.cli import main

        output = tmp_path / "review.docx"
        main(
            [
                "compare",
                "--gold",
                str(_GOLD_PATH),
                "--transcription",
                str(_TRANSCRIPTION_PATH),
                "--transcription-b",
                str(_GOLD_PATH),
                "--output",
                str(output),
            ]
        )
        assert output.exists()
        assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# _run_compare_core (GUI pipeline) with sample data
# ---------------------------------------------------------------------------


class TestRunCompareCoreWithSampleData:
    """Test the GUI's _run_compare_core function with real sample files."""

    @pytest.fixture(autouse=True)
    def _check_sample_files(self) -> None:
        if not _GOLD_PATH.exists() or not _TRANSCRIPTION_PATH.exists():
            pytest.skip("Sample demo files not present")

    def test_returns_low_wer(self) -> None:
        """Contract: WER on sample data is below 10%."""
        from transcriber_radrx.gui import _run_compare_core

        results = _run_compare_core(_GOLD_PATH, _TRANSCRIPTION_PATH, None, None)
        raw_wer = float(str(results["raw_wer"]))
        assert raw_wer < 0.10

    def test_corrections_applied(self) -> None:
        """Contract: result dict contains correction counts."""
        from transcriber_radrx.gui import _run_compare_core

        results = _run_compare_core(_GOLD_PATH, _TRANSCRIPTION_PATH, None, None)
        assert "phrase_corrections" in results
        assert "word_corrections" in results

    def test_term_recall_present_without_vocab(self) -> None:
        """Contract: result dict contains term recall fields (zero without vocabulary)."""
        from transcriber_radrx.gui import _run_compare_core

        results = _run_compare_core(_GOLD_PATH, _TRANSCRIPTION_PATH, None, None)
        assert "terms_found" in results
        assert "terms_total" in results
        # Without vocabulary, terms_total is 0 (no terms to check)
        assert int(str(results["terms_total"])) == 0

    def test_term_recall_with_vocabulary(self) -> None:
        """Contract: term recall is nonzero when vocabulary is provided."""
        from transcriber_radrx.gui import _run_compare_core

        vocab_path = _REPO_ROOT / "data" / "rt_vocabulary.txt"
        if not vocab_path.exists():
            pytest.skip("Vocabulary file not present")
        results = _run_compare_core(_GOLD_PATH, _TRANSCRIPTION_PATH, None, vocab_path)
        assert int(str(results["terms_total"])) > 0
        assert int(str(results["terms_found"])) > 0

    def test_two_transcriptions_produce_ensemble(self) -> None:
        """Contract: providing two transcriptions yields ensemble fields."""
        from transcriber_radrx.gui import _run_compare_core

        results = _run_compare_core(_GOLD_PATH, _TRANSCRIPTION_PATH, _GOLD_PATH, None)
        assert "ensemble_wer" in results
        assert "uwr" in results
        assert "review_count" in results
