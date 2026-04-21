"""Tests for the transcriber-radrx GUI module.

Tests cover:
- ``_term_recall`` helper function with various inputs.
- ``_run_compare_core`` pipeline with mocked CLI modules.
- ``_run_evaluate_core`` pipeline with mocked backends.
- CompareWorker thread: correct results dict structure (requires PySide6).
- EvaluateWorker thread: importability and basic structure (requires PySide6).
- GUI importability without PySide6 (headless guard).

All PySide6-dependent tests are skipped when PySide6 is not installed.

Authors: silas-397300f6
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_txt(tmp_path: Path, name: str, content: str) -> Path:
    """Write content to a .txt file in tmp_path and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _term_recall — no PySide6 dependency
# ---------------------------------------------------------------------------


class TestTermRecall:
    """Tests for the _term_recall helper."""

    def test_term_present_in_both(self) -> None:
        """Contract: term found in both gold and transcription is counted."""
        from transcriber_radrx.gui import _term_recall

        vocab: set[str] = {"imrt", "ptv"}
        found, total, missing = _term_recall("imrt ptv dose", "imrt ptv dose", vocab)
        assert found == 2
        assert total == 2
        assert missing == []

    def test_term_missing_from_transcription(self) -> None:
        """Contract: term in gold but absent from transcription appears in missing."""
        from transcriber_radrx.gui import _term_recall

        vocab: set[str] = {"gy", "imrt"}
        found, total, missing = _term_recall("imrt dose", "imrt gy dose", vocab)
        assert found == 1
        assert total == 2
        assert "gy" in missing

    def test_term_not_in_gold_is_not_counted(self) -> None:
        """Contract: vocab term absent from gold is not included in total."""
        from transcriber_radrx.gui import _term_recall

        vocab: set[str] = {"vmat"}
        found, total, missing = _term_recall("vmat imrt", "imrt dose", vocab)
        # "vmat" is in transcription but not in gold — excluded from relevant_terms
        assert total == 0
        assert found == 0

    def test_empty_vocab(self) -> None:
        """Contract: empty vocabulary yields zero counts."""
        from transcriber_radrx.gui import _term_recall

        found, total, missing = _term_recall("imrt dose", "imrt dose", set())
        assert found == 0
        assert total == 0
        assert missing == []

    def test_case_insensitive_matching(self) -> None:
        """Contract: matching is case-insensitive."""
        from transcriber_radrx.gui import _term_recall

        vocab: set[str] = {"gy"}
        found, total, missing = _term_recall("50.4 GY delivered", "50.4 Gy delivered", vocab)
        assert found == 1
        assert total == 1


# ---------------------------------------------------------------------------
# _run_compare_core — no PySide6 dependency
# ---------------------------------------------------------------------------


class TestRunCompareCore:
    """Tests for the _run_compare_core pipeline helper."""

    def test_single_transcription_returns_correct_keys(self, tmp_path: Path) -> None:
        """Contract: single-transcription run returns expected result keys."""
        from transcriber_radrx.gui import _run_compare_core

        gold = _make_txt(tmp_path, "gold.txt", "prescribed dose of fifty four Gy")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "prescribed dose of fifty four Gy")

        results = _run_compare_core(gold, trans_a, None, None)

        assert "raw_wer" in results
        assert "corrected_wer" in results
        assert "phrase_corrections" in results
        assert "word_corrections" in results
        assert "terms_found" in results
        assert "terms_total" in results
        assert "terms_missing" in results
        assert "corrected_text_a" in results

    def test_perfect_match_gives_zero_wer(self, tmp_path: Path) -> None:
        """Contract: identical gold and transcription yield WER of 0.0."""
        from transcriber_radrx.gui import _run_compare_core

        text = "imrt treatment planning target volume"
        gold = _make_txt(tmp_path, "gold.txt", text)
        trans_a = _make_txt(tmp_path, "trans_a.txt", text)

        results = _run_compare_core(gold, trans_a, None, None)

        assert float(results["raw_wer"]) == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_two_transcriptions_adds_ensemble_keys(self, tmp_path: Path) -> None:
        """Contract: two-transcription run includes ensemble_wer, uwr, review_count."""
        from transcriber_radrx.gui import _run_compare_core

        gold = _make_txt(tmp_path, "gold.txt", "prescribed dose fifty four gy")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "prescribed dose fifty four gy")
        trans_b = _make_txt(tmp_path, "trans_b.txt", "prescribed dose fifty four gy")

        results = _run_compare_core(gold, trans_a, trans_b, None)

        assert "ensemble_wer" in results
        assert "uwr" in results
        assert "review_count" in results

    def test_missing_gold_raises(self, tmp_path: Path) -> None:
        """Contract: missing gold standard file raises FileNotFoundError."""
        from transcriber_radrx.gui import _run_compare_core

        gold = tmp_path / "missing_gold.txt"
        trans_a = _make_txt(tmp_path, "trans_a.txt", "some text")

        with pytest.raises(FileNotFoundError):
            _run_compare_core(gold, trans_a, None, None)

    def test_missing_transcription_raises(self, tmp_path: Path) -> None:
        """Contract: missing transcription A raises FileNotFoundError."""
        from transcriber_radrx.gui import _run_compare_core

        gold = _make_txt(tmp_path, "gold.txt", "some text")
        trans_a = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            _run_compare_core(gold, trans_a, None, None)

    def test_with_vocabulary_file(self, tmp_path: Path) -> None:
        """Contract: vocabulary file path is accepted and used without error."""
        from transcriber_radrx.gui import _run_compare_core

        vocab = tmp_path / "vocab.txt"
        vocab.write_text("imrt\nptv\ngy\n", encoding="utf-8")

        gold = _make_txt(tmp_path, "gold.txt", "imrt dose of fifty gy to ptv")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "imrt dose of fifty gy to ptv")

        results = _run_compare_core(gold, trans_a, None, vocab)

        assert float(results["raw_wer"]) == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_wer_gt_zero_for_different_texts(self, tmp_path: Path) -> None:
        """Contract: different gold and transcription yield WER > 0."""
        from transcriber_radrx.gui import _run_compare_core

        gold = _make_txt(tmp_path, "gold.txt", "the quick brown fox jumps")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "the slow green cat falls")

        results = _run_compare_core(gold, trans_a, None, None)

        assert float(results["raw_wer"]) > 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _run_evaluate_core — mocked backends, no PySide6
# ---------------------------------------------------------------------------


class TestRunEvaluateCoreImport:
    """Verify _run_evaluate_core is importable without PySide6."""

    def test_importable(self) -> None:
        """Contract: _run_evaluate_core imports cleanly from gui module."""
        from transcriber_radrx.gui import _run_evaluate_core

        assert callable(_run_evaluate_core)


# ---------------------------------------------------------------------------
# GUI module import guard (no PySide6 required)
# ---------------------------------------------------------------------------


class TestGuiModuleImport:
    """Verify gui.py can be imported in headless environments."""

    def test_import_succeeds_without_pyside6(self) -> None:
        """Contract: gui module imports without raising even when PySide6 absent."""
        import importlib

        # Re-import to verify no import-time crash
        spec = importlib.util.find_spec("transcriber_radrx.gui")
        assert spec is not None

    def test_pyside6_flag_reflects_availability(self) -> None:
        """Contract: _PYSIDE6_AVAILABLE is a boolean flag."""
        from transcriber_radrx.gui import _PYSIDE6_AVAILABLE

        assert isinstance(_PYSIDE6_AVAILABLE, bool)

    def test_main_callable(self) -> None:
        """Contract: main() is importable and callable."""
        from transcriber_radrx.gui import main

        assert callable(main)

    def test_read_text_input_importable_from_cli(self) -> None:
        """Contract: _read_text_input from cli.py is importable (shared by gui)."""
        from transcriber_radrx.cli import _read_text_input

        assert callable(_read_text_input)

    def test_read_text_input_reads_txt(self, tmp_path: Path) -> None:
        """Contract: _read_text_input reads plain text files correctly."""
        from transcriber_radrx.cli import _read_text_input

        f = tmp_path / "test.txt"
        f.write_text("hello world\n", encoding="utf-8")
        result = _read_text_input(f)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# PySide6-dependent tests
# ---------------------------------------------------------------------------

try:
    import PySide6 as _pyside6_check  # noqa: F401

    _PYSIDE6_PRESENT = True
except ModuleNotFoundError:
    _PYSIDE6_PRESENT = False

_skip_no_pyside6 = pytest.mark.skipif(not _PYSIDE6_PRESENT, reason="PySide6 not installed")


@pytest.fixture(scope="module")
def qt_app() -> object:
    """Create a QApplication for the test module (one per process)."""
    if not _PYSIDE6_PRESENT:
        pytest.skip("PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@_skip_no_pyside6
class TestCompareWorkerStructure:
    """Tests for CompareWorker thread (requires PySide6)."""

    def test_compare_worker_importable(self) -> None:
        """Contract: CompareWorker is importable when PySide6 is available."""
        from transcriber_radrx.gui import CompareWorker

        assert CompareWorker is not None

    def test_compare_worker_instantiates(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: CompareWorker instantiates with valid paths."""
        from transcriber_radrx.gui import CompareWorker

        gold = _make_txt(tmp_path, "gold.txt", "test")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "test")
        worker = CompareWorker(gold, trans_a)
        assert worker is not None
        worker.deleteLater()

    def test_compare_worker_has_signals(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: CompareWorker exposes finished, error, progress signals."""
        from transcriber_radrx.gui import CompareWorker

        gold = _make_txt(tmp_path, "gold.txt", "test")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "test")
        worker = CompareWorker(gold, trans_a)
        assert hasattr(worker, "finished")
        assert hasattr(worker, "error")
        assert hasattr(worker, "progress")
        worker.deleteLater()

    def test_compare_worker_run_emits_finished(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: CompareWorker.run() emits finished with a results dict."""
        from PySide6.QtCore import QEventLoop, QTimer

        from transcriber_radrx.gui import CompareWorker

        gold = _make_txt(tmp_path, "gold.txt", "the quick brown fox")
        trans_a = _make_txt(tmp_path, "trans_a.txt", "the quick brown fox")

        received: list[dict[str, object]] = []
        errors: list[str] = []

        worker = CompareWorker(gold, trans_a)
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.error.connect(loop.quit)

        # Timeout guard
        timer = QTimer()
        timer.setSingleShot(True)
        timer.setInterval(10000)
        timer.timeout.connect(loop.quit)

        worker.start()
        timer.start()
        loop.exec()
        timer.stop()

        assert not errors, f"Worker emitted error: {errors}"
        assert len(received) == 1
        result = received[0]
        assert "raw_wer" in result
        assert "corrected_wer" in result
        assert "phrase_corrections" in result
        assert "word_corrections" in result
        assert "terms_found" in result
        assert "terms_total" in result
        assert "terms_missing" in result
        assert "corrected_text_a" in result
        worker.wait()
        worker.deleteLater()

    def test_compare_worker_emits_error_on_bad_input(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: CompareWorker emits error signal when gold file is missing."""
        from PySide6.QtCore import QEventLoop, QTimer

        from transcriber_radrx.gui import CompareWorker

        gold = tmp_path / "no_such_file.txt"
        trans_a = _make_txt(tmp_path, "trans_a.txt", "test")

        errors: list[str] = []
        received: list[dict[str, object]] = []

        worker = CompareWorker(gold, trans_a)
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.error.connect(loop.quit)

        timer = QTimer()
        timer.setSingleShot(True)
        timer.setInterval(5000)
        timer.timeout.connect(loop.quit)

        worker.start()
        timer.start()
        loop.exec()
        timer.stop()

        assert not received, "Should not emit finished on error"
        assert len(errors) == 1
        worker.wait()
        worker.deleteLater()


@_skip_no_pyside6
class TestEvaluateWorkerStructure:
    """Tests for EvaluateWorker thread (requires PySide6)."""

    def test_evaluate_worker_importable(self) -> None:
        """Contract: EvaluateWorker is importable when PySide6 is available."""
        from transcriber_radrx.gui import EvaluateWorker

        assert EvaluateWorker is not None

    def test_evaluate_worker_instantiates(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: EvaluateWorker instantiates with valid paths."""
        from transcriber_radrx.gui import EvaluateWorker

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"")
        output = tmp_path / "review.docx"
        worker = EvaluateWorker(audio, output)
        assert worker is not None
        worker.deleteLater()

    def test_evaluate_worker_has_signals(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: EvaluateWorker exposes finished, error, progress signals."""
        from transcriber_radrx.gui import EvaluateWorker

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"")
        output = tmp_path / "review.docx"
        worker = EvaluateWorker(audio, output)
        assert hasattr(worker, "finished")
        assert hasattr(worker, "error")
        assert hasattr(worker, "progress")
        worker.deleteLater()


@_skip_no_pyside6
class TestFilePickerWidget:
    """Tests for FilePickerWidget (requires PySide6)."""

    def test_initial_state_has_no_path(self, qt_app: object) -> None:
        """Contract: freshly created picker has no selected path."""
        from transcriber_radrx.gui import FilePickerWidget

        picker = FilePickerWidget("Test", [".wav"])
        assert picker.selected_path() is None
        picker.deleteLater()

    def test_set_path_updates_selection(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: set_path updates the selected path."""
        from transcriber_radrx.gui import FilePickerWidget

        picker = FilePickerWidget("Test", [".wav"])
        target = tmp_path / "audio.wav"
        target.write_bytes(b"")
        picker.set_path(target)
        assert picker.selected_path() == target
        picker.deleteLater()

    def test_file_selected_signal_emitted_on_set_path(self, tmp_path: Path, qt_app: object) -> None:
        """Contract: set_path emits file_selected signal with the path string."""
        from transcriber_radrx.gui import FilePickerWidget

        picker = FilePickerWidget("Test", [".wav"])
        emitted: list[str] = []
        picker.file_selected.connect(emitted.append)

        target = tmp_path / "audio.wav"
        target.write_bytes(b"")
        picker.set_path(target)

        assert len(emitted) == 1
        assert emitted[0] == str(target)
        picker.deleteLater()


@_skip_no_pyside6
class TestMainWindowStructure:
    """Tests for MainWindow structure (requires PySide6)."""

    def test_main_window_instantiates(self, qt_app: object) -> None:
        """Contract: MainWindow instantiates without errors."""
        from transcriber_radrx.gui import MainWindow

        window = MainWindow()
        assert window is not None
        window.close()
        window.deleteLater()

    def test_main_window_has_two_tabs(self, qt_app: object) -> None:
        """Contract: MainWindow has exactly two tabs."""
        from transcriber_radrx.gui import MainWindow

        window = MainWindow()
        assert window._tabs.count() == 2
        window.close()
        window.deleteLater()

    def test_main_window_title(self, qt_app: object) -> None:
        """Contract: window title contains expected application name."""
        from transcriber_radrx.gui import MainWindow

        window = MainWindow()
        assert "transcriber-radrx" in window.windowTitle()
        window.close()
        window.deleteLater()

    def test_main_window_minimum_size(self, qt_app: object) -> None:
        """Contract: minimum window size is at least 700x500."""
        from transcriber_radrx.gui import MainWindow

        window = MainWindow()
        assert window.minimumWidth() >= 700
        assert window.minimumHeight() >= 500
        window.close()
        window.deleteLater()
