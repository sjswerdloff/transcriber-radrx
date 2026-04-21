"""PySide6 GUI launcher for transcriber-radrx.

Wraps the ``evaluate`` and ``compare`` CLI subcommands in a two-tab
clinician-facing application.

Tab 1 — Transcribe: pick an audio file, run the dual-backend ensemble
pipeline, save a review .docx, and optionally hand off to the Compare tab.

Tab 2 — Compare: bring a gold standard and one or two transcriptions,
apply corrections, compute WER / term recall / UWR, and save a report.

All heavy work runs in QThread workers so the GUI stays responsive.
PySide6 imports are guarded so the CLI can still be imported in environments
without PySide6 installed (headless CI, servers).

Authors: silas-397300f6
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Guard: fail-fast if PySide6 is missing, but only when the module is
# imported as part of GUI execution, not when the CLI imports it accidentally.
# ---------------------------------------------------------------------------

try:
    from PySide6.QtCore import QElapsedTimer, QThread, QTimer, QUrl, Signal
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _PYSIDE6_AVAILABLE = True
except ModuleNotFoundError:
    _PYSIDE6_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GUIWorkerError(RuntimeError):
    """Raised when a background worker encounters an unrecoverable error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Internal helpers (no PySide6 dependency)
# ---------------------------------------------------------------------------


def _term_recall(text: str, gold_text: str, vocab: set[str]) -> tuple[int, int, list[str]]:
    """Compute term recall of *text* against vocabulary terms present in *gold_text*.

    Args:
        text: The transcription text to evaluate.
        gold_text: The gold-standard text (used to filter relevant vocab terms).
        vocab: Set of lowercase vocabulary terms.

    Returns:
        Tuple of (found_count, total_count, missing_terms).
    """
    text_lower = re.sub(r"[^\w\s.]", " ", text.lower())
    text_lower = " ".join(text_lower.split())
    gold_lower = re.sub(r"[^\w\s.]", " ", gold_text.lower())
    gold_lower = " ".join(gold_lower.split())
    relevant_terms = [t for t in vocab if t in gold_lower]
    found = 0
    missing: list[str] = []
    for term in relevant_terms:
        pattern = r"(?:^|\s)" + re.escape(term) + r"(?:\s|$|[.,;:!?])"
        if re.search(pattern, text_lower):
            found += 1
        else:
            missing.append(term)
    return found, len(relevant_terms), missing


def _run_compare_core(
    gold_path: Path,
    transcription_a_path: Path,
    transcription_b_path: Path | None,
    vocabulary_path: Path | None,
) -> dict[str, object]:
    """Execute the compare pipeline and return a results dict.

    Mirrors the logic of ``_run_compare`` in cli.py but returns a structured
    dict rather than printing to stderr, so the GUI can display results
    without subprocess.

    Args:
        gold_path: Path to gold-standard text file (.txt or .docx).
        transcription_a_path: Path to transcription A file (.txt or .docx).
        transcription_b_path: Optional path to transcription B file.
        vocabulary_path: Optional path to RT vocabulary file.

    Returns:
        Dict with keys: raw_wer, corrected_wer, phrase_corrections,
        word_corrections, terms_found, terms_total, terms_missing,
        ensemble_wer (optional), uwr (optional), review_count (optional),
        corrected_text_a.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If a .docx file cannot be parsed.
        GUIWorkerError: If the compare pipeline fails.
    """
    import jiwer

    from transcriber_radrx.cli import _read_text_input
    from transcriber_radrx.corrector import CorrectionDictionary
    from transcriber_radrx.phrase_corrector import PhraseCorrectorPipeline

    gold_text = _read_text_input(gold_path)
    transcription_a_text = _read_text_input(transcription_a_path)
    transcription_b_text: str | None = None
    if transcription_b_path is not None:
        transcription_b_text = _read_text_input(transcription_b_path)

    vocabulary_set: set[str] = set()
    if vocabulary_path is not None and vocabulary_path.exists():
        from transcriber_radrx.cli import _load_vocabulary_set

        vocabulary_set = _load_vocabulary_set(vocabulary_path)

    phrase_pipeline = PhraseCorrectorPipeline()
    corrected_a, phrase_corrections_a = phrase_pipeline.correct(transcription_a_text)
    corrected_b: str | None = None
    if transcription_b_text is not None:
        corrected_b, _ = phrase_pipeline.correct(transcription_b_text)

    from transcriber_radrx.corrector import Correction

    word_corrections_a: list[Correction] = []
    if vocabulary_path is not None and vocabulary_path.exists():
        corrector = CorrectionDictionary(str(vocabulary_path), enable_phonetic=True)
        corrected_a, word_corrections_a, _ = corrector.correct_full(transcription_a_text)
        if corrected_b is not None and transcription_b_text is not None:
            corrected_b, _, _ = corrector.correct_full(transcription_b_text)

    raw_wer = jiwer.wer(gold_text, transcription_a_text)
    corrected_wer = jiwer.wer(gold_text, corrected_a)

    terms_found, terms_total, terms_missing = _term_recall(corrected_a, gold_text, vocabulary_set)

    result: dict[str, object] = {
        "raw_wer": raw_wer,
        "corrected_wer": corrected_wer,
        "phrase_corrections": len(phrase_corrections_a),
        "word_corrections": len(word_corrections_a),
        "terms_found": terms_found,
        "terms_total": terms_total,
        "terms_missing": terms_missing,
        "corrected_text_a": corrected_a,
    }

    if corrected_b is not None:
        from transcriber_radrx.ensemble.decision_rules import ensemble_transcriptions

        ensemble_result = ensemble_transcriptions(
            text_voxtral=corrected_a,
            text_whisper=corrected_b,
            vocabulary=vocabulary_set,
            fixture_id=gold_path.stem,
            voice="clinical",
        )
        ensemble_wer = jiwer.wer(gold_text, ensemble_result.text_ensemble)
        total_words = len(ensemble_result.words)
        uwr_value = ensemble_result.review_count / total_words if total_words > 0 else 0.0
        result["ensemble_wer"] = ensemble_wer
        result["uwr"] = uwr_value
        result["review_count"] = ensemble_result.review_count
        result["ensemble_text"] = ensemble_result.text_ensemble

    return result


def _run_evaluate_subprocess(
    audio_path: Path,
    output_path: Path,
    vocabulary_path: Path | None,
    progress_callback: object | None = None,
) -> dict[str, object]:
    """Execute the evaluate pipeline in a subprocess.

    Runs ``transcribe-radrx evaluate`` as a child process to isolate MLX
    GPU operations from Qt's event loop (MLX + QThread causes bus errors
    on Apple Silicon). Parses results from the CLI's stderr output.

    Args:
        audio_path: Path to the audio WAV file.
        output_path: Path for the output review .docx.
        vocabulary_path: Optional path to RT vocabulary file.
        progress_callback: Optional callable(str) for progress messages.

    Returns:
        Dict with keys: ensemble_text, review_count, total_words, output_path.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        GUIWorkerError: If the subprocess fails.
    """
    import subprocess

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    def _emit(msg: str) -> None:
        if callable(progress_callback):
            progress_callback(msg)  # type: ignore[call-arg]

    _emit("Starting evaluation (this may take a few minutes) ...")

    cmd = [
        sys.executable,
        "-m",
        "transcriber_radrx.cli",
        "evaluate",
        "--audio",
        str(audio_path),
        "--output",
        str(output_path),
    ]
    if vocabulary_path is not None:
        cmd.extend(["--vocabulary", str(vocabulary_path)])

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        msg = "Evaluation timed out after 10 minutes"
        raise GUIWorkerError(msg) from exc

    # Parse stderr for progress and results
    ensemble_text = result.stdout.strip()
    review_count = 0
    total_words = 0

    for line in result.stderr.splitlines():
        # Forward progress lines
        if "Loading" in line or "Transcribing" in line or "Running" in line or "Rendering" in line:
            _emit(line.strip().removeprefix("INFO: "))

        # Parse result lines
        if "Review count" in line:
            with contextlib.suppress(ValueError):
                review_count = int(line.split(":")[-1].strip())
        elif "Total words" in line:
            with contextlib.suppress(ValueError):
                total_words = int(line.split(":")[-1].strip())

    if result.returncode != 0:
        error_lines = [line for line in result.stderr.splitlines() if "error" in line.lower() or "Error" in line]
        msg = "\n".join(error_lines) if error_lines else f"Evaluation failed (exit code {result.returncode})"
        raise GUIWorkerError(msg)

    return {
        "ensemble_text": ensemble_text,
        "review_count": review_count,
        "total_words": total_words,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# QThread workers (only defined when PySide6 is available)
# ---------------------------------------------------------------------------

if _PYSIDE6_AVAILABLE:

    class CompareWorker(QThread):
        """Run the compare pipeline in a background thread.

        Signals:
            finished: Emitted with results dict on success.
            error: Emitted with error message string on failure.
            progress: Emitted with progress message strings.
        """

        finished: Signal = Signal(dict)
        error: Signal = Signal(str)
        progress: Signal = Signal(str)

        def __init__(
            self,
            gold_path: Path,
            transcription_a_path: Path,
            transcription_b_path: Path | None = None,
            vocabulary_path: Path | None = None,
            parent: QWidget | None = None,
        ) -> None:
            """Initialise the worker with input paths.

            Args:
                gold_path: Path to gold-standard file.
                transcription_a_path: Path to transcription A.
                transcription_b_path: Optional path to transcription B.
                vocabulary_path: Optional vocabulary file path.
                parent: Optional parent QObject.
            """
            super().__init__(parent)
            self._gold_path = gold_path
            self._transcription_a_path = transcription_a_path
            self._transcription_b_path = transcription_b_path
            self._vocabulary_path = vocabulary_path

        def run(self) -> None:
            """Execute the compare pipeline."""
            try:
                self.progress.emit("Applying corrections ...")
                results = _run_compare_core(
                    self._gold_path,
                    self._transcription_a_path,
                    self._transcription_b_path,
                    self._vocabulary_path,
                )
                self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))

    class EvaluateWorker(QThread):
        """Run the evaluate pipeline (ASR + ensemble) in a background thread.

        Signals:
            finished: Emitted with results dict on success.
            error: Emitted with error message string on failure.
            progress: Emitted with progress message strings.
        """

        finished: Signal = Signal(dict)
        error: Signal = Signal(str)
        progress: Signal = Signal(str)

        def __init__(
            self,
            audio_path: Path,
            output_path: Path,
            vocabulary_path: Path | None = None,
            parent: QWidget | None = None,
        ) -> None:
            """Initialise the worker with audio and output paths.

            Args:
                audio_path: Path to the WAV audio file.
                output_path: Destination path for the review .docx.
                vocabulary_path: Optional vocabulary file path.
                parent: Optional parent QObject.
            """
            super().__init__(parent)
            self._audio_path = audio_path
            self._output_path = output_path
            self._vocabulary_path = vocabulary_path

        def run(self) -> None:
            """Execute the evaluate pipeline in a subprocess.

            MLX GPU operations conflict with Qt's threading model on Apple
            Silicon (bus error). Running as a subprocess isolates them.
            """
            try:
                results = _run_evaluate_subprocess(
                    self._audio_path,
                    self._output_path,
                    self._vocabulary_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# File-picker widget
# ---------------------------------------------------------------------------

if _PYSIDE6_AVAILABLE:

    class FilePickerWidget(QWidget):
        """A labelled file-picker row with Browse and Clear buttons.

        Supports drag-and-drop for the accepted file extensions.

        Attributes:
            file_selected: Signal emitted with the selected path string when
                a file is chosen or dropped.
        """

        file_selected: Signal = Signal(str)

        def __init__(
            self,
            label: str,
            accepted_extensions: list[str],
            parent: QWidget | None = None,
        ) -> None:
            """Initialise the file-picker widget.

            Args:
                label: Human-readable label shown above the path display.
                accepted_extensions: List of lowercase extensions (e.g. [".wav"]).
                parent: Optional parent widget.
            """
            super().__init__(parent)
            self._accepted_extensions = [ext.lower() for ext in accepted_extensions]
            self.setAcceptDrops(True)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(label)
            layout.addWidget(lbl)

            row = QHBoxLayout()
            self._path_edit = QLineEdit()
            self._path_edit.setReadOnly(True)
            self._path_edit.setPlaceholderText("No file selected")
            self._path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(self._path_edit)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(self._on_browse)
            row.addWidget(browse_btn)

            clear_btn = QPushButton("×")
            clear_btn.setFixedWidth(28)
            clear_btn.setToolTip("Clear selection")
            clear_btn.clicked.connect(self._on_clear)
            row.addWidget(clear_btn)

            layout.addLayout(row)

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def selected_path(self) -> Path | None:
            """Return the currently selected path, or None if empty.

            Returns:
                Selected Path, or None.
            """
            text = self._path_edit.text().strip()
            return Path(text) if text else None

        def set_path(self, path: Path) -> None:
            """Programmatically set the selected path.

            Args:
                path: The path to display and select.
            """
            self._path_edit.setText(str(path))
            self.file_selected.emit(str(path))

        # ------------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------------

        def _on_browse(self) -> None:
            """Open a file dialog and update the path display."""
            ext_filter_parts = []
            for ext in self._accepted_extensions:
                ext_filter_parts.append(f"*{ext}")
            ext_filter = f"Files ({' '.join(ext_filter_parts)});;All Files (*)"
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", ext_filter)
            if path:
                self._path_edit.setText(path)
                self.file_selected.emit(path)

        def _on_clear(self) -> None:
            """Clear the current selection."""
            self._path_edit.clear()

        # ------------------------------------------------------------------
        # Drag-and-drop
        # ------------------------------------------------------------------

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
            """Accept drag events for files with matching extensions.

            Args:
                event: The drag-enter event.
            """
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if urls and any(Path(urls[0].toLocalFile()).suffix.lower() in self._accepted_extensions for _ in [None]):
                    event.acceptProposedAction()
                    return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
            """Handle file drop events.

            Args:
                event: The drop event.
            """
            urls = event.mimeData().urls()
            if urls:
                path_str = urls[0].toLocalFile()
                if Path(path_str).suffix.lower() in self._accepted_extensions:
                    self._path_edit.setText(path_str)
                    self.file_selected.emit(path_str)
                    event.acceptProposedAction()
                    return
            event.ignore()


# ---------------------------------------------------------------------------
# Transcribe tab
# ---------------------------------------------------------------------------

if _PYSIDE6_AVAILABLE:

    class TranscribeTab(QWidget):
        """Tab 1: Evaluate pipeline — pick audio, run ASR ensemble, save docx."""

        switch_to_compare: Signal = Signal(str)

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialise the Transcribe tab.

            Args:
                parent: Optional parent widget.
            """
            super().__init__(parent)
            self._worker: EvaluateWorker | None = None
            self._last_output_path: Path | None = None

            layout = QVBoxLayout(self)
            layout.setSpacing(12)

            # Audio file picker group
            audio_group = QGroupBox("Audio Recording")
            audio_layout = QVBoxLayout(audio_group)
            self._audio_picker = FilePickerWidget("Audio file (.wav):", [".wav"])
            audio_layout.addWidget(self._audio_picker)

            # Record controls
            record_row = QHBoxLayout()
            self._record_btn = QPushButton("Record")
            self._record_btn.clicked.connect(self._on_record_toggle)
            record_row.addWidget(self._record_btn)

            self._record_duration = QLabel("")
            self._record_duration.setStyleSheet("color: #555; font-family: monospace;")
            record_row.addWidget(self._record_duration)
            record_row.addStretch()
            audio_layout.addLayout(record_row)

            layout.addWidget(audio_group)

            # Recording state
            self._is_recording = False
            self._recorder: object = None  # QMediaRecorder when active
            self._record_timer = QTimer(self)
            self._record_timer.setInterval(500)
            self._record_timer.timeout.connect(self._update_record_duration)
            self._elapsed_timer = QElapsedTimer()
            self._recording_path: Path | None = None

            # Status
            self._status_label = QLabel("Ready")
            self._status_label.setStyleSheet("color: #555;")
            layout.addWidget(self._status_label)

            # Buttons
            btn_row = QHBoxLayout()
            self._transcribe_btn = QPushButton("Transcribe")
            self._transcribe_btn.setDefault(True)
            self._transcribe_btn.clicked.connect(self._on_transcribe)
            btn_row.addWidget(self._transcribe_btn)

            self._compare_btn = QPushButton("Compare with Gold Standard →")
            self._compare_btn.setVisible(False)
            self._compare_btn.clicked.connect(self._on_compare)
            btn_row.addWidget(self._compare_btn)

            btn_row.addStretch()
            layout.addLayout(btn_row)
            layout.addStretch()

        def _on_record_toggle(self) -> None:
            """Start or stop microphone recording."""
            if self._is_recording:
                self._stop_recording()
            else:
                self._start_recording()

        def _start_recording(self) -> None:
            """Start recording from the default microphone."""
            try:
                from PySide6.QtMultimedia import QAudioInput, QMediaCaptureSession, QMediaRecorder
            except ImportError:
                QMessageBox.warning(
                    self,
                    "Recording unavailable",
                    "QtMultimedia is not available. Record audio with another tool and use the file picker.",
                )
                return

            import tempfile

            # Create a temp WAV file for the recording
            with tempfile.NamedTemporaryFile(suffix=".wav", prefix="radrx-recording-", delete=False) as tmp:
                self._recording_path = Path(tmp.name)

            # Set up capture session
            self._audio_input = QAudioInput(self)
            self._capture_session = QMediaCaptureSession(self)
            self._capture_session.setAudioInput(self._audio_input)

            recorder = QMediaRecorder(self)
            self._capture_session.setRecorder(recorder)

            from PySide6.QtMultimedia import QMediaFormat

            recorder.setMediaFormat(QMediaFormat(QMediaFormat.FileFormat.Wave))
            recorder.setOutputLocation(QUrl.fromLocalFile(str(self._recording_path)))
            self._recorder = recorder

            recorder.record()
            self._is_recording = True
            self._elapsed_timer.start()
            self._record_timer.start()
            self._record_btn.setText("Stop Recording")
            self._record_btn.setStyleSheet("background-color: #e74c3c; color: white;")
            self._record_duration.setText("0:00")
            self._status_label.setText("Recording ...")

        def _stop_recording(self) -> None:
            """Stop the current recording and update the file picker."""
            if self._recorder is not None:
                from PySide6.QtMultimedia import QMediaRecorder

                if isinstance(self._recorder, QMediaRecorder):
                    self._recorder.stop()

            self._record_timer.stop()
            self._is_recording = False
            self._record_btn.setText("Record")
            self._record_btn.setStyleSheet("")

            elapsed_ms = self._elapsed_timer.elapsed()
            secs = elapsed_ms // 1000
            mins = secs // 60
            secs = secs % 60
            self._record_duration.setText(f"{mins}:{secs:02d}")

            if self._recording_path and self._recording_path.exists() and self._recording_path.stat().st_size > 0:
                self._audio_picker.set_path(self._recording_path)
                self._status_label.setText(f"Recorded {mins}:{secs:02d} — ready to transcribe")
            else:
                self._status_label.setText("Recording may have failed — check microphone permissions")

        def _update_record_duration(self) -> None:
            """Update the recording duration display (called by timer)."""
            elapsed_ms = self._elapsed_timer.elapsed()
            secs = elapsed_ms // 1000
            mins = secs // 60
            secs = secs % 60
            self._record_duration.setText(f"{mins}:{secs:02d}")

        def _on_transcribe(self) -> None:
            """Prompt for output path and start the evaluate worker."""
            audio_path = self._audio_picker.selected_path()
            if audio_path is None:
                QMessageBox.warning(self, "No audio file", "Please select an audio file first.")
                return

            output_path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save review document",
                str(audio_path.with_suffix(".docx")),
                "Word Document (*.docx)",
            )
            if not output_path_str:
                return

            output_path = Path(output_path_str)

            self._compare_btn.setVisible(False)
            self._status_label.setText("Loading models ...")
            self._transcribe_btn.setEnabled(False)

            self._worker = EvaluateWorker(audio_path, output_path)
            self._worker.progress.connect(self._on_progress)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        def _on_progress(self, message: str) -> None:
            """Update the status label with a progress message.

            Args:
                message: Progress text to display.
            """
            self._status_label.setText(message)

        def _on_finished(self, results: dict[str, object]) -> None:
            """Handle successful pipeline completion.

            Args:
                results: Results dict from the evaluate pipeline.
            """
            self._transcribe_btn.setEnabled(True)
            output_path_str = str(results.get("output_path", ""))
            self._last_output_path = Path(output_path_str) if output_path_str else None
            review_count = results.get("review_count", 0)
            total_words = results.get("total_words", 0)
            self._status_label.setText(
                f"Done — {review_count} review words out of {total_words}. Saved to {Path(output_path_str).name}"
            )
            self._compare_btn.setVisible(True)

            # Open in system default app
            if self._last_output_path and self._last_output_path.exists():
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_path)))

        def _on_error(self, message: str) -> None:
            """Handle pipeline error.

            Args:
                message: Error description.
            """
            self._transcribe_btn.setEnabled(True)
            self._status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Transcription failed", message)

        def _on_compare(self) -> None:
            """Emit signal to switch to Compare tab with last output pre-loaded."""
            if self._last_output_path:
                self.switch_to_compare.emit(str(self._last_output_path))


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------

if _PYSIDE6_AVAILABLE:

    class CompareTab(QWidget):
        """Tab 2: Compare pipeline — gold standard + transcriptions, compute WER."""

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialise the Compare tab.

            Args:
                parent: Optional parent widget.
            """
            super().__init__(parent)
            self._worker: CompareWorker | None = None

            layout = QVBoxLayout(self)
            layout.setSpacing(12)

            # File pickers group
            files_group = QGroupBox("Input Files")
            files_layout = QVBoxLayout(files_group)

            self._gold_picker = FilePickerWidget("Gold Standard Text (.txt or .docx):", [".txt", ".docx"])
            files_layout.addWidget(self._gold_picker)

            self._trans_a_picker = FilePickerWidget("Transcription (System A) (.txt or .docx):", [".txt", ".docx"])
            files_layout.addWidget(self._trans_a_picker)

            self._trans_b_picker = FilePickerWidget("Transcription (System B) — optional (.txt or .docx):", [".txt", ".docx"])
            files_layout.addWidget(self._trans_b_picker)

            layout.addWidget(files_group)

            # Results
            results_group = QGroupBox("Results")
            results_layout = QVBoxLayout(results_group)
            self._results_edit = QTextEdit()
            self._results_edit.setReadOnly(True)
            self._results_edit.setPlaceholderText("Results will appear here after comparison.")
            self._results_edit.setMinimumHeight(160)
            results_layout.addWidget(self._results_edit)
            layout.addWidget(results_group)

            # Buttons
            btn_row = QHBoxLayout()
            self._compare_btn = QPushButton("Compare")
            self._compare_btn.setDefault(True)
            self._compare_btn.clicked.connect(self._on_compare)
            btn_row.addWidget(self._compare_btn)

            self._save_btn = QPushButton("Save Report ...")
            self._save_btn.setEnabled(False)
            self._save_btn.clicked.connect(self._on_save_report)
            btn_row.addWidget(self._save_btn)

            btn_row.addStretch()
            layout.addLayout(btn_row)

            self._last_results: dict[str, object] = {}
            self._last_gold_path: Path | None = None
            self._last_output_path: Path | None = None

        def preload_transcription(self, path_str: str) -> None:
            """Pre-populate the System A picker (called from Transcribe tab).

            Args:
                path_str: String path to set in the System A picker.
            """
            self._trans_a_picker.set_path(Path(path_str))

        def _on_compare(self) -> None:
            """Validate inputs and start the compare worker."""
            gold_path = self._gold_picker.selected_path()
            trans_a_path = self._trans_a_picker.selected_path()

            if gold_path is None:
                QMessageBox.warning(self, "Missing input", "Please select a Gold Standard file.")
                return
            if trans_a_path is None:
                QMessageBox.warning(self, "Missing input", "Please select Transcription A.")
                return

            trans_b_path = self._trans_b_picker.selected_path()
            self._last_gold_path = gold_path

            self._compare_btn.setEnabled(False)
            self._save_btn.setEnabled(False)
            self._results_edit.setPlainText("Running comparison ...")

            self._worker = CompareWorker(gold_path, trans_a_path, trans_b_path)
            self._worker.progress.connect(self._on_progress)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        def _on_progress(self, message: str) -> None:
            """Append a progress message to the results panel.

            Args:
                message: Progress text.
            """
            self._results_edit.append(message)

        def _on_finished(self, results: dict[str, object]) -> None:
            """Render the results dict into the text panel.

            Args:
                results: Results dict from the compare pipeline.
            """
            self._compare_btn.setEnabled(True)
            self._last_results = results

            raw_wer = float(str(results.get("raw_wer", 0.0)))
            corrected_wer = float(str(results.get("corrected_wer", 0.0)))
            phrase_fixes = int(str(results.get("phrase_corrections", 0)))
            word_fixes = int(str(results.get("word_corrections", 0)))
            terms_found = int(str(results.get("terms_found", 0)))
            terms_total = int(str(results.get("terms_total", 0)))
            raw_missing = results.get("terms_missing", [])
            terms_missing: list[str] = [str(t) for t in raw_missing] if isinstance(raw_missing, list) else []

            lines: list[str] = [
                f"Raw WER:        {raw_wer:.4f}  ({raw_wer * 100:.1f}%)",
                f"Corrected WER:  {corrected_wer:.4f}  ({corrected_wer * 100:.1f}%)",
            ]
            if phrase_fixes:
                lines.append(f"Phrase fixes:   {phrase_fixes}")
            if word_fixes:
                lines.append(f"Word fixes:     {word_fixes}")
            if terms_total > 0:
                recall = terms_found / terms_total
                lines.append(f"Term recall:    {terms_found}/{terms_total}  ({recall * 100:.1f}%)")
                if terms_missing:
                    lines.append(f"Terms missing:  {', '.join(terms_missing[:10])}")

            if "ensemble_wer" in results:
                ensemble_wer = float(str(results["ensemble_wer"]))
                uwr = float(str(results.get("uwr", 0.0)))
                review_count = int(str(results.get("review_count", 0)))
                lines += [
                    "",
                    f"Ensemble WER:   {ensemble_wer:.4f}  ({ensemble_wer * 100:.1f}%)",
                    f"UWR:            {uwr:.4f}  ({uwr * 100:.1f}%)",
                    f"Review words:   {review_count}",
                ]

            self._results_edit.setPlainText("\n".join(lines))
            self._save_btn.setEnabled("ensemble_text" in results or "corrected_text_a" in results)

        def _on_error(self, message: str) -> None:
            """Handle compare pipeline error.

            Args:
                message: Error description.
            """
            self._compare_btn.setEnabled(True)
            self._results_edit.setPlainText(f"Error: {message}")
            QMessageBox.critical(self, "Comparison failed", message)

        def _on_save_report(self) -> None:
            """Save or open the ensemble .docx report."""
            if not self._last_results or self._last_gold_path is None:
                return

            trans_b_path = self._trans_b_picker.selected_path()
            if trans_b_path is None:
                QMessageBox.information(
                    self,
                    "No ensemble",
                    "Report .docx is only generated when two transcriptions are compared.",
                )
                return

            output_path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save comparison report",
                str(self._last_gold_path.with_suffix(".compare.docx")),
                "Word Document (*.docx)",
            )
            if not output_path_str:
                return

            output_path = Path(output_path_str)
            self._last_output_path = output_path

            gold_path = self._gold_picker.selected_path()
            trans_a_path = self._trans_a_picker.selected_path()
            if gold_path is None or trans_a_path is None:
                return

            try:
                _save_compare_docx(
                    gold_path=gold_path,
                    transcription_a_path=trans_a_path,
                    transcription_b_path=trans_b_path,
                    output_path=output_path,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return

            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path)))


# ---------------------------------------------------------------------------
# Docx save helper (no PySide6 dependency)
# ---------------------------------------------------------------------------


def _save_compare_docx(
    gold_path: Path,
    transcription_a_path: Path,
    transcription_b_path: Path,
    output_path: Path,
) -> None:
    """Re-run the ensemble pipeline and render a .docx to *output_path*.

    Args:
        gold_path: Path to gold-standard text file.
        transcription_a_path: Path to transcription A.
        transcription_b_path: Path to transcription B.
        output_path: Destination .docx path.

    Raises:
        GUIWorkerError: If the rendering fails.
    """
    import jiwer

    from transcriber_radrx.cli import _load_vocabulary_set, _read_text_input, _resolve_vocabulary
    from transcriber_radrx.corrector import CorrectionDictionary
    from transcriber_radrx.ensemble.decision_rules import ensemble_transcriptions
    from transcriber_radrx.ensemble.docx_renderer import render_ensemble_docx
    from transcriber_radrx.phrase_corrector import PhraseCorrectorPipeline

    gold_text = _read_text_input(gold_path)
    text_a = _read_text_input(transcription_a_path)
    text_b = _read_text_input(transcription_b_path)

    resolved_vocab = _resolve_vocabulary(None)
    vocabulary_set: set[str] = set()
    if resolved_vocab is not None:
        vocabulary_set = _load_vocabulary_set(resolved_vocab)

    phrase_pipeline = PhraseCorrectorPipeline()
    corrected_a, _ = phrase_pipeline.correct(text_a)
    corrected_b, _ = phrase_pipeline.correct(text_b)

    if resolved_vocab is not None:
        corrector = CorrectionDictionary(str(resolved_vocab), enable_phonetic=True)
        corrected_a, _, _ = corrector.correct_full(text_a)
        corrected_b, _, _ = corrector.correct_full(text_b)

    ensemble_result = ensemble_transcriptions(
        text_voxtral=corrected_a,
        text_whisper=corrected_b,
        vocabulary=vocabulary_set,
        fixture_id=gold_path.stem,
        voice="clinical",
    )

    _ = jiwer.wer(gold_text, ensemble_result.text_ensemble)  # validate pipeline works

    gold_texts = {(gold_path.stem, "clinical"): gold_text}
    render_ensemble_docx(
        [ensemble_result],
        output_path,
        mode="review",
        show_gold=True,
        gold_texts=gold_texts,
    )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

if _PYSIDE6_AVAILABLE:

    class MainWindow(QMainWindow):
        """Top-level application window with two tabs."""

        def __init__(self) -> None:
            """Initialise the main window."""
            super().__init__()
            self.setWindowTitle("transcriber-radrx — Clinical ASR Evaluation")
            self.setMinimumSize(700, 500)

            self._tabs = QTabWidget()
            self.setCentralWidget(self._tabs)

            self._transcribe_tab = TranscribeTab()
            self._compare_tab = CompareTab()

            self._tabs.addTab(self._transcribe_tab, "Transcribe")
            self._tabs.addTab(self._compare_tab, "Compare")

            self._transcribe_tab.switch_to_compare.connect(self._on_switch_to_compare)

        def _on_switch_to_compare(self, path_str: str) -> None:
            """Switch to the Compare tab and pre-load the transcription.

            Args:
                path_str: Path to the transcription .docx to pre-load.
            """
            self._compare_tab.preload_transcription(path_str)
            self._tabs.setCurrentWidget(self._compare_tab)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the transcribe-radrx GUI application.

    Raises:
        SystemExit: If PySide6 is not installed.
    """
    if not _PYSIDE6_AVAILABLE:
        print(
            "PySide6 is required to run the GUI.\nInstall with: uv sync --extra gui",
            file=sys.stderr,
        )
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
