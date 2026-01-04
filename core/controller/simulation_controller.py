"""
Simulation Controller - manages threading and communication.
"""
import time
import threading
import os
from typing import Optional
try:
    from PyQt5.QtCore import QObject, pyqtSignal, QThread
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Fallback for non-GUI mode
    class QObject:
        pass


class SimulationController(QObject if PYQT_AVAILABLE else object):
    """
    Controls simulation execution in a separate thread.
    Emits signals for GUI updates.
    """

    if PYQT_AVAILABLE:
        # Qt Signals for thread-safe communication
        metrics_updated = pyqtSignal(dict)  # Current metrics
        training_finished = pyqtSignal()  # Training complete
        error_occurred = pyqtSignal(str)  # Error message
        status_changed = pyqtSignal(str)  # Status message

    def __init__(self, trainer, save_dir='saved_models', save_interval=10):
        """
        Initialize controller.

        Args:
            trainer: TrainerBase instance (GA/CNN/DQN)
            save_dir: Directory to save models
            save_interval: Save model every N generations/episodes
        """
        if PYQT_AVAILABLE:
            super().__init__()

        self.trainer = trainer
        self.worker_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.is_paused = threading.Event()
        self.is_running = False

        # Speed control
        self.target_fps = 60
        self.frame_delay = 1.0 / self.target_fps

        # Auto-save settings
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.setup_save_directory()

    def start(self):
        """Start training in a separate thread."""
        if self.is_running:
            self._emit_status("Already running")
            return

        self.should_stop.clear()
        self.is_paused.clear()
        self.is_running = True

        self.worker_thread = threading.Thread(target=self._run_training, daemon=True)
        self.worker_thread.start()

        self._emit_status("Training started")

    def pause(self):
        """Pause training."""
        if not self.is_running:
            return

        self.is_paused.set()
        self._emit_status("Training paused")

    def resume(self):
        """Resume training."""
        if not self.is_running:
            return

        self.is_paused.clear()
        self._emit_status("Training resumed")

    def stop(self):
        """Stop training."""
        if not self.is_running:
            return

        self.should_stop.set()
        self._emit_status("Stopping training...")

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        self.is_running = False
        self._emit_status("Training stopped")

    def set_speed(self, fps: int):
        """
        Set simulation speed (frames per second).

        Args:
            fps: Target FPS (1-1000)
        """
        self.target_fps = max(1, min(1000, fps))
        self.frame_delay = 1.0 / self.target_fps
        self._emit_status(f"Speed set to {self.target_fps} FPS")

    def setup_save_directory(self):
        """Create save directory if it doesn't exist."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self._emit_status(f"Created save directory: {self.save_dir}")

    def _run_training(self):
        """Main training loop (runs in worker thread)."""
        try:
            while not self.should_stop.is_set() and not self.trainer.is_finished():
                # Check pause
                if self.is_paused.is_set():
                    time.sleep(0.1)
                    continue

                # Execute one training step
                start_time = time.time()
                metrics = self.trainer.train_step()

                # Emit metrics to GUI
                self._emit_metrics(metrics)

                # Auto-save checkpoint
                self._auto_save(metrics)

                # Frame rate limiting
                elapsed = time.time() - start_time
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)

            # Final save
            self._emit_status("Saving final model...")
            self.trainer.save_checkpoint(self.save_dir)

            # Training finished
            self.is_running = False
            self._emit_training_finished()
            self._emit_status("Training complete")

        except Exception as e:
            self.is_running = False
            self._emit_error(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _auto_save(self, metrics: dict):
        """Auto-save model periodically."""
        # Determine step number (generation or episode)
        step = metrics.get('generation', metrics.get('episode', 0))

        # Save at intervals
        if step > 0 and step % self.save_interval == 0:
            try:
                self.trainer.save_checkpoint(self.save_dir)
                self._emit_status(f"Auto-saved at step {step}")
            except Exception as e:
                self._emit_error(f"Auto-save failed: {e}")

    def _emit_metrics(self, metrics: dict):
        """Emit metrics signal (thread-safe)."""
        if PYQT_AVAILABLE and hasattr(self, 'metrics_updated'):
            self.metrics_updated.emit(metrics)
        else:
            # Fallback: just print
            print(f"Metrics: {metrics}")

    def _emit_training_finished(self):
        """Emit training finished signal."""
        if PYQT_AVAILABLE and hasattr(self, 'training_finished'):
            self.training_finished.emit()

    def _emit_error(self, message: str):
        """Emit error signal."""
        if PYQT_AVAILABLE and hasattr(self, 'error_occurred'):
            self.error_occurred.emit(message)
        else:
            print(f"ERROR: {message}")

    def _emit_status(self, message: str):
        """Emit status message."""
        if PYQT_AVAILABLE and hasattr(self, 'status_changed'):
            self.status_changed.emit(message)
        else:
            print(f"STATUS: {message}")

    def get_current_metrics(self) -> dict:
        """
        Get current metrics from trainer.

        Returns:
            dict: Current metrics
        """
        return self.trainer.get_metrics()

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: Path to save to
        """
        try:
            self.trainer.save_checkpoint(path)
            self._emit_status(f"Checkpoint saved to {path}")
        except Exception as e:
            self._emit_error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to load from
        """
        try:
            self.trainer.load_checkpoint(path)
            self._emit_status(f"Checkpoint loaded from {path}")
        except Exception as e:
            self._emit_error(f"Failed to load checkpoint: {e}")
