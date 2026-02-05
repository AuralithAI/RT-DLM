"""
RT-DLM Prometheus Metrics Exporter

Provides metrics for training monitoring via Prometheus.
Integrates with existing production metrics trackers.
"""

from typing import Optional, Dict, Any
import time
import threading
from functools import wraps

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        start_http_server,
        REGISTRY,
        CollectorRegistry,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsExporter:
    """
    Prometheus metrics exporter for RT-DLM training.
    
    Exposes metrics for:
    - Training progress (loss, perplexity, accuracy)
    - Resource utilization (GPU memory, CPU)
    - Throughput (tokens/sec, batches/sec)
    - Model health (gradient norms, NaN counts)
    
    Usage:
        exporter = MetricsExporter(port=8000)
        exporter.start()
        
        # During training:
        exporter.record_training_step(loss=2.5, perplexity=12.0)
        exporter.record_batch_time(0.5)
        exporter.update_gpu_memory(used=8e9, total=16e9)
    """
    
    def __init__(
        self,
        port: int = 8000,
        namespace: str = "rtdlm",
        registry: Optional[Any] = None,
    ):
        """
        Initialize the metrics exporter.
        
        Args:
            port: Port to expose metrics on
            namespace: Prometheus metric namespace prefix
            registry: Optional custom Prometheus registry
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )
        
        self.port = port
        self.namespace = namespace
        self.registry = registry or REGISTRY
        self._server_started = False
        self._lock = threading.Lock()
        
        # Initialize metrics
        self._init_training_metrics()
        self._init_resource_metrics()
        self._init_throughput_metrics()
        self._init_health_metrics()
        self._init_info_metrics()
    
    def _init_training_metrics(self):
        """Initialize training-related metrics."""
        # Training loss
        self.training_loss = Gauge(
            f"{self.namespace}_training_loss",
            "Current training loss",
            ["model_name", "phase"],
            registry=self.registry,
        )
        
        # Validation loss
        self.validation_loss = Gauge(
            f"{self.namespace}_validation_loss",
            "Current validation loss",
            ["model_name"],
            registry=self.registry,
        )
        
        # Perplexity
        self.perplexity = Gauge(
            f"{self.namespace}_perplexity",
            "Model perplexity",
            ["model_name", "split"],
            registry=self.registry,
        )
        
        # Training steps counter
        self.training_steps = Counter(
            f"{self.namespace}_training_steps_total",
            "Total training steps completed",
            ["model_name"],
            registry=self.registry,
        )
        
        # Epochs counter
        self.epochs_completed = Gauge(
            f"{self.namespace}_epochs_completed",
            "Number of epochs completed",
            ["model_name"],
            registry=self.registry,
        )
        
        # Learning rate
        self.learning_rate = Gauge(
            f"{self.namespace}_learning_rate",
            "Current learning rate",
            ["model_name"],
            registry=self.registry,
        )
    
    def _init_resource_metrics(self):
        """Initialize resource utilization metrics."""
        # GPU memory usage
        self.gpu_memory_used = Gauge(
            f"{self.namespace}_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["device_id"],
            registry=self.registry,
        )
        
        self.gpu_memory_total = Gauge(
            f"{self.namespace}_gpu_memory_total_bytes",
            "Total GPU memory in bytes",
            ["device_id"],
            registry=self.registry,
        )
        
        self.gpu_memory_utilization = Gauge(
            f"{self.namespace}_gpu_memory_utilization",
            "GPU memory utilization (0-1)",
            ["device_id"],
            registry=self.registry,
        )
        
        # CPU utilization
        self.cpu_utilization = Gauge(
            f"{self.namespace}_cpu_utilization",
            "CPU utilization percentage",
            registry=self.registry,
        )
        
        # RAM usage
        self.ram_used = Gauge(
            f"{self.namespace}_ram_used_bytes",
            "RAM used in bytes",
            registry=self.registry,
        )
    
    def _init_throughput_metrics(self):
        """Initialize throughput metrics."""
        # Tokens per second
        self.tokens_per_second = Gauge(
            f"{self.namespace}_tokens_per_second",
            "Training throughput in tokens per second",
            ["model_name"],
            registry=self.registry,
        )
        
        # Batches per second
        self.batches_per_second = Gauge(
            f"{self.namespace}_batches_per_second",
            "Training throughput in batches per second",
            ["model_name"],
            registry=self.registry,
        )
        
        # Batch processing time histogram
        self.batch_time = Histogram(
            f"{self.namespace}_batch_processing_seconds",
            "Batch processing time in seconds",
            ["model_name"],
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        
        # Data loading time
        self.data_loading_time = Summary(
            f"{self.namespace}_data_loading_seconds",
            "Data loading time in seconds",
            registry=self.registry,
        )
    
    def _init_health_metrics(self):
        """Initialize model health metrics."""
        # Gradient norm
        self.gradient_norm = Gauge(
            f"{self.namespace}_gradient_norm",
            "Global gradient norm",
            ["model_name"],
            registry=self.registry,
        )
        
        # NaN gradient count
        self.nan_gradient_count = Counter(
            f"{self.namespace}_nan_gradients_total",
            "Total NaN gradients encountered",
            ["model_name"],
            registry=self.registry,
        )
        
        # Gradient clipping events
        self.gradient_clips = Counter(
            f"{self.namespace}_gradient_clips_total",
            "Total gradient clipping events",
            ["model_name"],
            registry=self.registry,
        )
        
        # Checkpoint saves
        self.checkpoints_saved = Counter(
            f"{self.namespace}_checkpoints_saved_total",
            "Total checkpoints saved",
            ["model_name"],
            registry=self.registry,
        )
        
        # Early stopping metric
        self.early_stopping_patience = Gauge(
            f"{self.namespace}_early_stopping_patience_remaining",
            "Remaining early stopping patience",
            ["model_name"],
            registry=self.registry,
        )
    
    def _init_info_metrics(self):
        """Initialize info/metadata metrics."""
        self.model_info = Info(
            f"{self.namespace}_model",
            "Model metadata",
            registry=self.registry,
        )
        
        self.training_info = Info(
            f"{self.namespace}_training",
            "Training configuration",
            registry=self.registry,
        )
    
    def start(self, daemon: bool = True):
        """
        Start the Prometheus HTTP server.
        
        Args:
            daemon: Whether to run as daemon thread
        """
        with self._lock:
            if self._server_started:
                return
            
            start_http_server(self.port, registry=self.registry)
            self._server_started = True
            print(f"📊 Prometheus metrics available at http://localhost:{self.port}/metrics")
    
    # =========================================================================
    # Training Metrics Recording
    # =========================================================================
    
    def record_training_step(
        self,
        loss: float,
        perplexity: Optional[float] = None,
        learning_rate: Optional[float] = None,
        model_name: str = "rtdlm",
        phase: str = "train",
    ):
        """Record metrics for a training step."""
        self.training_loss.labels(model_name=model_name, phase=phase).set(loss)
        self.training_steps.labels(model_name=model_name).inc()
        
        if perplexity is not None:
            self.perplexity.labels(model_name=model_name, split="train").set(perplexity)
        
        if learning_rate is not None:
            self.learning_rate.labels(model_name=model_name).set(learning_rate)
    
    def record_validation(
        self,
        loss: float,
        perplexity: Optional[float] = None,
        model_name: str = "rtdlm",
    ):
        """Record validation metrics."""
        self.validation_loss.labels(model_name=model_name).set(loss)
        
        if perplexity is not None:
            self.perplexity.labels(model_name=model_name, split="validation").set(perplexity)
    
    def record_epoch(self, epoch: int, model_name: str = "rtdlm"):
        """Record epoch completion."""
        self.epochs_completed.labels(model_name=model_name).set(epoch)
    
    # =========================================================================
    # Resource Metrics Recording
    # =========================================================================
    
    def update_gpu_memory(
        self,
        used: float,
        total: float,
        device_id: str = "0",
    ):
        """Update GPU memory metrics."""
        self.gpu_memory_used.labels(device_id=device_id).set(used)
        self.gpu_memory_total.labels(device_id=device_id).set(total)
        utilization = used / total if total > 0 else 0
        self.gpu_memory_utilization.labels(device_id=device_id).set(utilization)
    
    def update_cpu_ram(self, cpu_percent: float, ram_bytes: float):
        """Update CPU and RAM metrics."""
        self.cpu_utilization.set(cpu_percent)
        self.ram_used.set(ram_bytes)
    
    # =========================================================================
    # Throughput Metrics Recording
    # =========================================================================
    
    def record_batch_time(self, seconds: float, model_name: str = "rtdlm"):
        """Record batch processing time."""
        self.batch_time.labels(model_name=model_name).observe(seconds)
        if seconds > 0:
            self.batches_per_second.labels(model_name=model_name).set(1.0 / seconds)
    
    def record_throughput(
        self,
        tokens_per_sec: float,
        batches_per_sec: Optional[float] = None,
        model_name: str = "rtdlm",
    ):
        """Record throughput metrics."""
        self.tokens_per_second.labels(model_name=model_name).set(tokens_per_sec)
        if batches_per_sec is not None:
            self.batches_per_second.labels(model_name=model_name).set(batches_per_sec)
    
    def record_data_loading(self, seconds: float):
        """Record data loading time."""
        self.data_loading_time.observe(seconds)
    
    # =========================================================================
    # Health Metrics Recording
    # =========================================================================
    
    def record_gradient_norm(self, norm: float, model_name: str = "rtdlm"):
        """Record gradient norm."""
        self.gradient_norm.labels(model_name=model_name).set(norm)
    
    def record_nan_gradient(self, model_name: str = "rtdlm"):
        """Increment NaN gradient counter."""
        self.nan_gradient_count.labels(model_name=model_name).inc()
    
    def record_gradient_clip(self, model_name: str = "rtdlm"):
        """Increment gradient clipping counter."""
        self.gradient_clips.labels(model_name=model_name).inc()
    
    def record_checkpoint(self, model_name: str = "rtdlm"):
        """Record checkpoint save."""
        self.checkpoints_saved.labels(model_name=model_name).inc()
    
    def update_early_stopping(self, patience_remaining: int, model_name: str = "rtdlm"):
        """Update early stopping patience."""
        self.early_stopping_patience.labels(model_name=model_name).set(patience_remaining)
    
    # =========================================================================
    # Info Metrics
    # =========================================================================
    
    def set_model_info(self, info: Dict[str, str]):
        """Set model metadata."""
        self.model_info.info(info)
    
    def set_training_info(self, info: Dict[str, str]):
        """Set training configuration info."""
        self.training_info.info(info)


# =============================================================================
# Decorator for timing functions
# =============================================================================

def timed_metric(metric_func):
    """
    Decorator to time a function and record to a metric.
    
    Usage:
        exporter = MetricsExporter()
        
        @timed_metric(exporter.record_batch_time)
        def train_step():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            metric_func(elapsed)
            return result
        return wrapper
    return decorator


# =============================================================================
# Integration with existing trackers
# =============================================================================

class PrometheusTrainingCallback:
    """
    Callback to integrate Prometheus metrics with training loop.
    
    Usage:
        callback = PrometheusTrainingCallback(port=8000)
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = train_step(batch)
                callback.on_batch_end(loss=loss, batch_time=0.5)
            callback.on_epoch_end(epoch, val_loss=val_loss)
    """
    
    def __init__(
        self,
        port: int = 8000,
        model_name: str = "rtdlm",
        auto_start: bool = True,
    ):
        self.exporter = MetricsExporter(port=port)
        self.model_name = model_name
        self._step = 0
        
        if auto_start:
            self.exporter.start()
    
    def on_train_begin(self, config: Dict[str, Any]):
        """Called at training start."""
        self.exporter.set_training_info({
            "batch_size": str(config.get("batch_size", "unknown")),
            "learning_rate": str(config.get("learning_rate", "unknown")),
            "epochs": str(config.get("epochs", "unknown")),
            "optimizer": config.get("optimizer", "unknown"),
        })
        
        self.exporter.set_model_info({
            "name": self.model_name,
            "hidden_dim": str(config.get("hidden_dim", "unknown")),
            "num_layers": str(config.get("num_layers", "unknown")),
            "num_heads": str(config.get("num_heads", "unknown")),
        })
    
    def on_batch_end(
        self,
        loss: float,
        batch_time: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ):
        """Called after each batch."""
        self._step += 1
        
        # Calculate perplexity from loss
        import math
        perplexity = math.exp(min(loss, 20))  # Cap to avoid overflow
        
        self.exporter.record_training_step(
            loss=loss,
            perplexity=perplexity,
            learning_rate=learning_rate,
            model_name=self.model_name,
        )
        
        if batch_time is not None:
            self.exporter.record_batch_time(batch_time, self.model_name)
        
        if gradient_norm is not None:
            self.exporter.record_gradient_norm(gradient_norm, self.model_name)
        
        if tokens_per_sec is not None:
            self.exporter.record_throughput(tokens_per_sec, model_name=self.model_name)
    
    def on_epoch_end(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
    ):
        """Called after each epoch."""
        self.exporter.record_epoch(epoch, self.model_name)
        
        if val_loss is not None:
            self.exporter.record_validation(
                loss=val_loss,
                perplexity=val_perplexity,
                model_name=self.model_name,
            )
    
    def on_checkpoint(self):
        """Called when checkpoint is saved."""
        self.exporter.record_checkpoint(self.model_name)
    
    def on_nan_gradient(self):
        """Called when NaN gradient is detected."""
        self.exporter.record_nan_gradient(self.model_name)
    
    def on_gradient_clip(self):
        """Called when gradient is clipped."""
        self.exporter.record_gradient_clip(self.model_name)
    
    def update_resources(
        self,
        gpu_used: Optional[float] = None,
        gpu_total: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        ram_bytes: Optional[float] = None,
    ):
        """Update resource utilization metrics."""
        if gpu_used is not None and gpu_total is not None:
            self.exporter.update_gpu_memory(gpu_used, gpu_total)
        
        if cpu_percent is not None and ram_bytes is not None:
            self.exporter.update_cpu_ram(cpu_percent, ram_bytes)


# =============================================================================
# Standalone server for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="RT-DLM Metrics Exporter")
    parser.add_argument("--port", type=int, default=8000, help="Metrics port")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    args = parser.parse_args()
    
    exporter = MetricsExporter(port=args.port)
    exporter.start()
    
    if args.demo:
        print("Running in demo mode - generating fake metrics...")
        exporter.set_model_info({
            "name": "rtdlm-demo",
            "hidden_dim": "512",
            "num_layers": "6",
        })
        
        step = 0
        while True:
            # Simulate training metrics
            loss = 5.0 * (0.99 ** step) + random.uniform(0, 0.5)
            exporter.record_training_step(
                loss=loss,
                perplexity=2 ** loss,
                learning_rate=1e-4 * (0.999 ** step),
                model_name="rtdlm-demo",
            )
            exporter.record_batch_time(random.uniform(0.3, 0.7), "rtdlm-demo")
            exporter.record_gradient_norm(random.uniform(0.5, 2.0), "rtdlm-demo")
            exporter.update_gpu_memory(
                used=random.uniform(6e9, 10e9),
                total=16e9,
            )
            
            step += 1
            time.sleep(1)
    else:
        print(f"Metrics server running on port {args.port}")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping metrics server")
