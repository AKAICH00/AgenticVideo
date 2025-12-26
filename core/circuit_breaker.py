"""
Circuit Breaker Pattern for API Resilience

Protects against cascading failures when external APIs (Runway, Sora, Kling, etc.)
become unavailable or slow.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing, requests are rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed

Based on research from Agent 3: Critical for protecting video generation APIs
that can take minutes to respond and may have rate limits.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 60.0  # Request timeout in seconds
    excluded_exceptions: tuple = ()  # Exceptions that don't trigger the breaker


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for the circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    half_open_calls: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    state_changed_at: float = field(default_factory=time.time)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker is OPEN for {service_name}. "
            f"Retry after {retry_after:.1f} seconds."
        )


class CircuitBreaker:
    """
    Circuit breaker implementation for API protection.

    Usage:
        breaker = CircuitBreaker("runway")

        # As decorator
        @breaker
        async def call_runway_api():
            ...

        # As context manager
        async with breaker:
            await call_runway_api()

        # Direct call
        result = await breaker.call(call_runway_api, arg1, arg2)
    """

    # Global registry of circuit breakers
    _instances: dict[str, "CircuitBreaker"] = {}

    def __init__(
        self,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

        # Register in global registry
        CircuitBreaker._instances[service_name] = self

    @classmethod
    def get(cls, service_name: str) -> "CircuitBreaker":
        """Get or create a circuit breaker for a service."""
        if service_name not in cls._instances:
            cls._instances[service_name] = CircuitBreaker(service_name)
        return cls._instances[service_name]

    @classmethod
    def get_all_stats(cls) -> dict[str, CircuitBreakerStats]:
        """Get stats for all circuit breakers."""
        return {name: cb.stats for name, cb in cls._instances.items()}

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        return self.stats.state

    @property
    def is_closed(self) -> bool:
        return self.stats.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.stats.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.stats.state == CircuitState.HALF_OPEN

    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self.stats.state != CircuitState.OPEN:
            return False
        elapsed = time.time() - self.stats.state_changed_at
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.stats.state
        self.stats.state = new_state
        self.stats.state_changed_at = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self.stats.half_open_calls = 0
            self.stats.success_count = 0

        logger.info(
            f"Circuit breaker [{self.service_name}]: {old_state.value} -> {new_state.value}"
        )

    async def _before_call(self):
        """Called before each request. May raise CircuitBreakerOpen."""
        async with self._lock:
            self.stats.total_calls += 1

            if self.stats.state == CircuitState.OPEN:
                if self._should_try_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    retry_after = (
                        self.config.recovery_timeout
                        - (time.time() - self.stats.state_changed_at)
                    )
                    raise CircuitBreakerOpen(self.service_name, retry_after)

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        self.service_name,
                        self.config.recovery_timeout,
                    )
                self.stats.half_open_calls += 1

    async def _on_success(self):
        """Called after a successful request."""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = time.time()
            self.stats.failure_count = 0  # Reset consecutive failures

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, error: Exception):
        """Called after a failed request."""
        # Don't count excluded exceptions
        if isinstance(error, self.config.excluded_exceptions):
            return

        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self.stats.state == CircuitState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            logger.warning(
                f"Circuit breaker [{self.service_name}] failure: {error}. "
                f"Failure count: {self.stats.failure_count}/{self.config.failure_threshold}"
            )

    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result from the function

        Raises:
            CircuitBreakerOpen: If the circuit is open
            TimeoutError: If the function times out
            Exception: Any exception from the function
        """
        await self._before_call()

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout,
            )
            await self._on_success()
            return result
        except asyncio.TimeoutError as e:
            await self._on_failure(e)
            raise
        except Exception as e:
            await self._on_failure(e)
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator."""

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper

    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        self.stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker [{self.service_name}] manually reset")

    def force_open(self):
        """Manually force the circuit breaker to open state."""
        self._transition_to(CircuitState.OPEN)

    def get_status(self) -> dict:
        """Get current status as a dictionary."""
        return {
            "service": self.service_name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure": self.stats.last_failure_time,
            "last_success": self.stats.last_success_time,
            "state_changed_at": self.stats.state_changed_at,
        }


# Pre-configured circuit breakers for video generation APIs
def get_video_api_breaker(provider: str) -> CircuitBreaker:
    """
    Get a circuit breaker configured for video generation APIs.

    Video APIs have specific characteristics:
    - Long processing times (30s - 5min)
    - Rate limits
    - Occasional outages

    Args:
        provider: API provider name ('runway', 'sora', 'kling', 'fal', 'wan_local')
    """
    configs = {
        "runway": CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            timeout=300.0,  # 5 min timeout for video generation
        ),
        "sora": CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            timeout=600.0,  # 10 min timeout for long videos
        ),
        "kling": CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=45.0,
            timeout=180.0,  # 3 min timeout
        ),
        "fal": CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            timeout=300.0,
        ),
        "kie": CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            timeout=300.0,  # 5 min timeout for Kie/Kling video generation
        ),
        "wan_local": CircuitBreakerConfig(
            failure_threshold=2,  # Local failures are more serious
            recovery_timeout=120.0,  # Longer recovery for local GPU issues
            timeout=600.0,  # Local can be slow
        ),
        "elevenlabs": CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            timeout=60.0,
        ),
    }

    config = configs.get(provider, CircuitBreakerConfig())

    # Check if breaker already exists with correct config
    if provider in CircuitBreaker._instances:
        existing = CircuitBreaker._instances[provider]
        # Update config if timeout differs (fixes issue where default 60s was used)
        if existing.config.timeout != config.timeout:
            logger.info(
                f"Updating circuit breaker [{provider}] timeout: "
                f"{existing.config.timeout}s -> {config.timeout}s"
            )
            existing.config = config
        return existing

    # Create new breaker with correct config
    return CircuitBreaker(provider, config)
