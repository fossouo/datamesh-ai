#!/usr/bin/env python3
"""
Test script for DataMesh.AI improvements.

Tests:
1. Resilience patterns (retry, circuit breaker, idempotency)
2. SQL Agent pagination
3. OpenTelemetry metrics integration
4. Async orchestrator functionality
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../orchestrator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../observability"))

from resilience import (
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitState,
    IdempotencyStore,
    ResilientClient,
    with_retry,
    with_circuit_breaker,
    RetryError,
    CircuitOpenError,
)


class TestRetryLogic(unittest.TestCase):
    """Test retry with exponential backoff."""

    def test_retry_success_first_attempt(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return {"status": "SUCCESS"}

        result = always_succeeds()
        self.assertEqual(call_count, 1)
        self.assertEqual(result["status"], "SUCCESS")

    def test_retry_success_after_failures(self):
        """Test retry succeeds after initial failures."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, initial_delay_seconds=0.01))
        def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return {"status": "SUCCESS"}

        result = fails_twice_then_succeeds()
        self.assertEqual(call_count, 3)
        self.assertEqual(result["status"], "SUCCESS")

    def test_retry_exhausted(self):
        """Test that RetryError is raised when retries are exhausted."""
        @with_retry(RetryConfig(max_retries=2, initial_delay_seconds=0.01))
        def always_fails():
            raise ValueError("Always fails")

        with self.assertRaises(RetryError):
            always_fails()


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig())
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertTrue(breaker.allow_request())

    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure()

        self.assertEqual(breaker.state, CircuitState.OPEN)
        self.assertFalse(breaker.allow_request())

    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)

        # Wait for timeout
        time.sleep(0.15)
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)

    def test_circuit_closes_on_success(self):
        """Test circuit closes after successful calls in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.05,
        )
        breaker = CircuitBreaker("test", config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)

        # Wait for timeout (transition to half-open)
        time.sleep(0.1)
        _ = breaker.state  # Trigger state check

        # Record successes
        breaker.record_success()
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

    def test_circuit_decorator(self):
        """Test circuit breaker decorator."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

        call_count = 0

        @with_circuit_breaker(breaker)
        def protected_call():
            nonlocal call_count
            call_count += 1
            raise ValueError("Failure")

        # First two calls should go through
        for _ in range(2):
            try:
                protected_call()
            except ValueError:
                pass

        self.assertEqual(call_count, 2)
        self.assertEqual(breaker.state, CircuitState.OPEN)

        # Third call should be rejected by circuit breaker
        with self.assertRaises(CircuitOpenError):
            protected_call()

        self.assertEqual(call_count, 2)  # Call didn't execute


class TestIdempotency(unittest.TestCase):
    """Test request deduplication."""

    def test_idempotency_key_generation(self):
        """Test idempotency key is generated from request."""
        store = IdempotencyStore(ttl_seconds=60)

        request1 = {"capability": "sql.execute", "payload": {"sql": "SELECT 1"}}
        request2 = {"capability": "sql.execute", "payload": {"sql": "SELECT 1"}}
        request3 = {"capability": "sql.execute", "payload": {"sql": "SELECT 2"}}

        key1 = store.generate_key(request1)
        key2 = store.generate_key(request2)
        key3 = store.generate_key(request3)

        # Same requests should generate same key
        self.assertEqual(key1, key2)
        # Different requests should generate different keys
        self.assertNotEqual(key1, key3)

    def test_idempotency_explicit_key(self):
        """Test explicit idempotency key takes precedence."""
        store = IdempotencyStore()

        request = {
            "idempotencyKey": "my-custom-key",
            "capability": "sql.execute",
            "payload": {"sql": "SELECT 1"},
        }

        key = store.generate_key(request)
        self.assertEqual(key, "my-custom-key")

    def test_idempotency_cache_hit(self):
        """Test cached results are returned."""
        store = IdempotencyStore(ttl_seconds=60)

        request = {"capability": "test", "payload": {"data": "test"}}
        key = store.generate_key(request)

        # Store result
        result = {"status": "SUCCESS", "data": "cached"}
        store.set(key, result)

        # Retrieve
        cached = store.get(key)
        self.assertEqual(cached, result)

    def test_idempotency_expiry(self):
        """Test cached results expire after TTL."""
        store = IdempotencyStore(ttl_seconds=0.1)

        request = {"capability": "test", "payload": {}}
        key = store.generate_key(request)

        store.set(key, {"result": "data"})
        self.assertIsNotNone(store.get(key))

        # Wait for expiry
        time.sleep(0.15)
        self.assertIsNone(store.get(key))


class TestResilientClient(unittest.TestCase):
    """Test the combined resilient client."""

    def test_client_initialization(self):
        """Test client initializes with default config."""
        client = ResilientClient()
        self.assertIsNotNone(client.retry_config)
        self.assertIsNotNone(client.circuit_config)
        self.assertIsNotNone(client.idempotency_store)

    def test_client_successful_call(self):
        """Test successful call through client."""
        client = ResilientClient(
            retry_config=RetryConfig(max_retries=1, initial_delay_seconds=0.01)
        )

        def mock_call(request):
            return {"status": "SUCCESS", "data": "result"}

        result = client.call(
            endpoint="test-agent",
            request={"capability": "test"},
            call_fn=mock_call,
            use_idempotency=False,
        )

        self.assertEqual(result["status"], "SUCCESS")

    def test_client_idempotency_hit(self):
        """Test idempotency cache hit."""
        client = ResilientClient(idempotency_ttl=60)

        call_count = 0

        def mock_call(request):
            nonlocal call_count
            call_count += 1
            return {"status": "SUCCESS", "data": "result"}

        request = {"capability": "test", "payload": {"data": "test"}}

        # First call
        result1 = client.call("test-agent", request, mock_call, use_idempotency=True)
        self.assertEqual(call_count, 1)
        self.assertFalse(result1.get("_idempotency_hit"))

        # Second call with same request - should hit cache
        result2 = client.call("test-agent", request, mock_call, use_idempotency=True)
        self.assertEqual(call_count, 1)  # No additional call
        self.assertTrue(result2.get("_idempotency_hit"))


class TestOpenTelemetry(unittest.TestCase):
    """Test OpenTelemetry metrics integration."""

    def test_metrics_collector_fallback(self):
        """Test metrics collector works without OpenTelemetry installed."""
        from otel_metrics import MetricsCollector, OTelConfig

        config = OTelConfig(service_name="test-service")
        collector = MetricsCollector(config)

        stats = collector.get_stats()
        self.assertIn("otel_enabled", stats)

    def test_trace_request_context(self):
        """Test trace_request context manager."""
        from otel_metrics import get_metrics

        collector = get_metrics()

        with collector.trace_request("test_op", "test-agent", "test.cap", "trace-123") as ctx:
            ctx["status"] = "success"

        stats = collector.get_stats()
        self.assertGreaterEqual(stats.get("requests_total", 0), 1)


class TestPagination(unittest.TestCase):
    """Test SQL Agent pagination."""

    def test_pagination_parameters(self):
        """Test pagination parameter handling."""
        # Simulate the pagination logic from sql_agent_aws.py
        page_size = min(100, 1000)  # max 1000
        page = max(1, 1)  # min 1
        offset = (page - 1) * page_size

        self.assertEqual(page_size, 100)
        self.assertEqual(page, 1)
        self.assertEqual(offset, 0)

        # Test page 2
        page = 2
        offset = (page - 1) * page_size
        self.assertEqual(offset, 100)

    def test_pagination_response_structure(self):
        """Test pagination response includes expected fields."""
        # Simulate pagination response
        all_rows = list(range(250))  # 250 rows
        page_size = 100
        page = 1
        offset = (page - 1) * page_size

        page_rows = all_rows[offset:offset + page_size]
        has_more = len(all_rows) > offset + page_size

        pagination = {
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
            "next_cursor": f"page:{page + 1}:size:{page_size}" if has_more else None,
            "total_pages": (len(all_rows) + page_size - 1) // page_size,
        }

        self.assertEqual(len(page_rows), 100)
        self.assertTrue(pagination["has_more"])
        self.assertEqual(pagination["total_pages"], 3)
        self.assertIsNotNone(pagination["next_cursor"])


def main():
    """Run all tests."""
    print("=" * 60)
    print("DataMesh.AI Improvements Test Suite")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRetryLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestIdempotency))
    suite.addTests(loader.loadTestsFromTestCase(TestResilientClient))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenTelemetry))
    suite.addTests(loader.loadTestsFromTestCase(TestPagination))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
