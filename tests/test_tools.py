"""Tests for tool modules."""

import pytest

from agentic_cli.tools.executor import (
    MockPythonExecutor,
    SafePythonExecutor,
    TimeoutError,
)
from agentic_cli.tools.registry import (
    ErrorCode,
    ToolCategory,
    ToolDefinition,
    ToolError,
    ToolRegistry,
    ToolResult,
    register_tool,
    with_result_wrapper,
)
from agentic_cli.tools.resilience import (
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    RetryConfig,
    retry,
)


class TestSafePythonExecutor:
    """Tests for SafePythonExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create executor instance for tests."""
        return SafePythonExecutor(default_timeout=5)

    def test_executor_initialization(self, executor: SafePythonExecutor):
        """Test executor initializes correctly."""
        assert executor.default_timeout == 5
        assert "math" in executor._namespace
        assert "json" in executor._namespace

    def test_simple_expression(self, executor: SafePythonExecutor):
        """Test executing a simple expression."""
        result = executor.execute("1 + 1")

        assert result["success"] is True
        assert result["result"] == "2"
        assert result["error"] == ""

    def test_math_operations(self, executor: SafePythonExecutor):
        """Test math module is available."""
        result = executor.execute("math.sqrt(16)")

        assert result["success"] is True
        assert result["result"] == "4.0"

    def test_print_output(self, executor: SafePythonExecutor):
        """Test stdout capture."""
        result = executor.execute("print('Hello, World!')")

        assert result["success"] is True
        assert "Hello, World!" in result["output"]

    def test_multiline_code(self, executor: SafePythonExecutor):
        """Test multiline code execution."""
        code = """
x = 5
y = 10
x + y
"""
        result = executor.execute(code)

        assert result["success"] is True
        assert result["result"] == "15"

    def test_function_definition(self, executor: SafePythonExecutor):
        """Test function definition and calling."""
        code = """
def add(a, b):
    return a + b
add(3, 4)
"""
        result = executor.execute(code)

        assert result["success"] is True
        assert result["result"] == "7"

    def test_syntax_error(self, executor: SafePythonExecutor):
        """Test syntax error handling."""
        result = executor.execute("def incomplete(")

        assert result["success"] is False
        assert "Syntax error" in result["error"] or "SyntaxError" in result["error"]

    def test_runtime_error(self, executor: SafePythonExecutor):
        """Test runtime error handling."""
        result = executor.execute("1 / 0")

        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_context_injection(self, executor: SafePythonExecutor):
        """Test injecting context variables."""
        result = executor.execute("x * 2", context={"x": 21})

        assert result["success"] is True
        assert result["result"] == "42"

    def test_execution_time_recorded(self, executor: SafePythonExecutor):
        """Test that execution time is recorded."""
        result = executor.execute("sum(range(100))")

        assert result["success"] is True
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] >= 0


class TestCodeValidation:
    """Tests for code validation."""

    @pytest.fixture
    def executor(self):
        """Create executor instance for tests."""
        return SafePythonExecutor()

    def test_valid_code(self, executor: SafePythonExecutor):
        """Test validation of valid code."""
        is_valid, error = executor.validate_code("x = 1 + 1")

        assert is_valid is True
        assert error == ""

    def test_invalid_syntax(self, executor: SafePythonExecutor):
        """Test validation catches syntax errors."""
        is_valid, error = executor.validate_code("def bad(")

        assert is_valid is False
        assert "Syntax error" in error

    def test_blocked_import(self, executor: SafePythonExecutor):
        """Test validation blocks unauthorized imports."""
        is_valid, error = executor.validate_code("import os")

        assert is_valid is False
        assert "not allowed" in error

    def test_allowed_import(self, executor: SafePythonExecutor):
        """Test validation allows whitelisted imports."""
        is_valid, error = executor.validate_code("import math")

        assert is_valid is True

    def test_blocked_from_import(self, executor: SafePythonExecutor):
        """Test validation blocks unauthorized from imports."""
        is_valid, error = executor.validate_code("from os import path")

        assert is_valid is False
        assert "not allowed" in error

    def test_allowed_from_import(self, executor: SafePythonExecutor):
        """Test validation allows whitelisted from imports."""
        is_valid, error = executor.validate_code("from math import sqrt")

        assert is_valid is True

    def test_private_attribute_blocked(self, executor: SafePythonExecutor):
        """Test validation blocks private attribute access."""
        is_valid, error = executor.validate_code("obj._private")

        assert is_valid is False
        assert "private" in error.lower()

    def test_dunder_attribute_blocked(self, executor: SafePythonExecutor):
        """Test validation blocks dunder attribute access."""
        is_valid, error = executor.validate_code("obj.__class__")

        assert is_valid is False

    def test_blocked_builtin_call(self, executor: SafePythonExecutor):
        """Test validation blocks dangerous builtin calls."""
        is_valid, error = executor.validate_code("eval('1+1')")

        assert is_valid is False
        assert "eval" in error


class TestBlockedOperations:
    """Tests for blocked operations in executor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance for tests."""
        return SafePythonExecutor()

    def test_eval_blocked(self, executor: SafePythonExecutor):
        """Test eval is blocked."""
        result = executor.execute("eval('1+1')")

        assert result["success"] is False

    def test_exec_blocked(self, executor: SafePythonExecutor):
        """Test exec is blocked."""
        result = executor.execute("exec('x=1')")

        assert result["success"] is False

    def test_open_blocked(self, executor: SafePythonExecutor):
        """Test open is blocked."""
        result = executor.execute("open('/etc/passwd')")

        assert result["success"] is False

    def test_import_os_blocked(self, executor: SafePythonExecutor):
        """Test os import is blocked."""
        result = executor.execute("import os")

        assert result["success"] is False

    def test_import_subprocess_blocked(self, executor: SafePythonExecutor):
        """Test subprocess import is blocked."""
        result = executor.execute("import subprocess")

        assert result["success"] is False


class TestAllowedOperations:
    """Tests for allowed operations in executor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance for tests."""
        return SafePythonExecutor()

    def test_numpy_available(self, executor: SafePythonExecutor):
        """Test numpy is available if installed."""
        result = executor.execute("import numpy as np; int(np.array([1,2,3]).sum())")

        # May fail if numpy not installed, but should not be blocked
        if result["success"]:
            assert result["result"] == "6"
        else:
            # Should fail due to numpy not installed, not security
            assert "not allowed" not in result["error"]

    def test_json_available(self, executor: SafePythonExecutor):
        """Test json module is available."""
        result = executor.execute("json.dumps({'a': 1})")

        assert result["success"] is True
        assert result["result"] == '\'{"a": 1}\''

    def test_datetime_available(self, executor: SafePythonExecutor):
        """Test datetime module is available."""
        result = executor.execute("datetime.datetime.now().year")

        assert result["success"] is True
        assert int(result["result"]) >= 2024

    def test_re_available(self, executor: SafePythonExecutor):
        """Test re module is available."""
        result = executor.execute("re.match(r'\\d+', '123').group()")

        assert result["success"] is True
        assert result["result"] == "'123'"

    def test_collections_available(self, executor: SafePythonExecutor):
        """Test collections module is available."""
        result = executor.execute("collections.Counter('aab')['a']")

        assert result["success"] is True
        assert result["result"] == "2"


class TestMockPythonExecutor:
    """Tests for MockPythonExecutor class."""

    def test_mock_executor_valid_code(self):
        """Test mock executor accepts valid code."""
        executor = MockPythonExecutor()

        result = executor.execute("x = 1 + 1")

        assert result["success"] is True
        assert result["result"] == "42"  # Mock always returns 42
        assert result["output"] == "Mock execution output"

    def test_mock_executor_invalid_code(self):
        """Test mock executor catches syntax errors."""
        executor = MockPythonExecutor()

        result = executor.execute("def bad(")

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    def test_mock_executor_validate(self):
        """Test mock executor validation."""
        executor = MockPythonExecutor()

        is_valid, error = executor.validate_code("x = 1 + 1")
        assert is_valid is True

        is_valid, error = executor.validate_code("def bad(")
        assert is_valid is False


class TestResultTruncation:
    """Tests for result truncation."""

    def test_long_result_truncated(self):
        """Test that very long results are truncated."""
        executor = SafePythonExecutor()

        # Create a very long string
        result = executor.execute("'x' * 20000")

        assert result["success"] is True
        assert "truncated" in result["result"]
        assert len(result["result"]) <= 10100  # 10000 + some buffer for message


class TestToolError:
    """Tests for ToolError class."""

    def test_tool_error_basic(self):
        """Test basic ToolError creation."""
        error = ToolError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.error_code == "TOOL_ERROR"
        assert error.recoverable is False
        assert error.details == {}
        assert error.tool_name is None

    def test_tool_error_full(self):
        """Test ToolError with all attributes."""
        error = ToolError(
            message="API rate limit exceeded",
            error_code=ErrorCode.RATE_LIMITED,
            recoverable=True,
            details={"retry_after": 60},
            tool_name="web_search",
        )

        assert error.message == "API rate limit exceeded"
        assert error.error_code == ErrorCode.RATE_LIMITED
        assert error.recoverable is True
        assert error.details == {"retry_after": 60}
        assert error.tool_name == "web_search"

    def test_tool_error_to_dict(self):
        """Test ToolError serialization."""
        error = ToolError(
            message="Not found",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=False,
            tool_name="search",
        )

        data = error.to_dict()

        assert data["success"] is False
        assert data["error"]["message"] == "Not found"
        assert data["error"]["code"] == ErrorCode.NOT_FOUND
        assert data["error"]["recoverable"] is False
        assert data["error"]["tool_name"] == "search"


class TestToolResult:
    """Tests for ToolResult class."""

    def test_result_ok(self):
        """Test successful result."""
        result = ToolResult.ok({"items": [1, 2, 3]}, source="test")

        assert result.success is True
        assert result.data == {"items": [1, 2, 3]}
        assert result.error is None
        assert result.metadata == {"source": "test"}

    def test_result_fail(self):
        """Test failed result."""
        error = ToolError("Failed", error_code=ErrorCode.API_ERROR)
        result = ToolResult.fail(error, attempt=1)

        assert result.success is False
        assert result.data is None
        assert result.error == error
        assert result.metadata == {"attempt": 1}

    def test_result_to_dict_success(self):
        """Test successful result serialization."""
        result = ToolResult.ok({"count": 5})
        result.execution_time_ms = 10.123

        data = result.to_dict()

        assert data["success"] is True
        assert data["data"] == {"count": 5}
        assert data["execution_time_ms"] == 10.12
        assert "error" not in data

    def test_result_to_dict_failure(self):
        """Test failed result serialization."""
        error = ToolError("Timeout", error_code=ErrorCode.TIMEOUT)
        result = ToolResult.fail(error)

        data = result.to_dict()

        assert data["success"] is False
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.TIMEOUT


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_definition_basic(self):
        """Test basic tool definition."""

        def my_tool(query: str) -> dict:
            return {"result": query}

        definition = ToolDefinition(
            name="my_tool",
            description="A test tool",
            func=my_tool,
        )

        assert definition.name == "my_tool"
        assert definition.description == "A test tool"
        assert definition.func == my_tool
        assert definition.category == ToolCategory.OTHER
        assert definition.is_async is False

    def test_definition_async_detection(self):
        """Test async function detection."""

        async def async_tool(query: str) -> dict:
            return {"result": query}

        definition = ToolDefinition(
            name="async_tool",
            description="An async tool",
            func=async_tool,
        )

        assert definition.is_async is True

    def test_definition_with_metadata(self):
        """Test definition with full metadata."""

        def search(query: str) -> dict:
            return {}

        definition = ToolDefinition(
            name="search",
            description="Search tool",
            func=search,
            category=ToolCategory.SEARCH,
            requires_api_key="SERPER_API_KEY",
            timeout_seconds=60,
            rate_limit=100,
            metadata={"version": "1.0"},
        )

        assert definition.category == ToolCategory.SEARCH
        assert definition.requires_api_key == "SERPER_API_KEY"
        assert definition.timeout_seconds == 60
        assert definition.rate_limit == 100
        assert definition.metadata == {"version": "1.0"}


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_empty(self):
        """Test empty registry."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.list_tools() == []
        assert registry.get("nonexistent") is None

    def test_registry_register_direct(self):
        """Test direct registration."""
        registry = ToolRegistry()

        def my_tool(x: int) -> int:
            """Multiply by two."""
            return x * 2

        registry.register(my_tool, category=ToolCategory.ANALYSIS)

        assert len(registry) == 1
        assert "my_tool" in registry
        tool = registry.get("my_tool")
        assert tool is not None
        assert tool.description == "Multiply by two."
        assert tool.category == ToolCategory.ANALYSIS

    def test_registry_register_decorator(self):
        """Test decorator registration."""
        registry = ToolRegistry()

        @registry.register(category=ToolCategory.SEARCH)
        def search_tool(query: str) -> dict:
            """Search for things."""
            return {"results": []}

        assert len(registry) == 1
        assert "search_tool" in registry
        assert registry.get("search_tool").category == ToolCategory.SEARCH

    def test_registry_custom_name(self):
        """Test registration with custom name."""
        registry = ToolRegistry()

        @registry.register(name="custom_search", description="Custom search")
        def internal_search(q: str) -> dict:
            return {}

        assert "custom_search" in registry
        assert "internal_search" not in registry
        assert registry.get("custom_search").description == "Custom search"

    def test_registry_get_functions(self):
        """Test getting all functions."""
        registry = ToolRegistry()

        def tool_a():
            pass

        def tool_b():
            pass

        registry.register(tool_a)
        registry.register(tool_b)

        functions = registry.get_functions()
        assert len(functions) == 2
        assert tool_a in functions
        assert tool_b in functions

    def test_registry_list_by_category(self):
        """Test listing tools by category."""
        registry = ToolRegistry()

        @registry.register(category=ToolCategory.SEARCH)
        def search1():
            pass

        @registry.register(category=ToolCategory.SEARCH)
        def search2():
            pass

        @registry.register(category=ToolCategory.EXECUTION)
        def execute():
            pass

        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        assert len(search_tools) == 2

        exec_tools = registry.list_by_category(ToolCategory.EXECUTION)
        assert len(exec_tools) == 1


class TestWithResultWrapper:
    """Tests for with_result_wrapper decorator."""

    def test_wrapper_success(self):
        """Test wrapper with successful function."""

        @with_result_wrapper
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)

        assert result["success"] is True
        assert result["data"] == 5
        assert "execution_time_ms" in result

    def test_wrapper_tool_error(self):
        """Test wrapper with ToolError."""

        @with_result_wrapper
        def failing_tool():
            raise ToolError(
                "Intentional failure",
                error_code=ErrorCode.VALIDATION_FAILED,
            )

        result = failing_tool()

        assert result["success"] is False
        assert result["error"]["code"] == ErrorCode.VALIDATION_FAILED
        assert result["error"]["tool_name"] == "failing_tool"

    def test_wrapper_unexpected_error(self):
        """Test wrapper with unexpected exception."""

        @with_result_wrapper
        def buggy_tool():
            raise ValueError("Unexpected bug")

        result = buggy_tool()

        assert result["success"] is False
        assert result["error"]["code"] == ErrorCode.INTERNAL_ERROR
        assert "Unexpected bug" in result["error"]["message"]


class TestRetry:
    """Tests for retry decorator."""

    def test_retry_succeeds_first_try(self):
        """Test retry when function succeeds immediately."""
        call_count = 0

        @retry(max_attempts=3)
        def always_works():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_works()

        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Test retry succeeds after initial failures."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = fails_twice()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry raises after all attempts exhausted."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError, match="Permanent failure"):
            always_fails()

        assert call_count == 3

    def test_retry_non_retryable_error(self):
        """Test retry doesn't retry non-retryable errors."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            raises_value_error()

        assert call_count == 1  # Only one attempt

    def test_retry_tool_error_recoverable(self):
        """Test retry retries recoverable ToolError."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def recoverable_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ToolError("Rate limited", recoverable=True)
            return "success"

        result = recoverable_error()

        assert result == "success"
        assert call_count == 3

    def test_retry_tool_error_not_recoverable(self):
        """Test retry doesn't retry non-recoverable ToolError."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def non_recoverable_error():
            nonlocal call_count
            call_count += 1
            raise ToolError(
                "Invalid input",
                error_code=ErrorCode.INVALID_INPUT,
                recoverable=False,
            )

        with pytest.raises(ToolError, match="Invalid input"):
            non_recoverable_error()

        assert call_count == 1


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_circuit_starts_closed(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=60)

        @breaker
        def protected_call():
            return "success"

        # Open the circuit
        breaker.record_failure()

        with pytest.raises(ToolError) as exc_info:
            protected_call()

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_circuit_allows_when_closed(self):
        """Test circuit allows calls when closed."""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker
        def protected_call():
            return "success"

        result = protected_call()

        assert result == "success"

    def test_circuit_records_success(self):
        """Test circuit records successes."""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker
        def protected_call():
            return "success"

        # Add some failures
        breaker.record_failure()
        breaker.record_failure()

        # Success should reset failure count
        protected_call()

        assert breaker._failure_count == 0

    def test_circuit_reset(self):
        """Test circuit can be manually reset."""
        breaker = CircuitBreaker(failure_threshold=1)

        breaker.record_failure()
        assert breaker.is_open

        breaker.reset()

        assert breaker.is_closed
        assert breaker._failure_count == 0


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_limiter_allows_within_rate(self):
        """Test limiter allows calls within rate."""
        limiter = RateLimiter(rate=100, burst=10)

        # Should allow burst calls immediately
        for _ in range(10):
            assert limiter.acquire(blocking=False)

    def test_limiter_blocks_over_burst(self):
        """Test limiter blocks calls over burst."""
        limiter = RateLimiter(rate=10, burst=3)

        # Exhaust burst
        for _ in range(3):
            limiter.acquire(blocking=False)

        # Next call should fail (non-blocking)
        assert not limiter.acquire(blocking=False)

    def test_limiter_refills_tokens(self):
        """Test limiter refills tokens over time."""
        import time

        limiter = RateLimiter(rate=100, burst=5)  # 100/sec = refill quickly

        # Exhaust tokens
        for _ in range(5):
            limiter.acquire(blocking=False)

        # Wait for refill (100/sec = 0.01 sec per token)
        time.sleep(0.05)  # Should refill ~5 tokens

        # Should be able to acquire again
        assert limiter.acquire(blocking=False)

    def test_limiter_as_decorator(self):
        """Test limiter as decorator."""
        limiter = RateLimiter(rate=100, burst=5)

        @limiter
        def rate_limited_call():
            return "success"

        # Should allow calls within burst
        for _ in range(5):
            result = rate_limited_call()
            assert result == "success"


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert ErrorCode.RATE_LIMITED in config.retryable_codes

    def test_config_as_decorator(self):
        """Test config can be converted to decorator."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        call_count = 0

        @config.as_decorator()
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            failing_func()

        assert call_count == 2
