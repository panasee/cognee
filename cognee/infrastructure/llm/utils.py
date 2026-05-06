import asyncio

import litellm

from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    get_llm_client,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

CONNECTION_TEST_TIMEOUT_SECONDS = 30


def get_max_chunk_tokens() -> int:
    """
    Calculate the maximum number of tokens allowed in a chunk.

    The function determines the maximum chunk size based on the maximum token limit of the
    embedding engine and half of the LLM maximum context token size. It ensures that the
    chunk size does not exceed these constraints.

    Returns:
    --------

        - int: The maximum number of tokens that can be included in a chunk, determined by
          the smaller value of the embedding engine's max tokens and half of the LLM's
          maximum tokens.
    """
    # NOTE: Import must be done in function to avoid circular import issue
    from cognee.infrastructure.databases.vector import get_vector_engine

    # Calculate max chunk size based on the following formula
    embedding_engine = get_vector_engine().embedding_engine
    llm_client = get_llm_client(raise_api_key_error=False)

    # We need to make sure chunk size won't take more than half of LLM max context token size
    # but it also can't be bigger than the embedding engine max token size
    llm_cutoff_point = llm_client.max_completion_tokens // 2  # Round down the division
    max_chunk_tokens = min(embedding_engine.max_completion_tokens, llm_cutoff_point)

    return max_chunk_tokens


def get_model_max_completion_tokens(model_name: str) -> int | None:
    """
    Retrieve the maximum token limit for a specified model name if it exists.

    Checks if the provided model name is present in the predefined model cost dictionary. If
    found, it logs the maximum token count for that model and returns it. If the model name
    is not recognized, it logs an informational message and returns None.

    Parameters:
    -----------

        - model_name (str): Name of LLM or embedding model

    Returns:
    --------

        Number of max tokens of model, or None if model is unknown
    """
    max_completion_tokens: int | None = None

    if model_name in litellm.model_cost:
        if "max_tokens" in litellm.model_cost[model_name]:
            max_completion_tokens = litellm.model_cost[model_name]["max_tokens"]
            logger.debug(f"Max input tokens for {model_name}: {max_completion_tokens}")
        else:
            logger.debug(
                f"Model max_tokens not found in LiteLLM's model_cost for model {model_name}."
            )
    else:
        logger.debug("Model not found in LiteLLM's model_cost.")

    return max_completion_tokens


async def test_llm_connection() -> None:
    """
    Test connectivity to the LLM endpoint using a minimal completion call.

    Uses litellm.acompletion with a single-token ping rather than the structured-output
    path, so the preflight stays uniform across providers and isn't subject to the
    instructor + tenacity retry stack inside the provider adapters (which can mask a
    fast failure behind exponential backoff and trigger spurious timeouts).
    """
    llm_config = get_llm_config()
    model = llm_config.llm_model
    provider = llm_config.llm_provider
    # Some configs duplicate the provider in the model id (e.g. provider="anthropic",
    # model="anthropic/claude-..."); pass an unprefixed model and let
    # custom_llm_provider drive routing.
    if provider and model.startswith(f"{provider}/"):
        model = model[len(provider) + 1 :]

    try:
        logger.info("Testing connection to LLM endpoint...")
        await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                custom_llm_provider=provider or None,
                api_key=llm_config.llm_api_key,
                api_base=llm_config.llm_endpoint or None,
                api_version=llm_config.llm_api_version or None,
                messages=[{"role": "user", "content": "hi"}],
                # Reasoning models (o1/o3/gpt-5) consume the budget on hidden reasoning
                # before emitting output and 400 if the cap is too small, so leave headroom.
                max_tokens=256,
                num_retries=0,
            ),
            timeout=CONNECTION_TEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        msg = (
            f"LLM connection test timed out after {CONNECTION_TEST_TIMEOUT_SECONDS}s. "
            "Check that your LLM endpoint is reachable and responding. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise TimeoutError(msg)
    except litellm.exceptions.AuthenticationError as e:
        msg = (
            "LLM authentication failed. Check your LLM_API_KEY configuration. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise e
    except litellm.exceptions.BadRequestError as e:
        # A "max_tokens / output limit reached" 400 still proves the endpoint is reachable
        # and the credentials are accepted — treat it as a passing preflight rather than
        # failing the whole pipeline on an artifact of our tiny ping.
        if "max_tokens" in str(e).lower() or "output limit" in str(e).lower():
            logger.debug("LLM preflight hit token-limit response; treating as reachable.")
            return
        logger.error(e)
        logger.error("Connection to LLM could not be established.")
        raise e
    except Exception as e:
        logger.error(e)
        logger.error("Connection to LLM could not be established.")
        raise e


async def test_embedding_connection() -> None:
    """
    Test the connection to the embedding engine by embedding a sample text.

    Handles exceptions that may occur during the operation, logs the error, and re-raises
    the exception if the connection to the embedding handler cannot be established.
    Wrapped in a timeout to prevent indefinite hangs.
    """
    try:
        # NOTE: Vector engine import must be done in function to avoid circular import issue
        from cognee.infrastructure.databases.vector import get_vector_engine

        logger.info("Testing connection to Embedding endpoint...")
        await asyncio.wait_for(
            get_vector_engine().embedding_engine.embed_text(["test"]),
            timeout=CONNECTION_TEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        msg = (
            f"Embedding connection test timed out after {CONNECTION_TEST_TIMEOUT_SECONDS}s. "
            "Check that your embedding endpoint is reachable. "
            "Set COGNEE_SKIP_CONNECTION_TEST=true to bypass this check."
        )
        logger.error(msg)
        raise TimeoutError(msg)
    except Exception as e:
        logger.error(e)
        logger.error("Connection to Embedding handler could not be established.")
        raise e
