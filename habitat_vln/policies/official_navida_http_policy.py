"""Habitat-side HTTP client for the separately hosted Official NaVIDA policy."""

import base64
from io import BytesIO
import json
import math
from typing import Any, Dict, Mapping, Optional
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

try:
    from ..core import NavigationObservation, PolicyOutput
except ImportError:
    from core import NavigationObservation, PolicyOutput

from .prompts import VALID_ACTIONS


DEFAULT_OFFICIAL_NAVIDA_URL = "http://127.0.0.1:8008"


class _UrllibJSONTransport:
    """Small standard-library JSON transport used outside unit tests."""

    def request(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
        return json.loads(response_body)


class OfficialNaVIDAHTTPPolicy:
    """Send public RGB observations to an Official NaVIDA HTTP server."""

    policy_protocol = "paper_pure"

    def __init__(
        self,
        base_url: str = DEFAULT_OFFICIAL_NAVIDA_URL,
        timeout: float = 30.0,
        transport: Any = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0.0
        ):
            raise ValueError(
                "timeout must be a finite number greater than zero"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.transport = transport or _UrllibJSONTransport()
        self.episode_id = None  # type: Optional[str]
        self._episode_start_error = None  # type: Optional[str]
        self._closed = False

    def start_episode(self, episode_id: str) -> None:
        """Reset the server-owned history and action queue for one episode."""
        episode_id = str(episode_id)
        self.episode_id = episode_id
        self._episode_start_error = None
        try:
            response = self.transport.request(
                "POST",
                f"{self.base_url}/v1/episodes/start",
                payload={"episode_id": episode_id},
                timeout=self.timeout,
            )
        except Exception as exc:
            self._episode_start_error = (
                f"Official NaVIDA episode start request failed: {exc}"
            )
            return
        if not isinstance(response, Mapping):
            self._episode_start_error = (
                "Official NaVIDA episode response must be a JSON object"
            )
            return
        if (
            response.get("episode_id") != episode_id
            or response.get("reset") is not True
        ):
            self._episode_start_error = (
                "Official NaVIDA server did not reset the requested episode"
            )
            return

    def predict(self, observation: NavigationObservation) -> PolicyOutput:
        """Return at most one server-selected atomic action for this simulator step."""
        if not isinstance(observation, NavigationObservation):
            return self._invalid_output(
                "Official NaVIDA requires a NavigationObservation",
                termination_reason="invalid_observation",
            )
        if self.episode_id is None:
            return self._invalid_output(
                "start_episode must be called before predict",
                termination_reason="episode_not_started",
            )
        if self._episode_start_error is not None:
            return self._invalid_output(
                self._episode_start_error,
                termination_reason="episode_start_failed",
            )
        if not isinstance(observation.step, int) or isinstance(observation.step, bool):
            return self._invalid_output(
                "simulator step must be an integer",
                termination_reason="invalid_observation",
            )
        if observation.step < 0:
            return self._invalid_output(
                "simulator step must not be negative",
                termination_reason="invalid_observation",
            )
        if not isinstance(observation.instruction, str):
            return self._invalid_output(
                "instruction must be a string",
                termination_reason="invalid_observation",
            )

        try:
            rgb_png_base64 = self._encode_rgb_png(observation.rgb)
        except Exception as exc:
            return self._invalid_output(
                f"Official NaVIDA RGB encoding failed: {exc}",
                termination_reason="invalid_observation",
            )

        try:
            response = self.transport.request(
                "POST",
                f"{self.base_url}/v1/steps",
                payload={
                    "episode_id": self.episode_id,
                    "simulator_step": observation.step,
                    "instruction": observation.instruction,
                    "rgb_png_base64": rgb_png_base64,
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            return self._invalid_output(
                f"Official NaVIDA HTTP request failed: {exc}",
                termination_reason="http_request_failed",
            )

        return self._policy_output_from_response(response, observation.step)

    def clear_action_queue(self) -> None:
        """Do nothing because the Official NaVIDA server owns its action queue."""

    def close(self) -> None:
        """Close an injected transport once when it exposes a close hook."""
        if self._closed:
            return
        self._closed = True
        close_transport = getattr(self.transport, "close", None)
        if callable(close_transport):
            close_transport()

    @staticmethod
    def _encode_rgb_png(rgb: Any) -> str:
        """Encode one RGB array as a lossless PNG base64 string."""
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _policy_output_from_response(
        self,
        response: Any,
        simulator_step: int,
    ) -> PolicyOutput:
        if not isinstance(response, Mapping):
            return self._invalid_output("step response must be a JSON object")

        required_fields = {
            "episode_id",
            "simulator_step",
            "decision_step",
            "action",
            "valid",
            "inferred",
            "raw_output",
            "error",
            "termination_reason",
            "latency_seconds",
            "metadata",
        }
        missing_fields = sorted(required_fields - set(response))
        if missing_fields:
            return self._invalid_output(
                f"step response is missing fields: {', '.join(missing_fields)}",
                response=response,
            )

        error = self._response_validation_error(response, simulator_step)
        if error is not None:
            return self._invalid_output(error, response=response)

        metadata = self._response_metadata(response)
        is_valid = response["valid"]
        return PolicyOutput(
            action=response["action"] if is_valid else None,
            raw_text=response["raw_output"],
            is_valid=is_valid,
            termination_reason=response["termination_reason"],
            metadata=metadata,
        )

    def _response_validation_error(
        self,
        response: Mapping[str, Any],
        simulator_step: int,
    ) -> Optional[str]:
        if response["episode_id"] != self.episode_id:
            return "step response episode_id does not match the active episode"
        if not isinstance(response["simulator_step"], int) or isinstance(
            response["simulator_step"], bool
        ):
            return "step response simulator_step must be an integer"
        if response["simulator_step"] != simulator_step:
            return "step response simulator_step does not match the request"
        if not isinstance(response["decision_step"], int) or isinstance(
            response["decision_step"], bool
        ):
            return "step response decision_step must be an integer"
        if response["decision_step"] < 0:
            return "step response decision_step must not be negative"
        if not isinstance(response["valid"], bool):
            return "step response valid must be a boolean"
        if not isinstance(response["inferred"], bool):
            return "step response inferred must be a boolean"
        if not isinstance(response["raw_output"], str):
            return "step response raw_output must be a string"
        if response["error"] is not None and not isinstance(response["error"], str):
            return "step response error must be a string or null"
        if response["termination_reason"] is not None and not isinstance(
            response["termination_reason"], str
        ):
            return "step response termination_reason must be a string or null"
        if not isinstance(response["latency_seconds"], (int, float)) or isinstance(
            response["latency_seconds"], bool
        ):
            return "step response latency_seconds must be a number"
        latency_seconds = response["latency_seconds"]
        if latency_seconds < 0.0 or (
            isinstance(latency_seconds, float)
            and not math.isfinite(latency_seconds)
        ):
            return "step response latency_seconds must be finite and non-negative"
        if not isinstance(response["metadata"], Mapping):
            return "step response metadata must be a JSON object"
        if response["valid"]:
            if not isinstance(response["action"], str):
                return "a valid step response action must be a string"
            if response["action"] not in VALID_ACTIONS:
                return "step response contains an invalid action"
        elif response["action"] is not None:
            return "an invalid step response must not contain an action"
        return None

    @staticmethod
    def _response_metadata(response: Mapping[str, Any]) -> Dict[str, Any]:
        server_metadata = response.get("metadata", {})
        if not isinstance(server_metadata, Mapping):
            server_metadata = {}
        return {
            "raw_output": response.get("raw_output", ""),
            "termination_reason": response.get("termination_reason"),
            "error": response.get("error"),
            "latency_seconds": response.get("latency_seconds"),
            "inferred": response.get("inferred"),
            "decision_step": response.get("decision_step"),
            "server_metadata": dict(server_metadata),
        }

    def _invalid_output(
        self,
        error: str,
        termination_reason: Optional[str] = None,
        response: Any = None,
    ) -> PolicyOutput:
        response = response if isinstance(response, Mapping) else {}
        metadata = self._response_metadata(response)
        server_error = metadata.get("error")
        if server_error is not None and server_error != error:
            metadata["server_error"] = server_error
        metadata["error"] = error
        if termination_reason is None:
            response_reason = response.get("termination_reason")
            if isinstance(response_reason, str):
                termination_reason = response_reason
        return PolicyOutput(
            action=None,
            raw_text=(
                response.get("raw_output", "")
                if isinstance(response.get("raw_output", ""), str)
                else ""
            ),
            is_valid=False,
            termination_reason=termination_reason,
            metadata=metadata,
        )
