"""One independent, grounded answer per Dialogflow CX event request.

The payload ceiling is an application policy, not a claimed Google API limit.
Credentials use ADC and are never embedded in requests or error messages.
"""

import uuid


class ConversationError(RuntimeError):
    """Safe error boundary for the API; never include upstream exception text."""

    def __init__(self, public_message: str, status_code: int = 502):
        self.public_message = public_message
        self.status_code = status_code
        super().__init__(public_message)


def regional_endpoint(location: str) -> str:
    return "dialogflow.googleapis.com" if location == "global" else f"{location}-dialogflow.googleapis.com"


class ConversationAnswerClient:
    def __init__(self, settings, client=None):
        # Keep imports lazy so the explicit Gemini rollback can start without CX.
        from google.cloud import dialogflowcx_v3 as cx
        from google.api_core.client_options import ClientOptions

        self.settings = settings
        self._cx = cx
        self.client = client if client is not None else cx.SessionsClient(
            client_options=ClientOptions(api_endpoint=regional_endpoint(settings.conversation_location))
        )
        self._session_parent = (
            f"projects/{settings.conversation_project_id}/locations/{settings.conversation_location}"
            f"/agents/{settings.conversation_agent_id}/environments/{settings.conversation_environment_id}"
        )

    def generate_answer(self, question: str, evidence: str) -> str:
        from google.api_core import exceptions as api_errors
        from google.protobuf import json_format

        if not isinstance(question, str) or not question.strip() or not isinstance(evidence, str):
            raise ConversationError("The question and manual evidence must be valid text.", 400)

        try:
            request = self._cx.DetectIntentRequest(
                session=f"{self._session_parent}/sessions/{uuid.uuid4().hex}",
                query_input=self._cx.QueryInput(
                    event=self._cx.EventInput(event=self.settings.conversation_event_name),
                    language_code=self.settings.conversation_language_code,
                ),
                query_params=self._cx.QueryParameters(
                    parameters={"question": question, "manual_evidence": evidence}
                ),
            )
            # Count the entire UTF-8 JSON envelope, including escaped parameter text.
            payload_size = len(json_format.MessageToJson(
                self._cx.DetectIntentRequest.pb(request), ensure_ascii=False
            ).encode("utf-8"))
        except (UnicodeError, ValueError, TypeError):
            raise ConversationError("The question and manual evidence must be valid UTF-8 text.", 400) from None
        if payload_size > self.settings.conversation_max_payload_bytes:
            raise ConversationError("The question and retrieved evidence exceed the conversation payload limit.", 413)

        try:
            # detectIntent mutates session state and is documented as non-idempotent.
            response = self.client.detect_intent(
                request=request, retry=None, timeout=self.settings.conversation_timeout_seconds
            )
        except api_errors.ResourceExhausted:
            raise ConversationError("The conversation service is at capacity. Please retry later.", 429) from None
        except api_errors.DeadlineExceeded:
            raise ConversationError("The conversation service timed out. Please retry.", 504) from None
        except (api_errors.PermissionDenied, api_errors.Unauthenticated):
            raise ConversationError("The conversation service is not authorized. Check its cloud configuration.", 503) from None
        except api_errors.ServiceUnavailable:
            raise ConversationError("The conversation service is unavailable. Please retry later.", 503) from None
        except Exception:
            raise ConversationError("The conversation service could not generate a manual answer.", 502) from None

        if not isinstance(response, self._cx.DetectIntentResponse):
            raise ConversationError("The conversation service returned a malformed response.")
        result = response.query_result
        # These status objects are authoritative failure signals, not response prose.
        if any(status.code for status in result.webhook_statuses):
            raise ConversationError("The conversation service failed to produce a manual answer.")
        if any(block.end_state in (
            self._cx.OutputState.OUTPUT_STATE_CANCELLED,
            self._cx.OutputState.OUTPUT_STATE_FAILED,
            self._cx.OutputState.OUTPUT_STATE_ESCALATED,
        ) for block in result.trace_blocks):
            raise ConversationError("The conversation service failed to produce a manual answer.")
        params = result.parameters or {}
        answer = params.get("manual_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ConversationError("The conversation service returned no manual answer.")
        if answer.strip() == "$session.params.manual_answer":
            raise ConversationError("The conversation service returned an unresolved manual answer.")
        return answer.strip()
