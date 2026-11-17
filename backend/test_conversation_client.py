"""Tests use the actual CX protobuf boundary, mocking only network clients."""

from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.api_core import exceptions as api_errors
from google.cloud import dialogflowcx_v3 as cx
from google.protobuf import json_format

from conversation_client import ConversationAnswerClient, ConversationError
from provision_conversation import (
    DEFAULT_RESOURCE_ID, Provisioner, build_event_handler, build_generator, configure_flow,
    load_project_credentials, load_spec, main,
)


def settings(**overrides):
    values = dict(conversation_project_id="test-project", conversation_location="us-central1",
                  conversation_agent_id="agent", conversation_environment_id="production-id",
                  conversation_language_code="en", conversation_event_name="RAG_ANSWER",
                  conversation_timeout_seconds=30.0, conversation_max_payload_bytes=16384)
    values.update(overrides)
    return SimpleNamespace(**values)


def response(answer="Clean the brew group weekly. (p. 12)", **result):
    return cx.DetectIntentResponse(query_result=cx.QueryResult(
        parameters={"manual_answer": answer}, **result
    ))


def test_event_serialization_uses_separate_parameters_and_production_environment():
    network = Mock()
    network.detect_intent.return_value = response()
    ConversationAnswerClient(settings(), network).generate_answer("How?", "Page 12: rinse.")
    request = network.detect_intent.call_args.kwargs["request"]
    wire = cx.DetectIntentRequest.deserialize(cx.DetectIntentRequest.serialize(request))
    assert wire.session.startswith("projects/test-project/locations/us-central1/agents/agent/environments/production-id/sessions/")
    assert len(wire.session.rsplit("/", 1)[1]) <= 36
    assert cx.QueryInput.pb(wire.query_input).WhichOneof("input") == "event"
    assert wire.query_input.event.event == "RAG_ANSWER"
    assert wire.query_input.language_code == "en"
    assert dict(wire.query_params.parameters) == {"question": "How?", "manual_evidence": "Page 12: rinse."}
    assert network.detect_intent.call_args.kwargs["retry"] is None
    assert network.detect_intent.call_args.kwargs["timeout"] == 30.0


@pytest.mark.parametrize("region,endpoint", [
    ("us-central1", "us-central1-dialogflow.googleapis.com"),
    ("global", "dialogflow.googleapis.com"),
])
def test_constructor_uses_adc_with_location_endpoint(region, endpoint):
    with patch.object(cx, "SessionsClient") as factory:
        ConversationAnswerClient(settings(conversation_location=region))
    assert factory.call_args.kwargs["client_options"].api_endpoint == endpoint
    assert "credentials" not in factory.call_args.kwargs


def test_only_manual_answer_is_returned():
    network = Mock()
    network.detect_intent.return_value = response("  Grounded answer (p. 12)  ", response_messages=[
        cx.ResponseMessage(text=cx.ResponseMessage.Text(text=["Unrelated generic agent response"]))
    ])
    assert ConversationAnswerClient(settings(), network).generate_answer("Q", "E") == "Grounded answer (p. 12)"


@pytest.mark.parametrize("answer", [None, "", "   ", 42, True, {"answer": "wrong shape"}, ["answer"], "$session.params.manual_answer"])
def test_invalid_answer_is_rejected(answer):
    network = Mock()
    network.detect_intent.return_value = response(answer)
    with pytest.raises(ConversationError) as caught:
        ConversationAnswerClient(settings(), network).generate_answer("Q", "E")
    assert caught.value.status_code == 502


def test_generic_fallback_without_output_is_not_an_answer():
    network = Mock()
    network.detect_intent.return_value = cx.DetectIntentResponse(query_result=cx.QueryResult(
        response_messages=[cx.ResponseMessage(text=cx.ResponseMessage.Text(text=["Sorry, try again"]))]
    ))
    with pytest.raises(ConversationError):
        ConversationAnswerClient(settings(), network).generate_answer("Q", "E")


@pytest.mark.parametrize("result", [
    {"webhook_statuses": [{"code": 13, "message": "private upstream details"}]},
    {"trace_blocks": [{"end_state": cx.OutputState.OUTPUT_STATE_FAILED}]},
    {"trace_blocks": [{"end_state": cx.OutputState.OUTPUT_STATE_CANCELLED}]},
])
def test_failed_or_blocked_turn_is_rejected_even_if_output_exists(result):
    network = Mock()
    network.detect_intent.return_value = response(**result)
    with pytest.raises(ConversationError):
        ConversationAnswerClient(settings(), network).generate_answer("Q", "E")


@pytest.mark.parametrize("value", [None, {}, SimpleNamespace(query_result=SimpleNamespace(parameters={"manual_answer": "fake"}))])
def test_malformed_transport_response_is_rejected(value):
    network = Mock()
    network.detect_intent.return_value = value
    with pytest.raises(ConversationError):
        ConversationAnswerClient(settings(), network).generate_answer("Q", "E")


@pytest.mark.parametrize("upstream,status", [
    (api_errors.ResourceExhausted, 429), (api_errors.DeadlineExceeded, 504),
    (api_errors.PermissionDenied, 503), (api_errors.Unauthenticated, 503),
    (api_errors.ServiceUnavailable, 503), (api_errors.InvalidArgument, 502),
    (RuntimeError, 502),
])
def test_upstream_errors_are_safe_and_not_retried(upstream, status):
    network = Mock()
    network.detect_intent.side_effect = upstream("SECRET question evidence credential")
    with pytest.raises(ConversationError) as caught:
        ConversationAnswerClient(settings(), network).generate_answer("Q", "E")
    assert caught.value.status_code == status
    assert "SECRET" not in str(caught.value)
    assert caught.value.public_message == str(caught.value)
    assert network.detect_intent.call_count == 1


@pytest.mark.parametrize("question,evidence", [(None, "E"), ("", "E"), ("  ", "E"), ("Q", None), ("Q", 2), ("\ud800", "E")])
def test_invalid_input_never_calls_network(question, evidence):
    network = Mock()
    with pytest.raises(ConversationError) as caught:
        ConversationAnswerClient(settings(), network).generate_answer(question, evidence)
    assert caught.value.status_code == 400
    network.detect_intent.assert_not_called()


def test_utf8_envelope_size_is_enforced_at_boundary_without_truncation():
    network = Mock()
    network.detect_intent.return_value = response()
    config = settings()
    adapter = ConversationAnswerClient(config, network)
    question, evidence = "Café?", "☕" * 80
    adapter.generate_answer(question, evidence)
    request = network.detect_intent.call_args.kwargs["request"]
    size = len(json_format.MessageToJson(cx.DetectIntentRequest.pb(request), ensure_ascii=False).encode("utf-8"))
    config.conversation_max_payload_bytes = size
    adapter.generate_answer(question, evidence)
    config.conversation_max_payload_bytes = size - 1
    with pytest.raises(ConversationError) as caught:
        adapter.generate_answer(question, evidence)
    assert caught.value.status_code == 413
    assert network.detect_intent.call_count == 2


def test_concurrent_answers_have_unique_sessions_and_no_shared_parameters():
    network = Mock()
    def answer_call(*, request, **kwargs):
        # Serialize and decode to exercise real Struct and message marshalling.
        wire = cx.DetectIntentRequest.deserialize(cx.DetectIntentRequest.serialize(request))
        return response(wire.query_params.parameters["question"])
    network.detect_intent.side_effect = answer_call
    adapter = ConversationAnswerClient(settings(), network)
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda n: adapter.generate_answer(f"Question {n}", f"Evidence {n}"), range(20)))
    sessions = [call.kwargs["request"].session for call in network.detect_intent.call_args_list]
    assert len(set(sessions)) == 20
    assert answers == [f"Question {n}" for n in range(20)]


def test_generator_and_event_artifacts_cross_real_protobuf_boundary():
    config, prompt, digest = load_spec("gemini-2.5-flash")
    generator = build_generator(cx, config, prompt, "gemini-2.5-flash", digest)
    generator = cx.Generator.deserialize(cx.Generator.serialize(generator))
    assert {p.id: p.name for p in generator.placeholders} == {"question": "question", "manual_evidence": "manual_evidence"}
    assert "$last-user-utterance" not in generator.prompt_text.text
    event = build_event_handler(cx, config, "projects/p/locations/l/agents/a/generators/g")
    event = cx.EventHandler.deserialize(cx.EventHandler.serialize(event))
    assert event.event == "RAG_ANSWER"
    assert dict(event.trigger_fulfillment.generators[0].input_parameters) == {"question": "$session.params.question", "manual_evidence": "$session.params.manual_evidence"}
    assert event.trigger_fulfillment.generators[0].output_parameter == "manual_answer"
    assert event.trigger_fulfillment.messages[0].text.text == ["$session.params.manual_answer"]


def test_default_dry_run_is_offline_and_requires_no_cloud_client(capsys):
    with patch("provision_conversation.Provisioner", side_effect=AssertionError("must remain offline")), \
            patch("provision_conversation.load_project_credentials", side_effect=AssertionError("must not load credentials")):
        assert main(["--project", "p", "--location", "us-central1"]) == 0
    assert '"cloud_mutations": false' in capsys.readouterr().out


def test_cloud_mode_loads_project_credentials_before_constructing_clients():
    with patch("provision_conversation.load_project_credentials") as load_credentials, \
            patch("provision_conversation.Provisioner") as factory:
        def construct(*args, **kwargs):
            load_credentials.assert_called_once_with()
            return SimpleNamespace(inspect=lambda config: {"agent": None})
        factory.side_effect = construct
        assert main(["--project", "p", "--location", "us-central1", "--inspect"]) == 0


def test_relative_dotenv_credential_path_resolves_against_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    (tmp_path / ".env").write_text('GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"\n', encoding="utf-8")
    load_project_credentials(tmp_path)
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str((tmp_path / "credentials/service-account.json").resolve())


def test_shell_credentials_take_precedence_over_project_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "shell-credentials.json")
    (tmp_path / ".env").write_text("GOOGLE_APPLICATION_CREDENTIALS=project-credentials.json\n", encoding="utf-8")
    load_project_credentials(tmp_path)
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "shell-credentials.json"


def test_missing_project_credentials_leave_standard_adc_available(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    (tmp_path / ".env").write_text("ANSWER_PROVIDER=irrelevant\n", encoding="utf-8")
    load_project_credentials(tmp_path)
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_apply_requires_explicit_model_before_constructing_client():
    with patch("provision_conversation.Provisioner", side_effect=AssertionError("must not construct")):
        with pytest.raises(SystemExit) as caught:
            main(["--project", "p", "--location", "us-central1", "--apply"])
    assert caught.value.code == 2


def test_provisioning_refuses_unowned_agent_with_reserved_display_name():
    config, _, _ = load_spec("model")
    agents = Mock()
    agents.list_agents.return_value = [cx.Agent(display_name=config["display_name"], description="someone else's agent")]
    provisioner = Provisioner("p", "us-central1", clients={"agents": agents})
    with pytest.raises(ValueError, match="ownership"):
        provisioner.find_agent(config)
    assert len(agents.method_calls) == 1


def managed_resources():
    config, prompt, digest = load_spec("model")
    agent_name = "projects/p/locations/us-central1/agents/a"
    flow_name = f"{agent_name}/flows/{DEFAULT_RESOURCE_ID}"
    agent = cx.Agent(name=agent_name, display_name=config["display_name"], description=config["ownership_marker"],
                     start_flow=flow_name, time_zone=config["time_zone"], default_language_code=config["default_language_code"])
    generator = build_generator(cx, config, prompt, "model", digest)
    generator.name = f"{agent_name}/generators/g"
    flow = cx.Flow(name=flow_name, event_handlers=[build_event_handler(cx, config, generator.name)])
    flow, _ = configure_flow(cx, config, flow, agent_name, generator.name)
    version = cx.Version(name=f"{flow_name}/versions/1", display_name=f"coffee-maestro-{digest}", state=cx.Version.State.SUCCEEDED)
    environment = cx.Environment(name=f"{agent_name}/environments/e", display_name=config["environment_display_name"], version_configs=[{"version": version.name}])
    clients = {key: Mock() for key in ("agents", "generators", "flows", "versions", "environments")}
    clients["agents"].list_agents.return_value = [agent]
    clients["generators"].list_generators.return_value = [generator]
    clients["flows"].get_flow.return_value = flow
    clients["versions"].list_versions.return_value = [version]
    clients["environments"].list_environments.return_value = [environment]
    return config, prompt, digest, clients, generator, flow, version, environment


def test_apply_is_idempotent_after_real_proto_resources_already_exist():
    config, prompt, digest, clients, *_ = managed_resources()
    provisioner = Provisioner("p", "us-central1", clients=clients)
    first = provisioner.apply(config, prompt, "model", digest)
    second = provisioner.apply(config, prompt, "model", digest)
    assert first == second
    assert all(not call[0].startswith(("create", "update", "delete")) for client in clients.values() for call in client.method_calls)


def test_existing_version_with_drifted_draft_is_refused_before_mutation():
    config, prompt, digest, clients, _, flow, *_ = managed_resources()
    flow.event_handlers = []
    with pytest.raises(ValueError, match="drifted"):
        Provisioner("p", "us-central1", clients=clients).apply(config, prompt, "model", digest)
    assert all(not call[0].startswith(("create", "update", "delete")) for client in clients.values() for call in client.method_calls)


def test_generator_content_hash_drift_is_refused_before_mutation():
    config, prompt, digest, clients, generator, *_ = managed_resources()
    generator.prompt_text.text = "different prompt"
    with pytest.raises(ValueError, match="content hash"):
        Provisioner("p", "us-central1", clients=clients).apply(config, prompt, "model", digest)
    assert all(not call[0].startswith(("create", "update", "delete")) for client in clients.values() for call in client.method_calls)


def test_inspect_only_reads_resource_metadata():
    config, _, _, clients, _, _, version, environment = managed_resources()
    result = Provisioner("p", "us-central1", clients=clients).inspect(config)
    assert result["environments"][0]["versions"] == [version.name]
    assert result["environments"][0]["name"] == environment.name
    assert all(call[0].startswith("list") for client in clients.values() for call in client.method_calls)


def test_new_resources_are_created_and_deployed_using_actual_proto_schemas():
    config, prompt, digest, clients, generator, flow, version, environment = managed_resources()
    agent = clients["agents"].list_agents.return_value[0]
    for key in ("agents", "generators", "versions", "environments"):
        getattr(clients[key], f"list_{key}").return_value = []
    clients["agents"].create_agent.return_value = agent
    clients["generators"].create_generator.return_value = generator
    # New agent has the default flow and no managed event yet.
    flow.event_handlers = [cx.EventHandler(event="sys.no-match-default")]
    flow.transition_routes = []
    clients["versions"].create_version.return_value.result.return_value = version
    clients["environments"].create_environment.return_value.result.return_value = environment
    result = Provisioner("p", "us-central1", clients=clients).apply(config, prompt, "model", digest)
    assert result["environment"] == environment.name
    request = clients["flows"].update_flow.call_args.kwargs["request"]
    decoded = cx.UpdateFlowRequest.deserialize(cx.UpdateFlowRequest.serialize(cx.UpdateFlowRequest(request)))
    assert [handler.event for handler in decoded.flow.event_handlers] == ["sys.no-match-default", "RAG_ANSWER", "sys.no-input-default", "sys.long-utterance"]
    assert list(decoded.update_mask.paths) == ["event_handlers", "transition_routes"]
    for key, method, request_type in (
        ("agents", "create_agent", cx.CreateAgentRequest),
        ("generators", "create_generator", cx.CreateGeneratorRequest),
        ("versions", "create_version", cx.CreateVersionRequest),
        ("environments", "create_environment", cx.CreateEnvironmentRequest),
    ):
        arguments = getattr(clients[key], method).call_args.kwargs
        assert request_type.deserialize(request_type.serialize(request_type(arguments["request"])))
        assert arguments["retry"] is None


def test_existing_agent_target_requires_adoption_before_any_mutation():
    config, prompt, digest, clients, *_ = managed_resources()
    agent = cx.Agent(clients["agents"].list_agents.return_value[0])
    agent.description = "User-owned existing agent"
    clients["agents"].get_agent.return_value = agent
    with pytest.raises(ValueError, match="ownership"):
        Provisioner("p", "us-central1", agent_id="a", clients=clients).apply(config, prompt, "model", digest)
    clients["agents"].get_agent.assert_called_once_with(
        request={"name": "projects/p/locations/us-central1/agents/a"}, retry=None, timeout=60)
    assert all(not call[0].startswith(("create", "update", "delete", "list"))
               for client in clients.values() for call in client.method_calls)


def test_explicit_missing_agent_never_creates_a_replacement():
    config, prompt, digest, clients, *_ = managed_resources()
    clients["agents"].get_agent.side_effect = api_errors.NotFound("missing target")
    with pytest.raises(api_errors.NotFound):
        Provisioner("p", "us-central1", agent_id="missing", adopt_existing=True, clients=clients).apply(config, prompt, "model", digest)
    clients["agents"].create_agent.assert_not_called()
    clients["agents"].list_agents.assert_not_called()


def test_adoption_switches_entry_oneof_preserves_user_settings_and_is_idempotent():
    config, prompt, digest, clients, *_ = managed_resources()
    original = cx.Agent(clients["agents"].list_agents.return_value[0])
    original.display_name = "Coffee Maestro Agent"
    original.description = "Existing project description"
    original.time_zone = "Europe/Kaliningrad"
    original.start_playbook = f"{original.name}/playbooks/{DEFAULT_RESOURCE_ID}"
    original.enable_spell_correction = True
    state = {"agent": original}
    clients["agents"].get_agent.side_effect = lambda **kwargs: state["agent"]
    def update_agent(*, request, **kwargs):
        wire = cx.UpdateAgentRequest.deserialize(cx.UpdateAgentRequest.serialize(cx.UpdateAgentRequest(request)))
        state["agent"] = cx.Agent(wire.agent)
        return state["agent"]
    clients["agents"].update_agent.side_effect = update_agent
    provisioner = Provisioner("p", "us-central1", agent_id="a", adopt_existing=True, clients=clients)
    first = provisioner.apply(config, prompt, "model", digest)
    second = provisioner.apply(config, prompt, "model", digest)
    assert first == second
    clients["agents"].update_agent.assert_called_once()
    patch_request = clients["agents"].update_agent.call_args.kwargs["request"]
    assert set(patch_request["update_mask"]["paths"]) == {"description", "time_zone", "start_flow"}
    assert state["agent"].display_name == "Coffee Maestro Agent"
    assert state["agent"].enable_spell_correction is True
    assert state["agent"].description == "Existing project description\n" + config["ownership_marker"]
    assert state["agent"].time_zone == "Africa/Johannesburg"
    assert cx.Agent.pb(state["agent"]).WhichOneof("session_entry_resource") == "start_flow"
    assert original.start_playbook.endswith("/playbooks/" + DEFAULT_RESOURCE_ID)
    assert not any(call[0].startswith("delete") for client in clients.values() for call in client.method_calls)
    clients["agents"].create_agent.assert_not_called()
    clients["agents"].list_agents.assert_not_called()


def test_flow_branding_preserves_unrelated_handlers_routes_actions_and_payloads():
    config, _, _, _, generator, flow, *_ = managed_resources()
    agent_name = flow.name.rsplit("/flows/", 1)[0]
    unrelated = cx.EventHandler(event="MY_EVENT", target_page=f"{flow.name}/pages/custom",
                                trigger_fulfillment={"messages": [{"text": {"text": ["Unrelated"]}}]})
    welcome = cx.TransitionRoute(intent=f"{agent_name}/intents/{DEFAULT_RESOURCE_ID}",
                                 target_page=f"{flow.name}/pages/welcome",
                                 trigger_fulfillment={"webhook": "preserve-me", "tag": "existing",
                                                      "messages": [{"text": {"text": ["Old greeting"]}}, {"payload": {"card": "keep"}}]})
    other_route = cx.TransitionRoute(condition="true", target_flow="other-flow")
    flow.transition_routes = [welcome, other_route]
    flow.event_handlers.append(unrelated)
    flow.description = "Unrelated flow description"
    result, paths = configure_flow(cx, config, flow, agent_name, generator.name)
    assert paths == ["transition_routes"]
    assert result.event_handlers[-1] == unrelated
    assert result.transition_routes[-1] == other_route
    assert result.transition_routes[0].target_page == welcome.target_page
    assert result.transition_routes[0].trigger_fulfillment.webhook == "preserve-me"
    assert dict(result.transition_routes[0].trigger_fulfillment.messages[1].payload) == {"card": "keep"}
    assert result.description == "Unrelated flow description"
    assert result.transition_routes[0].trigger_fulfillment.messages[0].text.text == [config["welcome_message"]]


def test_inspect_can_target_unowned_existing_agent_without_adopting_it():
    config, _, _, clients, *_ = managed_resources()
    agent = cx.Agent(clients["agents"].list_agents.return_value[0])
    agent.description = "not managed"
    clients["agents"].get_agent.return_value = agent
    result = Provisioner("p", "us-central1", agent_id="a", clients=clients).inspect(config)
    assert result["owned"] is False
    clients["agents"].create_agent.assert_not_called()
    clients["agents"].update_agent.assert_not_called()


def test_production_publish_keeps_unrelated_flow_version_configs():
    config, prompt, digest, clients, _, flow, version, environment = managed_resources()
    other_version = "projects/p/locations/us-central1/agents/a/flows/other/versions/9"
    environment.version_configs = [{"version": other_version}, {"version": flow.name + "/versions/old"}]
    clients["environments"].update_environment.return_value.result.return_value = environment
    Provisioner("p", "us-central1", clients=clients).apply(config, prompt, "model", digest)
    request = clients["environments"].update_environment.call_args.kwargs["request"]
    decoded = cx.UpdateEnvironmentRequest.deserialize(cx.UpdateEnvironmentRequest.serialize(cx.UpdateEnvironmentRequest(request)))
    assert [v.version for v in decoded.environment.version_configs] == [other_version, version.name]
    assert list(decoded.update_mask.paths) == ["version_configs"]


def test_adoption_cli_requires_explicit_target_before_cloud_client():
    with patch("provision_conversation.Provisioner", side_effect=AssertionError("must not construct")):
        with pytest.raises(SystemExit) as caught:
            main(["--project", "p", "--location", "us-central1", "--adopt-existing"])
    assert caught.value.code == 2


def test_existing_agent_dry_run_records_exact_target_and_stays_offline(capsys):
    with patch("provision_conversation.Provisioner", side_effect=AssertionError("must remain offline")):
        main(["--project", "p", "--location", "europe-west2", "--agent-id", "supplied-agent", "--adopt-existing"])
    output = capsys.readouterr().out
    assert '"agent_id": "supplied-agent"' in output
    assert '"adopt_existing": true' in output
