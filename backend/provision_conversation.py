"""Inspect or provision only the dedicated Coffee Maestro Dialogflow CX agent.

Dry-run is offline and the default. --inspect is read-only; --apply is required
for mutations. A model must be explicitly selected and verified in the region.
No API enabling, IAM changes, deletion, or unrelated-agent edits occur here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from conversation_client import regional_endpoint


ARTIFACT_DIR = Path(__file__).resolve().parent / "conversation"
RPC_TIMEOUT = 60
OPERATION_TIMEOUT = 600
DEFAULT_RESOURCE_ID = "00000000-0000-0000-0000-000000000000"


def load_project_credentials(backend_dir=None):
    """Honor shell ADC first, then the backend's configured service account.

    Read only the credential-path setting: provisioning does not depend on
    answer-provider, Pinecone, or ingestion settings being valid. Relative
    paths from .env use the same backend directory as the app configuration.
    """
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        return
    from dotenv import dotenv_values

    backend_dir = Path(backend_dir) if backend_dir is not None else ARTIFACT_DIR.parent
    credential_path = dotenv_values(backend_dir / ".env").get("GOOGLE_APPLICATION_CREDENTIALS")
    if credential_path:
        candidate = Path(credential_path)
        if not candidate.is_absolute():
            candidate = (backend_dir / candidate).resolve()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)


def load_spec(model: str):
    config = json.loads((ARTIFACT_DIR / "agent_config.json").read_text(encoding="utf-8"))
    prompt = (ARTIFACT_DIR / "generator_prompt.txt").read_text(encoding="utf-8")
    spec = {"config": config, "model": model, "prompt": prompt}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return config, prompt, digest


def build_generator(cx, config, prompt, model, digest):
    return cx.Generator(
        display_name=f"{config['generator_display_name_prefix']} {digest}",
        prompt_text=cx.Phrase(text=prompt),
        placeholders=[cx.Generator.Placeholder(id=name, name=name)
                      for name in ("question", "manual_evidence")],
        llm_model_settings=cx.LlmModelSettings(model=model),
        model_parameter=cx.Generator.ModelParameter(
            temperature=config["temperature"], max_decode_steps=config["max_decode_steps"]
        ),
    )


def build_event_handler(cx, config, generator_name):
    return cx.EventHandler(
        event=config["event_name"],
        trigger_fulfillment=cx.Fulfillment(
            generators=[cx.Fulfillment.GeneratorSettings(
                generator=generator_name,
                # Live verification confirmed expressions are required: bare
                # names become literal generator inputs, not session bindings.
                input_parameters=config["generator_input_parameters"],
                output_parameter="manual_answer",
            )],
            messages=[cx.ResponseMessage(text=cx.ResponseMessage.Text(
                text=["$session.params.manual_answer"]
            ))],
        ),
    )


def unique_match(items, display_name, kind):
    matches = [item for item in items if item.display_name == display_name]
    if len(matches) > 1:
        raise ValueError(f"Multiple {kind} resources match; refusing ambiguous ownership.")
    return matches[0] if matches else None


def configure_flow(cx, config, flow, agent_name, generator_name):
    """Set managed responses while preserving all unrelated flow content."""
    wanted = cx.Flow(flow)
    handlers = list(wanted.event_handlers)
    event = config["event_name"]
    matches = [i for i, handler in enumerate(handlers) if handler.event == event]
    if len(matches) > 1:
        raise ValueError("Multiple managed RAG event handlers exist; refusing ambiguous configuration.")
    desired = build_event_handler(cx, config, generator_name)
    if matches:
        desired.name = handlers[matches[0]].name
        handlers[matches[0]] = desired
    else:
        handlers.append(desired)

    def set_response(fulfillment, text):
        # Keep payload/audio/channel-specific messages, webhook fields and actions.
        retained = [message for message in fulfillment.messages
                    if message.channel or cx.ResponseMessage.pb(message).WhichOneof("message") != "text"]
        fulfillment.messages = [cx.ResponseMessage(text=cx.ResponseMessage.Text(text=[text]))] + retained
        fulfillment.enable_generative_fallback = False

    for name, text in (
        ("sys.no-match-default", config["no_match_message"]),
        ("sys.no-input-default", config["no_input_message"]),
        ("sys.long-utterance", config["no_match_message"]),
    ):
        matches = [i for i, handler in enumerate(handlers) if handler.event == name]
        if len(matches) > 1:
            raise ValueError("Multiple default event handlers exist; refusing ambiguous configuration.")
        if matches:
            handler = cx.EventHandler(handlers[matches[0]])
            set_response(handler.trigger_fulfillment, text)
            handlers[matches[0]] = handler
        else:
            handler = cx.EventHandler(event=name)
            set_response(handler.trigger_fulfillment, text)
            handlers.append(handler)
    wanted.event_handlers = handlers

    routes = list(wanted.transition_routes)
    welcome_intent = f"{agent_name}/intents/{DEFAULT_RESOURCE_ID}"
    matches = [i for i, route in enumerate(routes) if route.intent == welcome_intent]
    if len(matches) > 1:
        raise ValueError("Multiple welcome routes exist; refusing ambiguous configuration.")
    if matches:
        route = cx.TransitionRoute(routes[matches[0]])
        set_response(route.trigger_fulfillment, config["welcome_message"])
        routes[matches[0]] = route
    else:
        route = cx.TransitionRoute(intent=welcome_intent)
        set_response(route.trigger_fulfillment, config["welcome_message"])
        routes.append(route)
    wanted.transition_routes = routes
    paths = [field for field in ("event_handlers", "transition_routes")
             if list(getattr(cx.Flow.pb(flow), field)) != list(getattr(cx.Flow.pb(wanted), field))]
    return wanted, paths


class Provisioner:
    def __init__(self, project, location, *, agent_id=None, adopt_existing=False, clients=None):
        from google.cloud import dialogflowcx_v3 as cx
        self.cx = cx
        self.project = project
        self.location = location
        self.parent = f"projects/{project}/locations/{location}"
        if agent_id is not None and (not agent_id or "/" in agent_id or any(char.isspace() for char in agent_id)):
            raise ValueError("Agent ID must be a single resource identifier.")
        if adopt_existing and not agent_id:
            raise ValueError("Adoption requires an explicit agent ID.")
        self.agent_id = agent_id
        self.adopt_existing = adopt_existing
        self.clients = clients if clients is not None else {
            key: factory(client_options={"api_endpoint": regional_endpoint(location)})
            for key, factory in {
                "agents": cx.AgentsClient, "generators": cx.GeneratorsClient,
                "flows": cx.FlowsClient, "versions": cx.VersionsClient,
                "environments": cx.EnvironmentsClient,
            }.items()
        }

    def call(self, service, method, **request):
        return getattr(self.clients[service], method)(request=request, retry=None, timeout=RPC_TIMEOUT)

    def find_agent(self, config, *, inspect_only=False):
        if self.agent_id:
            # An explicit target never falls back to discovery or duplicate creation.
            agent = self.call("agents", "get_agent", name=f"{self.parent}/agents/{self.agent_id}")
            expected_name = f"{self.parent}/agents/{self.agent_id}"
            if agent.name != expected_name:
                raise ValueError("The returned agent does not match the explicit resource target.")
        else:
            agent = unique_match(self.call("agents", "list_agents", parent=self.parent),
                                 config["display_name"], "agent")
        owned = agent and config["ownership_marker"] in agent.description.splitlines()
        if agent and not owned and not inspect_only and not self.adopt_existing:
            raise ValueError("An agent has the reserved name without the ownership marker; refusing to edit it.")
        return agent

    def inspect(self, config):
        agent = self.find_agent(config, inspect_only=True)
        if not agent:
            return {"agent": None, "project": self.project, "location": self.location}
        return {
            "agent": agent.name, "display_name": agent.display_name,
            "start_flow": agent.start_flow, "start_playbook": agent.start_playbook,
            "time_zone": agent.time_zone,
            "owned": config["ownership_marker"] in agent.description.splitlines(),
            "generators": [{"name": item.name, "display_name": item.display_name,
                            "model": item.llm_model_settings.model}
                           for item in self.call("generators", "list_generators", parent=agent.name)],
            "environments": [{"name": item.name, "display_name": item.display_name,
                              "versions": [v.version for v in item.version_configs]}
                             for item in self.call("environments", "list_environments", parent=agent.name)],
        }

    def apply(self, config, prompt, model, digest):
        cx = self.cx
        agent = self.find_agent(config)
        if agent is None:
            agent = self.call("agents", "create_agent", parent=self.parent, agent=cx.Agent(
                display_name=config["display_name"], description=config["ownership_marker"],
                default_language_code=config["default_language_code"], time_zone=config["time_zone"],
            ))
        desired_agent = cx.Agent(agent)
        agent_update_paths = []
        if config["ownership_marker"] not in desired_agent.description.splitlines():
            description = "\n".join(filter(None, (desired_agent.description, config["ownership_marker"])))
            if len(description) > 500:
                raise ValueError("Adoption marker would exceed the agent description limit; refusing to overwrite existing text.")
            desired_agent.description = description
            agent_update_paths.append("description")
        if desired_agent.time_zone != config["time_zone"]:
            desired_agent.time_zone = config["time_zone"]
            agent_update_paths.append("time_zone")
        if not desired_agent.start_flow:
            # Assigning this oneof clears the entry-point reference, not the
            # default playbook resource. The existing playbook is not deleted.
            desired_agent.start_flow = f"{agent.name}/flows/{DEFAULT_RESOURCE_ID}"
            agent_update_paths.append("start_flow")
        if desired_agent.default_language_code != config["default_language_code"]:
            raise ValueError("The target agent's default language differs; refusing an implicit language migration.")
        wanted_generator = build_generator(cx, config, prompt, model, digest)
        generator = unique_match(self.call("generators", "list_generators", parent=agent.name),
                                 wanted_generator.display_name, "generator")
        if generator is None:
            generator = self.call("generators", "create_generator", parent=agent.name,
                                  generator=wanted_generator)
        else:
            # An immutable, content-addressed generator must not drift unnoticed.
            comparable = cx.Generator(generator)
            comparable.name = ""
            if cx.Generator.pb(comparable) != cx.Generator.pb(wanted_generator):
                raise ValueError("Existing managed generator differs from its content hash; refusing to publish.")

        flow = self.call("flows", "get_flow", name=desired_agent.start_flow)
        flow, flow_update_paths = configure_flow(cx, config, flow, agent.name, generator.name)
        version_display_name = f"coffee-maestro-{digest}"
        version = unique_match(self.call("versions", "list_versions", parent=desired_agent.start_flow),
                               version_display_name, "version")
        if version is not None and flow_update_paths:
            raise ValueError("Draft event drifted from an existing version; review before publishing.")
        if agent_update_paths:
            agent = self.call("agents", "update_agent", agent=desired_agent,
                              update_mask={"paths": agent_update_paths})
        else:
            agent = desired_agent
        if flow_update_paths:
            self.call("flows", "update_flow", flow=flow, update_mask={"paths": flow_update_paths})

        if version is None:
            operation = self.call("versions", "create_version", parent=agent.start_flow, version=cx.Version(
                display_name=version_display_name, description=f"Coffee Maestro config SHA256 prefix {digest}"
            ))
            version = operation.result(timeout=OPERATION_TIMEOUT)
        if version.state != cx.Version.State.SUCCEEDED:
            raise ValueError("The managed flow version is not ready for deployment.")
        environment = unique_match(self.call("environments", "list_environments", parent=agent.name),
                                   config["environment_display_name"], "environment")
        versions = [cx.Environment.VersionConfig(version=version.name)]
        if environment is not None:
            versions = [item for item in environment.version_configs
                        if item.version.rsplit("/versions/", 1)[0] != agent.start_flow] + versions
        if environment is None:
            operation = self.call("environments", "create_environment", parent=agent.name,
                                  environment=cx.Environment(
                                      display_name=config["environment_display_name"],
                                      description="Coffee Maestro managed production manual answers",
                                      version_configs=versions,
                                  ))
            environment = operation.result(timeout=OPERATION_TIMEOUT)
        elif [v.version for v in environment.version_configs] != [v.version for v in versions]:
            environment.version_configs = versions
            operation = self.call("environments", "update_environment", environment=environment,
                                  update_mask={"paths": ["version_configs"]})
            environment = operation.result(timeout=OPERATION_TIMEOUT)
        return {
            "project": self.project, "location": self.location, "agent": agent.name,
            "generator": generator.name, "start_flow": agent.start_flow, "version": version.name,
            "environment": environment.name, "model": model, "config_hash": digest,
            "event": config["event_name"], "language_code": config["default_language_code"],
            "application_max_payload_bytes": config["application_max_payload_bytes"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--model", help="Explicit generator model verified as supported in the target region")
    parser.add_argument("--agent-id", help="Exact existing agent ID; never creates a second agent when provided")
    parser.add_argument("--adopt-existing", action="store_true", help="Explicitly adopt the targeted agent and append the ownership marker")
    parser.add_argument("--manifest", type=Path, default=ARTIFACT_DIR / "deployment_manifest.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Offline desired configuration (default)")
    modes.add_argument("--inspect", action="store_true", help="Read managed resource metadata; no mutations")
    modes.add_argument("--apply", action="store_true", help="Create/reuse managed resources and deploy the version")
    args = parser.parse_args(argv)
    if args.apply and not args.model:
        parser.error("--apply requires an explicit --model; verify availability first")
    if args.adopt_existing and not args.agent_id:
        parser.error("--adopt-existing requires --agent-id")
    config, prompt, digest = load_spec(args.model)
    if args.inspect or args.apply:
        load_project_credentials()
    if args.inspect:
        result = Provisioner(args.project, args.location, agent_id=args.agent_id).inspect(config)
    elif args.apply:
        result = Provisioner(args.project, args.location, agent_id=args.agent_id,
                             adopt_existing=args.adopt_existing).apply(config, prompt, args.model, digest)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.manifest)
    else:
        result = {"mode": "dry-run", "project": args.project, "location": args.location,
                  "config": config, "model": args.model, "config_hash": digest,
                  "agent_id": args.agent_id, "adopt_existing": args.adopt_existing,
                  "prompt_file": str(ARTIFACT_DIR / "generator_prompt.txt"),
                  "planned_actions": [
                                      f"configure exact existing agent {args.agent_id}" if args.agent_id else "create/reuse dedicated owned agent",
                                      "create/reuse immutable generator",
                                      "configure start-flow event", "create/reuse flow version",
                                      "create/update managed production environment", "write nonsecret manifest"],
                  "cloud_mutations": False}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Cloud exception messages can contain request metadata/help tokens.
        # Report only type; authenticated operators can inspect cloud-side logs.
        print(json.dumps({"error": "Conversation provisioning failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
