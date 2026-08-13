# Privacy Control Catalogue

Status: initial control set

This is the central control catalogue for verifiable privacy. Each control must have four parts before it can support a production privacy claim:

- **Enforcement** — the mechanism intended to make the claim true.
- **Evidence** — what can be independently inspected or verified.
- **Adversarial test** — how we try to make the claim false.
- **Residual trust** — what still has to be trusted.

Controls are intentionally engine-agnostic where possible. vLLM-specific notes describe the first Linux/CUDA target.

## Control matrix

| ID | Claim | Enforcement | Evidence | Adversarial test | Residual trust |
| --- | --- | --- | --- | --- | --- |
| VP-001 | Clients cannot mutate engine privacy settings | Wrapper exposes an allowlisted inference schema and strips/rejects unknown engine extensions | Wrapper source/build digest, policy digest, schema test result | Send logging/cache/debug/connector controls through every public endpoint and verify they are rejected/overridden | Wrapper correctness |
| VP-002 | The engine is not directly public | Engine binds only to a private namespace, loopback/UDS, or isolated service network; firewall/MAC blocks external ingress | Socket inventory, firewall/network-namespace measurement, deployment-profile digest | Attempt direct engine/API/metrics connection from client network | Guest kernel/network policy |
| VP-003 | Raw inference requests are not logged by the engine in normal mode | vLLM request logging remains disabled; approved logging config excludes request bodies | Canonical launch config, active process args/config, log-canary test | Send unique canary prompt and search all approved log sinks | Engine obeys config; privileged host outside boundary |
| VP-004 | Model outputs are not logged by the engine in normal mode | vLLM output logging disabled; request logging disabled | Canonical launch config and canary scan | Generate unique canary output and search log sinks | Same as VP-003 |
| VP-005 | HTTP access logs do not become a hidden inference-content store | Engine access logs disabled or strictly internal; wrapper access logs contain only approved metadata and never bodies | Logging config digest and log schema | Inject secrets into path/query/header/body variants and scan logs | Reverse proxy/platform log behavior |
| VP-006 | Prefix-cache reuse does not cross the configured tenant boundary | Wrapper derives cache isolation salt from authenticated tenant scope and never trusts client salt | Policy ID plus salted-scope derivation implementation identity; request receipt records isolation policy, not secret salt | Tenant A warms prefix; Tenant B probes same prefix and timing; attempt explicit salt collision/override | Engine cache-salt implementation and remaining timing noise |
| VP-007 | Cache hashing uses a cryptographic algorithm | Approved engine config requires SHA-256-class cache hashing | Engine config digest | Boot with non-approved hash and verify fail-closed | Hash implementation |
| VP-008 | External KV/cache connectors do not silently widen the trust boundary | Disabled in baseline deployment; later connectors require a separate approved profile | Config measurement and deployment policy | Enable connector/config drift and verify traffic admission stops | OS/process measurement fidelity |
| VP-009 | Metrics are not public inference endpoints | `/metrics` and engine telemetry bind only to private monitoring paths; public gateway does not proxy them | Network policy measurement; endpoint reachability test | Probe metrics/admin endpoints from customer network | Monitoring stack access control |
| VP-010 | Normal traces do not contain raw inference content | Trace attributes/exporters use allowlisted non-content fields | Tracing config digest and canary scan | Send unique prompt/output canaries and inspect exported traces | Tracing library/exporter behavior |
| VP-011 | Swap does not create an unmanaged plaintext retention path | Swap disabled for the serving guest/process profile, or a separately approved encrypted-memory policy is used | OS measurement | Force memory pressure and verify no swap path becomes active | Guest kernel; underlying storage/hypervisor |
| VP-012 | Core dumps do not create inference-memory artifacts | Core dumps disabled for wrapper and engine; crash collectors configured not to capture process memory | `ulimit`/systemd/kernel measurement, policy digest | Crash test process and verify no core artifact | Guest OS and crash subsystem |
| VP-013 | Engine filesystem writes are constrained | Read-only system paths, explicit writable dirs, MAC/sandbox policy | Deployment-profile measurement | Attempt writes to forbidden paths and secret locations | Guest kernel/MAC implementation |
| VP-014 | Engine network egress is constrained in steady state | Default-deny egress with explicit lifecycle/monitoring exceptions | Network policy measurement | Attempt DNS/HTTP/TCP egress to canary destinations | Guest kernel/network stack |
| VP-015 | Engine cannot read wrapper signing secrets | Separate process identity, filesystem permissions, MAC policy, no secret mount into engine | File ownership/MAC measurement | Attempt engine-side read of attestation key path/socket | Guest kernel/process isolation |
| VP-016 | A request receipt is bound to a measured deployment and policy | Attestor signs canonical receipt containing deployment, build, model, config, and policy identities | Signature + public verification key + published schema | Mutate any field and verify signature failure; replay against wrong deployment | Attestor key custody and measurement source |
| VP-017 | Evidence does not itself leak prompt/completion contents | Receipts contain no raw content; content commitments are optional and privacy-preserving by construction | Schema validation | Fuzz content into every receipt field and ensure no raw payload survives | Canonicalization/redaction implementation |
| VP-018 | Privacy-sensitive debug mode cannot be silently enabled | Debug/break-glass state is admin-only, time-bounded, and changes the attested policy state; baseline may prohibit it entirely | Policy transition event + changed deployment/receipt state | Enable debug and verify ordinary privacy receipt cannot be issued | Admin identity and policy engine |
| VP-019 | Configuration drift stops new customer traffic | Wrapper continuously or periodically revalidates critical measured controls and fails closed | Drift-check status in deployment evidence | Toggle critical flag/network rule while serving and verify admission stops | Measurement freshness |
| VP-020 | The wrapper artifact is independently inspectable | Wrapper source is public; release/build identity links runtime artifact to source/provenance | Source revision, artifact digest, provenance record | Rebuild/compare where reproducibility is supported | Build infrastructure until reproducibility is proven |
| VP-021 | Model and tokenizer identities are explicit | Model, tokenizer, chat template, and relevant config are digest-pinned | Request/deployment receipt identities | Change tokenizer/template/model artifact and verify evidence changes or admission fails | Artifact storage integrity |
| VP-022 | Admin changes are audited without recording inference payloads | Config-change audit log contains actor, action, old/new policy identity, timestamp, not customer body data | Audit schema and log sample | Put canary payloads in inference traffic while making admin changes; scan audit log | Admin/audit subsystem |
| VP-023 | Privacy claims declare infrastructure assumptions | Published policy names provider/hypervisor/hardware trust boundary and assurance level | Policy document + receipt assurance level | Review receipt against unsupported claim; schema must not imply higher level | Provider assertions outside application boundary |

## vLLM baseline configuration notes

The exact flags must be generated from the pinned vLLM version rather than copied blindly from documentation. The baseline intent is:

```text
request-information logging     disabled
model-output logging            disabled
uvicorn access log              disabled on the private engine API
prefix caching                  permitted only with wrapper-owned isolation salt
prefix-cache hash               cryptographic (SHA-256-class)
external KV connectors          disabled unless separately approved
engine metrics                  private only
profiling/debug content capture disabled
```

Current upstream vLLM documentation states:

- `--enable-log-requests` defaults to false; at INFO it logs request ID/parameters/LoRA information, while DEBUG can include prompt text or token IDs;
- `--enable-log-outputs` is separately controlled and requires request logging;
- uvicorn access logs have their own disable control;
- `cache_salt` is designed to scope Automatic Prefix Cache reuse and mitigate timing-based cache-content inference;
- cryptographically insecure cache hashing can increase privacy risk in multi-tenant deployments.

References:

- https://docs.vllm.ai/en/latest/cli/serve/
- https://docs.vllm.ai/en/stable/configuration/engine_args/
- https://docs.vllm.ai/en/stable/design/prefix_caching/
- https://docs.vllm.ai/en/stable/api/vllm/config/cache/

## Cache-isolation policy

The wrapper, not the client, chooses the cache-isolation scope.

A first implementation should support internal policy values such as:

- `request` — no reuse across requests; strongest isolation, lowest reuse;
- `principal` — reuse only for the authenticated user/service principal;
- `tenant` — reuse within one organization/account boundary;
- `disabled` — Automatic Prefix Caching disabled.

The public inference API does not expose these as low-level engine knobs. Product-level privacy tiers may map to them later, but the mapping remains server policy.

For vLLM, the wrapper should derive a high-entropy salt from an internal secret plus stable scope identity, for example conceptually:

```text
cache_salt = HMAC-SHA256(cache_isolation_key, policy_version || scope_type || scope_id)
```

The raw salt is internal. The receipt records the scope type and policy identity, not the secret or a stable tenant identifier.

A client-supplied `cache_salt` must be stripped or overwritten.

## Logging policy

The baseline should retain useful system observability while preventing inference-content logging.

Allowed examples:

- process start/stop;
- engine/model identity;
- health transitions;
- GPU/CPU/memory utilization;
- scheduler queue depth;
- aggregate latency/throughput histograms;
- sanitized error classes;
- policy/configuration transitions;
- security-control failures.

Forbidden in normal customer-serving mode:

- raw request/response bodies;
- prompt text;
- token IDs derived from user content;
- generated text;
- image/audio/file payloads;
- tool payload contents;
- raw KV-cache data;
- memory dumps.

Request identifiers should be pseudonymous and retention-bounded. Tenant identifiers should be minimized or transformed where operationally possible.

## Fail-closed rule

The wrapper must not issue a normal privacy receipt or accept new customer inference when a control marked **required** is:

- missing;
- unverifiable;
- stale beyond its measurement window;
- inconsistent with the approved config digest;
- explicitly in break-glass state.

The response to control failure should identify the failed control without exposing another tenant's data.

## Control lifecycle

Each control should eventually carry:

```text
id
status: proposed | implemented | verified | externally-reviewed
required_for_policy: [policy ids]
implementation_refs: [source paths]
measurement_refs: [probe paths]
test_refs: [test ids]
known_limitations
last_external_review
```

That turns this document from prose into an assurance inventory that can later be machine-readable.