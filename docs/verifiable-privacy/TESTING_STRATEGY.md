# Verifiable Privacy Testing Strategy

Status: initial validation plan

The project should spend money only when a test requires expensive hardware.

The operating principle is:

> Minimize cost per unknown, not merely hourly infrastructure cost.

Most privacy/control-plane work can be matured before renting meaningful GPU capacity.

## Budget target

Early validation should be designed around a small, disposable Linux environment with a target steady test spend below roughly **$100/month**, and preferably much lower when no GPU experiment is active.

GPU instances should be:

- temporary;
- selected for the minimum capability needed by the test;
- shut down automatically after the run;
- spot/preemptible where interruption does not invalidate the experiment;
- reserved/capacity-block style only when a deterministic validation window is worth the premium.

Do not run an expensive GPU merely to test wrapper routing, signing, policy parsing, or OS configuration.

## Test layers

### Layer 0 — Static architecture and policy tests

No cloud resources required.

Validate:

- receipt schema does not permit raw prompt/completion fields;
- policy files have canonical digests;
- every production privacy claim maps to a control ID;
- every required control has an adversarial test ID;
- engine-configuration generator rejects forbidden combinations;
- public inference schema does not expose engine privacy controls;
- deployment profiles are machine-parseable where possible.

### Layer 1 — Wrapper with a mock engine

Cheap CPU VM or local Linux container.

Test:

- auth/tenant resolution;
- request canonicalization;
- parameter allowlist;
- stripping of client `cache_salt` and other engine extensions;
- wrapper-generated internal cache scope;
- streaming correctness;
- response/request isolation under concurrency;
- receipt generation/signature verification;
- fail-closed behavior when a control fails;
- evidence freshness/expiry;
- key rotation;
- policy transition handling.

The mock engine should include malicious behaviors such as attempts to write files, connect externally, emit payloads to logs, and bind extra ports.

### Layer 2 — Linux deployment-profile tests

Still CPU-first where possible.

Evaluate candidate Linux profiles in disposable VMs.

Test:

- AppArmor/SELinux policy enforcement;
- engine/wrapper user separation;
- filesystem allow/deny rules;
- network ingress/egress rules;
- swap disabled;
- core dumps disabled;
- ptrace restrictions;
- service-manager hardening;
- log destinations and retention;
- crash behavior;
- configuration-drift detection.

This phase should select the initial blessed OS profile based on evidence, not preference.

### Layer 3 — Minimal real-engine functional test

Run a pinned vLLM version with the smallest practical model/device configuration that exercises the server controls under test.

Validate:

- exact launch/config capture;
- request logging disabled;
- output logging disabled;
- access-log behavior;
- metrics isolation;
- wrapper-only engine ingress;
- cache-salt injection/override;
- prefix-cache behavior;
- drift when flags/config change;
- model/tokenizer/build identity in receipts.

### Layer 4 — GPU privacy and cache tests

Rent the cheapest GPU that reproduces the real execution path.

These tests should answer GPU-specific unknowns, including:

- whether prefix-cache isolation behaves as expected under concurrency;
- whether timing probes can distinguish another tenant's cached prefix;
- whether GPU worker subprocesses create unexpected logs/files/sockets;
- whether profiling/debug tooling creates sensitive artifacts;
- whether process restarts leave unexpected cache/offload files;
- whether memory pressure triggers an unapproved persistence path.

### Layer 5 — Production-shape performance and failure tests

Only after earlier layers pass.

Use representative concurrency and models to measure:

- TTFT;
- decode throughput;
- cache hit rate;
- performance cost of request/principal/tenant cache isolation;
- wrapper overhead;
- attestation overhead;
- failover/restart behavior;
- policy-measurement frequency cost.

Privacy correctness takes priority over benchmark wins.

## Adversarial test catalogue

### T-001 — Engine-control smuggling

Attempt to pass every known vLLM privacy-sensitive field through OpenAI-compatible bodies, headers, query strings, model aliases, and extra-body extension mechanisms.

Expected: rejected or overwritten according to wrapper policy.

### T-002 — Direct engine bypass

Probe engine API, metrics, docs, health, admin, and debug endpoints from:

- public client network;
- another tenant/service container;
- unrelated local user.

Expected: only approved internal identities can connect.

### T-003 — Request-log canary

Send a unique high-entropy string in prompt content.

Search:

- engine stdout/stderr;
- journald/syslog;
- wrapper logs;
- access logs;
- reverse-proxy logs;
- tracing/APM;
- crash/error reports;
- cloud-agent logs included in the profile.

Expected: canary absent.

### T-004 — Output-log canary

Use a deterministic synthetic prompt that causes a unique output canary.

Expected: canary absent from all normal log/trace sinks.

### T-005 — Cross-tenant prefix timing probe

Tenant A repeatedly warms a long unique prefix. Tenant B submits the identical prefix under controlled load and measures TTFT/prefill behavior.

Run enough samples to distinguish natural variance from a cache hit.

Expected: Tenant B does not receive the cross-tenant cache benefit under an isolated policy.

### T-006 — Cache-salt override

Client supplies a chosen `cache_salt`, repeated salts, another tenant's guessed salt, null/empty values, encoding edge cases, and duplicate fields.

Expected: wrapper-controlled salt always wins; raw internal salt never appears in response/evidence/logs.

### T-007 — Hash-policy drift

Start/reconfigure engine with a non-approved prefix-cache hash.

Expected: privacy gate refuses new traffic.

### T-008 — Request-logging drift

Enable request logging after deployment verification.

Expected: drift is detected within the defined freshness window and new privacy-assured requests stop.

### T-009 — Debug-level escalation

Increase engine/application log level to DEBUG without changing request-logging flag, then with it enabled.

Expected: safe state remains content-free; unsafe combination triggers break-glass/fail-closed behavior.

### T-010 — Output-logging drift

Enable engine output logging.

Expected: normal privacy policy becomes invalid immediately or within the declared measurement window.

### T-011 — Public metrics leakage

Probe `/metrics` externally and inspect all exported labels internally for request/tenant identifiers.

Expected: public access denied; labels comply with metric policy.

### T-012 — Core-dump test

Crash a synthetic process with equivalent service limits and, where safe, the engine in a disposable environment.

Expected: no core/memory dump artifact is created or uploaded.

### T-013 — Swap pressure test

Create controlled memory pressure.

Expected: swap remains unavailable under the reference profile and the service degrades/fails according to policy instead of silently persisting pages.

### T-014 — Filesystem escape

From the engine identity, attempt writes to wrapper config, attestation-key location, system config, web root, arbitrary home directories, and executable paths.

Expected: denied.

### T-015 — Engine egress

Attempt DNS, HTTP, HTTPS, raw TCP, and metadata-service access from engine identity.

Expected: denied except explicit allowlisted dependencies.

### T-016 — Attestation-key isolation

Attempt to read or connect to signing-key material from engine identity and from an unrelated local service.

Expected: denied.

### T-017 — Receipt tamper

Modify each receipt field independently.

Expected: signature verification fails.

### T-018 — Receipt replay/context confusion

Replay valid receipt against another request, deployment, model, or policy.

Expected: verifier reports context mismatch rather than accepting it as evidence for the other run.

### T-019 — Evidence privacy fuzz

Insert canaries into every supported inference field and error path.

Expected: raw content never appears in receipt/deployment evidence.

### T-020 — Break-glass transition

Activate content-bearing debug profile in a disposable environment.

Expected: normal receipt issuance stops; evidence explicitly reports break-glass state; automatic expiry works.

### T-021 — Malicious mock engine

Mock engine tries to:

- bind a second listener;
- write outside approved dirs;
- access wrapper secret path;
- send payload to internet canary;
- emit request body to stderr.

Expected: OS confinement blocks all paths it is designed to block. Stderr emission demonstrates why application logging configuration remains separately required.

### T-022 — Concurrent response isolation

Interleave large streaming requests across tenants and force cancellation/retry/error paths.

Expected: no token stream, usage object, or receipt is associated with the wrong request.

### T-023 — Upgrade identity

Upgrade wrapper, engine, model, tokenizer, or deployment profile one at a time.

Expected: affected evidence digests change; stale evidence cannot authorize the new run.

## Red-team strategy

Red teaming is a required validation stage, not a launch-day flourish.

### Internal red team

For every control marked `verified`, assign someone or an automated harness the explicit goal of falsifying the claim.

The tester should not be constrained to the happy-path API documentation.

Include:

- malformed protocol inputs;
- concurrency races;
- process restarts;
- partial configuration changes;
- log-level changes;
- unexpected environment variables;
- symlink/mount/path tricks;
- network namespace mistakes;
- cache timing measurements;
- stale attestation state;
- failure injection during streaming;
- secret-bearing exceptions.

### External review

Before making strong public privacy claims, commission independent review of at least:

- threat model;
- wrapper reference-monitor logic;
- attestation/cryptographic design;
- Linux confinement profile;
- cache-isolation tests;
- build/provenance chain;
- claim wording versus actual evidence.

External review results and remediation status should be publishable wherever practical.

## Assurance matrix

The test harness should eventually generate a machine-readable matrix:

```text
control_id
test_id
source_revision
deployment_profile_digest
engine_digest
model_digest
result
started_at
finished_at
measurements
artifacts
estimated_cost
```

This prevents "we tested that once" from becoming a permanent claim.

## Cost-aware benchmark harness

Each cloud test run should record:

- provider/region/profile identifier;
- instance/device type;
- wall-clock duration;
- spot/on-demand/reserved mode;
- approximate run cost;
- environment/deployment digest;
- exact question being answered;
- pass/fail/measurement result.

The benchmark harness should make it obvious when an expensive test is answering a question already covered cheaply.

## Promotion gates

A control progresses through:

```text
proposed
  -> implemented
  -> locally tested
  -> cloud-profile tested
  -> adversarially verified
  -> externally reviewed (where required)
```

A privacy policy may only claim the level supported by the lowest required control in its set.

## First milestone

The first milestone does **not** require a large model or production GPU.

It requires:

- wrapper/mocked-engine reference path;
- signed receipt prototype;
- one candidate Linux confinement profile;
- request/output log canary tests;
- direct-engine-bypass test;
- configuration-drift fail-closed test;
- cache-salt override test at the wrapper boundary;
- documented evidence gaps.

Once that is repeatable and cheap, add real vLLM and GPU-specific validation.