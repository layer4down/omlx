# Verifiability and Attestation API

Status: API stub / design contract

The inference API answers **what did the model return?**

The verifiability API answers **what privacy policy and measured deployment state handled this request?**

Keeping these surfaces separate prevents inference compatibility from becoming coupled to the evidence model.

## Goals

The first API should let a client or independent auditor verify:

- the active privacy-policy identity;
- the wrapper source/build identity;
- the inference-engine identity;
- the model/tokenizer identity;
- the canonical runtime configuration identity;
- the deployment-profile identity and measured critical controls;
- the evidence issuer and signature;
- the evidence that applied to a particular inference request.

It must do this without requiring raw prompt or completion retention.

## Non-goals

The first API does not claim to prove:

- that a cloud hypervisor cannot observe guest memory;
- that a root user inside the guest cannot subvert processes;
- that GPU firmware cannot observe tensors or KV state;
- that physical memory was immediately zeroed after use;
- constant-time inference;
- zero information leakage through every workload-shape side channel.

These limits must be represented in the policy/assurance fields rather than left to documentation alone.

## Suggested endpoints

The exact URL prefix is intentionally not final. The logical resources are:

```text
GET /verifiability/v1/policy
GET /verifiability/v1/deployment
GET /verifiability/v1/builds/{component}
GET /verifiability/v1/models/{model_id}
GET /verifiability/v1/receipts/{request_id}
GET /verifiability/v1/keys
```

### `GET /policy`

Returns the public privacy-policy document and its canonical digest.

### `GET /deployment`

Returns the current signed deployment evidence, including measured critical controls and the assurance boundary.

### `GET /builds/{component}`

Returns source/build/provenance information for components such as `wrapper`, `attestor`, and `engine`.

### `GET /models/{model_id}`

Returns model/tokenizer/template/config identities used by the service.

### `GET /receipts/{request_id}`

Returns the signed receipt associated with one inference request.

### `GET /keys`

Returns public verification keys and key-rotation metadata. Private signing material is never exposed.

## Receipt lifecycle

A request receipt should be created only after:

1. client identity has been resolved;
2. the wrapper has selected the active privacy policy;
3. required deployment controls are fresh and passing;
4. the normalized inference request has been accepted;
5. the response has completed or terminated in a defined state.

The receipt must not be used as a substitute for a live fail-closed gate. Evidence describes an allowed run; it does not make an unsafe run safe.

## Initial receipt schema

Illustrative JSON:

```json
{
  "schema_version": "vp.receipt.v1",
  "request_id": "req_01...",
  "issued_at": "2026-08-12T00:00:00Z",
  "result": "completed",
  "privacy_policy": {
    "id": "vp-standard-v1",
    "digest": "sha256:...",
    "assurance_level": "application-measured"
  },
  "deployment": {
    "deployment_id": "dep_01...",
    "evidence_digest": "sha256:...",
    "profile": "vp-linux-reference-v1",
    "profile_digest": "sha256:..."
  },
  "software": {
    "wrapper": {
      "source_revision": "git:...",
      "artifact_digest": "sha256:..."
    },
    "attestor": {
      "artifact_digest": "sha256:..."
    },
    "engine": {
      "name": "vllm",
      "version": "pinned-at-deploy-time",
      "artifact_digest": "sha256:...",
      "config_digest": "sha256:..."
    }
  },
  "model": {
    "served_name": "example-model",
    "manifest_digest": "sha256:...",
    "tokenizer_digest": "sha256:...",
    "chat_template_digest": "sha256:..."
  },
  "controls": {
    "request_logging": "disabled",
    "output_logging": "disabled",
    "engine_access_log": "disabled",
    "cache_isolation_scope": "tenant",
    "cache_hash": "sha256",
    "external_kv_connector": "disabled",
    "swap": "disabled",
    "core_dumps": "disabled",
    "engine_public_ingress": "blocked",
    "engine_egress": "restricted",
    "break_glass": "inactive"
  },
  "content_commitment": null,
  "issuer": {
    "key_id": "vp-key-2026-08",
    "algorithm": "ed25519"
  },
  "signature": "base64url:..."
}
```

Field names and algorithms are proposals, not implementation commitments.

## Do not use plain content hashes as a privacy feature

A plain `SHA-256(prompt)` in a public receipt can create a dictionary-attack oracle for short or predictable prompts. The same applies to predictable completions.

Therefore the baseline receipt should contain **no content hash at all** unless the content-commitment design has been reviewed.

If users need to prove that a receipt corresponds to content they possess, safer candidate designs include:

- a commitment using a high-entropy client-held secret/nonce that is not published with the receipt;
- a client-side commitment supplied with the request and echoed into the signed receipt;
- an encrypted commitment readable only by the client;
- a standardized commitment scheme selected after cryptographic review.

The service should not invent a bespoke cryptographic protocol merely to make receipts look stronger.

## Tenant privacy inside evidence

Receipts should not expose a stable tenant identifier publicly.

A receipt may need to state the **isolation scope** (`request`, `principal`, `tenant`, or `disabled`) without exposing the identifier used to derive the cache salt.

If correlation is operationally required, prefer a rotation-bounded pseudonymous identifier scoped to the client that receives the receipt.

## Deployment evidence

A deployment evidence object should include more detail than each request receipt, for example:

```json
{
  "schema_version": "vp.deployment.v1",
  "deployment_id": "dep_01...",
  "measured_at": "2026-08-12T00:00:00Z",
  "valid_until": "2026-08-12T00:05:00Z",
  "policy_digest": "sha256:...",
  "components": {},
  "network": {},
  "processes": {},
  "os_controls": {},
  "logging": {},
  "cache": {},
  "known_exceptions": [],
  "trust_assumptions": [
    "cloud-hypervisor",
    "host-administrator",
    "cpu-gpu-firmware",
    "physical-host"
  ],
  "issuer": {},
  "signature": "base64url:..."
}
```

Critical measurements have an expiry/freshness window. A stale deployment object cannot authorize new requests.

## Assurance levels

Use names that describe what is actually measured rather than marketing labels such as "gold" or "military grade."

Proposed levels:

### `policy-declared`

The public policy and source are available, but there is no signed runtime measurement tying a request to a deployment.

### `application-measured`

Signed evidence binds the request to wrapper/engine/model/config/deployment measurements visible from the guest/application boundary.

This is the target for the first production-quality implementation.

### `hardware-attested`

Future level. Application evidence is additionally bound to a verified hardware/confidential-computing attestation chain.

This label must not be used until the exact TEE/GPU/provider path is implemented, validated, and threat-modeled.

## Signature and canonicalization

The signed payload requires a deterministic canonical representation.

Before implementation, choose:

- one canonical JSON or binary encoding;
- one signature algorithm with mature libraries;
- explicit domain separation between policy, deployment, and request evidence;
- key IDs and rotation rules;
- timestamp/freshness rules;
- replay semantics;
- revocation and compromised-key handling.

A likely simple starting point is a standardized canonical representation plus Ed25519, but the architecture does not depend on that choice.

## Public source-to-runtime linkage

The evidence API should eventually expose a chain such as:

```text
public Git commit/tag
    -> build provenance
    -> artifact digest
    -> deployment artifact digest
    -> canonical config digest
    -> measured deployment evidence
    -> per-request receipt
```

The verifier should be able to stop at any point and see what remains an assumption.

## Failure behavior

A request must not receive an ordinary `application-measured` receipt when:

- a required deployment measurement is stale;
- the active engine/config digest differs from policy;
- request/output logging is enabled contrary to policy;
- cache isolation cannot be established;
- break-glass mode is active;
- the signing key is unavailable or untrusted;
- the evidence chain cannot identify the active artifact.

The service may return a clear inference-unavailable error rather than silently downgrade assurance.

If downgrade modes are ever introduced, the client must opt into them explicitly at a product-policy level; they may not occur invisibly.

## Verification tooling

The repo should eventually include a standalone verifier that can operate without trusting the hosted service UI.

Conceptual commands:

```text
vp-verify policy policy.json
vp-verify deployment deployment.json
vp-verify receipt receipt.json
vp-verify chain receipt.json --source-check
```

The verifier should consume public keys and public source/build metadata and return both:

- what was cryptographically verified; and
- what remains a declared trust assumption.

That second result is as important as the first.