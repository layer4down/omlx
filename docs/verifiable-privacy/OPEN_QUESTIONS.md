# Open Questions

Status: deliberately unresolved

These questions should remain visible until testing or product requirements justify a decision. None should be silently converted into an implementation assumption.

## 1. What is the first blessed Linux profile?

Candidates:

- Ubuntu LTS / Debian-family with AppArmor;
- RHEL-compatible distribution with SELinux enforcing.

Decision inputs:

- vLLM/NVIDIA support;
- security confinement quality;
- package/update lifecycle;
- operational familiarity;
- cloud image availability;
- reproducibility;
- results of the cheap deployment-profile test matrix.

## 2. What is the default cache-isolation scope?

Candidates:

- per request;
- per authenticated principal;
- per tenant/account;
- prefix caching disabled.

We should measure performance benefit and timing leakage before choosing a default.

The public API will not expose the low-level salt regardless of the answer.

## 3. What exactly is a "tenant"?

Possibilities include:

- API key;
- user account;
- organization;
- billing account;
- explicit privacy domain.

Cache isolation, metrics, receipt pseudonyms, and rate limiting all depend on a precise definition.

## 4. Should the first inference surface include Anthropic compatibility?

OpenAI compatibility is sufficient for the first architecture.

Anthropic compatibility should be added only when there is demonstrated client demand or a concrete integration requirement. It must not create a second path that bypasses wrapper policy normalization.

## 5. How is attestation delivered with streaming inference?

Options include:

- request ID in response headers and receipt fetched later;
- terminal stream event containing receipt reference;
- separate client callback/subscription;
- synchronous receipt only after stream completion.

The design should not delay first-token latency unnecessarily.

## 6. Should receipts include any content commitment?

Baseline answer: no.

Plain content hashes are unsafe for predictable prompts because they can become dictionary-attack oracles.

Only add commitments after selecting a reviewed scheme that preserves privacy and offline verifiability.

## 7. Where are attestation signing keys held?

Candidates:

- software key in a tightly confined attestor process for development;
- cloud KMS/HSM;
- TPM-backed key;
- confidential-computing/TEE-bound key later.

The key choice determines how strongly a verifier can trust the receipt issuer.

## 8. Who measures the deployment verifier?

A process cannot simply declare itself trustworthy.

The first version may use a separately confined verifier/attestor whose artifact identity is included in evidence. Hardware-backed measurement is future work.

## 9. How fresh must critical measurements be?

Controls differ in change rate.

Examples:

- process command line/config: recheck on restart and periodically;
- sockets/network policy: periodic plus event-driven where possible;
- swap/core-dump policy: startup plus periodic;
- artifact/model digests: deployment lifecycle;
- break-glass state: immediate/event-driven.

The receipt must state freshness semantics.

## 10. What is the production stance on payload-bearing debugging?

Candidate policies:

- prohibited entirely on customer-serving nodes;
- allowed only after draining customer traffic and entering break-glass;
- allowed only in dedicated synthetic staging environments.

The architecture supports break-glass for development, but the production policy may remove it.

## 11. What metrics are necessary?

We want enough telemetry to operate the service without turning metrics into a tenant-activity side channel.

Decide which of these are required:

- aggregate GPU utilization;
- queue depth;
- model load state;
- request latency histograms;
- input/output token distributions;
- cache hit rate;
- per-model stats;
- per-tenant stats.

Per-tenant metrics deserve special scrutiny.

## 12. Are disk/offloaded KV caches permitted?

Baseline may be RAM-only until persistence semantics are tested.

If SSD/offload is enabled later, define:

- encryption requirements;
- tenant isolation;
- file permissions;
- retention/TTL;
- restart behavior;
- deletion semantics;
- backup/snapshot exclusion;
- evidence fields.

## 13. Are external KV connectors permitted?

Baseline: no.

They widen the privacy and attestation boundary and require a separate threat model.

## 14. What source/build provenance level is required at launch?

Minimum:

- public source revision;
- artifact digest;
- signed release/build identity.

Stronger target:

- reproducible build or SLSA-style provenance;
- transparency-log inclusion;
- SBOM;
- independent rebuild verification.

## 15. How should model identity be represented?

A model name is insufficient.

Need to decide the canonical set of digests for:

- weight files/manifest;
- config;
- tokenizer;
- chat template;
- generation defaults;
- optional LoRA/adapters;
- quantization metadata.

## 16. What changes invalidate an existing deployment identity?

Likely:

- wrapper/attestor/engine artifact;
- engine config;
- model/tokenizer/template;
- privacy policy;
- OS/MAC policy;
- network policy;
- cache configuration;
- signing key;
- cluster membership.

Define this explicitly before receipt implementation.

## 17. What does "memory handling" mean in a public claim?

Avoid vague language like "memory is secure."

Potential measurable statements:

- swap disabled;
- core dumps disabled;
- no unapproved disk KV offload;
- ptrace restricted;
- process users separated;
- writable mounts constrained;
- crash uploads disabled.

Physical DRAM/VRAM erasure is a separate hardware/runtime claim.

## 18. Does the wrapper need to be the only public process?

A reverse proxy/load balancer may terminate TLS before the wrapper.

If so, its logging, tracing, headers, buffering, and request-body behavior become part of the privacy boundary and evidence chain.

## 19. How do we handle cloud-provider load balancer/WAF logs?

If a managed edge service sees request bodies, its retention policy becomes part of the trust contract.

Prefer architectures where managed edge infrastructure does not create an undisclosed payload retention path.

## 20. What happens when evidence generation fails after inference has already run?

Possible policies:

- fail request before execution unless evidence prerequisites are healthy;
- if signing fails after generation, return result with explicit `unattested` status only under a separately selected policy;
- discard result and fail closed.

The first production privacy tier should likely fail closed before execution whenever evidence prerequisites are unhealthy.

## 21. How are receipts retained?

Options:

- client receives receipt and service retains minimal index only;
- short server retention;
- customer-configurable retention;
- append-only transparency of deployment state, not per-request activity.

Receipt retention must not create a new user-activity database by accident.

## 22. What is the relationship to the existing oMLX cluster trust model?

The current heterogeneous-cluster design already uses pairing, host identity, source/build digests, model identity, plan hashes, and fail-closed compatibility checks.

We should reuse those primitives where they fit rather than creating a parallel trust vocabulary, while preserving the inference-wrapper boundary and avoiding assumptions specific to MLX/Metal.

## 23. Should the wrapper live inside this repository long term?

This architecture can begin here because it intersects the existing cluster/inference work, but the wrapper is intentionally engine-agnostic.

Once implementation starts, evaluate whether it should become:

- a package/subproject in oMLX;
- a separately versioned repository;
- a standalone service consumed by oMLX and other engines.

Repository structure should follow the stable trust boundary, not determine it.

## Decision rule

Close an open question only when the decision records:

```text
decision
evidence / measurements
tradeoffs
privacy impact
controls affected
migration/reversal path
```

Architecture questions are cheaper to reopen than privacy promises.