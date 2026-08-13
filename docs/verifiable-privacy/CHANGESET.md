# Architecture Change Set: Verifiable Privacy

This file mirrors the intended GitHub tracking issue so the architecture scope remains versioned with the design even when repository issue settings or integration permissions prevent issue creation.

## Goal

Define an architecture-first design for a thin, engine-agnostic privacy wrapper around shared LLM inference. The wrapper should make privacy claims independently inspectable and later verifiable at runtime without requiring a fork of the inference engine.

The north star is:

> Maximize what can be verified, minimize what must be trusted, and state the remaining trust assumptions explicitly.

## Scope

Document the first-pass design for:

- a narrow OpenAI-compatible inference surface with engine controls kept private;
- a separate attestation/verifiability API;
- mandatory wrapper-mediated ingress/egress for user inference traffic;
- tenant-scoped cache policy and cache-side-channel controls;
- request/application/access logging policy;
- deployment hardening and process/OS trust boundaries;
- source, build, configuration, artifact, and runtime evidence;
- a central privacy-control catalogue mapping claims to enforcement, evidence, and adversarial tests;
- low-cost functional testing before GPU-scale testing;
- red-team/adversarial validation of every material privacy claim.

## Explicit non-goals for this change set

- no production wrapper implementation yet;
- no claim of mathematically provable privacy;
- no claim that application software alone can verify a cloud hypervisor, host administrator, GPU firmware, or physical hardware;
- no attempt to introspect or mutate the inference engine's heap/stack directly;
- no requirement to commit to vLLM permanently; vLLM is the first concrete engine target, but the trust layer should remain replaceable.

## Acceptance criteria

- [x] Problem statement and design principles are documented.
- [x] Trust boundaries and data flow are diagrammed.
- [x] Threat model covers inference payloads, logs, KV/prefix cache, debug/profiling modes, crash dumps/swap, network egress, admin access, and infrastructure trust assumptions.
- [x] Privacy-control catalogue maps each initial claim to enforcement mechanism, evidence, and a negative/adversarial test.
- [x] Attestation API is stubbed separately from the inference API.
- [x] Deployment profile defines Linux hardening candidates without pretending OS policy proves the hypervisor.
- [x] Testing strategy includes cheap CPU/control-plane tests, temporary/spot GPU tests, timing/cache isolation probes, configuration-drift tests, and red-team scenarios.
- [x] Future research captures reproducible builds, signed artifacts/provenance, confidential computing, hardware attestation, and alternative inference engines.
- [x] Open questions remain visible rather than being papered over.
- [ ] Select and validate the first Linux reference profile.
- [ ] Implement the wrapper reference monitor.
- [ ] Implement signed deployment evidence and request receipts.
- [ ] Run the adversarial test suite against a pinned vLLM deployment.
- [ ] Obtain independent review before making strong production privacy claims.

## Design principles

1. **Engine computes; wrapper enforces and proves.**
2. **The public inference API does not expose low-level engine configuration.**
3. **Every privacy claim must map to both evidence and an adversarial test.**
4. **Fail closed when required privacy controls cannot be established or verified.**
5. **Evidence must not contain raw prompts or completions by default.**
6. **The wrapper should be open source and independently inspectable.**
7. **Runtime evidence should bind back to source/build/configuration identities where feasible.**
8. **Trust assumptions are part of the product contract, not footnotes.**

## Documents in this change set

- `README.md` — architecture and evidence chain.
- `THREAT_MODEL.md` — adversaries, assets, trust boundaries, and non-goals.
- `PRIVACY_CONTROLS.md` — central assurance/control matrix.
- `ATTESTATION_API.md` — verifiability API and signed receipt design.
- `DEPLOYMENT_PROFILE.md` — Linux/process/network/logging/memory controls.
- `TESTING_STRATEGY.md` — cheap-first test ladder and red-team catalogue.
- `FUTURE_RESEARCH.md` — supply-chain, confidential-computing, and stronger-verification backlog.
- `OPEN_QUESTIONS.md` — unresolved design choices.

## Implementation order after architecture review

1. Machine-readable privacy policy and control definitions.
2. Minimal wrapper with a malicious/mock engine test double.
3. Deployment verifier and one cheap Linux candidate profile.
4. Signed deployment evidence and receipt prototype.
5. Real vLLM adapter with request/output/access-log and cache-isolation enforcement.
6. Adversarial harness and cost-reporting benchmark runner.
7. Temporary GPU validation.
8. Independent security/privacy review.

This ordering intentionally postpones expensive inference work until the trust/control plane is repeatable.