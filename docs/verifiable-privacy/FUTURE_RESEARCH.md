# Future Research

Status: intentionally out of scope for the first architecture implementation

This file keeps promising ideas visible without turning them into premature privacy claims.

## Software supply-chain provenance

Goal: strengthen the source-to-runtime link.

Research:

- reproducible wrapper builds;
- SLSA-compatible provenance;
- Sigstore signing/verification;
- transparency logs for releases and policy artifacts;
- hermetic dependency resolution;
- SBOM generation and verification;
- pinned base-image/package identities;
- dependency-update policy and revocation handling.

The desired chain is:

```text
public source revision
  -> reproducible or provenance-backed build
  -> signed artifact digest
  -> measured deployed artifact
  -> signed deployment evidence
  -> request receipt
```

References:

- https://slsa.dev/
- https://www.sigstore.dev/

## Confidential computing and hardware attestation

Application-level evidence cannot prove the hypervisor or hardware is unable to observe memory.

Research whether a useful production chain can bind the wrapper/deployment measurement to technologies such as:

- AMD SEV-SNP;
- Intel TDX;
- cloud-provider confidential VM offerings;
- GPU confidential-computing modes supported by the selected NVIDIA generation;
- TPM/measured boot;
- remote attestation services.

Questions:

- Can CPU guest attestation be meaningfully bound to GPU execution?
- What is the actual provider/hypervisor trust reduction?
- What debugging/observability features are lost?
- How is attestation freshness handled?
- Can model weights and KV state remain encrypted outside the trusted execution boundary?
- What parts of the DMA/IOMMU path remain trusted?
- Can the claim be independently verified without trusting a proprietary dashboard?

Do not advertise `hardware-attested` assurance until the entire chain is implemented and reviewed.

## GPU memory lifecycle

Research what guarantees are realistically available for:

- VRAM allocation/reuse between processes;
- GPU reset behavior;
- memory scrubbing/zeroing;
- worker-process termination;
- CUDA graph/runtime memory pools;
- NCCL buffers;
- MIG/partitioning where applicable;
- DMA-visible buffers.

The goal is to distinguish enforceable controls from folklore. Avoid promising immediate physical erasure unless the hardware/runtime contract supports it.

## Stronger content commitments

The baseline receipt deliberately avoids plain hashes of prompts/completions.

Research standardized privacy-preserving commitment designs that let a user later prove that a receipt corresponds to content they possess without publishing a dictionary oracle.

Requirements:

- no custom cryptography if a standard primitive/protocol fits;
- user-verifiable offline;
- no server retention of raw content required;
- clear replay/domain separation;
- compatible with streaming responses;
- independently reviewed.

## Transparency log for deployments

A public append-only log could make silent policy/build rollback harder.

Research publishing signed statements such as:

```text
deployment id
wrapper artifact digest
engine artifact digest
policy digest
profile digest
model catalogue identity
valid-from / valid-until
```

Privacy requirement: the log must not reveal tenant/request activity.

Potential building blocks include Sigstore/Rekor-style transparency mechanisms or a project-specific Merkle log only if justified.

## Independent/verifier-first tooling

Build a standalone verifier before building a glossy hosted assurance UI.

The verifier should:

- validate signatures;
- validate canonical encoding;
- follow source/build provenance;
- check freshness/revocation;
- explain assurance level;
- list unverified trust assumptions;
- work offline when supplied the required public material.

A browser UI can consume the same verifier library later.

## Alternative inference engines

The wrapper should remain compatible with other engines if they can expose equivalent controls.

Candidates to evaluate later include:

- SGLang;
- TensorRT-LLM;
- llama.cpp-family serving for smaller CPU/GPU deployments;
- engine-specific cloud runtimes.

For each engine, create a control adapter answering:

```text
How are request/output logs disabled?
How is cache sharing isolated?
Can engine metrics/admin endpoints be privately bound?
What external cache/offload paths exist?
How is exact config measured?
What debug/profiling modes can expose content?
What artifact/config identity can be attested?
```

Do not weaken the privacy contract merely to support another engine.

## External KV cache and disaggregated inference

Modern inference systems can move KV state across processes, machines, and storage tiers. This can improve performance while dramatically widening the privacy boundary.

Before enabling external KV connectors or prefill/decode disaggregation, threat-model:

- peer authentication;
- encryption in transit;
- cache-salt/isolation semantics across instances;
- remote retention;
- deletion/expiry;
- crash/replay behavior;
- metadata/timing leakage;
- evidence identity for every participating node;
- failover paths that might silently fall back to a less private cache.

Baseline policy keeps this disabled.

## Multi-node attestation

A future clustered deployment needs a receipt that represents all workers that touched a request.

Research:

- coordinator-signed aggregate measurement versus worker co-signatures;
- Merkle aggregation of worker evidence;
- worker identity/freshness;
- model-shard identities;
- node replacement during a request;
- partial failure semantics;
- heterogeneous CPU/GPU trust levels;
- binding cluster plan digest into receipts.

## Side-channel characterization

Cache salting addresses one important cross-tenant timing channel, not all timing leakage.

Research controlled measurements for:

- scheduler interference;
- batch occupancy;
- model loading/eviction;
- speculative decoding behavior;
- GPU memory pressure;
- token-length inference;
- shared adapter/LoRA state;
- external KV connectors;
- multi-node link contention.

The goal is to state which workload-shape leakage is mitigated, bounded, or explicitly accepted.

## Privacy-preserving observability

Research operational signals that preserve debuggability without payload access:

- structured error codes;
- deterministic synthetic probes;
- aggregate latency histograms;
- queue/scheduler state;
- GPU/kernel error counters;
- request state machines without content;
- privacy-safe distributed tracing;
- differential privacy for aggregate usage analytics if genuinely needed.

The default should remain "do not collect what is not needed."

## Formal methods

Once the wrapper state machine stabilizes, consider formal/specification work for the small high-value core:

- policy transition state machine;
- fail-closed admission logic;
- receipt canonicalization;
- cache-scope derivation;
- request-to-response association under concurrency;
- break-glass transitions.

Formal methods would strengthen a narrow component. They would not make the entire cloud inference stack "provably private."

## Independent audit and public challenge

Before strong external privacy positioning:

- publish the threat model and control catalogue;
- commission independent security/privacy review;
- remediate findings publicly where possible;
- consider a focused bug bounty for wrapper boundary and evidence validation;
- publish reproducible adversarial test cases;
- invite cache-timing and evidence-chain challenges.

The objective is to make skepticism productive rather than treating the architecture as something users must accept on faith.