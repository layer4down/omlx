# Reference Deployment Profile

Status: candidate profile for validation

This profile describes the guest-OS and process controls that should surround the privacy wrapper and inference engine. It is part of the evidence chain, not an invisible operations checklist.

The first objective is to make the engine's allowed interfaces small enough to reason about and test.

## Platform strategy

Do not commit to an operating-system family before measuring it.

The first inexpensive test matrix should compare at least:

- a current Ubuntu LTS / Debian-family Linux with AppArmor;
- a current RHEL-compatible Linux with SELinux enforcing.

The selected production reference should be the smallest well-supported Linux profile that passes:

- GPU/runtime compatibility;
- vLLM functional tests;
- process-confinement tests;
- logging/coredump/swap tests;
- network namespace/firewall tests;
- package/update reproducibility tests;
- operational recovery tests.

BSD/other Unix-like systems remain research candidates, but Linux is the first target because the inference/GPU ecosystem and vLLM production path are Linux-centered.

## Important MAC limitation

SELinux/AppArmor can strongly constrain what files, sockets, capabilities, devices, and processes the engine may access. They cannot classify arbitrary bytes written to an already-inherited stdout/stderr stream as "safe" versus "private."

Therefore OS confinement **does not replace engine logging configuration**.

Payload-bearing request/output logging must be disabled at the engine/application layer. OS controls then constrain alternative sinks and exfiltration paths.

## Process layout

Recommended shape:

```text
public network
    |
    v
reverse proxy / TLS (optional dedicated process)
    |
    v
privacy wrapper + policy gate
    |
    +----> local attestor / evidence signer
    |
    v
private engine socket / namespace
    |
    v
vLLM or other inference engine
```

The wrapper and engine use separate Unix users/process identities.

The engine must not receive:

- attestation private keys;
- public API credentials except a narrow internal service credential if required;
- cloud control-plane credentials;
- arbitrary host filesystem mounts;
- write access to wrapper source/config/evidence stores.

## Engine ingress

Preferred order:

1. Unix-domain socket if supported cleanly by the selected server topology;
2. loopback in a shared private namespace;
3. dedicated container/service network with firewall policy.

The engine must not bind an internet-reachable listener.

Validation:

- enumerate listening sockets;
- probe from public/client network;
- probe from an unrelated local service identity;
- verify only the wrapper path succeeds.

## Engine egress

Steady-state serving should use default-deny egress where operationally feasible.

Separate lifecycle phases:

### Build/install phase

May require package repositories and source registries.

### Model acquisition/staging phase

May require approved artifact/model origins.

### Serving phase

Should need only explicit internal dependencies such as:

- local wrapper/engine traffic;
- private metrics collector;
- approved cluster/KV peers if the policy allows them;
- time/DNS only where required by the deployment design.

Arbitrary internet egress from the engine is not a normal serving requirement and should be blocked.

## Filesystem policy

Baseline intent:

- OS/application roots read-only to the engine;
- model artifacts read-only;
- only explicitly required runtime/cache directories writable;
- wrapper configuration not writable by engine identity;
- attestation keys not readable by engine identity;
- temporary data on bounded ephemeral storage;
- no general-purpose home directory requirement;
- no write path into web roots, package directories, SSH configuration, or system logs beyond explicitly approved stdout/stderr collection.

Cache/offload directories require their own retention and cleanup policy because they may contain information derived from user prompts.

## Capabilities and privilege

The engine should run unprivileged.

Baseline goals:

- no `CAP_SYS_PTRACE`;
- no `CAP_SYS_ADMIN`;
- no raw socket capability;
- no ability to load kernel modules;
- no ability to mount filesystems;
- no ability to modify network/firewall policy;
- `NoNewPrivileges`-equivalent protection;
- explicit GPU device access only.

The wrapper should also run unprivileged, with only the additional access required to contact the attestor and perform approved measurements.

A privileged setup/bootstrap process should exit before serving begins.

## Process inspection and ptrace

Reduce cross-process memory inspection from ordinary service accounts:

- restrictive ptrace/Yama policy where available;
- separate users for wrapper, engine, and monitoring agents;
- hide/restrict `/proc` cross-user details where compatible with operations;
- never grant engine ptrace capability;
- debugging through a separately controlled break-glass profile only.

These controls do not protect from root inside the guest.

## Swap and hibernation

Reference privacy profile:

- swap disabled during serving unless an explicitly reviewed encrypted-memory design replaces this rule;
- no guest hibernation/suspend-to-disk on customer-serving nodes;
- memory-pressure tests verify the host fails predictably rather than silently adding an unapproved swap path.

This reduces accidental plaintext persistence. It does not prove immediate physical DRAM/VRAM erasure.

## Core dumps and crash capture

Disable core dumps for wrapper and engine.

Verify all relevant layers:

- process `RLIMIT_CORE`/ulimit;
- systemd service configuration where applicable;
- kernel core pattern;
- crash-reporting agents;
- container runtime settings;
- cloud-agent crash upload features.

A crash test is required. Configuration inspection alone is insufficient.

## Logging

### Engine

Baseline:

- request-information logging disabled;
- model-output logging disabled;
- private engine HTTP access logging disabled unless a later test proves the retained fields are safe and useful;
- normal application log level selected to preserve health/error visibility without content capture;
- custom logging configuration reviewed and hashed when used.

### Wrapper

Allowed fields should be schema-based rather than "best effort redaction."

Examples:

```text
timestamp
pseudonymous request id
policy id
deployment id
model id
status/error class
latency
input/output token counts if policy permits
cache scope type (not salt or tenant id)
control-gate result
```

Raw request/response bodies are prohibited in normal mode.

### System logs

stdout/stderr may be captured by journald/systemd/container runtimes. That is acceptable only because the engine/wrapper are configured not to emit customer content there.

Retention, access control, and export destinations for these logs belong to the deployment profile.

## Metrics

Metrics are operational data, not automatically public data.

Requirements:

- vLLM `/metrics` is reachable only from the private monitoring plane;
- public ingress does not route engine metrics/admin/debug endpoints;
- metric labels must be reviewed for stable tenant/request identifiers;
- per-request histograms should be evaluated for traffic-analysis implications;
- metrics retention and access are documented.

## Tracing and APM

No tracing/APM agent is assumed safe by default.

Before enabling one, verify:

- request/response body capture is disabled;
- HTTP header capture excludes auth and sensitive user headers;
- local variables/stack capture does not serialize prompt objects;
- exception capture is sanitized;
- exporter destination is approved;
- the agent's own logs do not contain payloads.

## Cache and temporary storage

Cache policy must state:

- whether prefix caching is enabled;
- isolation scope;
- cryptographic hash choice;
- RAM-only versus disk/offload behavior;
- retention across process restart;
- deletion/eviction semantics;
- whether external KV/cache connectors exist;
- who can read cache directories or peer traffic.

An "ephemeral" label is not sufficient evidence. The deployment verifier should measure actual mounts/storage configuration.

## Containerization

Containers may be used, but a container boundary is not treated as a separate hardware trust domain.

Useful controls include:

- read-only root filesystem;
- dropped Linux capabilities;
- seccomp profile;
- AppArmor/SELinux policy;
- explicit device mounts;
- no Docker/socket/control-plane mounts;
- private network namespace;
- tmpfs/ephemeral writable paths;
- pinned image digest.

The image digest becomes one link in the evidence chain.

## SELinux/AppArmor goals

The MAC profile should enforce behaviors such as:

- engine can read approved model paths;
- engine can access approved GPU devices;
- engine can write only approved runtime/cache paths;
- engine can connect only to approved internal sockets/peers;
- engine cannot read wrapper/attestor secrets;
- engine cannot bind public network interfaces;
- engine cannot execute arbitrary downloaded binaries from writable directories;
- wrapper can reach engine and attestor but cannot arbitrarily administer the host.

The policy source should live in the public repository and have its own digest in deployment evidence.

## Service-manager hardening

For systemd-based profiles, evaluate and test controls such as:

```text
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes   # only if compatible with runtime/JIT requirements
RestrictAddressFamilies=...
CapabilityBoundingSet=...
SystemCallFilter=...         # only after compatibility testing
```

Do not enable a hardening directive merely because it sounds strong. Every directive must pass real engine startup, model load, inference, GPU, and recovery tests.

## Deployment measurement

Before accepting customer traffic, the verifier should capture at least:

- OS distribution/version/kernel identity;
- wrapper/attestor/engine artifact digests;
- exact engine command/config identity;
- model/tokenizer/template digests;
- service account identities;
- listening sockets;
- engine egress/firewall state;
- MAC enforcement mode and policy digest;
- swap state;
- core-dump policy;
- relevant writable mounts;
- cache/offload configuration;
- request/output/access logging state;
- break-glass state;
- evidence-signing key ID.

Critical measurements have a freshness window and are rechecked after relevant lifecycle events.

## Configuration drift

Events that should trigger immediate revalidation include:

- engine restart;
- wrapper restart;
- model change;
- engine version/image change;
- privacy policy change;
- network policy change;
- MAC policy change;
- logging configuration change;
- debug/profiler activation;
- cache connector change;
- attestation key rotation.

If the system cannot determine whether a critical control still holds, it fails closed for new customer requests.

## Break-glass profile

The preferred long-term design is to debug production without payload visibility.

If content-bearing diagnostics are temporarily required during development/testing:

- use synthetic/non-customer traffic where possible;
- isolate the node from customer serving;
- activate an explicit `break-glass` deployment profile;
- expire it automatically;
- record the policy transition;
- do not issue normal privacy receipts while active;
- purge generated diagnostic artifacts according to a documented test procedure before returning the node to service.

A future production policy may remove this mode entirely.

## Cloud boundary

Guest-level measurements do not prove cloud-host behavior.

The deployment evidence should explicitly list provider/hypervisor/physical-host trust as an assumption until a supported hardware-attestation path is implemented.

This is a feature, not an embarrassment: the evidence chain is useful precisely because it distinguishes measured controls from trusted infrastructure.