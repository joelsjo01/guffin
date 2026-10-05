# Engine Principles

This file defines the project's permanent engineering and coding principles.

## Maintain the Software

- Give each module a coherent responsibility and keep dependencies directed
  toward the component that owns the behavior.
- Define contracts for shapes, precision, cache positions, committed state,
  cancellation and failure. Keep ownership and resource lifetimes explicit.
- Keep authoritative configuration and model capabilities in one place;
  adapters should pass the resolved configuration rather than recreate it.
- Share mechanisms when their behavior is shared. Keep architectural differences
  explicit and introduce abstractions when existing cases justify them.
- Remove abandoned execution paths and their exclusive tests and probes after
  tracing callers. Preserve useful findings in documentation and history.
- Keep independent numerical references, fault injection and developer probes
  outside the production execution dependency.
- Preserve behavior and hardware costs when refactoring: review allocations,
  transfers, synchronization and dispatches as part of the contract.
- Test observable behavior and meaningful invariants so internal structure can
  improve without rewriting tests merely to mirror the implementation.
- Keep builds and installation reproducible from a fresh checkout. Compiler
  dependency tracking must cover public and private headers.

## Optimize the System

- Treat end-to-end inference as the product. Optimize a subsystem only when
  evidence connects it to meaningful whole-path cost.
- Profile before choosing work. Attribute aggregate time, dispatches, data
  movement, allocation, synchronization, and CPU/GPU gaps.
- Estimate the maximum end-to-end gain first. If it is within run-to-run noise
  or does not justify the complexity, choose a larger bottleneck.
- Use microbenchmarks to explain a cost, compare candidates, or validate a
  hypothesis. Accept a performance change only when the real production path
  improves under comparable conditions.
- Prefer removing work, transfers, waits, allocations, and passes over making
  unnecessary work slightly faster.
- Measure release builds on representative inputs. Warm the path, repeat runs,
  report variance, control thermals and system state when practical, and
  interleave alternatives when drift could bias the result.

## Protect the Architecture

- Put behavior with the component that owns it. Make layout, lifetime,
  residency, synchronization, and mutation explicit at boundaries.
- Keep engine semantics independent from CLI, UI, benchmark, and diagnostic
  concerns. Diagnostics may observe production execution; they must not become
  a second execution path without a strong reason.
- Design the hot path for the intended hardware and workload, but keep
  interfaces narrow so specialization does not leak everywhere.
- Specialize when measured shapes or formats justify it. Generalize only when
  repeated real cases reveal a stable boundary; do not build speculative layers.
- Prefer the simplest design that preserves required invariants. Never trade a
  clear ownership model for local convenience or a small isolated win.
- Make performance-sensitive state persistent when its lifetime is persistent.
  Avoid hidden copies, implicit format conversions, temporary allocations, and
  synchronization inside cheap-looking abstractions.
- Complete the agreed implementation end to end. Let architecture and useful
  outcomes determine the scope of a change; avoid artificial iteration limits,
  temporary substitutes, and unnecessary approval boundaries.
- Do not mix performance work with unrelated cleanup, broad refactors, or new
  features. Remove rejected candidates and temporary instrumentation.

## Prove Changes

- Encode invariants in types, validation, ownership, and tests where practical.
  Treat dimensions, strides, packing, alignment, precision, aliasing, cache
  position, and synchronization as correctness properties.
- Validate assumptions where bad data enters and fail with enough context to
  identify the violated model, tensor, layout, backend, or lifecycle contract.
- Add deterministic tests for changed behavior and likely regressions. Cover
  boundary shapes, tails, precision limits, and unavailable devices as relevant.
- Use a small independent reference, golden output, or parity test to validate
  optimized numerical code. Do not use the optimized implementation as its own
  oracle.
- Choose tolerances from the expected numerical error and accumulation method;
  never loosen a tolerance merely to make a test pass.
- Test the public path that owns the behavior. Unit tests isolate the mechanism;
  integration tests prove the system connects correctly.
- Keep fallback behavior explicit. It must not silently change backend,
  residency, precision, or performance characteristics.
- A performance change is complete only when correctness checks pass, the real
  benchmark improves beyond noise, and the measurements support the claimed
  cause.

## Comment for the Next Decision

Write comments that preserve information a future reader cannot recover by
reading the code alone:

- why a design or algorithm was chosen over a plausible alternative;
- numerical assumptions, precision choices, and acceptable error;
- tensor shape, order, stride, packing, alignment, and aliasing invariants;
- buffer ownership, residency, lifetime, and synchronization requirements;
- hardware, backend, or API constraints and fallback behavior;
- non-obvious performance tradeoffs and when they hold.

Do not restate names, syntax, or obvious control flow. Prefer clearer code over
comments that explain confusing code. Put a short comment before the relevant
invariant or tricky block, and update or delete it with the code. Record durable
design reasons, not benchmark results or a development diary.

## Before Committing to a Design

Ask:

1. What measured whole-path problem does this solve?
2. What is the plausible upper bound on the end-to-end improvement?
3. Which ownership, layout, lifetime, or synchronization invariant changes?
4. Can the same result come from removing work or simplifying the data flow?
5. Is the abstraction supported by repeated cases, or only imagined future use?
6. How will correctness and real-path performance be proved independently?
7. If the idea fails, can all of its complexity be removed cleanly?

If these questions do not have concrete answers, gather evidence before adding
architecture.
