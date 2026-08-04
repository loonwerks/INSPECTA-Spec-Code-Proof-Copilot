# Process Verification and Repair Rule Book

## 1. Purpose and Scope

This rule book captures auditable process-definition Meta-Rules for translating English engineering, architecture, and training descriptions into HAMR-compatible SysML v2 / AADL-style process definitions and adjacent artifacts. The focus is process/thread/port/connection/scheduling/binding structure, with GUMBO attachment handled only when it concerns the correct behavior-owning thread.

## 2. Source Context Used

- goal_2.txt: found
- AmerTahat_Bridging_Natural_Language_to_Verified_Implementation__A_Neuro_Symbolic_High_Assurance_MBSE_Pipeline_in_SysML_v2.pdf: found
- Gumbo_FSE_agent_Plan.txt: found
- sireum_verification_plan.txt: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/modeling-tips.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/component-implementation-guide.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/hamr-overview.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/technical-approach.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/sysmlv2-aadl-concepts.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/modeling-tips.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/component-implementation-guide.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/hamr-overview.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/technical-approach.md: found
- /home/amertahat/SPEC_CODE_PROOF/HAMR-agent-configuration-experiments/hamr-claude-training/doc/sysmlv2-aadl-concepts.md: found
- golden_examples: found
- training: missing
- process-definition-examples: missing
- architecture-examples: missing
- learning-rules/hamr-claude-training: missing
- Steve_Miller_FAA_docAR-08-32.txt: found
- Steve_Miller_FAA_docAR-08-32.pdf: found
- tools/self_adapt.py: found
- embedding_distance.py: found
- golden_examples: found
- learning-rules: found
- tools: found
- eval_results: found
- eva-results: found

## 3. Rules Table

| Rule ID            | Name                                                                      | Domain              | Construct                 | Confidence |
| ------------------ | ------------------------------------------------------------------------- | ------------------- | ------------------------- | ---------- |
| MR-VERIFY-PROC-001 | Treat HAMR/Sireum/Logika diagnostics as process-definition repair signals | verification-repair | process-definition repair | 0.83       |

## 4. English Trigger Patterns

- type-check error; HAMR subset violation; invalid port; missing connection; namespace error

## 5. Formal Output Patterns

- `classified failure -> smallest safe repair to generated SysML process/thread/port/connection or GUMBO attachment`

## 6. Examples From Training Only

- `PD-009-DEVICES`: `Devices.sysml` (device; constructs: connection, package, port, process, scheduling, thread)
- `PD-013-MONITOR`: `Monitor.sysml` (monitor; constructs: GUMBO, connection, package, port, process, scheduling, thread)

## 7. Anti-Patterns and Unsupported Forms

- Do not train on held-out regulator examples when monitor/regulator split is available.
- Do not accept semantic/cosine similarity when required process, thread, port, or connection structure is missing.
- Do not invent unsupported SysML v2, HAMR, GUMBO, Scala/Slang, seL4, Rust, or Verus syntax.
- Do not weaken GUMBO contracts to hide imported-library or profile/sourcepath limitations.
- Do not edit generated code or existing application logic unless explicitly approved.

## 8. Naming Conventions

- Derive process, thread, and port names from the English description and repository examples.
- Preserve Monitor/Regulator/Operator subsystem naming where present.
- Use package-qualified data types and enum values when repository examples require them.
- Preserve generated-code compatibility names for Scala/Slang and seL4/Rust/Verus targets.

## 9. Type and Namespace Conventions

- Use the repository data model and active HAMR SysML profile.
- Preserve package/namespace placement used in the training examples.
- Treat EventDataPort/DataPort selection as a structural decision, not a formatting choice.

## 10. Process/Thread Containment Conventions

- Use one project-owned behavioral thread per process for seL4/Microkit targets when required by the training material.
- Keep behavior contracts attached to the behavior-owning thread, not arbitrary packages or process wrappers.
- Preserve flat architecture restrictions when HAMR tooling requires them.

## 11. Port and Connection Conventions

- Map English input verbs to input ports and output verbs to output ports.
- Map data-flow phrases to source/destination endpoints.
- Preserve direction, data type, and DataPort/EventDataPort classification as mandatory structural fields.

## 12. Scheduling and Binding Conventions

- Map timing phrases such as periodic/every N ms to dispatch and period attributes only when supported by repository syntax.
- Map deployment/protection-domain/domain/processor text to binding attributes only when supported by the active target profile.

## 13. GUMBO Attachment Conventions

- Use canonical `language "GUMBO" /*{ ... }*/` only at repository-supported attachment locations.
- Attach behavior contracts to project-owned behavioral threads.
- Keep GUMBO section order stable when behavior is included: state, functions, integration, initialize, compute, compute_cases.

## 14. Traceability Conventions

- Preserve English source snippets, REMH/FAA references, requirement IDs, or architecture IDs when available.
- Record rule IDs applied and unresolved ambiguities in generated output folders.

## 15. Verification Expectations

- Parse/type-check SysML v2 before code generation when available.
- Regenerate HAMR artifacts for JVM/Slang or Microkit only as requested.
- Run Logika/Verus/GUMBOX checks when configured.
- Verification/type-checking is the hard gate; semantic distance is a generalization control.

## 16. Structural Scoring Expectations

- Compute structural precision, recall, F1, and mandatory-field pass/fail for process, thread, port, connection, scheduling, binding, and GUMBO attachment fields.
- Start epsilon_struct = 0.00 for mandatory fields as an explicit structure-check gate, separate from semantic/cosine epsilon.
- Allow up to 0.10 only for non-critical naming/formatting differences when type-checking and traceability are preserved.

## 17. Repair Guidance

- Classify failures before editing: process syntax, thread syntax, containment, port direction/type/status, DataPort/EventDataPort, connection endpoint, scheduling/binding, HAMR subset, namespace/type, GUMBO attachment, profile/sourcepath, marker-region, application logic, or environment blocker.
- Prefer repairing generated SysML process fragments and GUMBO attachment locations.
- Stop for approval before changing existing application logic, generated code, or held-out examples.

## 18. Known Limitations

- This deterministic scaffold does not call an LLM API and does not run verification commands automatically.
- If no golden process-definition examples are found, seed rules remain context-derived and require human review.
- Monitor-to-regulator results are cross-subsystem/intra-workflow evidence, not broad cross-system generality.

## 19. Prompt Template for Applying the Rule Book

Use the process-definition Meta-Rule book below to generate a HAMR-compatible SysML v2 process definition from the following English engineering or architecture description. Preserve traceability to the English source. Use the learned naming, package, structure, requirement, constraint, port, flow, process, thread, scheduling, binding, GUMBO attachment, Scala/Slang, and seL4/Microkit conventions. Generate only well-formed formal specification text unless explanatory comments are required for traceability. Report the rule IDs applied, unresolved ambiguities, required verification checks, and expected generated artifacts.

## Rule Details

### MR-VERIFY-PROC-001 - Treat HAMR/Sireum/Logika diagnostics as process-definition repair signals
- Domain: verification-repair
- Target construct: process-definition repair
- Source examples: sireum verification plan, verification logs
- English triggers: type-check error; HAMR subset violation; invalid port; missing connection; namespace error
- Formal pattern: `classified failure -> smallest safe repair to generated SysML process/thread/port/connection or GUMBO attachment`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: verification log, generated artifact, rule ids applied
- Generated outputs: failure classification, repair plan, verification expectation
- Required process fields: process name, package/namespace, process/thread containment
- Required thread fields: thread name, dispatch semantics when provided, period/domain when provided
- Required port fields: port name, direction, type, DataPort/EventDataPort classification when supported
- Required connection fields: none
- Syntax constraints: Use repository examples as the source of truth for HAMR SysML syntax.; Type-check before HAMR code generation when the tool is available.
- HAMR subset constraints: Use the HAMR-compatible SysML v2/AADL subset rather than arbitrary SysML v2.; Keep system architecture flat when required by HAMR tooling.
- seL4/Microkit constraints: For seL4 Microkit targets, model each process as a protection-domain candidate and keep one thread per process when required.
- Naming conventions: Derive process, thread, and port names from repository examples and English terms; do not invent unrelated names.
- Namespace conventions: Preserve package-qualified type names and enum values used in repository examples.
- Traceability: Preserve English description, requirement id, source section, or REMH comment as traceability metadata when available.
- Verification: Parse/type-check the SysML v2 fragment.; Regenerate HAMR artifacts for the selected target when available.; Run Logika/Verus/GUMBOX only when the repository has the required target configured.
- Structural scoring: Mandatory process/thread/port/connection fields must match structurally, not only semantically.; A low semantic distance is not sufficient if required structure is missing.
- Repair hints: Repair generated SysML process fragments before touching application logic.; Do not weaken contracts to compensate for profile/sourcepath problems.; Stop and request approval before modifying generated code or held-out golden artifacts.
- Confidence: 0.83
