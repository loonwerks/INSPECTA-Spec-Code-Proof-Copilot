# SysML v2 General Rule Book

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

| Rule ID       | Name                                                                                         | Domain              | Construct    | Confidence |
| ------------- | -------------------------------------------------------------------------------------------- | ------------------- | ------------ | ---------- |
| MR-PROC-001   | Create process definitions from English subsystem or deployment-bound component descriptions | HAMR                | process      | 0.88       |
| MR-THREAD-001 | Create a behavioral thread inside each project-owned process                                 | HAMR                | thread       | 0.9        |
| MR-PORT-001   | Map English interface verbs to input/output DataPort or EventDataPort declarations           | SysMLv2             | port         | 0.89       |
| MR-CONN-001   | Map English data-flow descriptions to process/thread connection endpoints                    | SysMLv2             | connection   | 0.82       |
| MR-TRACE-001  | Preserve traceability from English text to process/thread/port/connection decisions          | repository-specific | traceability | 0.86       |

## 4. English Trigger Patterns

- subsystem, component, process, protection domain, partition, or deployment unit; English description groups behavior and interfaces under a named unit
- function shall compute, monitor, regulate, control, or transform values; periodic or time-triggered behavior is described
- receives, reads, obtains, samples, input; sends, outputs, publishes, writes, provides; event, message, sporadic, asynchronous trigger
- flows to; connected to; feeds; uses output of; sends to; publishes to
- requirement id; REMH section; source paragraph; architecture description

## 5. Formal Output Patterns

- `part def <ProcessName> :> Process or repository-equivalent process declaration with package/namespace placement`
- `part def <ThreadName>_i :> Thread with typed input/output ports and optional GUMBO block`
- `<direction> port <name> : <DataPort|EventDataPort>[<type>] or repository-equivalent port refinement`
- `connection <name> connects <sourcePort> to <destinationPort> or repository-equivalent connection syntax`
- `trace comment or metadata linking generated process fields to English source text`

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

### MR-PROC-001 - Create process definitions from English subsystem or deployment-bound component descriptions
- Domain: HAMR
- Target construct: process
- Source examples: monitor-related process examples, architecture examples
- English triggers: subsystem, component, process, protection domain, partition, or deployment unit; English description groups behavior and interfaces under a named unit
- Formal pattern: `part def <ProcessName> :> Process or repository-equivalent process declaration with package/namespace placement`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: component/subsystem name, architecture context, target platform when provided
- Generated outputs: process declaration, process part, traceability comment
- Required process fields: process name, package/namespace, process/thread containment
- Required thread fields: thread name, dispatch semantics when provided, period/domain when provided
- Required port fields: port name, direction, type, DataPort/EventDataPort classification when supported
- Required connection fields: none
- Syntax constraints: Use repository examples as the source of truth for HAMR SysML syntax.; Type-check before HAMR code generation when the tool is available.
- HAMR subset constraints: Use the HAMR-compatible SysML v2/AADL subset rather than arbitrary SysML v2.; Keep system architecture flat when required by HAMR tooling.
- seL4/Microkit constraints: For seL4 Microkit targets, model each process as a protection-domain candidate and keep one thread per process when required.
- Naming conventions: Use repository process naming style; preserve Monitor/Regulator/Operator names when present.
- Namespace conventions: Preserve package-qualified type names and enum values used in repository examples.
- Traceability: Preserve English description, requirement id, source section, or REMH comment as traceability metadata when available.
- Verification: Parse/type-check the SysML v2 fragment.; Regenerate HAMR artifacts for the selected target when available.; Run Logika/Verus/GUMBOX only when the repository has the required target configured.
- Structural scoring: Mandatory process/thread/port/connection fields must match structurally, not only semantically.; A low semantic distance is not sufficient if required structure is missing.
- Repair hints: Repair generated process/thread/port/connection fragments before modifying existing application logic.; Classify verification failures before editing artifacts.
- Confidence: 0.88

### MR-THREAD-001 - Create a behavioral thread inside each project-owned process
- Domain: HAMR
- Target construct: thread
- Source examples: Monitor and Regulator behavioral thread examples
- English triggers: function shall compute, monitor, regulate, control, or transform values; periodic or time-triggered behavior is described
- Formal pattern: `part def <ThreadName>_i :> Thread with typed input/output ports and optional GUMBO block`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: behavioral function name, ports, dispatch/period when available
- Generated outputs: thread declaration, process-thread containment, initialize/timeTriggered compatibility note
- Required process fields: process name, package/namespace, process/thread containment
- Required thread fields: thread name, dispatch semantics when provided, period/domain when provided
- Required port fields: port name, direction, type, DataPort/EventDataPort classification when supported
- Required connection fields: none
- Syntax constraints: Use repository examples as the source of truth for HAMR SysML syntax.; Type-check before HAMR code generation when the tool is available.
- HAMR subset constraints: Use the HAMR-compatible SysML v2/AADL subset rather than arbitrary SysML v2.; Keep system architecture flat when required by HAMR tooling.
- seL4/Microkit constraints: For seL4 Microkit targets, model each process as a protection-domain candidate and keep one thread per process when required.
- Naming conventions: Use the repository's thread suffix/prefix style, often <Function>_i for behavior-owning threads.
- Namespace conventions: Preserve package-qualified type names and enum values used in repository examples.
- Traceability: Preserve English description, requirement id, source section, or REMH comment as traceability metadata when available.
- Verification: Parse/type-check the SysML v2 fragment.; Regenerate HAMR artifacts for the selected target when available.; Run Logika/Verus/GUMBOX only when the repository has the required target configured.
- Structural scoring: Mandatory process/thread/port/connection fields must match structurally, not only semantically.; A low semantic distance is not sufficient if required structure is missing.
- Repair hints: Repair generated process/thread/port/connection fragments before modifying existing application logic.; Classify verification failures before editing artifacts.
- Confidence: 0.9

### MR-PORT-001 - Map English interface verbs to input/output DataPort or EventDataPort declarations
- Domain: SysMLv2
- Target construct: port
- Source examples: process port examples, GUMBO attachment examples
- English triggers: receives, reads, obtains, samples, input; sends, outputs, publishes, writes, provides; event, message, sporadic, asynchronous trigger
- Formal pattern: `<direction> port <name> : <DataPort|EventDataPort>[<type>] or repository-equivalent port refinement`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: English interface phrase, data type, direction, event vs data semantics
- Generated outputs: input port, output port, port type, DataPort/EventDataPort classification
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
- Repair hints: If HAMR rejects EventDataPort for a target, fall back only when repository examples show the accepted convention.; Do not change port direction to satisfy text similarity; direction is mandatory structure.
- Confidence: 0.89

### MR-CONN-001 - Map English data-flow descriptions to process/thread connection endpoints
- Domain: SysMLv2
- Target construct: connection
- Source examples: architecture examples, process-to-process connection examples
- English triggers: flows to; connected to; feeds; uses output of; sends to; publishes to
- Formal pattern: `connection <name> connects <sourcePort> to <destinationPort> or repository-equivalent connection syntax`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: producer component, consumer component, source port, destination port
- Generated outputs: connection declaration, source endpoint, destination endpoint
- Required process fields: process name, package/namespace, process/thread containment
- Required thread fields: thread name, dispatch semantics when provided, period/domain when provided
- Required port fields: port name, direction, type, DataPort/EventDataPort classification when supported
- Required connection fields: source endpoint, destination endpoint
- Syntax constraints: Use repository examples as the source of truth for HAMR SysML syntax.; Type-check before HAMR code generation when the tool is available.
- HAMR subset constraints: Use the HAMR-compatible SysML v2/AADL subset rather than arbitrary SysML v2.; Keep system architecture flat when required by HAMR tooling.
- seL4/Microkit constraints: For seL4 Microkit targets, model each process as a protection-domain candidate and keep one thread per process when required.
- Naming conventions: Derive process, thread, and port names from repository examples and English terms; do not invent unrelated names.
- Namespace conventions: Preserve package-qualified type names and enum values used in repository examples.
- Traceability: Preserve English description, requirement id, source section, or REMH comment as traceability metadata when available.
- Verification: Parse/type-check the SysML v2 fragment.; Regenerate HAMR artifacts for the selected target when available.; Run Logika/Verus/GUMBOX only when the repository has the required target configured.
- Structural scoring: Mandatory process/thread/port/connection fields must match structurally, not only semantically.; A low semantic distance is not sufficient if required structure is missing.
- Repair hints: Repair generated process/thread/port/connection fragments before modifying existing application logic.; Classify verification failures before editing artifacts.
- Confidence: 0.82

### MR-TRACE-001 - Preserve traceability from English text to process/thread/port/connection decisions
- Domain: repository-specific
- Target construct: traceability
- Source examples: FAA/REMH comments, training material references
- English triggers: requirement id; REMH section; source paragraph; architecture description
- Formal pattern: `trace comment or metadata linking generated process fields to English source text`
- Applicable context: HAMR-compatible SysML v2 process-definition workflow
- Forbidden context: Unsupported SysML v2/HAMR syntax, generated-code edits, or held-out test artifacts during rule extraction
- Required inputs: source file, English snippet, generated construct
- Generated outputs: source traceability note, requirement id, decision rationale
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
- Repair hints: Repair generated process/thread/port/connection fragments before modifying existing application logic.; Classify verification failures before editing artifacts.
- Confidence: 0.86
