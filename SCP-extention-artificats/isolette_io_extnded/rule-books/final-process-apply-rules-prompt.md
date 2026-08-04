# PROCESS_APPLYRULE / SYNTHESIZEPROCESSWITHRULES Prompt

Use the process-definition Meta-Rule book below to generate a HAMR-compatible SysML v2 process definition from the following English engineering or architecture description. Preserve traceability to the English source. Use the learned naming, package, structure, requirement, constraint, port, flow, process, thread, scheduling, binding, GUMBO attachment, Scala/Slang, and seL4/Microkit conventions. Generate only well-formed formal specification text unless explanatory comments are required for traceability. Report the rule IDs applied, unresolved ambiguities, required verification checks, and expected generated artifacts.

Inputs:
- selected process-definition rule book;
- English process or architecture description;
- training-material conventions;
- architecture context;
- data model context;
- existing package and namespace context;
- relevant existing SysML v2 files;
- relevant GUMBO attachment examples;
- optional Steve Miller / FAA REMH context;
- optional verification plan.

Outputs:
- generated SysML v2 process definition or process-related model fragment;
- generated thread/port/connection fragments, as appropriate;
- generated scheduling/binding fragments, as appropriate;
- GUMBO attachment location recommendation, as appropriate;
- applied rule IDs;
- source traceability notes;
- unresolved ambiguities;
- expected verifier obligations;
- confidence notes.
