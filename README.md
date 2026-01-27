# INSPECTA-Spec-Code-Proof-Copilot
Spec, Code, and Proofs copilot for SysML v2. This multi-agent neuro-symbolic copilot leverages the INSPECTA symbolic toolchain, coupled with highly automated human-in-the-loop mechanisms, to generate trustworthy infrastructure and  system code directly from SysML v2 models.

> **Low entry point:** a simple prompt that triggers the SCP Copilot self-healing loop:
> *Follow Gumbo_FSE_agent_Plan.txt to formalize contracts in Regulate.sysml, then execute the Sireum verification plan to complete system- and code-level verification.


### 🎬 Video Demo

[![Watch the video](https://img.youtube.com/vi/BGtiUfd8LCQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=BGtiUfd8LCQ)

**Click the image above to play the SCP Copilot video demo.**

This demo shows **English-to-System and Code-Level Verification for provided Scala/Slang application logic**, where **SCP Copilot**:
- Parses English document requirements, Meta-Rules, and sysml v2 spec (but with *NO* GUMBO contracts)  
- Applies Verification Plans and supervised self-healing loop:  
- Generates Gumbo contracts  
- Inserts the generated contracts into SysML v2 files  
- Runs HAMR code generation  
- Performs code-level verification and Logika model integration verification  
- Detects errors, repairs formulas, and iterates automatically  
- Continues until code-level verification successfully completes  

**Related artifacts used in the demo:**
-  **English requirements file:** check this → `Steve_Miller_FAA_docAR-08-32.pdf`
-  **Input SysML v2 files:** check this → `isolette/sysml`
-  **SysML v2 files with generated GUMBO contracts (output):** check this → `isolette/sysml`
-  **Meta-Rules file English to Gumbo Formal Specifications (User Mode):** check these → `Gumbo_FSE_agent_Plan.txt`
-  **Verification plans:** check these → `sireum_verification_plan.txt` and `sireum_instructions_plan.md`
-  **Supervised self-adaptation plan (Meta-Rule Developer Mode):** check these → `SCP_Cosine_Self_Adaptation_Plan.txt`



