# INSPECTA-Spec-Code-Proof-Copilot
Spec, Code, and Proofs copilot for SysML v2. This multi-agent neuro-symbolic copilot leverages the INSPECTA symbolic toolchain, coupled with highly automated human-in-the-loop mechanisms, to generate trustworthy infrastructure and  system code directly from SysML v2 models with a low entry point.

### 🎬 Video Demo

[![Watch the video](https://img.youtube.com/vi/BGtiUfd8LCQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=BGtiUfd8LCQ)

**Click the image above to play the SCP Copilot video demo.**

This demo shows **English-to-System and Code-Level Verification for provided Scala/Slang application logic**, where **SCP Copilot**:

- Is triggered by a **low entry-point prompt** (a single instruction that initiates the self-healing formal spec–code–proof loop):
- Parses English document requirements, Meta-Rules, and sysml v2 spec (but with *NO* GUMBO contracts)  
- Applies Verification Plans and supervised self-healing loop:  
- Generates Gumbo contracts  
- Inserts the generated contracts into SysML v2 files  
- Runs HAMR code generation  
- Performs code-level verification and Logika model integration verification  
- Detects errors, repairs formulas, and iterates automatically  
- Continues until code-level verification successfully completes  

**Related artifacts isolette_io files used in the demo:**
-  **English requirements file (input):** check this → `isolette_io/Steve_Miller_FAA_docAR-08-32.pdf`
-  **SysML v2 files with generated GUMBO contracts (output):** check this → `isolette_io/*.sysml`
-  **Input SysML v2 files:** check this → `isolette_io/sysml-aadl-libraries @ d732430/*.sysml` and (`isolette_io/*.sysml` same as output but remove GUMBO contracts)
-  **Meta-Rules file English to Gumbo Formal Specifications (User Mode):** check these → `isolette_io/Gumbo_FSE_agent_Plan.txt`
-  **Verification plan:** check these → `sireum_verification_plan.txt`
-  **Supervised self-adaptation plan (Meta-Rule Developer Mode):**
     check these → → [Supervised Meta-Rules Adaptation Assurance Case](scp_demo_artifacts/isolette_io/SCP_Supervised_Meta_Rules_Adaptation_Assurance_Case.txt) 
     `isolette_io/SCP_Supervised_Meta_Rules_Adaptation_Assurance_Case.txt` and `SCP_Cosine_Meta_Rules_Adaptation_Plan.txt`



