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
-  **English requirements file (input):** 
     > check this file → [Steve_Miller_FAA_docAR-08-32.pdf](scp_demo_artifacts/isolette_io/Steve_Miller_FAA_docAR-08-32.pdf)
-  **SysML v2 files with generated GUMBO contracts (output):**
     > check this dir →  [Isolette SysML files](scp_demo_artifacts/isolette_io) e.g., `isolette_io/Monitor.sysml` and `isolette_io/Regulate.sysml`
-  **Input SysML v2 files:**
     > check this dir → [sysml-aadl-libraries submodule](scp_demo_artifacts/isolette_io/sysml-aadl-libraries) 
     > and this dir → [Isolette SysML files](scp_demo_artifacts/isolette_io), same as output but remove GUMBO contracts
-  **Meta-Rules file English to Gumbo Formal Specifications (User Mode):** 
     > check this file → [Gumbo_FSE_agent_Plan.txt](scp_demo_artifacts/isolette_io/Gumbo_FSE_agent_Plan.txt)
-  **Verification plan:**
     > check this file → [sireum_verification_plan.txt](scp_demo_artifacts/isolette_io/sireum_verification_plan.txt)
-  **Supervised self-adaptation plan (Meta-Rule Developer Mode):**
     > check this file → [Supervised Meta-Rules Adaptation Assurance Case](scp_demo_artifacts/isolette_io/SCP_Supervised_Meta_Rules_Adaptation_Assurance_Case.txt) 
     > and this file → [Cosine Meta-Rules Adaptation Plan](scp_demo_artifacts/isolette_io/SCP_Cosine_Meta_Rules_Adaptation_Plan.txt)


### Supervised Meta-Rules Adaptation Plan Design Rationale, Risk , and Comparative Analysis

-  **Meta-Rules vs. Foundation Model Fine-Tuning (Comparison):**  
     > Check this file → [Meta_Rules_vs_Foundational_Model_Fine_Tuning_Comparison.txt](scp_demo_artifacts/isolette_io/docs/Meta_Rules_vs_Foundational_Model_Fine_Tuning_Comparison.txt)

-  **Risk Mitigation and Governance (Meta-Rule Adaptation):**  
     > Check this file → [SCP_Meta_Rules_Risk_Mitigation_Mapping.txt](scp_demo_artifacts/isolette_io/docs/SCP_Meta_Rules_Risk_Mitigation_Mapping.txt)  

