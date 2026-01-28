// Regulate.sysml

package Regulate {
    
    private import Isolette_Data_Model::*;
    private import AADL::*;
    private import AADL_Project::*;
    private import AADL_Project::Time_Units::*;
    
    // ===========================================================================================================
    //
    //   R e g u l a t e    T e m p e r a t u r e    P r o c e s s
    //
    // @description process that holds threads that regulate air temperature
    // A.5.1 REGULATE TEMPERATURE FUNCTION
    // 
    // ===========================================================================================================
    
        // process that holds threads that regulate air temperature
    part def Regulate_Temperature_i :> System {
        
        //thread to manage the interactions between with operator interface and the rest of the regulator process
        part mri : Manage_Regulator_Interface_Process;
        // thread to manage the heat source.  Receives the desired range, current determine, and determines whether or not to
        //send an on/off command to the heat source
        part mhs : Manage_Heat_Source_Process_i;
        //thread to manage the regular mode. Based inputs describing the failure status of the interface, subsystem internals,
        //and temperature  status,  determines the mode of the regulator process (i.e., if the mode is Init, Normal, or Failed modes)
        part mrm : Manage_Regulator_Mode_Process_i;
        
        part drf : Detect_Regulator_Failure_Process_i;

        // ===== INPUTs (monitored variables) ...based in part on Figure A.2 and Table A.5 
        //  desired temperature range from operator interface 		
        in port lower_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        in port upper_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        //  actual temperature from temp sensor
        in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
             
        // ===== OUTPUTs (controlled variables)  ...based in part on Figure A.2 and Table A.6
        // status of regulate temperature function as reported to operator interface
        out port regulator_status : DataPort { out :> type : Isolette_Data_Model::Status; }
        // value of current temperature to display on operator interface
        out port displayed_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        // actuation command to heater
        out port heat_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }

        
        // ==== INPUT INTERFACE values to internal components ==== 		
        // send incoming desired temperature range to regulator interface manager
        connection rudt : PortConnection
            connect upper_desired_tempWstatus to mri.upper_desired_tempWstatus;
        connection rldt : PortConnection
            connect lower_desired_tempWstatus to mri.lower_desired_tempWstatus;
    
        // send incoming current temperature (with status value)
        //   ...to interface manager
        connection rcti : PortConnection 
            connect current_tempWstatus to mri.current_tempWstatus;
        //   ...and to heat source manager
        connection rcth : PortConnection 
            connect current_tempWstatus to mhs.current_tempWstatus;
        //   ...and to mode manager
        connection rctm : PortConnection
            connect current_tempWstatus to mrm.current_tempWstatus;
       
        // ==== OUTPUT INTERFACE values from internal components ==== 
        // output temperature to display on operator interface
        connection rdt : PortConnection
            connect mri.displayed_temp to displayed_temp;
        // output regulator subsystem status to operator interface  
        connection rrs : PortConnection
            connect mri.regulator_status to regulator_status;
        // output on/off command to heat source
        connection rhc : PortConnection
            connect mhs.heat_control to heat_control;
      
        // ==== INTERNAL communication ====
        // send validated desired temperature from interface manager to heat source manager
        connection mudt : PortConnection 
            connect mri.upper_desired_temp to mhs.upper_desired_temp;
        connection mldt : PortConnection
               connect mri.lower_desired_temp to mhs.lower_desired_temp;
      
        // send regulator mode from mode manager 
        //  ... to interface manager
        connection rrmi : PortConnection 
            connect mrm.regulator_mode to mri.regulator_mode;
        //  ... and to heat source manager
        connection rrmh : PortConnection 
            connect mrm.regulator_mode to mhs.regulator_mode;
    
        // send regulator interface failure to mode manager
        connection rif : PortConnection
            connect mri.interface_failure to mrm.interface_failure;

        connection mif: PortConnection
            connect drf.internal_failure to mrm.internal_failure;
    }

    
    // ===========================================================================================================
    //
    //   M a n a g e    R e g u l a t o r   I n t e r f a c e   T h r e a d
    //
    //  
    // A.5.1.1 Manage Regulator Interface Function
    // 
    // ===========================================================================================================

    part def Manage_Regulator_Interface_Process :> Process {
        
        part mri: Manage_Regulator_Interface_i;
        
        // ==== INPUTS ====	(see Figure A.3)
        //  desired temperature from operator interface
        in port upper_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        in port lower_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
          // current temperature from temp sensor
        in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        // current mode of regulator subsystem
        in port regulator_mode : DataPort { in :> type : Isolette_Data_Model::Regulator_Mode; }
      
        // ==== OUTPUTS ====	(see Figure A.3)
        // validated ("valid" status confirmed) desired temperature range from operator interface
        out port upper_desired_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        out port lower_desired_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        // current temperature to display on operator interface
        out port displayed_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        // overall status of regulator subsystem sent to operator interface
        out port regulator_status : DataPort { out :> type : Isolette_Data_Model::Status; }
        // flag indicating improper inputs; sent to determine overall subsystem mode 
        out port interface_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

        connection udtw: PortConnection connect upper_desired_tempWstatus to mri.upper_desired_tempWstatus;
        connection ldtw: PortConnection connect lower_desired_tempWstatus to mri.lower_desired_tempWstatus;
        connection ctw: PortConnection connect current_tempWstatus to mri.current_tempWstatus;
        connection rm: PortConnection connect regulator_mode to mri.regulator_mode;

        connection udt: PortConnection connect mri.upper_desired_temp to upper_desired_temp;
        connection ldt: PortConnection connect mri.lower_desired_temp to lower_desired_temp;
        connection dt: PortConnection connect mri.displayed_temp to displayed_temp;
        connection rs: PortConnection connect mri. regulator_status to regulator_status;
        connection inf: PortConnection connect mri.interface_failure to interface_failure;

        attribute Domain: CASE_Scheduling::Domain = 7;
    }

    part def Manage_Regulator_Interface_i :> Thread {
        attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
        attribute :>> Period = 1000 [millisecond];
        attribute Domain: CASE_Scheduling::Domain = 7;
        attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

        // ==== INPUTS ====	(see Figure A.3)
        //  desired temperature from operator interface
        in port upper_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        in port lower_desired_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
          // current temperature from temp sensor
        in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        // current mode of regulator subsystem
        in port regulator_mode : DataPort { in :> type : Isolette_Data_Model::Regulator_Mode; }
      
        // ==== OUTPUTS ====	(see Figure A.3)
        // validated ("valid" status confirmed) desired temperature range from operator interface
        out port upper_desired_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        out port lower_desired_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        // current temperature to display on operator interface
        out port displayed_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
        // overall status of regulator subsystem sent to operator interface
        out port regulator_status : DataPort { out :> type : Isolette_Data_Model::Status; }
        // flag indicating improper inputs; sent to determine overall subsystem mode 
        out port interface_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

        language "GUMBO" /*{
          integration
            assume Table_A_7_LowerDesiredTemp "Range [96..101] |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=108>":
              96 [s32] <= lower_desired_tempWstatus.degrees & lower_desired_tempWstatus.degrees <= 101 [s32];
            assume Table_A_7_UpperDesiredTemp "Range [97..102] |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=108>":
              97 [s32] <= upper_desired_tempWstatus.degrees & upper_desired_tempWstatus.degrees <= 102 [s32];
          initialize
            guarantee regulatorStatusInitiallyInit:
              regulator_status == Isolette_Data_Model::Status.Init_Status;

          compute
            compute_cases
              case REQ_MRI_1 "If mode INIT => status Init |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Init_Regulator_Mode;
                guarantee regulator_status == Isolette_Data_Model::Status.Init_Status;

              case REQ_MRI_2 "If mode NORMAL => status On |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode;
                guarantee regulator_status == Isolette_Data_Model::Status.On_Status;

              case REQ_MRI_3 "If mode FAILED => status Failed |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;
                guarantee regulator_status == Isolette_Data_Model::Status.Failed_Status;

              case REQ_MRI_4 "If mode NORMAL => display temp rounded |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode;
                guarantee displayed_temp.degrees == current_tempWstatus.degrees;

              case REQ_MRI_5 "If mode not NORMAL => display temp UNSPECIFIED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume T;
                guarantee (regulator_mode != Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode) implies true;

              case REQ_MRI_6 "If lower OR upper status Invalid => interface failure True |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lower_desired_tempWstatus.status == Isolette_Data_Model::ValueStatus.Invalid |
                       upper_desired_tempWstatus.status == Isolette_Data_Model::ValueStatus.Invalid;
                guarantee interface_failure.flag;

              case REQ_MRI_7 "If lower AND upper status Valid => interface failure False |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lower_desired_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid &
                       upper_desired_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid;
                guarantee not interface_failure.flag;

              case REQ_MRI_8 "If NOT interface failure => copy desired range (mux) |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume T;
                guarantee (not interface_failure.flag) implies
                  (lower_desired_temp.degrees == lower_desired_tempWstatus.degrees &
                   upper_desired_temp.degrees == upper_desired_tempWstatus.degrees);

              case REQ_MRI_9 "If interface failure True => UNSPECIFIED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume T;
                guarantee interface_failure.flag implies true;
        }*/

            
    }
    
    // ===========================================================================================================
    //
    //   M a n a g e    R e g u l a t o r   M o d e   T h r e a d
    //
    //  
    //   A.5.1.2 Manage Regulator Mode Function
    // 
    // ===========================================================================================================

    part def Manage_Regulator_Mode_Process_i :> Process {

        part mrm: Manage_Regulator_Mode_i;

        // ==== INPUTS ========		
            // current temperature from temp sensor
          in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
          // status of operator interface interactions
          in port interface_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
          // internal status of regulator
          in port internal_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
  
          // ==== OUTPUTS ========		
          // mode of regulator (Init, Normal, Failed)
          out port regulator_mode : DataPort { out :> type : Isolette_Data_Model::Regulator_Mode; }

        connection ctw: PortConnection connect current_tempWstatus to mrm.current_tempWstatus;
        connection iff: PortConnection connect interface_failure to mrm.interface_failure;
        connection inf: PortConnection connect internal_failure to mrm.internal_failure;

        connection rm: PortConnection connect mrm.regulator_mode to regulator_mode;

        attribute Domain: CASE_Scheduling::Domain = 8;
    }

    part def Manage_Regulator_Mode_i :> Thread {
        attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
        attribute :>> Period = 1000 [millisecond];
        attribute Domain: CASE_Scheduling::Domain = 8;
        attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

        // ==== INPUTS ========		
            // current temperature from temp sensor
          in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
          // status of operator interface interactions
          in port interface_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
          // internal status of regulator
          in port internal_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
  
        // ==== OUTPUTS ========		
          // mode of regulator (Init, Normal, Failed)
          out port regulator_mode : DataPort { out :> type : Isolette_Data_Model::Regulator_Mode; }

        language "GUMBO" /*{
          state
            lastRegulatorMode: Isolette_Data_Model::Regulator_Mode;

          functions
            def timeout_condition_satisfied(): Base_Types::Boolean := F;

          initialize
            guarantee regulatorModeInitiallyInit:
              regulator_mode == Isolette_Data_Model::Regulator_Mode.Init_Regulator_Mode;

          compute
            compute_cases
              case REQ_MRM_2 "If mode INIT and status True => mode NORMAL |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lastRegulatorMode == Isolette_Data_Model::Regulator_Mode.Init_Regulator_Mode &
                       (not interface_failure.flag) &
                       (not internal_failure.flag) &
                       (current_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid);
                guarantee regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode;

              case REQ_MRM_3 "If mode NORMAL and status False => mode FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lastRegulatorMode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode &
                       (interface_failure.flag |
                        internal_failure.flag |
                        current_tempWstatus.status != Isolette_Data_Model::ValueStatus.Valid);
                guarantee regulator_mode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;

              case REQ_MRM_4 "If mode INIT and timeout > 1.0s => mode FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lastRegulatorMode == Isolette_Data_Model::Regulator_Mode.Init_Regulator_Mode &
                       timeout_condition_satisfied();
                guarantee regulator_mode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;

              case REQ_MRM_5 "If mode FAILED => remain FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=109>":
                assume lastRegulatorMode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;
                guarantee regulator_mode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;
        }*/

          		
    }
    
    // ===========================================================================================================
    //
    //   M a n a g e    H e a t    S o u r c e     T h r e a d
    //
    //  
    //    A.5.1.3 Manage Heat Source Function
    // 
    // ===========================================================================================================
    
    part def Manage_Heat_Source_Process_i :> Process {
    
        part mhs: Manage_Heat_Source_i;
        
        // ======== INPUTS =======
        // current temperature (from temp sensor)
        in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        // lowest and upper bound of desired temperature range
         in port lower_desired_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
        in port upper_desired_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
        // subsystem mode 
         in port regulator_mode : DataPort { in :> type : Isolette_Data_Model::Regulator_Mode; }
    
        // ======== OUTPUTS =======
        // command to turn heater on/off (actuation command)
          out port heat_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }
          
        connection ctw: PortConnection connect current_tempWstatus to mhs.current_tempWstatus;
        connection ldt: PortConnection connect lower_desired_temp to mhs.lower_desired_temp;
        connection udt: PortConnection connect upper_desired_temp to mhs.upper_desired_temp;
        connection rm: PortConnection connect regulator_mode to mhs.regulator_mode;

        connection hc: PortConnection connect mhs.heat_control to heat_control;

        attribute Domain: CASE_Scheduling::Domain = 9;
    }

    part def Manage_Heat_Source_i :> Thread {
        attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
        attribute :>> Period = 1000 [millisecond];
        attribute Domain: CASE_Scheduling::Domain = 9;
        attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;
        
        // ======== INPUTS =======
        // current temperature (from temp sensor)
        in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
        // lowest and upper bound of desired temperature range
         in port lower_desired_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
        in port upper_desired_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
        // subsystem mode 
         in port regulator_mode : DataPort { in :> type : Isolette_Data_Model::Regulator_Mode; }
    
        // ======== OUTPUTS =======
        // command to turn heater on/off (actuation command)
          out port heat_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }
          
        language "GUMBO" /*{
          state
            lastCmd: Isolette_Data_Model::On_Off;

          initialize
            guarantee initlastCmd:
              lastCmd == Isolette_Data_Model::On_Off.Off;

          compute
            compute_cases
              case REQ_MHS_1 "Init => heat Off |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Init_Regulator_Mode;
                guarantee heat_control == Isolette_Data_Model::On_Off.Off &
                          lastCmd == Isolette_Data_Model::On_Off.Off;

              case REQ_MHS_2 "Normal & temp < lower desired => heat On |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode &
                       current_tempWstatus.degrees < lower_desired_temp.degrees;
                guarantee heat_control == Isolette_Data_Model::On_Off.Onn &
                          lastCmd == Isolette_Data_Model::On_Off.Onn;

              case REQ_MHS_3 "Normal & temp > upper desired => heat Off |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode &
                       current_tempWstatus.degrees > upper_desired_temp.degrees;
                guarantee heat_control == Isolette_Data_Model::On_Off.Off &
                          lastCmd == Isolette_Data_Model::On_Off.Off;

              case REQ_MHS_4 "Normal & temp in desired band => heat holds |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Normal_Regulator_Mode &
                       (current_tempWstatus.degrees >= lower_desired_temp.degrees &
                        current_tempWstatus.degrees <= upper_desired_temp.degrees);
                guarantee heat_control == lastCmd;

              case REQ_MHS_5 "Failed => heat Off |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=111>":
                assume regulator_mode == Isolette_Data_Model::Regulator_Mode.Failed_Regulator_Mode;
                guarantee heat_control == Isolette_Data_Model::On_Off.Off &
                          lastCmd == Isolette_Data_Model::On_Off.Off;

              case REQ_MHS_LATENCY "Heat response <= 6 s |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=110>":
                assume T;
                guarantee true;
        }*/


            }

    // ===========================================================================================================
    //
    //   D e t e c t     R e g u l a t o r      F a i l u r e     T h r e a d
    //
    //  
    //  AR-08-32  A.5.1.4 Detect Regulator Failure Function
    // 
    //  "The Detect Regulator Failure Function identifies internal failures, (e.g., a memory check failure)
    //   in the Regulate Temperature Function. It defines a single Boolean-valued internal variable,
    //   Regulator Internal Failure, which is set to True if an internal failure is detected."
    //
    // http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=111	

    part def Detect_Regulator_Failure_Process_i :> Process {

        part drf: Detect_Regulator_Failure_i;

        out port internal_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

        connection inf: PortConnection connect drf.internal_failure to internal_failure;

        attribute Domain: CASE_Scheduling::Domain = 10;
    }

    part def Detect_Regulator_Failure_i :> Thread {

        attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
        attribute :>> Period = 1000 [millisecond];
        attribute Domain: CASE_Scheduling::Domain = 10;
        attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;
                
        out port internal_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

    }
}
