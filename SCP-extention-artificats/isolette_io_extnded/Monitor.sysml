// Monitor.sysml
package Monitor {
	
	private import Isolette_Data_Model::*;
	private import AADL::*;
	private import AADL_Project::*;
    private import AADL_Project::Time_Units::*;
		
	//A.5.2 Monitor Temperature Function
	// thread to manage the interactions between with operator interface and the rest
    // of the monitor process
	part def Monitor_Temperature_i :> System {

	    //thread for monitor interface implementation
	    part mmi : Manage_Monitor_Interface_Process_i;

	    //thread for manage alarm implementation
	    part ma : Manage_Alarm_Process_i;
	    //thread for monitor mode implementation
	    part mmm : Manage_Monitor_Mode_Process_i;

		part dmf: Detect_Monitor_Failure_Process_i;

		// ===== INPUTs (monitored variables) ...based in part on Figure A.2 and Table A.5 
	    // value of current temp (with status flag) from air temp sensor
	    in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
	    // operator-specific alarm range for temperature
	    in port lower_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
	    in port upper_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
	    
	    // ===== OUTPUTs (controlled variables)  ...based in part on Figure A.2 and Table A.6
	    // control signal (on/off) to alarm component 
	    out port alarm_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }
	    // enumerated type describing status/mode of monitor (Init, On/operating, or Failed)
	    out port monitor_status : DataPort { out :> type : Isolette_Data_Model::Status; }
	    
	    // ==== INPUT INTERFACE values to internal components ====
	    // send incoming alarm temperature range to monitor interface manager
	    connection muat : PortConnection
	    	connect upper_alarm_tempWstatus to mmi.upper_alarm_tempWstatus;
	    connection mlat : PortConnection
	    	connect lower_alarm_tempWstatus to mmi.lower_alarm_tempWstatus;
	  
	    // send incoming current temperature (with status value)
	    //   ...to interface manager
	    connection mcti : PortConnection
	    	connect current_tempWstatus to mmi.current_tempWstatus;
  	    //   ...to alarm manager (controls alarm)
	    connection mcta : PortConnection
	    	connect current_tempWstatus to ma.current_tempWstatus;
	    //   ...to mode manager
	    connection mctm : PortConnection
	    	connect current_tempWstatus to mmm.current_tempWstatus;
	
	    // ==== OUTPUT INTERFACE values from internal components ==== 
	    // send alarm actuation command (on/off) 
	    connection malrm : PortConnection
	    	connect ma.alarm_control to alarm_control;
	    // send status of monitor subsystem  to operator interface
	    connection mms : PortConnection
	    	connect mmi.monitor_status to monitor_status;
	  
	    // ==== INTERNAL communication ====
	    // send validated alarm range from interface manager to alarm manager (controller)
   	    connection maul : PortConnection
   	    	connect mmi.upper_alarm_temp to ma.upper_alarm_temp;
	    connection mall : PortConnection
	    	connect mmi.lower_alarm_temp to ma.lower_alarm_temp;
	
	    // send monitor mode from mode manager 
	    //  ... to interface manager
	    connection mmmi : PortConnection
	    	connect mmm.monitor_mode to mmi.monitor_mode;
	    //  ... to alarm manager
	    connection mmma : PortConnection
	    	connect mmm.monitor_mode to ma.monitor_mode;
	  
	    // send regulator interface failure to mode manager
	    connection intff : PortConnection
	    	connect mmi.interface_failure to mmm.interface_failure;

		connection mif: PortConnection 
			connect dmf.internal_failure to mmm.internal_failure;
	}
	
	// ===========================================================================================================
    // 
	//   M a n a g e    M o n i t o r   I n t e r f a c e   T h r e a d
	// 
	// 
	// A.5.2.1 Manage Monitor Interface Function
	// 
	// ===========================================================================================================
	
	part def Manage_Monitor_Interface_Process_i :> Process {

		part mmi: Manage_Monitor_Interface_i;

		// ==== INPUTS ====	(see Figure A.5)
		//  operator specified alarm range
		in port upper_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		in port lower_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// current temperature from temp sensor
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// current mode of monitor subsystem (init, normal or failed)
		in port monitor_mode : DataPort { in :> type : Isolette_Data_Model::Monitor_Mode; }
		
		// ==== OUTPUTS ====	(see Figure A.5)  
		// validated ("valid" status confirmed & removed) alarm temperature range from operator interface
		out port upper_alarm_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
		out port lower_alarm_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
		// overall status of monitor subsystem sent to operator interface
	    out port monitor_status : DataPort { out :> type : Isolette_Data_Model::Status; }
	    // flag indicating improper inputs; sent to determine overall subsystem mode 
		out port interface_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

		connection uatw: PortConnection connect upper_alarm_tempWstatus to mmi.upper_alarm_tempWstatus;
		connection latw: PortConnection connect lower_alarm_tempWstatus to mmi.lower_alarm_tempWstatus;
		connection ctw: PortConnection connect current_tempWstatus to mmi.current_tempWstatus;
		connection mm: PortConnection connect monitor_mode to mmi.monitor_mode;

		connection uat: PortConnection connect mmi.upper_alarm_temp to upper_alarm_temp;
		connection lat: PortConnection connect mmi.lower_alarm_temp to lower_alarm_temp;
		connection ms: PortConnection connect mmi.monitor_status to monitor_status;
		connection ifc: PortConnection connect mmi.interface_failure to interface_failure;

		attribute Domain: CASE_Scheduling::Domain = 4;
	}

	part def Manage_Monitor_Interface_i :> Thread {

		attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
		attribute :>> Period = 1000 [millisecond];
		attribute Domain: CASE_Scheduling::Domain = 4;
		attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

		// ==== INPUTS ====	(see Figure A.5)
		//  operator specified alarm range
		in port upper_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		in port lower_alarm_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// current temperature from temp sensor
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// current mode of monitor subsystem (init, normal or failed)
		in port monitor_mode : DataPort { in :> type : Isolette_Data_Model::Monitor_Mode; }
		
		// ==== OUTPUTS ====	(see Figure A.5)  
		// validated ("valid" status confirmed & removed) alarm temperature range from operator interface
		out port upper_alarm_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
		out port lower_alarm_temp : DataPort { out :> type : Isolette_Data_Model::Temp_i; }
		// overall status of monitor subsystem sent to operator interface
	    out port monitor_status : DataPort { out :> type : Isolette_Data_Model::Status; }
		// flag indicating improper inputs; sent to determine overall subsystem mode 
		out port interface_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

		language "GUMBO" /*{
		  integration
		    assume Table_A_12_LowerAlarmTemp "Range [96..101] |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=112>":
		      96 [s32] <= lower_alarm_tempWstatus.degrees & lower_alarm_tempWstatus.degrees <= 101 [s32];
		    assume Table_A_12_UpperAlarmTemp "Range [97..102] |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=112>":
		      97 [s32] <= upper_alarm_tempWstatus.degrees & upper_alarm_tempWstatus.degrees <= 102 [s32];

		  initialize
		    guarantee monitorStatusInitiallyInit:
		      monitor_status == Isolette_Data_Model::Status.Init_Status;

		  compute
		    compute_cases
		      case REQ_MMI_1 "If mode INIT => status Init |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Init_Monitor_Mode;
		        guarantee monitor_status == Isolette_Data_Model::Status.Init_Status;

		      case REQ_MMI_2 "If mode NORMAL => status On |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode;
		        guarantee monitor_status == Isolette_Data_Model::Status.On_Status;

		      case REQ_MMI_3 "If mode FAILED => status Failed |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;
		        guarantee monitor_status == Isolette_Data_Model::Status.Failed_Status;

		      case REQ_MMI_4 "If lower OR upper status Invalid => interface failure True |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume lower_alarm_tempWstatus.status == Isolette_Data_Model::ValueStatus.Invalid |
		               upper_alarm_tempWstatus.status == Isolette_Data_Model::ValueStatus.Invalid;
		        guarantee interface_failure.flag;

		      case REQ_MMI_5 "If lower AND upper status Valid => interface failure False |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume lower_alarm_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid &
		               upper_alarm_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid;
		        guarantee not interface_failure.flag;

		      case REQ_MMI_6 "If NOT interface failure => copy alarm range (mux) |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume T;
		        guarantee (not interface_failure.flag) implies
		          (lower_alarm_temp.degrees == lower_alarm_tempWstatus.degrees &
		           upper_alarm_temp.degrees == upper_alarm_tempWstatus.degrees);

		      case REQ_MMI_7 "If interface failure True => UNSPECIFIED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=113>":
		        assume T;
		        guarantee interface_failure.flag implies true;
		}*/
	}
	
	// ===========================================================================================================
	//
	//   M a n a g e    M o n i t o r   M o d e   T h r e a d
	//
	//  
	//   A.5.2.2 Manage Monitor Mode Function
	// 
	// ===========================================================================================================
	
	// A.5.2.2 Manage Monitor Mode Function
	
	part def Manage_Monitor_Mode_Process_i :> Process {
		
		part mmm: Manage_Monitor_Mode_i;

		// ==== INPUTS ========		
		// current temperature from temp sensor
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// status of operator interface interactions
		in port interface_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
		// in case of internal failure, a flag is raised
		in port internal_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
		   
		// ==== OUTPUTS ========		
		// mode of regulator (Init, Normal, Failed)   
		out port monitor_mode : DataPort { out :> type : Isolette_Data_Model::Monitor_Mode; }

		connection ctw: PortConnection connect current_tempWstatus to mmm.current_tempWstatus;
		connection iff: PortConnection connect interface_failure to mmm.interface_failure;
		connection inf: PortConnection connect internal_failure to mmm.internal_failure;

		connection mm: PortConnection connect mmm.monitor_mode to monitor_mode;

		attribute Domain: CASE_Scheduling::Domain = 3;
	}

	part def Manage_Monitor_Mode_i :> Thread {
		
		attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
		attribute :>> Period = 1000 [millisecond];
		attribute Domain: CASE_Scheduling::Domain = 3;
		attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

		// ==== INPUTS ========		
		// current temperature from temp sensor
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// status of operator interface interactions
		in port interface_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
		// in case of internal failure, a flag is raised
		in port internal_failure : DataPort { in :> type : Isolette_Data_Model::Failure_Flag_i; }
		   
		// ==== OUTPUTS ========		
		// mode of regulator (Init, Normal, Failed)   
		out port monitor_mode : DataPort { out :> type : Isolette_Data_Model::Monitor_Mode; }
		
		language "GUMBO" /*{
		  state
		    lastMonitorMode: Isolette_Data_Model::Monitor_Mode;

		  functions
		    def timeout_condition_satisfied(): Base_Types::Boolean := F;

		  initialize
		    guarantee monitorModeInitiallyInit:
		      monitor_mode == Isolette_Data_Model::Monitor_Mode.Init_Monitor_Mode;

		  compute
		    compute_cases
		      case REQ_MMM_1 "If mode INIT and status Valid => mode NORMAL |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=114>":
		        assume lastMonitorMode == Isolette_Data_Model::Monitor_Mode.Init_Monitor_Mode &
		               (not interface_failure.flag) &
		               (not internal_failure.flag) &
		               (current_tempWstatus.status == Isolette_Data_Model::ValueStatus.Valid);
		        guarantee monitor_mode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode;

		      case REQ_MMM_2 "If mode INIT and timeout > 1.0s => mode FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=114>":
		        assume lastMonitorMode == Isolette_Data_Model::Monitor_Mode.Init_Monitor_Mode &
		               timeout_condition_satisfied();
		        guarantee monitor_mode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;

		      case REQ_MMM_3 "If mode NORMAL and status Invalid => mode FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=114>":
		        assume lastMonitorMode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode &
		               (interface_failure.flag |
		                internal_failure.flag |
		                current_tempWstatus.status != Isolette_Data_Model::ValueStatus.Valid);
		        guarantee monitor_mode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;

		      case REQ_MMM_4 "If mode FAILED => remain FAILED |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=115>":
		        assume lastMonitorMode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;
		        guarantee monitor_mode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;
		}*/
	}
	
	// ===========================================================================================================
	//
	//   M a n a g e    A l a r m    T h r e a d
	//
	//  
	//    A.5.2.3 Manage Alarm Function
	// 
	// ===========================================================================================================
	
	
	// A.5.2.3 Manage Alarm Function
	// The Manage Alarm Function turns the Alarm Control on when the Current
	// Temperature of the Room falls below or rises above the Alarm Temperature Range.
	
	part def Manage_Alarm_Process_i :> Process {

		part ma: Manage_Alarm_i;

		// ======== INPUTS =======
		// current temperature (from temp sensor)
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// lowest and upper bound of operator specified alarm temperature range
		in port lower_alarm_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
		in port upper_alarm_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
		// subsystem mode
		in port monitor_mode : DataPort { in :> type : Isolette_Data_Model::Monitor_Mode; }
		  
		// ======== OUTPUTS =======
		// command to turn alarm on/off (actuation command)  
		out port alarm_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }

		connection ctw: PortConnection connect current_tempWstatus to ma.current_tempWstatus;
		connection lat: PortConnection connect lower_alarm_temp to ma.lower_alarm_temp;
		connection uat: PortConnection connect upper_alarm_temp to ma.upper_alarm_temp;
		connection mm: PortConnection connect monitor_mode to ma.monitor_mode;

		connection ac: PortConnection connect ma.alarm_control to alarm_control;

		attribute Domain: CASE_Scheduling::Domain = 5;
	}

	part def Manage_Alarm_i :> Thread {
		attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
		attribute :>> Period = 1000 [millisecond];
		attribute Domain: CASE_Scheduling::Domain = 5;
		attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

		// ======== INPUTS =======
		// current temperature (from temp sensor)
		in port current_tempWstatus : DataPort { in :> type : Isolette_Data_Model::TempWstatus_i; }
		// lowest and upper bound of operator specified alarm temperature range
		in port lower_alarm_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
		in port upper_alarm_temp : DataPort { in :> type : Isolette_Data_Model::Temp_i; }
		// subsystem mode
		in port monitor_mode : DataPort { in :> type : Isolette_Data_Model::Monitor_Mode; }
		  
		// ======== OUTPUTS =======
		// command to turn alarm on/off (actuation command)  
		out port alarm_control : DataPort { out :> type : Isolette_Data_Model::On_Off; }

		language "GUMBO" /*{
		  state
		    lastCmd: Isolette_Data_Model::On_Off;

		  compute
		    compute_cases
		      case REQ_MA_1 "If mode INIT => alarm_control Off |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=115>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Init_Monitor_Mode;
		        guarantee alarm_control == Isolette_Data_Model::On_Off.Off;

		      case REQ_MA_2 "If mode NORMAL and temp out of alarm range => alarm_control On |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=115>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode &
		               (current_tempWstatus.degrees < lower_alarm_temp.degrees |
		                current_tempWstatus.degrees > upper_alarm_temp.degrees);
		        guarantee alarm_control == Isolette_Data_Model::On_Off.Onn;

		      case REQ_MA_3 "If mode NORMAL and temp in hysteresis band => no change |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=115>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode &
		               ((current_tempWstatus.degrees * 2 [s32] >= lower_alarm_temp.degrees * 2 [s32] &
		                 current_tempWstatus.degrees * 2 [s32] < lower_alarm_temp.degrees * 2 [s32] + 1 [s32]) |
		                (current_tempWstatus.degrees * 2 [s32] > upper_alarm_temp.degrees * 2 [s32] - 1 [s32] &
		                 current_tempWstatus.degrees * 2 [s32] <= upper_alarm_temp.degrees * 2 [s32]));
		        guarantee alarm_control == lastCmd;

		      case REQ_MA_4 "If mode NORMAL and temp in reset band => alarm_control Off |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=115>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Normal_Monitor_Mode &
		               (current_tempWstatus.degrees * 2 [s32] >= lower_alarm_temp.degrees * 2 [s32] + 1 [s32] &
		                current_tempWstatus.degrees * 2 [s32] <= upper_alarm_temp.degrees * 2 [s32] - 1 [s32]);
		        guarantee alarm_control == Isolette_Data_Model::On_Off.Off;

		      case REQ_MA_5 "If mode FAILED => alarm_control On |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=116>":
		        assume monitor_mode == Isolette_Data_Model::Monitor_Mode.Failed_Monitor_Mode;
		        guarantee alarm_control == Isolette_Data_Model::On_Off.Onn;

		      case REQ_MA_LATENCY "Alarm activation within 5 seconds |<http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=116>":
		        assume T;
		        guarantee true;
		}*/

	}

	// ===========================================================================================================
	//
	//   D e t e c t    M o n i t o r    T h r e a d
	//
	//  
	//    A.5.2.4 Detect Monitor Failure Function
	//    "The Detect Monitor Failure Function identifies internal failures, (e.g., a memory check failure)
	//     in the Monitor Temperature Function. It defines a single Boolean-valued internal variable,
	//     Monitor Internal Failure, which is set to True if an internal failure is detected. "
	// 
	//    http://pub.santoslab.org/high-assurance/module-requirements/reading/FAA-DoT-Requirements-AR-08-32.pdf#page=116
	// ===========================================================================================================	

	part def Detect_Monitor_Failure_Process_i :> Process {

		part dmf: Detect_Monitor_Failure_i;

		out port internal_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

		connection inf: PortConnection connect dmf.internal_failure to internal_failure;

		attribute Domain: CASE_Scheduling::Domain = 6;
	}

	part def Detect_Monitor_Failure_i :> Thread {

		attribute :>> Dispatch_Protocol = Supported_Dispatch_Protocols::Periodic;
		attribute :>> Period = 1000 [millisecond];
		attribute Domain: CASE_Scheduling::Domain = 6;
		attribute Microkit_Language: HAMR::Microkit_Language = HAMR::Microkit_Languages::Rust;

		out port internal_failure : DataPort { out :> type : Isolette_Data_Model::Failure_Flag_i; }

	}
}
