//
// Created by Frederic on 12/19/2023.
//
// System headers
#include <stdio.h>
#include <vector>
#include <string.h>
#include <iostream>
#include <iomanip>
//#include <format>
//Application headers
#include "../global.cuh"
#include "../include/Network.cuh"
#include "../include/Exception.cuh"
#include "../include/resmap_Sizes.cuh"
//#include "../include/Sockets.cuh"
//#include "../include/Adapters.cuh"

#if defined (WINDOWS)
//TODO: insert the headers here for the Windows environment
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <rpcdce.h>
#define MALLOC(x) HeapAlloc(GetProcessHeap(), 0, (x))
#define FREE(x) HeapFree(GetProcessHeap(), 0, (x))
#elif defined (LINUX)
//TODO: insert the headers here for the Linux environment
#endif
//////////////////////////////////////////////////////////////////////////////
// Namespace delaration
//////////////////////////////////////////////////////////////////////////////
namespace namespace_Network {
  ////////////////////////////////////////////////////////////////////////////////
  // Class Network_ClassTemplate_t mirrored valued type class definition
  ////////////////////////////////////////////////////////////////////////////////
  /*
  *
  * PS C:\Program Files (x86)\Nmap> ./nmap -sP 192.168.1.0/24|Select-String 'Nmap scan report for'
  *
  * Nmap scan report for 192.168.1.1
  * Nmap scan report for 192.168.1.59
  * Nmap scan report for 192.168.1.147
  * Nmap scan report for 192.168.1.155
  * Nmap scan report for 192.168.1.156
  * Nmap scan report for 192.168.1.164
  * Nmap scan report for 192.168.1.232
  * Nmap scan report for 192.168.1.248
  * Nmap scan report for 192.168.1.22
  * Nmap scan report for host.docker.internal (192.168.1.221)
  *
  * PS C:\Program Files (x86)\Nmap> ./nmap -sP 192.168.1.0/24|Select-String 'MAC Address:'
  *
  * MAC Address: 00:06:91:BF:04:2F (PT Inovacao)
  * MAC Address: 24:5E:BE:33:95:7B (Qnap Systems)
  * MAC Address: BC:A5:11:14:0E:32 (Netgear)
  * MAC Address: 60:35:C0:EE:BE:E4 (SFR)
  * MAC Address: 90:09:D0:35:A6:75 (Synology Incorporated)
  * MAC Address: 90:09:D0:35:A6:76 (Synology Incorporated)
  * MAC Address: 44:1C:A8:E2:AA:D3 (Hon Hai Precision Ind.)
  * MAC Address: 44:5C:E9:7F:A4:70 (Samsung Electronics)
  * MAC Address: 6C:0B:84:67:C4:BF (Universal Global Scientific Industrial)
  *
  * PS C:\Program Files (x86)\Nmap>
  *
  * FILE *pin = popen("nmap -p 123 10.0.1.0/24","r");
  * if ( pin ) {
  *    while (!feof(pin)) {
  *       const char *line = readLine(pin);
  *       printf("%s\n", line);
  *    }
  *    pclose(pin);
  * }
  *
  * Starting Nmap 7.94 ( https://nmap.org ) at 2024-02-11 14:19 W. Europe Standard Time
  * Initiating SYN Stealth Scan at 14:19
  * Scanning kubernetes.docker.internal (127.0.0.1) [1000 ports]
  * Discovered open port 135/tcp on 127.0.0.1
  * Discovered open port 445/tcp on 127.0.0.1
  * Discovered open port 3306/tcp on 127.0.0.1
  * Discovered open port 1123/tcp on 127.0.0.1
  * Discovered open port 5432/tcp on 127.0.0.1
  * Discovered open port 9090/tcp on 127.0.0.1
  * Discovered open port 1310/tcp on 127.0.0.1
  * Discovered open port 1521/tcp on 127.0.0.1
  * Discovered open port 5500/tcp on 127.0.0.1
  * Discovered open port 49152/tcp on 127.0.0.1
  * Completed SYN Stealth Scan at 14:19, 0.34s elapsed (1000 total ports)
  * Nmap scan report for kubernetes.docker.internal (127.0.0.1)
  * Host is up (0.0015s latency).
  * Not shown: 990 closed tcp ports (reset)
  * PORT      STATE SERVICE
  * 135/tcp   open  msrpc
  * 445/tcp   open  microsoft-ds
  * 1123/tcp  open  murray
  * 1310/tcp  open  husky
  * 1521/tcp  open  oracle
  * 3306/tcp  open  mysql
  * 5432/tcp  open  postgresql
  * 5500/tcp  open  hotline
  * 9090/tcp  open  zeus-admin
  * 49152/tcp open  unknown
  *
  * Read data files from: C:\Program Files (x86)\Nmap
  * Nmap done: 1 IP address (1 host up) scanned in 1.19 seconds
  *           Raw packets sent: 1000 (44.000KB) | Rcvd: 2032 (85.438KB)
  *
  ********************************************************************************
  */
  ////////////////////////////////////////////////////////////////////////////////
  /// Template class constructors
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T, typename S> int Network_ClassTemplate_t<T,S>::_initialize_t() {} /* end of Network_ClassTemplate_t<T,S>::_initialize_t() method */
    template <typename T, typename S> int Network_ClassTemplate_t<T,S>::_initialize_t(network_struct *s_network) {
        int rc = RC_SUCCESS;
      //TODO: initialize here the number of machines and the like to getthe number of machines and the like.
      //TODO: instantiate the template class Socket
      /* namespace_Network::Socket_ClassTemplate_t<float, std::string> *cSocketLayer_t =
                    new namespace_Network::Socket_ClassTemplate_t<float, std::string>();  */
        return rc;
    } /* end of Network_ClassTemplate_t<T,S>::_initialize_t(network_struct *s_network) method */
  template <typename T, typename S> int Network_ClassTemplate_t<T,S>::_initialize_t(
    machine_struct *s_machine,
    IPAddresses_struct *s_IPAddresses,
    adapters_struct *s_adapters,
    socket_struct *s_socket,
    network_struct *s_network) {
      int rc = RC_SUCCESS;
      rc = _initialize_t(s_network); if (rc != RC_SUCCESS){rc = RC_FAIL;}
      // Here we get the adapters and IPadresses of the the system by instantiating on the adapter template class
      cOverAdaptersLayer_t = new namespace_Network::Adapters_ClassTemplate_t<T,S>(s_IPAddresses, s_adapters);

      //TODO: initialize the other data structures Socket as well by calling the template classes
      //TODO: continue here
      return rc;
    } /* end of Network_ClassTemplate_t<T,S>::_initialize_t(machine_struct *s_machine,
    IPAddresses_struct *s_IPAddresses, adapters_struct *s_adapters, socket_struct *s_socket,
    network_struct *s_network) method */
  template <typename T, typename S> int Network_ClassTemplate_t<T,S>::_finalize_t() {int rc = RC_SUCCESS; return rc;}
  /* constructors Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t */
  template <typename T, typename S> Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t() {}
  template <typename T, typename S>
        Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t(network_struct *s_network) {
      int rc = RC_SUCCESS;
      rc = _initialize_t(s_network); if (rc != RC_SUCCESS){rc = RC_FAIL;}
    } /* end of Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t(network_struct *s_network) constructor */
  template <typename T, typename S>
  Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t(machine_struct *s_machine, IPAddresses_struct *s_IPAddresses,
    adapters_struct *s_adapters, socket_struct *s_socket, network_struct *s_network) {
      int rc = RC_SUCCESS;
      rc = _initialize_t(s_machine, s_IPAddresses, s_adapters, s_socket, s_network); if (rc != RC_SUCCESS){rc = RC_FAIL;}

      if (s_debug_gpu->debug_high_adapters_struct_i == 1) {
        cOverAdaptersLayer_t->print_adapters_data_structure_t(s_adapters); }

    rc = get_network_map_nmap(s_machine, s_adapters); if (rc != RC_SUCCESS) {rc = RC_FAIL;}
    rc = get_network_foreachIPs(s_machine, s_adapters, s_network); ; if (rc != RC_SUCCESS) {rc = RC_FAIL;}

  } /* end of Network_ClassTemplate_t(machine_struct *s_machine, ..., network_struct *s_network) constructor */
  ////////////////////////////////////////////////////////////////////////////////
  /// Template class getters
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T, typename S> int
  Network_ClassTemplate_t<T,S>::get_network_foreachIPs(machine_struct *s_mach, adapters_struct *s_adptrs,
    network_struct *s_network) {
    int rc = RC_SUCCESS;
    std::string ip;
    //std::cout<<"line 172 ---> "<<std::endl;
    std::cout<<B_BLUE<<"*---------- Network map screen (forAIP)*"<<std::endl;
    ip = s_adptrs->my_GatewayList.IpAddress.String;     //s_adptrs->my_DhcpServer.IpAddress.String;
    std::string cmd = "Some initial string in get_network_localIPs";
    std::string netmask = "/24";
    std::string delimiter = "IPs";
    char *scan;
    // Temporary stuff
    std::string result = "";
    char buffer[MAX_ADAPTER_NAME_LENGTH]; // MAX_ADAPTER_NAME_LENGTH         256 // arb.
    std::vector<std::string> v_temp;
     // the iterators and pointers for the vectors
    std::basic_string <char>:: pointer p_array_ip;
    std::basic_string <char>:: size_type nArray_ip;
    // index variables
    int i;
    /*
     * for each IP address coming from s_mach->array_machines_ipv4[@]
     * nmap -T4 -v s_mach->array_machines_ipv4[@]
     *
     */
#if defined (LINUX)
    //nmap -T4 -v 192.168.1.0/24 |grep "Nmap scan report"|grep -v "host down"
    cmd = "nmap -T4 -v "+ip+netmask+" |grep \"Nmap scan report\"|grep -v \"host down\"";
#elif defined (WINDOWS)
    //nmap -T4 -v 192.168.1.*/24 |Select-String "Nmap scan report"|Select-String -NotMatch "host down";
    //cmd = "nmap -T4 -v "+ip+netmask+" |Select-String \"Nmap scan report\"|Select-String -NotMatch \"host down\"";
    // nmap -T4 -v 192.168.1.*/24
    //cmd = "nmap -T4 -v "+ip+netmask+" | Select-String -NotMatch \"host down\"";
    //cmd = "nmap -T4 -v "+ip+netmask+" | grep -v \"host down\"";
    //cmd = "nmap -sP "+ip+netmask+" | grep \"Nmap scan report for\"";
    // nmap -V 192.168.1.0/24|grep "Platform"
    //cmd = "nmap -T4 -v "+ip+netmask+" | grep \""+delimiter+"\"";
    //    //&'C:\Program Files (x86)\Nmap\nmap.exe' -T4 -v 192.168.1.1/24
    cmd = "nmap -T4 -v "+ip;
    //cmd = "&\'C:\\Program Files (x86)\\Nmap\\nmap.exe\' -T4 -v "+ip;
#endif

    std::cout<<B_BLUE<<"Fonction                           ---> "<<B_MAGENTA<<__FUNCTION__<<std::endl;
    std::cout<<B_BLUE<<"initial string                     ---> "<<B_GREEN<<cmd<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->DhcpEnabled_ui           ---> "<<B_YELLOW<<s_adptrs->DhcpEnabled_ui<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->DhcpEnabled_char         ---> "<<B_YELLOW<<s_adptrs->DhcpEnabled_char<<std::endl;
    //std::cout<<B_BLUE<<"my_DhcpServer.IpAddress.String     ---> "<<B_YELLOW<<s_adptrs->my_DhcpServer.IpAddress.String<<std::endl;
    std::cout<<B_BLUE<<"my_GatewayList.IpAddress.String    ---> "<<B_YELLOW<<s_adptrs->my_GatewayList.IpAddress.String<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->LeaseObtained_char       ---> "<<B_YELLOW<<s_adptrs->LeaseObtained_char<<std::endl;
    std::cout<<B_BLUE<<"s_mach->nMachine_ipv4              ---> "<<B_YELLOW<<s_mach->nMachine_ipv4<<std::endl;
    std::cout<<B_BLUE<<"my_IpAddressList.IpAddress.String  ---> "<<B_YELLOW<<std::string(s_adptrs->my_IpAddressList.IpAddress.String)<<std::endl;

    s_network->p_v_network_scan_result = new std::vector<std::pair<std::string, std::string>>();
    for (i = 0 ; i < s_mach->nMachine_ipv4; i++ ) {
      cmd = "nmap -T4 -v "+std::string(s_mach->array_machines_ipv4[i]);

      std::cout
      <<B_BLUE<<"Machine ID["<<B_GREEN<<i<<B_BLUE<<"]                      ---> "
      <<B_YELLOW<<std::setw(MAX_IPV4_ADDRESS_LENGTH)<<s_mach->array_machines_ipv4[i]<<B_BLUE<<" ---> "
      <<B_MAGENTA<<cmd
      <<COLOR_RESET<<std::endl;

      cmd.push_back('\0');
      scan = (char*)malloc(cmd.size()*sizeof(char));
      p_array_ip = scan;
      nArray_ip = cmd.copy(p_array_ip, cmd.size(), 0); if (rc != RC_SUCCESS) {rc = RC_FAIL;}
      // Openinng the scan file in the process open signature file
      FILE *pin = _popen(scan, "r");
      if (!pin) throw std::runtime_error("_popen() failed");
      try {
        while (fgets(buffer, sizeof buffer, pin) != NULL) {
          result += buffer;
          v_temp.push_back(trim(buffer).c_str());
        }
      } catch (...) { _pclose(pin); throw;}
      _pclose(pin);

      s_network->
      p_v_network_scan_result->
      push_back(std::make_pair(std::string(s_mach->array_machines_ipv4[i]),result));
      if (s_debug_gpu->debug_high_s_network_struct_i == 1) {
        std::cout<<result<<std::endl;
        std::cout<<B_RED<<"  typeid(result).name()  "<<typeid(result).name()<<COLOR_RESET<<std::endl;
      }
      std::cout<<B_RED<<"  result.size()          "<<result.size()<<COLOR_RESET<<std::endl;
    }
    // Accessing elements in the vector
    for (const auto& entry : *s_network->p_v_network_scan_result) {
      std::cout << "Key: " << entry.first << ", Value: " << entry.second << std::endl;
    }

    // Don't forget to free the memory if you're done using the vector
    delete s_network->p_v_network_scan_result;

    return rc;
  } /* end of get_network_foreachIPs method */
  ////////////////////////////////////////////////////////////////////////////////
  /// Template class getters
  ////////////////////////////////////////////////////////////////////////////////
  /* getters for the nmap */
  template <typename T, typename S> int
  Network_ClassTemplate_t<T,S>::get_network_localIPs(machine_struct *s_mach,
    adapters_struct *s_adptrs, std::string delimiter_string_in) {
    int rc = RC_SUCCESS;
    std::string ip;
    //std::cout<<"line 294 ---> "<<std::endl;
    std::cout<<B_BLUE<<"*---------- Network map screen (locIP)-*"<<std::endl;
    ip = s_adptrs->my_GatewayList.IpAddress.String;  // s_adptrs->my_DhcpServer.IpAddress.String;
    ip = s_adptrs->my_DhcpServer.IpAddress.String;

    /*
    if (s_adptrs->DhcpEnabled_char == "No") {
      std::cout<<B_BLUE<<"my_DhcpServer.IpAddress.String     ---> "<<B_YELLOW<<s_adptrs->my_DhcpServer.IpAddress.String<<std::endl;
      ip = s_adptrs->my_GatewayList.IpAddress.String;  // s_adptrs->my_DhcpServer.IpAddress.String;
    } else if (s_adptrs->DhcpEnabled_char == "Yes") {
      ip = s_adptrs->my_DhcpServer.IpAddress.String;
      std::cout<<B_BLUE<<"my_GatewayList.IpAddress.String     ---> "<<B_YELLOW<<s_adptrs->my_GatewayList.IpAddress.String<<std::endl;
    }
    */
    std::string cmd = "Some initial string in get_network_localIPs";
    std::string netmask = "/24";
    std::string delimiter = "IPs";
    char *scan;
    // Temporary stuff
    std::string result = "";
    char buffer[MAX_ADAPTER_NAME_LENGTH]; // MAX_ADAPTER_NAME_LENGTH         256 // arb.
    std::vector<std::string> v_temp;
    std::vector<std::string> v_ith_temp;
    std::vector<std::string> v_nmap_ipv4_char;
    std::vector<std::string> v_nmap_macs_char;
    // the iterators and pointers for the vectors
    std::basic_string <char>:: pointer p_array_ip;
    std::basic_string <char>:: size_type nArray_ip;
    // local index variables
    int i, j;
    ///////////////////////////////////////////
    // Start of the execution commands
    ///////////////////////////////////////////
    if (delimiter_string_in == "IPs") {
      delimiter = "Nmap scan report for";
    } else if (delimiter_string_in == "MACs") {
      delimiter = "MAC Address: ";
    }
    // The command with delimiter in windows both grep and Select-String work
#if defined (LINUX)
    cmd = "nmap -sP "+ip+netmask+" |grep \"Nmap scan report\"|grep -v \"host down\"";
#elif defined (WINDOWS)
    //cmd = "nmap -sP "+ip+netmask+" | grep \""+delimiter+"\"";
    //&'C:\Program Files (x86)\Nmap\nmap.exe' -sP 192.168.1.1/24
    //cmd = "nmap -sP "+ip+netmask+" |Select-String -NotMatch \"host down\"";
    //cmd = "nmap -sP "+ip+netmask;
    cmd = "nmap -sP "+ip+netmask+" |grep -v \"host down\"";

#endif

    std::cout<<B_BLUE<<"Fonction                           ---> "<<B_MAGENTA<<__FUNCTION__<<COLOR_RESET<<std::endl;
    std::cout<<B_BLUE<<"initial string                     ---> "<<B_GREEN<<cmd<<COLOR_RESET<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->DhcpEnabled_ui           ---> "<<B_YELLOW<<s_adptrs->DhcpEnabled_ui<<COLOR_RESET<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->DhcpEnabled_char         ---> "<<B_YELLOW<<s_adptrs->DhcpEnabled_char<<COLOR_RESET<<std::endl;
    //std::cout<<B_BLUE<<"my_DhcpServer.IpAddress.String     ---> "<<B_YELLOW<<s_adptrs->my_DhcpServer.IpAddress.String<<std::endl;
    std::cout<<B_BLUE<<"s_adptrs->LeaseObtained_char       ---> "<<B_YELLOW<<s_adptrs->LeaseObtained_char<<COLOR_RESET<<std::endl;

    // allocating the memeory size on the string and mapping tehe string into scan */
    cmd.push_back('\0');
    scan = (char*)malloc(cmd.size()*sizeof(char));
    p_array_ip = scan;
    nArray_ip = cmd.copy(p_array_ip, cmd.size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}

    /* opening the outout file into the process open signature file */
    FILE *fp = _popen(scan, "r");

    if (!fp) throw std::runtime_error("_popen() failed!");
    try {
      while (fgets(buffer, sizeof buffer, fp) != NULL) {
        result += buffer;
        v_temp.push_back(trim(buffer).c_str());
      }
    } catch (...) { _pclose(fp); throw;}
    _pclose(fp);

    for (i = 0 ; i < v_temp.size() ; i++) {
      v_ith_temp = split(v_temp[i], delimiter);
      if (delimiter_string_in == "IPs") { v_nmap_ipv4_char.push_back(v_ith_temp[1]);}
      if (delimiter_string_in == "MACs") { v_nmap_macs_char.push_back(v_ith_temp[1]);}
    }
    // fillling the number of scanned IPs ibn the network
    if (delimiter_string_in == "IPs") { s_mach->nMachine_ipv4 = v_nmap_ipv4_char.size(); }
    if (delimiter_string_in == "MACs") { s_mach->nMachine_macs = v_nmap_macs_char.size(); }

    if (s_debug_gpu->debug_high_s_network_struct_i == 1 ) {
      for (i = 0 ; i < v_nmap_ipv4_char.size() ; i++) {
        std::cout<<v_nmap_ipv4_char[i]<<std::endl;
      }
      for (i = 0 ; i < v_nmap_macs_char.size() ; i++) {
        std::cout<<v_nmap_macs_char[i]<<std::endl;
      }
    }
    // Allocating the memory for the array and the vector mapping
    if (delimiter_string_in == "IPs") {
      s_mach->array_machines_ipv4 =
        (char**)malloc(s_mach->nMachine_ipv4*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      for (j = 0; j < v_nmap_ipv4_char.size(); j++) {
        if (s_debug_gpu->debug_high_s_network_struct_i == 1 ) {
          std::cout<<"v_nmap_ipv4_char["<<j<<"] ---> "<<v_nmap_ipv4_char[j]<<" ---> size ---> "<<v_nmap_ipv4_char[j].size()<<std::endl;
        }
        v_nmap_ipv4_char[j].push_back('\0');
        s_mach->array_machines_ipv4[j] = (char*)malloc(v_nmap_ipv4_char[j].length()*sizeof(char));
        p_array_ip = s_mach->array_machines_ipv4[j];
        nArray_ip = v_nmap_ipv4_char[j].copy(p_array_ip, v_nmap_ipv4_char[j].length(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
    } else if (delimiter_string_in == "MACs") {
      // Optional call but will not be used this way instead the IP will be used differently in a
      // subsequent call. The IP adddress array s_mach->array_machines_ipv4[@] to be exploited
      s_mach->array_machines_macs =
        (char**)malloc(s_mach->nMachine_macs*(MAX_ADAPTER_DESCRIPTION_LENGTH+4)*sizeof(char));
      for (j = 0; j < v_nmap_macs_char.size(); j++) {
        if (s_debug_gpu->debug_high_s_network_struct_i == 1 ) {
          std::cout<<"v_nmap_macs_char["<<j<<"] ---> "<<v_nmap_macs_char[j]<<" ---> size ---> "<<v_nmap_macs_char[j].size()<<std::endl;
        }
        v_nmap_macs_char[j].push_back('\0');
        s_mach->array_machines_macs[j] = (char*)malloc(v_nmap_macs_char[j].length()*sizeof(char));
        p_array_ip = s_mach->array_machines_macs[j];
        nArray_ip = v_nmap_macs_char[j].copy(p_array_ip, v_nmap_macs_char[j].length(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
    }
    // Printing some stuff back on screen
    if (delimiter_string_in == "IPs") {
      for (int i = 0; i < s_mach->nMachine_ipv4; i++ ) {
        std::cout
          <<B_BLUE<<"s_mach->array_machines_ipv4["<<B_GREEN<<i<<B_BLUE"]     ---> "
          <<B_YELLOW<<std::setw(MAX_IPV4_ADDRESS_LENGTH)<<s_mach->array_machines_ipv4[i]
          <<COLOR_RESET<<std::endl;
      }
      std::cout<<B_BLUE<<"s_mach->nMachine_ipv4              ---> "<<B_YELLOW<<s_mach->nMachine_ipv4<<COLOR_RESET<<std::endl;
    }
    if (delimiter_string_in == "MACs") {
      for (int i = 0; i < s_mach->nMachine_macs; i++ ) {
        std::cout
          <<B_BLUE<<"s_mach->array_machines_macs["<<B_GREEN<<i<<B_BLUE"]     ---> "
          <<B_YELLOW<<s_mach->array_machines_macs[i]
          <<COLOR_RESET<<std::endl;
      }
      std::cout<<B_BLUE<<"s_mach->nMachine_macs              ---> "<<B_YELLOW<<s_mach->nMachine_macs<<COLOR_RESET<<std::endl;
    }
    std::cout<<B_BLUE<<"*--------------------------------------*"<<std::endl;
    std::cout<<COLOR_RESET;

    // releasing the memory on string caharcter scan and other vectors
    free(scan);
    v_nmap_macs_char.clear();
    v_nmap_macs_char.clear();

    return rc;
  } /* end of get_network_localIPs(adapters_struct *s_adptrs) method */
  template <typename T, typename S> int
  Network_ClassTemplate_t<T,S>::get_network_map_nmap(machine_struct *s_mach, adapters_struct *s_adptrs) {
    int rc = RC_SUCCESS;
    /* starting  by getting the local IPs and filling the s_machine data_structure */
    rc = get_network_localIPs(s_mach, s_adptrs, "IPs"); if (rc != RC_SUCCESS){ rc = RC_FAIL;}
    //rc = get_network_localIPs(s_mach, s_adptrs, "MACs"); if (rc != RC_SUCCESS){ rc = RC_FAIL;}
    return rc;
  } /* end of get_network_map_nmap(adapters_struct *s_adptrs) method */
  template <typename T, typename S> int
  Network_ClassTemplate_t<T,S>::get_network_map_nmap() {
    int rc = RC_SUCCESS;
    return rc;
  } /* end of get_network_map_nmap method */
  ////////////////////////////////////////////////////////////////////////////////
  /// Helper methods
  ////////////////////////////////////////////////////////////////////////////////
  // for string delimiter
  template <typename T, typename S>  std::vector<std::string> Network_ClassTemplate_t<T,S>::split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
      token = s.substr (pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      res.push_back (token);
    }
    res.push_back (s.substr (pos_start));
    return res;
  } /* end of split method */
  template <typename T, typename S> std::string Network_ClassTemplate_t<T,S>::trim(const std::string &s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {start++;}
    auto end = s.end();
    do {end--;} while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
  } /* end trim method */
  ////////////////////////////////////////////////////////////////////////////////
  /// Template class destructors
  ////////////////////////////////////////////////////////////////////////////////
  /* destructor Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t */
  template <typename T, typename S> Network_ClassTemplate_t<T,S>::~Network_ClassTemplate_t() {}
  ////////////////////////////////////////////////////////////////////////////////
  // Class Network constructors overloaded methods
  ////////////////////////////////////////////////////////////////////////////////
  Network::Network() {
        int rc = RC_SUCCESS;
        namespace_Network::Network_ClassTemplate_t<float, std::string>
        *cBaseNetLayer_t = new namespace_Network::Network_ClassTemplate_t<float, std::string>();
        rc = _initialize();
        std::cout<<B_YELLOW<<"Class Network::Network() has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
  } /* end of Network::Network() constructor */
  Network::Network(network_struct *s_network) {
      int rc = RC_SUCCESS;
      namespace_Network::Network_ClassTemplate_t<float, std::string>
      *cOverNetLayer_t = new namespace_Network::Network_ClassTemplate_t<float, std::string>(s_network);
      rc = _initialize();
      std::cout<<B_CYAN<<"Class Network::Network(network_struct *s_network) has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
    } /* end of Network::Network(network_struct *s_network) constructor */
  Network::Network(machine_struct *s_machine, IPAddresses_struct *s_IPAddresses,
  adapters_struct *s_adapters, socket_struct *s_socket, network_struct *s_network) {
      int rc = RC_SUCCESS;
      namespace_Network::Network_ClassTemplate_t<float, std::string>
      *cOverNetLayer_t = new namespace_Network::Network_ClassTemplate_t<float, std::string>(s_machine, s_IPAddresses,
        s_adapters, s_socket, s_network);
      rc = _initialize();
      std::cout<<B_CYAN<<"Class Network::Network(machine_struct *s_machine,"
                         " ... , network_struct *s_network) has been instantiated,"
                         " return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
    } /* end of Network::Network(machine_struct *s_machine, .... , network_struct *s_network) constructor */
  ////////////////////////////////////////////////////////////////////////////////
  /// Class checkers
  ////////////////////////////////////////////////////////////////////////////////
  int Network::hello() {
    int rc = RC_SUCCESS;
    return rc;
  } /* end of hello checker method */
////////////////////////////////////////////////////////////////////////////////
// Initialiser
////////////////////////////////////////////////////////////////////////////////
  int Network::_initialize() {
    int rc = RC_SUCCESS;
    //rc = print_object_header_deepL(__FUNCTION__, __FILE__);
    return rc;
  } /* end of _initialize method */
////////////////////////////////////////////////////////////////////////////////
//Finaliser deaalicate sthe arrays and cleans up the environement
////////////////////////////////////////////////////////////////////////////////
  int Network::_finalize() {
    int rc = RC_SUCCESS;
    try{
      //TODO: need to free m_tinfo calloc
    } /* end of try */
    catch(Exception<std::invalid_argument> &e){std::cerr<<e.what()<<std::endl;}
    catch(...){
      std::cerr<<B_YELLOW
        "Program error! Unknown type of exception occured."<<std::endl;
      std::cerr<<B_RED"Aborting."<<std::endl;
      rc = RC_FAIL;
      return rc;
    }
    return rc;
  } /* end of _finalize method */
////////////////////////////////////////////////////////////////////////////////
//Destructor
////////////////////////////////////////////////////////////////////////////////
  Network::~Network() {
    int rc = RC_SUCCESS;
    //finalising the the method and remmmoving all of of the alocated arrays
    rc = _finalize();
    if (rc != RC_SUCCESS) {
      std::cerr<<B_RED"return code: "<<rc
               <<" line: "<<__LINE__<<" file: "<<__FILE__<<C_RESET<<std::endl;
      exit(rc);
    } else {rc = RC_SUCCESS; print_destructor_message("Network");}
    rc = get_returnCode(rc, "Network", 0);
  } /* end of ~deepL_FileHandler method */
  ////////////////////////////////////////////////////////////////////////////////
  // Methods that gets interfaced to extern C code for the API and the helper
  ////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
  extern "C" {
#endif


#ifdef __cplusplus
  }
#endif


  ////////////////////////////////////////////////////////////////////////////////
  // end of deepLEM namespace
  ////////////////////////////////////////////////////////////////////////////////
} /* End of namespace namespace_Network */
////////////////////////////////////////////////////////////////////////////////

