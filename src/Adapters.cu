//
// Created by Frederic on 12/23/2023.
//
// System headers
#include <vector>
#include <string.h>
#include <iostream>
//#include <format>
//Application headers
#include "../global.cuh"
#include "../include/Adapters.cuh"

#include <iomanip>

#include "../include/Sockets.cuh"
#include "../include/Exception.cuh"
#include "../include/resmap_Sizes.cuh"

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
  // Class Adapters_ClassTemplate_t mirrored valued type class definition
  ////////////////////////////////////////////////////////////////////////////////
  /*
  ********************************************************************************
  */
    template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::_initialize_t() {}
    template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::_initialize_t(adapters_struct *s_adapters, IPAddresses_struct *s_IPAddrs) {
      int rc = RC_SUCCESS;
      /*//TODO: instantiate the template class Socket namespace_Network::Socket_ClassTemplate_t<float, std::string> *cSocketLayer_t =
                 new namespace_Network::Socket_ClassTemplate_t<float, std::string>();  */
      /* making contact with the socket object using inheritance from the object declaration */
      rc = namespace_Network::Socket::get_sockets(); if (rc != RC_SUCCESS) {rc = RC_FAIL;}
      /* calling the adapeters struct filler method */
      rc = get_IPAddresses_Struct_t(s_IPAddrs); if (rc != RC_SUCCESS) {rc = RC_WARNING;}
      /* calling the adapeters struct filler method */
      std::cout<<B_BLUE<<"*--------------------------------------*"<<std::endl;
      std::cout<<B_BLUE<<"machineIPAddr                      ---> "<<B_YELLOW<<get_Adapters_Struct_IPAddr_t(s_adapters, s_IPAddrs)<<std::endl;
      std::cout<<B_BLUE<<"*--------------------------------------*"<<std::endl;
      return rc;
    }
    /* constructors Network_ClassTemplate_t<T,S>::Network_ClassTemplate_t */
    template <typename T, typename S> Adapters_ClassTemplate_t<T,S>::Adapters_ClassTemplate_t() {}
    template <typename T, typename S> Adapters_ClassTemplate_t<T,S>::Adapters_ClassTemplate_t(IPAddresses_struct *s_IPAddrs, adapters_struct *s_adapters) {
      int rc = RC_SUCCESS;
      std::cout<<B_B_RED<<"START IMPLEMENMTING HERE THE NETWORK CLASS (Adapters_ClassTemplate_t ....."<<COLOR_RESET<<std::endl;
      std::cout<<B_B_BLUE<<"s_adapters->nAdapters (@init)      ---> "<<B_B_MAGENTA<<s_adapters->nAdapters<<COLOR_RESET<<std::endl;
      /* calling the initilisation method */
      rc = _initialize_t(s_adapters, s_IPAddrs); if (rc != RC_SUCCESS) {rc = RC_FAIL;}
      /* printing the data structure */
      rc = print_IPAddresses_data_structure_t(s_IPAddrs);; if (rc != RC_SUCCESS) {rc = RC_WARNING;}
      rc = print_adapters_data_structure_t(s_adapters); if (rc != RC_SUCCESS) {rc = RC_WARNING;}
    } /* end of Network_ClassTemplate_t constructor */
  template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::get_IPAddresses_Struct_t(IPAddresses_struct *s_IPAddrs) {
      int rc = RC_SUCCESS;

      /* Variables used by GetIpAddrTable */
      PMIB_IPADDRTABLE p_IPAddrTable; //, p2_IPAddrTable;
      DWORD dw_Size = 0, dw_Size_tmp = 0;
      DWORD dw_RetVal = 0;
      IN_ADDR IPAddr;
      DWORD ifIndex;
      //Interface stuff
      PIP_INTERFACE_INFO p_interface_info;
      unsigned long outBuffLen_ul = 0;
      // Local variables
      std::string ip = "Some.IP.000.000";
      std::basic_string <char>:: pointer p_array_ip;
      std::basic_string <char>:: size_type nArray_ip;

      // start of the execution commands
      std::cout<<B_BLUE<<"dw_Size (@init)                    ---> "<<B_MAGENTA<<dw_Size<<COLOR_RESET<<std::endl;
      std::cout<<B_BLUE<<"dw_RetVal (@init)                  ---> "<<B_MAGENTA<<dw_RetVal<<COLOR_RESET<<std::endl;
      // Before calling AddIPAddress we use GetIpAddrTable to get
      // an adapter to which we can add the IP.
      p_IPAddrTable = (MIB_IPADDRTABLE *) MALLOC(sizeof (MIB_IPADDRTABLE));
      if (p_IPAddrTable == NULL) {
        printf("Error allocating memory needed to call GetIpAddrTable\n");
        exit (RC_FAIL);
      } else {
        dw_Size = 0;
        // Make an initial call to GetIpAddrTable to get the
        // necessary size into the dwSize variable
        if (GetIpAddrTable(p_IPAddrTable, &dw_Size, 0) == ERROR_INSUFFICIENT_BUFFER) {
          FREE(p_IPAddrTable);
          p_IPAddrTable = (MIB_IPADDRTABLE *) MALLOC(dw_Size);
        }
        if (p_IPAddrTable == NULL) {
          printf("Memory allocation failed for GetIpAddrTable\n");
          exit(RC_FAIL);
        }
      }
      dw_Size_tmp = dw_Size;
      std::cout<<B_BLUE<<"dw_Size (@after first call)        ---> "<<B_MAGENTA<<dw_Size<<COLOR_RESET<<std::endl;
      std::cout<<B_BLUE<<"dw_Size_tmp (@after first call)    ---> "<<B_MAGENTA<<dw_Size<<COLOR_RESET<<std::endl;
      std::cout<<B_BLUE<<"dw_RetVal (@after first call)      ---> "<<B_MAGENTA<<dw_RetVal<<COLOR_RESET<<std::endl;

      // Make a second call to GetIpAddrTable to get the
      // actual data we want
      if ((dw_RetVal = GetIpAddrTable(p_IPAddrTable, &dw_Size_tmp, 0)) == NO_ERROR) {
        s_IPAddrs->nIPs = p_IPAddrTable->dwNumEntries;

        ifIndex = p_IPAddrTable->table[0].dwIndex;
        s_IPAddrs->ithIPIndex = ifIndex;

        //IPAddr.S_un.S_addr = (u_long) p_IPAddrTable->table[0].dwMask;
        //s_IPAddrs->current_Mask_string = inet_ntoa(IPAddr);
        s_IPAddrs->current_Mask_ul = p_IPAddrTable->table[0].dwMask;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->current_Mask_ul,2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        p_array_ip = s_IPAddrs->current_Mask_string;
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        s_IPAddrs->current_BCastAddr_ul = p_IPAddrTable->table[0].dwBCastAddr;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->current_BCastAddr_ul,2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        p_array_ip = s_IPAddrs->current_BCastAddr_string;
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        s_IPAddrs->current_ReassemblySize_ul = p_IPAddrTable->table[0].dwReasmSize;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->current_ReassemblySize_ul,2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        p_array_ip = s_IPAddrs->current_ReassemblySize_string;
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        s_IPAddrs->current_unused1_us = p_IPAddrTable->table[0].unused1;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->current_unused1_us,2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        p_array_ip = s_IPAddrs->current_unused1_string;
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        s_IPAddrs->current_Type_us = p_IPAddrTable->table[0].wType;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->current_Type_us,2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        p_array_ip = s_IPAddrs->current_Type_string;
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        IPAddr.S_un.S_addr = (u_long) p_IPAddrTable->table[0].dwAddr;
        s_IPAddrs->current_ipv4_string = inet_ntoa(IPAddr);
        s_IPAddrs->current_ipv4_ul = p_IPAddrTable->table[0].dwAddr;

        if (s_debug_gpu->debug_high_IPAddresses_struct_i == 1) {
          printf("\n\tInterface Index                --->: \t%ld\n", s_IPAddrs->ithIPIndex);
          printf("\tSubnet Mask                    --->: \t%s (%lu%)\n", s_IPAddrs->current_Mask_string, s_IPAddrs->current_Mask_ul);
          printf("\tBroadCast Address              --->: \t%s (%lu%)\n", s_IPAddrs->current_BCastAddr_string, s_IPAddrs->current_BCastAddr_ul);
          printf("\tReassembly size                --->: \t%s (%lu%)\n", s_IPAddrs->current_ReassemblySize_string, s_IPAddrs->current_ReassemblySize_ul);
          printf("\tUnused1 Address                --->: \t%s (%lu%)\n", s_IPAddrs->current_unused1_string, s_IPAddrs->current_unused1_us);
          printf("\tType Address                   --->: \t%s (%lu%)\n", s_IPAddrs->current_Type_string, s_IPAddrs->current_Type_us);
          printf("\tIPv4 Address                   --->: \t%s (%lu%)\n", s_IPAddrs->current_ipv4_string, s_IPAddrs->current_ipv4_ul);
          printf("\n");
        }
      } else {
        printf("Call to GetIpAddrTable failed with error %d.\n", dw_RetVal);
        if (p_IPAddrTable) FREE(p_IPAddrTable);
        exit(RC_FAIL);
      }

      // allocating the array of ip address of size nIPs obtained from GetIpAddrTable
      s_IPAddrs->ip_address_ipv4_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_ipv4_ul = (unsigned long*)malloc(s_IPAddrs->nIPs*sizeof(unsigned long));

      s_IPAddrs->ip_address_mask_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_mask_ul = (unsigned long*)malloc(s_IPAddrs->nIPs*sizeof(unsigned long));

      s_IPAddrs->ip_address_BCastAddr_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_BCastAddr_ul = (unsigned long*)malloc(s_IPAddrs->nIPs*sizeof(unsigned long));

      s_IPAddrs->ip_address_ReassemblySize_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_ReassemblySize_ul = (unsigned long*)malloc(s_IPAddrs->nIPs*sizeof(unsigned long));

      s_IPAddrs->ip_address_unused1_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_unused1_us = (unsigned short*)malloc(s_IPAddrs->nIPs*sizeof(unsigned short));

      s_IPAddrs->ip_address_Type_array = (char**)malloc(s_IPAddrs->nIPs*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_IPAddrs->array_Type_us = (unsigned short*)malloc(s_IPAddrs->nIPs*sizeof(unsigned short));

      for (int i = 0; i < p_IPAddrTable->dwNumEntries; i++) {
        // Address IPv4
        s_IPAddrs->ip_address_ipv4_array[i] =  "123.345.567.890"; // initilisation just in case
        s_IPAddrs->array_ipv4_ul[i] = (u_long) p_IPAddrTable->table[i].dwAddr; //IPAddr.S_un.S_addr;
        //0001 0110 0000 0001 1010 1000 1100 0000
        //0000 0001 0011 1000 1010 1000 1100 0000
        //0000 0001 1001 0000 0001 0101 1010 1100
        //0000 0001 0000 0000 0000 0000 0111 1111
        //1101 1110 0000 0001 1010 1000 1100 0000
        // Converting decimals to IP addresses
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_ipv4_ul[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_ipv4_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_ipv4_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        // Mask array
        s_IPAddrs->ip_address_mask_array[i] =  "055.055.055.055"; // initilisation just in case
        s_IPAddrs->array_mask_ul[i] = (u_long) p_IPAddrTable->table[i].dwMask;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_mask_ul[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_mask_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_mask_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        // Broadcast array
        s_IPAddrs->ip_address_BCastAddr_array[i] = "1.2.3.4"; // initilisation just in case
        s_IPAddrs->array_BCastAddr_ul[i] = (u_long)p_IPAddrTable->table[i].dwBCastAddr;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_BCastAddr_ul[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_BCastAddr_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_BCastAddr_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        // Reassembly size
        s_IPAddrs->ip_address_ReassemblySize_array[i] = "155.155.0.0"; // initilisation just in case
        s_IPAddrs->array_ReassemblySize_ul[i] = (u_long)p_IPAddrTable->table[i].dwReasmSize;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_ReassemblySize_ul[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_ReassemblySize_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_ReassemblySize_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        // Unised 1
        s_IPAddrs->ip_address_unused1_array[i] = "1.2.3.4"; // initilisation just in case
        s_IPAddrs->array_unused1_us[i] = (u_short)p_IPAddrTable->table[i].unused1;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_unused1_us[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_unused1_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_unused1_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        // Type
        s_IPAddrs->ip_address_Type_array[i] = "5.5.5.5"; // initilisation just in case
        s_IPAddrs->array_Type_us[i] = (u_short)p_IPAddrTable->table[i].wType;
        ip = convertBinaryToASCII_IP_32bit(decimalToBinary_32bit(s_IPAddrs->array_Type_us[i],2));
        ip.push_back('\0'); // to get rud of the junk at the end of string what a pain that was!!
        //Allocating the neccessary memory for the array mapping
        s_IPAddrs->ip_address_Type_array[i] = (char*)malloc(ip.size()*sizeof(char));
        p_array_ip = s_IPAddrs->ip_address_Type_array[i];
        nArray_ip = ip.copy ( p_array_ip , ip.size(), 0 ); if (nArray_ip < 1 ){rc = RC_FAIL;}

        if (s_debug_gpu->debug_high_IPAddresses_struct_i == 1) {
          printf("\tIP Address     [%i]              --->: \t%s (%lu%)\n",i, s_IPAddrs->ip_address_ipv4_array[i], s_IPAddrs->array_ipv4_ul[i]);
          printf("\tSubnet Mask    [%i]              --->: \t%s (%lu%)\n",i, s_IPAddrs->ip_address_mask_array[i], s_IPAddrs->array_mask_ul[i]);
          printf("\tBCastAddr      [%i]              --->: \t%s (%lu%)\n",i, s_IPAddrs->ip_address_BCastAddr_array[i], s_IPAddrs->array_BCastAddr_ul[i]);
          printf("\tReassemblySize [%i]              --->: \t%s (%lu%)\n",i, s_IPAddrs->ip_address_ReassemblySize_array[i], s_IPAddrs->array_ReassemblySize_ul[i]);
          printf("\tunused1        [%i]              --->: \t%s (%u%)\n",i, s_IPAddrs->ip_address_unused1_array[i], s_IPAddrs->array_unused1_us[i]);
          printf("\tType           [%i]              --->: \t%s (%u%)\n",i, s_IPAddrs->ip_address_Type_array[i], s_IPAddrs->array_Type_us[i]);
        }
      }

      // now getting the adaptor info from a specific comboindex

      return rc;
    } /* end of get_IPAddresses_Struct_t(IPAddresses_struct *s_IPAdds) method */
  template <typename T, typename S> std::string Adapters_ClassTemplate_t<T,S>::decimalToBinary_32bit(unsigned long n, int base) {
      std::string binary = "";
      int rc = RC_SUCCESS;
      unsigned long arr[32]; int i = 0; // num = n;
      int npad = 0;
      std::string pad_string = "0";
      std::string::iterator it;

      while(n != 0) {
        arr[i] = n % base;
        i++;
        n = n / base;
      }
      for(i = i - 1; i >= 0;i--){ binary += std::to_string(arr[i]); }

      npad = 0;
      if (binary.size() < 32 ) {
        npad = 32 - binary.size();
        //std::cout<<std::endl<<" binary.size() < 32 ===> "<< binary.size()<< " so padding by ===> " << npad<<std::endl;
        binary.insert (0, npad, '0');
      } else if (binary.size() == 32) {
        npad = 0; // Redundant but I put it anyway in it cause I removed the comments
        //std::cout<<std::endl<<" binary.size() = 32 ===> "<< binary.size()<< " so padding by ===> " << npad<<std::endl;
      }
      return binary;
    } /* end of decimalToBinary_32bit method */
  template <typename T, typename S> std::string Adapters_ClassTemplate_t<T,S>::convertBinaryToASCII_IP_32bit(std::string binary) {
      std::string ip = "";
      std::vector<std::string> ret;
      std::vector<std::string> sub_ip;
      std::string full_ip = "etst";

      int k = 0, j = 0, l = 0;
      int splitlength = 8;
      int sum = 0;

      int numsubstrings = binary.length() / splitlength;
      //std::cout<<" binary          ---> "<<binary<<std::endl;
      //std::cout<<" binary.length() ---> "<<binary.length()<<std::endl;
      //std::cout<<" numsubstrings   ---> "<<numsubstrings<<std::endl;

      for (auto i = 0; i < numsubstrings; i++) {
        ret.push_back(binary.substr(i * splitlength, splitlength));
      }

      // if there are leftover characters, create a shorter item at the end.
      if (binary.length() % splitlength != 0) {
        ret.push_back(binary.substr(splitlength * numsubstrings));
      }

      for (auto i = 0 ; i < ret.size() ; i++ ) {
        //std::cout<<"binary           ---> " << ret[i]; //<<std::endl;

        k= splitlength - 1;;
        sum = 0;
        l = 0;
        for (j = ret[i].size() - 1 ; j >= 0 ; j--) {
          //std::cout<< " ||| ret["<<i<<"]"<<"["<<l<<"] --> "<<ret[i][l]<<" k = "<< k <<" ";
          if ( ret[i][l]  == '1' ) { sum += pow(2,k); }
          k--; l++;
        }
        //std::cout<< " ---- decimal ---> " << sum <<std::endl;
        sub_ip.push_back(std::to_string(sum));
      }
      //for (k = 0 ; k < sub_ip.size() ; k++) {std::cout<< " sub_ip["<<k<<"]: --> " << sub_ip[k]<< std::endl;}

      // now constructing the IP string name for each ip addresses
      // full IPv4 IP adress
      full_ip = sub_ip[3]+"."+sub_ip[2]+"."+sub_ip[1]+"."+sub_ip[0];

      return full_ip;
    } /* end of convertBinaryToASCII_IP_32bit method */
  template <typename T, typename S> std::string Adapters_ClassTemplate_t<T,S>::trim(const std::string &s) {
      auto start = s.begin();
      while (start != s.end() && std::isspace(*start)) {start++;}
      auto end = s.end();
      do {end--;} while (std::distance(start, end) > 0 && std::isspace(*end));
      return std::string(start, end + 1);
    } /* end trim method */
  template <typename T, typename S> std::string Adapters_ClassTemplate_t<T,S>::get_Adapters_Struct_IPAddr_t(adapters_struct *s_adapters, IPAddresses_struct *s_IPAddrs) {
      // initial maping of variables
      int rc = RC_SUCCESS;
      std::string outAdapterIPAddr = s_IPAddrs->current_ipv4_string;
      // Declaration of local vectors and intermediate vectors
      std::vector<unsigned long> v_ComboIndex;
      std::vector<std::string> v_AdapterName;
      std::vector<std::string> v_AdapterDesc;
      std::vector<std::string> v_AdapterIpAddr;

      std::vector<std::string> v_AdapterType;
      std::vector<std::string> v_AdapterMACAddress;

      std::vector<std::string> v_my_IpAddressList_ip;
      std::vector<std::string> v_my_IpAddressList_mask;
      std::vector<std::string> v_my_GatewayList_ip;
      std::vector<std::string> v_my_GatewayList_mask;
      // DHCP vectors
      std::vector<std::string> v_DhcpEnabled_char;
      std::vector<std::string> v_DhcpEnabled_server_ip;
      std::vector<std::string> v_DhcpEnabled_LeaseObtained_char;
      std::vector<std::string> v_DhcpEnabled_LeaseExpires_char;
      // index variables
      int j;
      // The data structures
      PIP_ADAPTER_INFO p_AdapterInfo;
      PIP_ADAPTER_INFO p_Adapter = NULL;
      DWORD dwRetVal = 0;
      UINT i;
      // --------------------------------------------------------
      // start of the execution commands
      // --------------------------------------------------------

      // variables used to print DHCP time info
      struct tm newtime;
      char buffer[32];
      errno_t error;
      std::string err_str = "Err: Exiting function!";

      ULONG outBufLen_ul = sizeof(IP_ADAPTER_INFO);
      p_AdapterInfo = (IP_ADAPTER_INFO*)MALLOC(sizeof(IP_ADAPTER_INFO));
      if (p_AdapterInfo == NULL) {
        printf("Error allocating memory needed to call GetAdaptersinfo\n");
        return err_str;
      }
      // Make an initial call to GetAdaptersInfo to get
      // the necessary size into the ulOutBufLen variable
      if (GetAdaptersInfo(p_AdapterInfo, &outBufLen_ul) == ERROR_BUFFER_OVERFLOW) {
        FREE(p_AdapterInfo);
        p_AdapterInfo = (IP_ADAPTER_INFO*)MALLOC(outBufLen_ul);
        if (p_AdapterInfo == NULL) {
          printf("Error allocating memory needed to call GetAdaptersinfo\n");
          return err_str;
        }
      }

      PIP_INTERFACE_INFO pInfo = NULL;
      ULONG ulOutBufLen_pinfo = 0;
      DWORD interfaceRetVal = GetInterfaceInfo(pInfo, &ulOutBufLen_pinfo);
      // the iterators and pointers for the vectors
      std::basic_string <char>:: pointer p_array_ip;
      std::basic_string <char>:: size_type nArray_ip;

      if ((dwRetVal = GetAdaptersInfo(p_AdapterInfo, &outBufLen_ul)) == NO_ERROR) {
        p_Adapter = p_AdapterInfo;
        int count_nadapters = 0;

        while (p_Adapter) {
          count_nadapters ++; //countting the number of adapters on the machine
          if (s_debug_gpu->debug_adapters_struct_i == 1) {
            printf("\tComboIndex  : \t%d\n", p_Adapter->ComboIndex);
            printf("\tAdapter Name: \t%s\n", p_Adapter->AdapterName);
            printf("\tAdapter Desc: \t%s\n", p_Adapter->Description);
          }
          v_ComboIndex.push_back(p_Adapter->ComboIndex);
          v_AdapterName.push_back(p_Adapter->AdapterName);
          v_AdapterDesc.push_back(p_Adapter->Description);
          if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
            strcpy(s_IPAddrs->current_adapter_name_uuid ,p_Adapter->AdapterName);
            s_IPAddrs->current_adapter_name_Desc = p_Adapter->Description;
            s_adapters->adapter_name_raw = p_Adapter->Description;
            strcpy(s_adapters->adapter_name_uuid, p_Adapter->AdapterName);
            s_adapters->adapter_index = p_Adapter->Index;
            s_adapters->ComboIndex =  p_Adapter->ComboIndex;
            //IP stuff
            s_adapters->my_IpAddressList.IpAddress.String = (char*)malloc(16*sizeof(char));
            s_adapters->my_IpAddressList.IpMask.String = (char*)malloc(16*sizeof(char));
            strcpy(s_adapters->my_IpAddressList.IpAddress.String, p_Adapter->IpAddressList.IpAddress.String);
            strcpy(s_adapters->my_IpAddressList.IpMask.String, p_Adapter->IpAddressList.IpMask.String);
            // Gateway stuff
            s_adapters->my_GatewayList.IpAddress.String = (char*)malloc(16*sizeof(char));
            s_adapters->my_GatewayList.IpMask.String = (char*)malloc(16*sizeof(char));
            strcpy(s_adapters->my_GatewayList.IpAddress.String, p_Adapter->GatewayList.IpAddress.String);
            strcpy(s_adapters->my_GatewayList.IpMask.String, p_Adapter->GatewayList.IpMask.String);
          }
          if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\tAdapter Addr: \t"); }
          // Getting the Mac addresses
          std::stringstream ss;
          for (i = 0; i < p_Adapter->AddressLength; i++) {
            if (i == (p_Adapter->AddressLength - 1)) {
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("%.2X\n", (int)p_Adapter->Address[i]); }
              ss<<std::hex << (int)p_Adapter->Address[i];
            }
            else {
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("%.2X-", (int)p_Adapter->Address[i]); }
              ss<<std::hex <<(int)p_Adapter->Address[i]<<"-";
            }
          }
          if (s_debug_gpu->debug_adapters_struct_i == 1) {if (ss.str().empty()){printf("\n");}}
          if (ss.str().empty()){ss<<"-";}
          v_AdapterMACAddress.push_back(ss.str());

          // Now getting the Type of adapaters
          if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\tType        : \t");}
          switch (p_Adapter->Type) {
            case MIB_IF_TYPE_OTHER:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Other %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Other");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {s_adapters->Type_ui = p_Adapter->Type;  s_adapters->Type_char = "Other"; }
            break;
            case MIB_IF_TYPE_ETHERNET:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Ethernet %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Ethernet");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "Ethernet";
            }
            break;
            case MIB_IF_TYPE_TOKENRING:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Token Ring %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Token Ring");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "Token Ring";
            }
            break;
            case MIB_IF_TYPE_FDDI:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("FDDI %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("FDDI");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "FDDI";
            }
            break;
            case MIB_IF_TYPE_PPP:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("PPP %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("PPP");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "PPP";
            }
            break;
            case MIB_IF_TYPE_LOOPBACK:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Lookback %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Lookback");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "Lookback";
            }
            break;
            case MIB_IF_TYPE_SLIP:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Slip %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Slip");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "Slip";
            }
            break;
            default:
              if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("Unknown type %ld\n", p_Adapter->Type);}
            v_AdapterType.push_back("Unknown type "+std::to_string((p_Adapter->Type)));
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->Type_ui = p_Adapter->Type; s_adapters->Type_char = "Unknown type";
            }
            break;
          }

          // TODO: implement the vector to fill in the data striucture from the data strcuture
          v_my_IpAddressList_ip.push_back(p_Adapter->IpAddressList.IpAddress.String);
          v_my_IpAddressList_mask.push_back(p_Adapter->IpAddressList.IpMask.String);
          v_my_GatewayList_ip.push_back(p_Adapter->GatewayList.IpAddress.String);
          v_my_GatewayList_mask.push_back(p_Adapter->GatewayList.IpMask.String);
	  
          if (s_debug_gpu->debug_high_adapters_struct_i == 1) {
            printf("\tCombo index : \t%d IP Address : \t%s\n", p_Adapter->ComboIndex, p_Adapter->IpAddressList.IpAddress.String);
            printf("\tIP Mask     : \t%s\n", p_Adapter->IpAddressList.IpMask.String);
            printf("\tGateway IP  : \t%s\n", p_Adapter->GatewayList.IpAddress.String);
            printf("\tGateway MSK : \t%s\n", p_Adapter->GatewayList.IpMask.String);
            printf("\t***\n");
          }


          //std::cout<<"line 540 ---> "<<std::endl;

          if (p_Adapter->DhcpEnabled) {
            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\tDHCP Enabled: Yes\n");}

            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->DhcpEnabled_ui = 1;
              s_adapters->DhcpEnabled_char = "Yes";
            }
            v_DhcpEnabled_char.push_back("Yes");

            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\t  DHCP Server: \t%s\n", p_Adapter->DhcpServer.IpAddress.String);}

            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->my_DhcpServer.IpAddress.String = (char*)malloc(MAX_IPV4_ADDRESS_LENGTH*sizeof(char));
              strcpy(s_adapters->my_DhcpServer.IpAddress.String, p_Adapter->DhcpServer.IpAddress.String);
            }
            if ( strlen(p_Adapter->DhcpServer.IpAddress.String) == 0 ) {
              v_DhcpEnabled_server_ip.push_back("-");
            } else {v_DhcpEnabled_server_ip.push_back(p_Adapter->DhcpServer.IpAddress.String);}
            // Getting the DHCP leases information
            // DHCP leasing obtained details defaults on Thu Jan  1 01:00:00 1970
            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\t  Lease Obtained: ");}
            error = _localtime32_s(&newtime, (__time32_t*)&p_Adapter->LeaseObtained);
            if (error) {
              printf("Invalid Argument to _localtime32_s\n");
            } else {
              // Convert to an ASCII representation
              error = asctime_s(buffer, 32, &newtime);
              if (error) {
                printf("Invalid Argument to asctime_s\n");
              } else {
                // asctime_s returns the string terminated by \n\0
                if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("%s", buffer);}
                v_DhcpEnabled_LeaseObtained_char.push_back(trim(buffer).c_str());
                if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
                  s_adapters->LeaseObtained_char = (char*)malloc((DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
                  strcpy(s_adapters->LeaseObtained_char, trim(buffer).c_str());
                }
              }
            }
            // DCHP leasing expiration details
            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\t  Lease Expires :  ");}
            error = _localtime32_s(&newtime, (__time32_t*)&p_Adapter->LeaseExpires);
            if (error) {
              printf("Invalid Argument to _localtime32_s\n");
            } else {
              // Convert to an ASCII representation
              error = asctime_s(buffer, 32, &newtime);
              if (error) {
                printf("Invalid Argument to asctime_s\n");
              } else {
                // asctime_s returns the string terminated by \n\0
                //std::cout<<"line 591 ---> "<<std::endl;
                if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("%s", buffer);}
                v_DhcpEnabled_LeaseExpires_char.push_back(trim(buffer).c_str());
                if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
                  s_adapters->LeaseExpires_char = (char*)malloc((DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
                  strcpy(s_adapters->LeaseExpires_char, trim(buffer).c_str());
                }
              }
            }
          } else {
            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\tDHCP Enabled: No\n");}
            v_DhcpEnabled_char.push_back("No");
            v_DhcpEnabled_server_ip.push_back("-");
            v_DhcpEnabled_LeaseObtained_char.push_back("-");
            v_DhcpEnabled_LeaseExpires_char.push_back("-");
            if (p_Adapter->ComboIndex == s_IPAddrs->ithIPIndex) {
              s_adapters->DhcpEnabled_ui = 0;
              s_adapters->DhcpEnabled_char = "No";
            }
          }

          if (p_Adapter->HaveWins) {
            if (s_debug_gpu->debug_adapters_struct_i == 1) {
              printf("\tHave Wins   : Yes\n");
              printf("\t  Primary Wins Server:    %s\n", p_Adapter->PrimaryWinsServer.IpAddress.String);
              printf("\t  Secondary Wins Server:  %s\n", p_Adapter->SecondaryWinsServer.IpAddress.String);
            }
          } else {
            if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\tHave Wins   : No\n");}
          }
          p_Adapter = p_Adapter->Next;
          if (s_debug_gpu->debug_adapters_struct_i == 1) {printf("\n");}
        } /* end of while loop while (p_Adapter) */
        s_adapters->nAdapters = count_nadapters;
      } else { printf("GetAdaptersInfo failed with error: %d\n", dwRetVal); } /* end of if ((dwRetVal = GetAdaptersInfo(p_AdapterInfo, &ulOutBufLen_pinfo)) if block */

      // Display the data
      //std::cout<<"size of v_ComboIndex vector ---> "<<v_ComboIndex.size()<<std::endl;
      s_adapters->array_ComboIndex_ul = (unsigned long*)malloc(s_adapters->nAdapters*sizeof(unsigned long));
      s_adapters->array_AdapterName = (char**)malloc(s_adapters->nAdapters*(MAX_ADAPTER_NAME_LENGTH+4)*sizeof(char));
      s_adapters->array_AdapterDesc = (char**)malloc(s_adapters->nAdapters*(MAX_ADAPTER_DESCRIPTION_LENGTH+4)*sizeof(char));
      s_adapters->array_MacAddress = (char**)malloc(s_adapters->nAdapters*(MAX_ADAPTER_DESCRIPTION_LENGTH+4)*sizeof(char));
      s_adapters->array_Type = (char**)malloc(s_adapters->nAdapters*(MAX_ADAPTER_DESCRIPTION_LENGTH+4)*sizeof(char));
      s_adapters->array_IpAddressList_ip = (char**)malloc(s_adapters->nAdapters*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      s_adapters->array_IpAddressList_mask = (char**)malloc(s_adapters->nAdapters*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      s_adapters->array_GatewayList_ip = (char**)malloc(s_adapters->nAdapters*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      s_adapters->array_GatewayList_mask = (char**)malloc(s_adapters->nAdapters*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      s_adapters->array_DhcpEnabled_char = (char**)malloc(s_adapters->nAdapters*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_adapters->array_DhcpEnabled_server_ip = (char**)malloc(s_adapters->nAdapters*(MAX_IPV4_ADDRESS_LENGTH+4)*sizeof(char));
      s_adapters->array_DhcpEnabled_LeaseObtained_char = (char**)malloc(s_adapters->nAdapters*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));
      s_adapters->array_DhcpEnabled_LeaseExpires_char = (char**)malloc(s_adapters->nAdapters*(DEFAULT_MINIMUM_ENTITIES+4)*sizeof(char));

      for (j = 0; j < v_ComboIndex.size(); j++ ) {
        //std::cout<<"v_ComboIndex["<<j<<"] ---> "<<v_ComboIndex[j]<<std::endl;
        s_adapters->array_ComboIndex_ul[j] = v_ComboIndex[j];
      }
      for (j = 0; j < v_AdapterName.size(); j++) {
        //std::cout<<"v_AdapterName["<<j<<"] ---> "<<v_AdapterName[j]<<" ---> size ---> "<<v_AdapterName[j].size()<<std::endl;
        v_AdapterName[j].push_back('\0');
        s_adapters->array_AdapterName[j] = (char*)malloc(v_AdapterName[j].length()*sizeof(char));
        p_array_ip = s_adapters->array_AdapterName[j];
        nArray_ip = v_AdapterName[j].copy(p_array_ip, v_AdapterName[j].length(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_AdapterDesc.size(); j++) {
        //std::cout<<"v_adapterDesc["<<j<<"] ---> "<<v_AdapterDesc[j]<<" ---> size ---> "<<v_AdapterDesc[j].size()<<std::endl;
        v_AdapterDesc[j].push_back('\0');
        s_adapters->array_AdapterDesc[j] = (char*)malloc(v_AdapterDesc[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_AdapterDesc[j];
        nArray_ip = v_AdapterDesc[j].copy(p_array_ip, v_AdapterDesc[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_AdapterType.size(); j++) {
        //std::cout<<"v_adapterType["<<j<<"] ---> "<<v_AdapterType[j]<<" ---> size ---> "<<v_AdapterType[j].size()<<std::endl;
        v_AdapterType[j].push_back('\0');
        s_adapters->array_Type[j] = (char*)malloc(v_AdapterType[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_Type[j];
        nArray_ip = v_AdapterType[j].copy(p_array_ip, v_AdapterType[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_AdapterMACAddress.size(); j++) {
        //std::cout<<"v_AdapterMACAddress["<<j<<"] ---> "<<v_AdapterMACAddress[j]<<" ---> size ---> "<<v_AdapterMACAddress[j].size()<<std::endl;
        v_AdapterMACAddress[j].push_back('\0');
        s_adapters->array_MacAddress[j] = (char*)malloc(v_AdapterMACAddress[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_MacAddress[j];
        nArray_ip = v_AdapterMACAddress[j].copy(p_array_ip, v_AdapterMACAddress[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_my_IpAddressList_ip.size(); j++) {
        //std::cout<<"v_my_IpAddressList_ip["<<j<<"] ---> "<<v_my_IpAddressList_ip[j]<<" ---> size ---> "<<v_my_IpAddressList_ip[j].size()<<std::endl;
        v_my_IpAddressList_ip[j].push_back('\0');
        s_adapters->array_IpAddressList_ip[j] = (char*)malloc(v_my_IpAddressList_ip[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_IpAddressList_ip[j];
        nArray_ip = v_my_IpAddressList_ip[j].copy(p_array_ip, v_my_IpAddressList_ip[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_my_IpAddressList_mask.size(); j++) {
        //std::cout<<"v_my_IpAddressList_mask["<<j<<"] ---> "<<v_my_IpAddressList_mask[j]<<" ---> size ---> "<<v_my_IpAddressList_mask[j].size()<<std::endl;
        v_my_IpAddressList_mask[j].push_back('\0');
        s_adapters->array_IpAddressList_mask[j] = (char*)malloc(v_my_IpAddressList_mask[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_IpAddressList_mask[j];
        nArray_ip = v_my_IpAddressList_mask[j].copy(p_array_ip, v_my_IpAddressList_mask[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_my_GatewayList_ip.size(); j++) {
        //std::cout<<"v_my_GatewayList_ip["<<j<<"] ---> "<<v_my_GatewayList_ip[j]<<" ---> size ---> "<<v_my_GatewayList_ip[j].size()<<std::endl;
        v_my_GatewayList_ip[j].push_back('\0');
        s_adapters->array_GatewayList_ip[j] = (char*)malloc(v_my_GatewayList_ip[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_GatewayList_ip[j];
        nArray_ip = v_my_GatewayList_ip[j].copy(p_array_ip, v_my_GatewayList_ip[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_my_GatewayList_mask.size(); j++) {
        //std::cout<<"v_my_GatewayList_mask["<<j<<"] ---> "<<v_my_GatewayList_mask[j]<<" ---> size ---> "<<v_my_GatewayList_mask[j].size()<<std::endl;
        v_my_GatewayList_mask[j].push_back('\0');
        s_adapters->array_GatewayList_mask[j] = (char*)malloc(v_my_GatewayList_mask[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_GatewayList_mask[j];
        nArray_ip = v_my_GatewayList_mask[j].copy(p_array_ip, v_my_GatewayList_mask[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_DhcpEnabled_char.size(); j++) {
        //std::cout<<"v_DhcpEnabled_char["<<j<<"] ---> "<<v_DhcpEnabled_char[j]<<" ---> size ---> "<<v_DhcpEnabled_char[j].size()<<std::endl;
        v_DhcpEnabled_char[j].push_back('\0');
        s_adapters->array_DhcpEnabled_char[j] = (char*)malloc(v_DhcpEnabled_char[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_DhcpEnabled_char[j];
        nArray_ip = v_DhcpEnabled_char[j].copy(p_array_ip, v_DhcpEnabled_char[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_DhcpEnabled_server_ip.size(); j++) {
        //std::cout<<"v_DhcpEnabled_server_ip["<<j<<"] ---> "<<v_DhcpEnabled_server_ip[j]<<" ---> size ---> "<<v_DhcpEnabled_server_ip[j].size()<<std::endl;
        v_DhcpEnabled_server_ip[j].push_back('\0');
        s_adapters->array_DhcpEnabled_server_ip[j] = (char*)malloc(v_DhcpEnabled_server_ip[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_DhcpEnabled_server_ip[j];
        nArray_ip = v_DhcpEnabled_server_ip[j].copy(p_array_ip, v_DhcpEnabled_server_ip[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_DhcpEnabled_LeaseObtained_char.size(); j++) {
        //std::cout<<"v_DhcpEnabled_LeaseObtained_char["<<j<<"] ---> "<<v_DhcpEnabled_LeaseObtained_char[j]<<" ---> size ---> "<<v_DhcpEnabled_LeaseObtained_char[j].size()<<std::endl;
        v_DhcpEnabled_LeaseObtained_char[j].push_back('\0');
        s_adapters->array_DhcpEnabled_LeaseObtained_char[j] = (char*)malloc(v_DhcpEnabled_LeaseObtained_char[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_DhcpEnabled_LeaseObtained_char[j];
        nArray_ip = v_DhcpEnabled_LeaseObtained_char[j].copy(p_array_ip, v_DhcpEnabled_LeaseObtained_char[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }
      for (j = 0; j < v_DhcpEnabled_LeaseExpires_char.size(); j++) {
        //std::cout<<"v_DhcpEnabled_LeaseExpires_char["<<j<<"] ---> "<<v_DhcpEnabled_LeaseExpires_char[j]<<" ---> size ---> "<<v_DhcpEnabled_LeaseExpires_char[j].size()<<std::endl;
        v_DhcpEnabled_LeaseExpires_char[j].push_back('\0');
        s_adapters->array_DhcpEnabled_LeaseExpires_char[j] = (char*)malloc(v_DhcpEnabled_LeaseExpires_char[j].size()*sizeof(char));
        p_array_ip = s_adapters->array_DhcpEnabled_LeaseExpires_char[j];
        nArray_ip = v_DhcpEnabled_LeaseExpires_char[j].copy(p_array_ip, v_DhcpEnabled_LeaseExpires_char[j].size(), 0); if (nArray_ip < 1 ){rc = RC_FAIL;}
      }

      // releasing the pointer
      if (p_AdapterInfo) FREE(p_AdapterInfo);
      if (p_Adapter) FREE(p_Adapter);

      return outAdapterIPAddr;
  } /* end of getMachineIPAddr_t(adapters_struct *s_adapters, IPAddresses_struct *s_IPAdds) method */
  template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::print_IPAddresses_data_structure_t(IPAddresses_struct *s_IPAddrs) {
    int rc = RC_SUCCESS;
    std::cout<<B_BLUE<<"*------- IPAddresses struct -----------*"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->nIPs                     ---> "<<B_YELLOW<<s_IPAddrs->nIPs<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->ithIPIndex               ---> "<<B_YELLOW<<s_IPAddrs->ithIPIndex<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_ipv4_string      ---> "<<B_YELLOW<<s_IPAddrs->current_ipv4_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_ipv4_ul<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_ipv6_string      ---> "<<B_YELLOW<<s_IPAddrs->current_ipv6_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_ipv6_ul<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_Mask_string      ---> "<<B_YELLOW<<s_IPAddrs->current_Mask_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_Mask_ul<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_BCastAddr_string ---> "<<B_YELLOW<<s_IPAddrs->current_BCastAddr_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_BCastAddr_ul<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_RessBlySize_str  ---> "<<B_YELLOW<<s_IPAddrs->current_ReassemblySize_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_ReassemblySize_ul<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_unused1_string   ---> "<<B_YELLOW<<s_IPAddrs->current_unused1_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_unused1_us<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_Type_string      ---> "<<B_YELLOW<<s_IPAddrs->current_Type_string<<B_BLUE" ("<<B_MAGENTA<<s_IPAddrs->current_Type_us<<B_BLUE")"<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_adapter_name_uuid---> "<<B_YELLOW<<s_IPAddrs->current_adapter_name_uuid<<std::endl;
    std::cout<<B_BLUE<<"s_IPAdds->current_adapter_name_Desc---> "<<B_YELLOW<<s_IPAddrs->current_adapter_name_Desc<<std::endl;

      for (int i = 0; i < s_IPAddrs->nIPs; i++ ) {
        std::cout
          <<B_BLUE<<"s_IPAddrs->ip_address_ipv4_array["<<B_GREEN<<i<<B_BLUE"]---> "
          <<B_YELLOW<<std::setw(MAX_IPV4_ADDRESS_LENGTH)<<s_IPAddrs->ip_address_ipv4_array[i]
          <<B_BLUE<<" ("               <<B_MAGENTA<<std::setw(12)<<s_IPAddrs->array_ipv4_ul[i]<<B_BLUE")"
          <<B_BLUE<<" Mask: "          <<B_GREEN  <<std::setw(MAX_IPV4_ADDRESS_LENGTH)<<s_IPAddrs->ip_address_mask_array[i]
          <<B_BLUE<<" BCastAddr: "     <<B_CYAN   <<s_IPAddrs->ip_address_BCastAddr_array[i]
          <<B_BLUE<<" ReassemblySize: "<<B_B_RED  <<s_IPAddrs->ip_address_ReassemblySize_array[i]
          <<B_BLUE<<" unused1: "       <<B_YELLOW <<s_IPAddrs->ip_address_unused1_array[i]
          <<B_BLUE<<" Type: "          <<B_MAGENTA<<s_IPAddrs->ip_address_Type_array[i]
          <<COLOR_RESET<<std::endl;
      }
      //std::cout<<B_BLUE<<"s_adapters->adapter_name_uuid      ---> "<<B_YELLOW<<s_adapters->adapter_name_uuid<<std::endl;
    //std::cout<<B_BLUE<<"s_adapters->adapter_index          ---> "<<B_YELLOW<<s_adapters->adapter_index<<std::endl;
    //std::cout<<B_BLUE<<"devices->ncuda_cores               ---> "<<B_YELLOW<<*(devices->ncuda_cores)<<std::endl;
    std::cout<<B_BLUE<<"*--------------------------------------*"<<std::endl;
    std::cout<<COLOR_RESET;
    
    return rc;
  } /* end of print_IPAddresses_data_structure_t(IPAddresses_struct *s_IPAdds) constructor */
  template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::print_adapters_data_structure_t(adapters_struct *s_adapters) {
    int rc = RC_SUCCESS;
    int i;
    std::cout<<B_BLUE<<"*------- adapters struct --------------*"<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->nAdapters              ---> "<<B_YELLOW<<s_adapters->nAdapters<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->adapter_name_raw       ---> "<<B_YELLOW<<s_adapters->adapter_name_raw<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->adapter_name_uuid      ---> "<<B_YELLOW<<s_adapters->adapter_name_uuid<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->adapter_index          ---> "<<B_YELLOW<<s_adapters->adapter_index<<std::endl;
    //std::cout<<B_BLUE<<"devices->ncuda_cores               ---> "<<B_YELLOW<<*(devices->ncuda_cores)<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->ComboIndex             ---> "<<B_YELLOW<<s_adapters->ComboIndex<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->AddressLength          ---> "<<B_YELLOW<<s_adapters->AddressLength<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->Index                  ---> "<<B_YELLOW<<s_adapters->Index<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->Type_ui                ---> "<<B_YELLOW<<s_adapters->Type_ui<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->Type_char              ---> "<<B_YELLOW<<s_adapters->Type_char<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->DhcpEnabled_ui         ---> "<<B_YELLOW<<s_adapters->DhcpEnabled_ui<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->DhcpEnabled_char       ---> "<<B_YELLOW<<s_adapters->DhcpEnabled_char<<std::endl;
    //std::cout<<"line 798 ---> "<<std::endl;
      if (s_adapters->DhcpEnabled_char == "No") {
        std::cout<<B_BLUE<<"my_DhcpServer.IpAddress.String     ---> "<<B_YELLOW<<s_adapters->my_DhcpServer.IpAddress.String<<std::endl;
      } else if (s_adapters->DhcpEnabled_char == "Yes") {
        std::cout<<B_BLUE<<"my_GatewayList.IpAddress.String     ---> "<<B_YELLOW<<s_adapters->my_GatewayList.IpAddress.String<<std::endl;
      }
      //std::cout<<"line 800 ---> "<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->LeaseObtained_char     ---> "<<B_YELLOW<<s_adapters->LeaseObtained_char<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->LeaseExpires_char      ---> "<<B_YELLOW<<s_adapters->LeaseExpires_char<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->HaveWins_int           ---> "<<B_YELLOW<<s_adapters->HaveWins_int<<std::endl;
    std::cout<<B_BLUE<<"s_adapters->HaveWins_char          ---> "<<B_YELLOW<<s_adapters->HaveWins_char<<std::endl;
    std::cout<<B_BLUE<<"my_IpAddressList.IpAddress.String  ---> "<<B_YELLOW<<s_adapters->my_IpAddressList.IpAddress.String<<std::endl;
    std::cout<<B_BLUE<<"my_IpAddressList.IpMask.String     ---> "<<B_YELLOW<<s_adapters->my_IpAddressList.IpMask.String<<std::endl;
    std::cout<<B_BLUE<<"my_GatewayList.IpAddress.String    ---> "<<B_YELLOW<<s_adapters->my_GatewayList.IpAddress.String<<std::endl;
    std::cout<<B_BLUE<<"my_GatewayList.IpMask.String       ---> "<<B_YELLOW<<s_adapters->my_GatewayList.IpMask.String<<std::endl;

    //std::cout<<"line 811 ---> "<<std::endl;

    for (i = 0; i < s_adapters->nAdapters; i++ ) {
      std::cout<<B_BLUE<<"s_adapters->array_ComboIndex_ul["<<i<<"] ---> "
      <<B_YELLOW<<std::setw(3)<<s_adapters->array_ComboIndex_ul[i]
      <<B_BLUE<<" IP-IP: " <<B_YELLOW<<std::setw(15)<<s_adapters->array_IpAddressList_ip[i]
      <<B_BLUE<<" IP-MSK: "<<B_CYAN<<std::setw(15)<<s_adapters->array_IpAddressList_mask[i]
      <<B_BLUE<<" Name: "  <<B_MAGENTA<<s_adapters->array_AdapterName[i]
      <<B_BLUE<<" Desc: "  <<B_GREEN<<std::setw(45)<<s_adapters->array_AdapterDesc[i]
      <<B_BLUE<<" Type: "  <<B_MAGENTA<<std::setw(15)<<s_adapters->array_Type[i]
      <<B_BLUE<<" Mac: "   <<B_CYAN<<std::setw(17)<<s_adapters->array_MacAddress[i]
      <<COLOR_RESET<<std::endl;
    }
    for (i = 0; i < s_adapters->nAdapters; i++ ) {
      std::cout<<B_BLUE<<"s_adapters->array_ComboIndex_ul["<<i<<"] ---> "
      <<B_YELLOW<<std::setw(3)<<s_adapters->array_ComboIndex_ul[i]
      <<B_BLUE<<" DHCP: " <<B_YELLOW<<std::setw(5)<<s_adapters->array_DhcpEnabled_char[i]
      <<B_BLUE<<" DHCP-server: "<<B_CYAN<<std::setw(15)<<s_adapters->array_DhcpEnabled_server_ip[i]
      <<B_BLUE<<" DHCP-LeaseObtained: "<<B_CYAN<<std::setw(24)<<s_adapters->array_DhcpEnabled_LeaseObtained_char[i]
      <<B_BLUE<<" DHCP-LeaseExpires: "<<B_CYAN<<std::setw(24)<<s_adapters->array_DhcpEnabled_LeaseExpires_char[i]
      <<B_BLUE<<" GW-IP: " <<B_YELLOW<<std::setw(15)<<s_adapters->array_GatewayList_ip[i]
      <<B_BLUE<<" GW-MSK: "<<B_CYAN<<std::setw(15)<<s_adapters->array_GatewayList_mask[i]
      <<COLOR_RESET<<std::endl;
    }
    std::cout<<B_BLUE<<"*--------------------------------------*"<<std::endl;
    std::cout<<COLOR_RESET;

    return rc;
  } /* end of print_adapters_data_structure_t(adapters_struct *s_adapters) method */
  template <typename T, typename S> int Adapters_ClassTemplate_t<T,S>::_finalize_t() {
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
  } /* end of the _finalize_t method */
  template <typename T, typename S> Adapters_ClassTemplate_t<T,S>::~Adapters_ClassTemplate_t() {
    int rc = RC_SUCCESS;
    //finalising the the method and remmmoving all of of the alocated arrays
    rc = _finalize();
    if (rc != RC_SUCCESS) {
      std::cerr<<B_RED"return code: "<<rc
	       <<" line: "<<__LINE__<<" file: "<<__FILE__<<C_RESET<<std::endl;
      exit(rc);
    } else {rc = RC_SUCCESS; print_destructor_message("~Adapters_ClassTemplate_t");}
    rc = get_returnCode(rc, "Network", 0);
    
  } /* end of ~Adapters_ClassTemplate_t destructor method */
  ////////////////////////////////////////////////////////////////////////////////
  // Class Network definition extended from the CUDA NPP sdk library
  ////////////////////////////////////////////////////////////////////////////////
  Adapters::Adapters() {
    int rc = RC_SUCCESS;
    namespace_Network::Adapters_ClassTemplate_t<float, std::string>
      *cBaseAdapterLayer_t = new namespace_Network::Adapters_ClassTemplate_t<float, std::string>();
    rc = _initialize();
    std::cout<<B_YELLOW<<"Class Adapters::Adapters() has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
  } /* end of deepL_ClassTemplate constructor */
  Adapters::Adapters(IPAddresses_struct *s_IPAddresses, adapters_struct *s_adapters) {
    int rc = RC_SUCCESS;
    namespace_Network::Adapters_ClassTemplate_t<float, std::string>
      *cOverAdapterLayer_t = new namespace_Network::Adapters_ClassTemplate_t<float, std::string>(s_IPAddresses, s_adapters);
    rc = _initialize();
    std::cout<<B_CYAN<<"Class Adapters::Adapters(...) has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
  } /* end of deepL_ClassTemplate constructor */
  ////////////////////////////////////////////////////////////////////////////////
  // Class checkers
  ////////////////////////////////////////////////////////////////////////////////
  int Adapters::hello() {
    int rc = RC_SUCCESS;
    return rc;
  } /* end of hello checker method */
  ////////////////////////////////////////////////////////////////////////////////
  // Initialiser
  ////////////////////////////////////////////////////////////////////////////////
  int Adapters::_initialize() {
    int rc = RC_SUCCESS;
    //rc = print_object_header_deepL(__FUNCTION__, __FILE__);
    return rc;
  } /* end of _initialize method */
  ////////////////////////////////////////////////////////////////////////////////
  //Finaliser deaalicate sthe arrays and cleans up the environement
  ////////////////////////////////////////////////////////////////////////////////
  int Adapters::_finalize() {
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
  Adapters::~Adapters() {
    int rc = RC_SUCCESS;
    //finalising the the method and remmmoving all of of the alocated arrays
    rc = _finalize();
    if (rc != RC_SUCCESS) {
      std::cerr<<B_RED"return code: "<<rc
	       <<" line: "<<__LINE__<<" file: "<<__FILE__<<C_RESET<<std::endl;
      exit(rc);
    } else {rc = RC_SUCCESS; print_destructor_message("Adapters");}
    rc = get_returnCode(rc, "Network", 0);
  } /* end of ~deepL_FileHandler method */
  ////////////////////////////////////////////////////////////////////////////////
  // Methods that gets interfaced to extern C code for the API and the helper
  ////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
  extern "C" {
#endif
    /* here add the API C/Fortran binding methods */
#ifdef __cplusplus
  }
#endif
  ////////////////////////////////////////////////////////////////////////////////
  // end of deepLEM namespace
  ////////////////////////////////////////////////////////////////////////////////
} /* End of namespace namespace_Network */
////////////////////////////////////////////////////////////////////////////////


