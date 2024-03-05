//
// Created by Frederic on 12/2/2023.
//
#include <iostream>
#include "../global.cuh"
//#include "../include/Sockets.cuh"
//#include "../include/common.cuh"
//#include "../include/Exception.cuh"
//#include "../include/resmap_Sizes.cuh"

#if defined (WINDOWS)
//TODO: insert the headers here for the Windows environment

#elif defined (LINUX)
//TODO: insert the headers here for the Linux environment
#include <errno.h>      ///< errno
#include <sys/socket.h> ///< socket
#include <netinet/in.h> ///< sockaddr_in
#include <arpa/inet.h>  ///< getsockname
#include <unistd.h>     ///< close
#endif
//////////////////////////////////////////////////////////////////////////////
// Namespace delaration
//////////////////////////////////////////////////////////////////////////////
namespace namespace_Network {
    ////////////////////////////////////////////////////////////////////////////////
    // Class Socket_ClassTemplate_t mirrored valued type class definition
    ////////////////////////////////////////////////////////////////////////////////
    /*
    ********************************************************************************
    */
    //TODO: implement the mirrorored class here where needed Socket_ClassTemplate_t(socket_struct *s_sock)
    ////////////////////////////////////////////////////////////////////////////////
    // Class Socket definition extended from the CUDA NPP sdk library
    ////////////////////////////////////////////////////////////////////////////////
    /* constructor */
    Socket::Socket() {
        int rc = RC_SUCCESS;
        rc = Socket::_initialize(); if (rc != RC_SUCCESS) {rc = RC_WARNING;}
        namespace_Network::Socket_ClassTemplate_t<float, std::string>
        *cBaseSocketLayer_t = new namespace_Network::Socket_ClassTemplate_t<float, std::string>();
        std::cout<<B_BLUE<<"Class Socket::Socket() has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
    } /* end of Socket::Socket() constructor */
    Socket::Socket(socket_struct *s_sock) {
        int rc = RC_SUCCESS;
        rc = Socket::_initialize(); if (rc != RC_SUCCESS) {rc = RC_WARNING;}
        namespace_Network::Socket_ClassTemplate_t<float, std::string>
        *cOverSocketLayer_t = new namespace_Network::Socket_ClassTemplate_t<float, std::string>(s_sock);
        std::cout<<B_MAGENTA<<"Class Socket::Socket(socket_struct *s_sock) has been instantiated, return code: "<<B_GREEN<<rc<<COLOR_RESET<<std::endl;
    } /* end of Socket::Socket(socket_struct *s_sock) constructor */
int Socket::_initialize() {
    int rc = RC_SUCCESS;
    /* TODO: insert the initialisers in the method when needed */
/*
 *Depends what you are using the address for.
 *If it's part of a struct sockaddr_in, then it has to be unsigned long.
 *Typically kernel system calls take IP addresses as 4-byte integers rather
 * than strings.
 *
 * ntohl stands for "network to host long" where the long refers to the fact
 *
 * unsigned char a, b, c, d;
    sscanf( ipAddrString, "%hhu.%hhu.%hhu.%hhu", &a, &b, &c, &d );
    unsigned long ipAddr = ( a << 24 ) || ( b << 16 ) || ( c << 8 ) || d;
*/
    return rc;
} /* end of Socket::_initialize method */
int Socket::get_sockets() {
    int rc = RC_SUCCESS;

        std::cout<<B_RED<<"Socket::get_sockets()              ---> "<<B_B_CYAN<<"socket information"<<COLOR_RESET<<std::endl;

    return rc;
}

int Socket::_finalize() {
    int rc = RC_SUCCESS;
    try{
        //TODO: free the alocated pointers here, bring in the Eception class
    }
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

/*!\brief Destructor for the Template class and calls _finalize() to clean up
  allocated pointers
  \return        Returns one of {RC_SUCCESS, RC_FAIL, RC_STOP} via int rc
 */
Socket::~Socket() {
    int rc = RC_SUCCESS;
    //finalising the the method and remmmoving all of of the alocated arrays
    rc = _finalize();
    if (rc != RC_SUCCESS) {
        std::cerr<<B_RED"return code: "<<rc
                 <<" line: "<<__LINE__<<" file: "<<__FILE__<<C_RESET<<std::endl;
        exit(rc);
    } else {rc = RC_SUCCESS; print_destructor_message("Socket");}
    rc = get_returnCode(rc, "Socket", 0);
} /* end of ~sockets destructor */


#ifdef __cplusplus
    extern "C" {
#endif



#ifdef __cplusplus
    }
#endif


} /* End of namespace namespace_Network */



