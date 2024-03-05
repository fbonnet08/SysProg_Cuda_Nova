//
// Created by Frederic on 12/2/2023.
//
//#include "common.cuh"
//#include "common_krnl.cuh"
//#include "Network.cuh"
//#include "../global.cuh"
#include "../include/get_systemQuery_cpu.cuh"
#include "../include/get_deviceQuery_gpu.cuh"
#include "../include/deviceTools_gpu.cuh"

#ifndef SOCKETS_CUH
#define SOCKETS_CUH

namespace namespace_Network {
    /*
enum __socket_type {
    SOCK_STREAM = 1,
#define SOCK_STREAM SOCK_STREAM
    SOCK_DGRAM = 2
#define SOCK_DGRAM SOCK_DGRAM
};
*/
    /* */
    struct sockAddress {};

    template<typename T, typename S>
    class Socket_ClassTemplate_t
    :
    public namespace_System_cpu::SystemQuery_cpu,
    public namespace_System_gpu::SystemQuery_gpu,
    public namespace_System_gpu::DeviceTools_gpu
    {
        private:
        public:
            explicit Socket_ClassTemplate_t() { ;}
            explicit Socket_ClassTemplate_t(socket_struct *s_sock) {;}
        protected:
        int _initialize_t(){;}
        int _initialize_t(socket_struct *s_sock){;}
        int _finalize_t(){;}

        }; /* end of Socket_ClassTemplate_t mirrored class */

    /* Clas declaration */
    class Socket {
    private:
    public:
        /* constructor */
        Socket();
        Socket(socket_struct *s_sock);
        /* methods */
        int _initialize();
        int _finalize();
        int get_sockets();
        /* destructor */
        ~Socket();
    }; /* end of Socket class */


    ////////////////////////////////////////////////////////////////////////////////
    // Methods that gets interfaced to extern C code for the API and the helper
    ////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
    extern "C" {
#endif


#ifdef __cplusplus
    }
#endif

}/* End of namespace namespace_network */

#endif //SOCKETS_CUH
