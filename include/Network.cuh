//
// Created by Frederic on 12/19/2023.
//
#ifndef NETWORK_CUH
#define NETWORK_CUH
//#include "../global.cuh"
//#include "common.cuh"
#include "Sockets.cuh"
#include "Adapters.cuh"
#include "../include/get_systemQuery_cpu.cuh"
#include "../include/get_deviceQuery_gpu.cuh"
#include "../include/deviceTools_gpu.cuh"
namespace namespace_Network {

    template<typename T, typename S>
    class Network_ClassTemplate_t
        :
    public Socket,
    public Adapters,
    public namespace_System_cpu::SystemQuery_cpu,
    public namespace_System_gpu::SystemQuery_gpu,
    public namespace_System_gpu::DeviceTools_gpu
    {
    private:
    public:
        /* constructors */
        explicit Network_ClassTemplate_t();
        explicit Network_ClassTemplate_t(network_struct *s_network);
        explicit Network_ClassTemplate_t(machine_struct *s_machine, IPAddresses_struct *s_IPAddresses,
                adapters_struct *s_adapters, socket_struct *s_socket, network_struct *s_network);
        /* destructors */
        ~Network_ClassTemplate_t();
        /* methods */
        int get_network_foreachIPs(machine_struct *s_mach, adapters_struct *s_adptrs, network_struct *s_network);
        /* getters */
        int get_network_map_nmap();
        int get_network_map_nmap(machine_struct *s_machine, adapters_struct *s_adptrs);
        int get_network_localIPs(machine_struct *s_machine, adapters_struct *s_adptrs, std::string delimiter_string_in);
        /* helper methods */
        std::string trim(const std::string &s);
        std::vector<std::string> split(std::string s, std::string delimiter);
        /* some global variables to the class */
        namespace_Network::Adapters_ClassTemplate_t<T,S> *cOverAdaptersLayer_t;
    protected:
        int _initialize_t();
        int _initialize_t(network_struct *s_network);
        int _initialize_t(machine_struct *s_machine, IPAddresses_struct *s_IPAddresses,
                        adapters_struct *s_adapters, socket_struct *s_socket, network_struct *s_network);
        /* finalizers */
        int _finalize_t();
    }; /* end of Network_ClassTemplate_t mirrored class */

    class Network {
        private:
        public:
        /* constructors */
        Network();
        Network(network_struct *s_network);
        Network(machine_struct *s_machine, IPAddresses_struct *s_IPAddresses,
        adapters_struct *s_adapters, socket_struct *s_socket, network_struct *s_network);
        /* destructors */
        ~Network();
        /* checkers */
        int hello();
        protected:
        int _initialize();
        int _finalize();
    }; /* end of Network class */
    ////////////////////////////////////////////////////////////////////////////////
    // Methods that gets interfaced to extern C code for the API and the helper
    ////////////////////////////////////////////////////////////////////////////////
    #ifdef __cplusplus
    extern "C" {
    #endif

#ifdef __cplusplus
    }
#endif

} /* End of namespace namespace_Network */

#endif //NETWORK_CUH

