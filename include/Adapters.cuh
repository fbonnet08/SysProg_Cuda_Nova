//
// Created by Frederic on 12/23/2023.
//

#ifndef ADAPTERS_CUH
#define ADAPTERS_CUH

//#include "../global.cuh"
//#include "Sockets.cuh"
#include "../include/get_systemQuery_cpu.cuh"
#include "../include/get_deviceQuery_gpu.cuh"
#include "../include/deviceTools_gpu.cuh"
namespace namespace_Network {

    template<typename T, typename S>
    class Adapters_ClassTemplate_t
        :
    public Socket,
    public namespace_System_cpu::SystemQuery_cpu,
    public namespace_System_gpu::SystemQuery_gpu,
    public namespace_System_gpu::DeviceTools_gpu
    {
    private:
    public:
        /* constructors */
        explicit Adapters_ClassTemplate_t();
        explicit Adapters_ClassTemplate_t(IPAddresses_struct *s_IPAddrs,
        adapters_struct *s_adapters);
        /* destructors */
        ~Adapters_ClassTemplate_t();
        /* methods */
        std::string decimalToBinary_32bit(unsigned long n, int base);
        std::string convertBinaryToASCII_IP_32bit(std::string binary);
        std::string trim(const std::string &s);
        /* getters */
        int get_IPAddresses_Struct_t(IPAddresses_struct *s_IPAddrs);
        std::string get_Adapters_Struct_IPAddr_t(adapters_struct *s_adapters, IPAddresses_struct *s_IPAddrs);
        /* printers */
        int print_adapters_data_structure_t(adapters_struct *s_adapters);
        int print_IPAddresses_data_structure_t(IPAddresses_struct *s_IPAddrs);
    protected:
        int _initialize_t();
        int _initialize_t(adapters_struct *s_adapters, IPAddresses_struct *s_IPAddrs );
        int _finalize_t();
    }; /* end of Adapters__ClassTemplate_t mirrored class */

    class Adapters {
    private:
    public:
        /* constructors */
        Adapters();
        Adapters(IPAddresses_struct *s_IPAddresses,
        adapters_struct *s_adapters);
        /* destructors */
        ~Adapters();
        /* checkers */
        int hello();
    protected:
        int _initialize();
        int _finalize();
    }; /* end of Adapters_ class */
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























#endif //ADAPTERS_CUH
