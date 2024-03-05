/*
 *   -- DEEPL addon
 *      Author: Frederic Bonnet, Date: 13th of May 2017
 *      Monash University
 *      May 2017
 *
 *      Routine which test the cuDNN deep learning library
 *
 *      Non Special case
 * @precisions normal z -> s d c
*/
//system headers
#if defined (CUDA) /* prepropressing for the cuda environment */
#include <cuda.h> // need CUDA_VERSION 7.0
#include <cudnn.h>
#endif

//aplication headers
#include "saxs_HelperSimul.h"
#include "saxs_exception.h"
#include "cmdLine.h"
#include "deepL.h"
//#include "deepL_FileHandler.h"
//#include "deepL_ImageIO.h"
#include "deepL_Layer.hpp"
#include "deepL_ClassTemplate.hpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Namespace for the extension of deepLEM package
////////////////////////////////////////////////////////////////////////////////
namespace deepLEM {
////////////////////////////////////////////////////////////////////////////////
// Class deepL_ClassTemplate_t mirrored valued type class definition
////////////////////////////////////////////////////////////////////////////////
/*
********************************************************************************
*/
////////////////////////////////////////////////////////////////////////////////
// Class Networks definition extended from the CUDA NPP sdk library
////////////////////////////////////////////////////////////////////////////////
  deepL_ClassTemplate::deepL_ClassTemplate() {
    int rc = RC_SUCCESS;
    deepLEM::deepL_ClassTemplate_t<float>
      *cBaseLayer_t = new deepLEM::deepL_ClassTemplate_t<float>();
    rc = _initialize();
  } /* end of deepL_ClassTemplate constructor */
////////////////////////////////////////////////////////////////////////////////
// Class checkers
////////////////////////////////////////////////////////////////////////////////
  int deepL_ClassTemplate::hello() {
    int rc = RC_SUCCESS;
    return rc;
  } /* end of hello checker method */
////////////////////////////////////////////////////////////////////////////////
// Initialiser
////////////////////////////////////////////////////////////////////////////////
  int deepL_ClassTemplate::_initialize() {
    int rc = RC_SUCCESS;
    rc = print_object_header_deepL(__FUNCTION__, __FILE__);
    return rc;
  } /* end of _initialize method */
////////////////////////////////////////////////////////////////////////////////
//Finaliser deaalicate sthe arrays and cleans up the environement
////////////////////////////////////////////////////////////////////////////////
  int deepL_ClassTemplate::_finalize() {
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
  deepL_ClassTemplate::~deepL_ClassTemplate() {
    int rc = RC_SUCCESS;
    //finalising the the method and remmmoving all of of the alocated arrays
    rc = _finalize();
    if (rc != RC_SUCCESS) {
      std::cerr<<B_RED"return code: "<<rc
               <<" line: "<<__LINE__<<" file: "<<__FILE__<<C_RESET<<std::endl;
      exit(rc);
    } else {rc = RC_SUCCESS; print_destructor_message("deepL_ClassTemplate");}
    
  } /* end of ~deepL_FileHandler method */
////////////////////////////////////////////////////////////////////////////////
// end of deepLEM namespace
////////////////////////////////////////////////////////////////////////////////
} /* end of deepLEM namespace */
////////////////////////////////////////////////////////////////////////////////
// Methods that gets interfaced to extern C code for the API and the helper
////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
