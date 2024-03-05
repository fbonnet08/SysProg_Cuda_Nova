/*******************************************************************************
 *     Author: Frederic D.R. Bonnet date: 23rd of May 2017. 14:35pm
 *
 * Name:
 * deepL_EMData.h - base definitions used in all modules.
 *
 * Description:
 * header file for file handling for the DeepLearning package
 *
 * Part of codes taking from the NVIDIA sample source code and adapted to and
 * accordingly to process CUDA code in the OpenGL environment
 *
 * Section of code has been taken from the NVIDIA Sample code and adaprted 
 * accordingly to fit local purpose of SAXS data handling and processing
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 *******************************************************************************
 */
#ifndef DEEPL_CLASSTEMPLATE_HPP_
#define DEEPL_CLASSTEMPLATE_HPP_

namespace deepLEM {

  template<typename value_type>
  class deepL_ClassTemplate_t : public deepL_Layer {

  public:
    explicit deepL_ClassTemplate_t() {;}
  protected:

  private:

  }; /* end of deepL_ClassTemplate_t mirrored class */

  class deepL_ClassTemplate {

  public:
    /* constructors */
    deepL_ClassTemplate();
    /* destructors */
    ~deepL_ClassTemplate();
    /* checkers */
    int hello();
  protected:
    int _initialize();
    int _finalize();
  private:

  }; /* end of deepL_ClassTemplate class */
  
} /* end of deepL namespace */

  
#endif  // DEEPL_CLASSTEMPLATE_HPP_
