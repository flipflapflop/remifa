/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "common.hxx"
#include "utils.hxx"

// STD
#include <string>
#include <cstring>
#include <iostream>

namespace remifa {
namespace tests {

   class TestOpts {
   public:
      TestOpts():
         ncpu(1), ngpu(1), m(256), n(256), k(256), nb(256), check(true),
         prec(remifa::tests::prec::FP32), algo(remifa::tests::algo::REMIFA),
         usetc(true), mat(remifa::tests::COND), cond(1),
         out_upd(remifa::compute_type::FP32), in_upd(remifa::compute_type::FP32),
         out_pnl(remifa::tests::prec::DEFAULT), in_pnl(remifa::tests::prec::DEFAULT),
         out_fac(remifa::tests::prec::DEFAULT), in_fac(remifa::tests::prec::DEFAULT),
         usecutlass(false), gamma(1.0), check_conv(false), latmr_mode(1), latms_mode(1),
         cast_f16(false), scale(false), split_k(false), split_k_slices(0),
         slv(remifa::compute_type::FP32)
      {}

      int parse_opts(std::string const& context, int argc, char** argv) {
         
         int ret = 0;

         std::string help = 
            "Usage: " + context + " [options]\n" 
R"(
Options:
--algo=cusolver        Run cuSOLVER algo (full-precision)
--algo=cusolver-hp     Run cuSOLVER algo (half-precision)
--algo=remifa-ll       Run REMIFA, Left-Looking algo
--algo=remifa-rl       Run REMIFA, Right-Looking algo
--cond COND            Generate matrix with condition number exponent equals to 
                       COND i.e. condition number equal to 10^COND
--cutlass              Use CUTLASS GEMM implementation 
--diagdom              Generate diagonally dominant matrix
--fp16                 Use fp16 working precision
--fp32                 Use fp32 working precision
--in-upd=              Inner update compute type among fp16, fp32, TC16 and TC32
--out-upd=             Outer update compute type among fp16, fp32, TC16 and TC32
--in-pnl=              Inner panel type among fp16, fp32
--out-pnl=             Outer panel type among fp16, fp32
--in-fac=              Inner factor precision among fp16, fp32
--out-fac=             Outer factor precision among fp16, fp32
--m                    Number of rows in test matrix
--n                    Number of columns in test matrix
--no-check             Disable error checking
--posrand              Generate matrix with random entries in [0,1]
--rand                 Generate matrix with random entries in [-1,1]
)";
         
         for( int i = 1; i < argc; ++i ) {

            if ( !strcmp("--help", argv[i]) ) {

               std::cout << help;
               return 1;
            }

            // Matrix properties

            // Matrix dimensions
            else if ( !strcmp("--m", argv[i]) && i+1 < argc ) {
               m =  std::atoi( argv[++i] );
               std::cout << "Number of rows = " << m  << std::endl;

            }
            else if ( !strcmp("-m", argv[i]) && i+1 < argc ) {
               m =  std::atoi( argv[++i] );
               std::cout << "Number of rows = " << m  << std::endl;
            }
            else if ( !strcmp("--n", argv[i]) && i+1 < argc ) {
               n =  std::atoi( argv[++i] );
               std::cout << "Number of columns = " << n << std::endl;
            }
            else if ( !strcmp("-n", argv[i]) && i+1 < argc ) {
               n =  std::atoi( argv[++i] );
               std::cout << "Number of columns = " << n << std::endl;
            }
            else if ( !strcmp("--k", argv[i]) && i+1 < argc ) {
               k =  std::atoi( argv[++i] );
               std::cout << "Parameter k set to " << k << std::endl;
            }
            else if ( !strcmp("-k", argv[i]) && i+1 < argc ) {
               k =  std::atoi( argv[++i] );
               std::cout << "Parameter k set to " << k << std::endl;
            }

            // Block-size
            else if ( !strcmp("--nb", argv[i]) && i+1 < argc ) {
               nb =  std::atoi( argv[++i] );
               std::cout << "Block-size set to " << nb << std::endl;
            }

            
            // Generate random matrix with entries in [-1,1]
            else if ( !strcmp("--rand", argv[i]) ) {
               mat = remifa::tests::mat::RAND;
               std::cout << "Generate random matrix with entries in [-1,1]" << std::endl;
            }
            // Generate symmetric random matrix with entries in [-1,1]
            else if ( !strcmp("--symrand", argv[i]) ) {
               mat = remifa::tests::mat::SYMRAND;
               std::cout << "Generate symmetric random matrix with entries in [-1,1]" << std::endl;
            }
            // Generate random matrix with entries in [0,1]
            else if ( !strcmp("--posrand", argv[i]) ) {
               mat = remifa::tests::mat::POSRAND;
               std::cout << "Generate random matrix with entries in [0,1]" << std::endl;
            }

            // Condition number exponent (matrix cond is 10^cond)
            else if ( !strcmp("--cond", argv[i]) && i+1 < argc ) {
               mat = remifa::tests::mat::COND;
               cond =  std::stod( argv[++i] );
               std::cout << "Generate random matrix with condition number exponent = " << cond << std::endl;
            }
            // Condition number exponent (matrix cond is 10^cond)
            else if ( !strcmp("--gamma", argv[i]) && i+1 < argc ) {
               gamma =  std::stod( argv[++i] );
               std::cout << "Generate random matrix with gamma  = " << gamma << std::endl;
            }
            
            // Read sparse matrix passed in argument (in matrix market
            // format) and convert it to dense.
            else if ( !strcmp("--mtx", argv[i]) && i+1 < argc ) {
               mat = remifa::tests::mat::MTX;
               matfile =  argv[++i];
               std::cout << "Read sparse matrix = " << matfile << std::endl;
            }
            

            // Generate diagonally dominant matrix
            else if ( !strcmp("--diagdom", argv[i]) ) {
               mat = remifa::tests::mat::DIAGDOM;
               std::cout << "Generate diaginally dominant matrix" << std::endl;
            }
            // Generate diagonally dominant matrix with non-negative entries
            else if ( !strcmp("--diagdom-pos", argv[i]) ) {
               mat = remifa::tests::mat::DIAGDOM_POS;
               std::cout << "Generate diaginally dominant matrix with non-negative entries"
                         << std::endl;
            }

            // Generate tri-diagonally dominant matrix
            else if ( !strcmp("--tridiagdom", argv[i]) ) {
               mat = remifa::tests::mat::TRIDIAGDOM;
               std::cout << "Generate tri-diaginally dominant matrix" << std::endl;
            }
            // Generate tri-diagonally dominant matrix
            else if ( !strcmp("--arrowhead", argv[i]) ) {
               mat = remifa::tests::mat::ARROWHEAD;
               std::cout << "Generate DD arrowhead matrix" << std::endl;
            }
            // Generate arrowhead dominant matrix in [0,1]
            else if ( !strcmp("--arrowhead-pos", argv[i]) ) {
               mat = remifa::tests::mat::ARROWHEAD_POS;
               std::cout << "Generate DD arrowhead matrix with positive entries" << std::endl;
            }
            // Generate arrowhead dominant matrix in [0,1]
            else if ( !strcmp("--arrowhead-pos-01", argv[i]) ) {
               mat = remifa::tests::mat::ARROWHEAD_POS_01;
               std::cout << "Generate DD arrowhead matrix with positive entries in [0,0.1]" << std::endl;
            }
            // Generate arrowhead dominant matrix in [-1,1]
            else if ( !strcmp("--arrowhead-pos-neg", argv[i]) ) {
               mat = remifa::tests::mat::ARROWHEAD_POS_NEG;
               std::cout << "Generate DD arrowhead matrix with positive and negative entries" << std::endl;
            }
            // Generate random matrix using using LATMR routine
            else if ( !strcmp("--latmr", argv[i]) && i+1 < argc) {
               latmr_mode =  std::atoi( argv[++i] );
               mat = remifa::tests::mat::LATMR;
               std::cout << "Generate random matrix using LATMR, mode = " << latmr_mode << std::endl;
            }
            // Generate random matrix using using LATMR routine
            else if ( !strcmp("--latms", argv[i]) && i+1 < argc) {
               latms_mode =  std::atoi( argv[++i] );
               mat = remifa::tests::mat::LATMS;
               std::cout << "Generate random matrix using LATMS, mode = " << latmr_mode << std::endl;
            }

            // Machine

            // Number of CPU
            else if ((!strcmp("--ncpu", argv[i]) || !strcmp("-ncpu", argv[i])) &&
                     i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
               std::cout << "Number of CPU = " << ncpu  << std::endl;
            }
            // Number of GPU
            else if ((!strcmp("--ngpu", argv[i]) || !strcmp("-ngpu", argv[i])) &&
                      i+1 < argc )  {
               ngpu =  std::atoi( argv[++i] );
               std::cout << "Number of GPU = " << ngpu  << std::endl;
            }
            // Tensor core mode (ON/OFF)
            else if ( !strcmp("--enable-tc", argv[i]) ) {
               std::cout << "Tensor cores enabled" << std::endl;
               usetc =  true;
            }
            else if ( !strcmp("--disable-tc", argv[i]) ) {
               std::cout << "Tensor cores deactivated" << std::endl;
               usetc =  false;
            }

            // Test 
            
            // Error checking
            else if ( !strcmp("--no-check", argv[i]) ) {
               std::cout << "Error checking disabled" << std::endl;
               check = false;
            }
            else if ( !strcmp("--check-conv", argv[i]) ) {
               std::cout << "Enable conversion error checking" << std::endl;
               check_conv = true;
            }

            
            // Working precision
            else if ( !strcmp("--fp16", argv[i]) ) {
               prec = remifa::tests::prec::FP16;
               std::cout << "Working precision set to FP16" << std::endl;
            }
            else if ( !strcmp("--fp32", argv[i]) ) {
               prec = remifa::tests::prec::FP32;
               std::cout << "Working precision set to FP32" << std::endl;
            }
            else if ( !strcmp("--fp64", argv[i]) ) {
               prec = remifa::tests::prec::FP64;
               std::cout << "Working precision set to FP64" << std::endl;
            }

            // Select algorithm to be tested
            else if ( !strcmp("--algo=cusolver", argv[i]) ) {
               algo = remifa::tests::algo::cuSOLVER;
               std::cout << "CuSOLVER algorithm selected (full-precision)" << std::endl;
            }
            else if ( !strcmp("--algo=cusolver-hp", argv[i]) ) {
               algo = remifa::tests::algo::cuSOLVER_HP;
               std::cout << "CuSOLVER algorithm selected (half-precision)" << std::endl;
            }
            else if ( !strcmp("--algo=remifa", argv[i]) ) {
               algo = remifa::tests::algo::REMIFA;
               std::cout << "REMIFA algorithm selected (full-precision)" << std::endl;
            }
            else if ( !strcmp("--algo=remifa-ll", argv[i]) ) {
               algo = remifa::tests::algo::REMIFA_LL;
               std::cout << "REMIFA left-looking algorithm selected" << std::endl;
            }
            else if ( !strcmp("--algo=remifa-rl", argv[i]) ) {
               algo = remifa::tests::algo::REMIFA_RL;
               std::cout << "REMIFA right-looking algorithm selected" << std::endl;
            }

            //
            // Outer update compute type
            //
            else if ( !strcmp("--out-upd=fp16", argv[i]) ) {
               out_upd = remifa::compute_type::FP16;
               std::cout << "Set outer update compute type to FP16" << std::endl;
            }
            else if ( !strcmp("--out-upd=fp32", argv[i]) ) {
               out_upd = remifa::compute_type::FP32;
               std::cout << "Set outer update compute type to FP32" << std::endl;
            }
            else if ( !strcmp("--out-upd=tc16", argv[i]) ) {
               out_upd = remifa::compute_type::TC16;
               std::cout << "Set outer update compute type to TC16" << std::endl;
            }
            else if ( !strcmp("--out-upd=tc32", argv[i]) ) {
               out_upd = remifa::compute_type::TC32;
               std::cout << "Set outer update compute type to TC32" << std::endl;
            }

            //
            // Inner update compute type
            //
            else if ( !strcmp("--in-upd=fp16", argv[i]) ) {
               in_upd = remifa::compute_type::FP16;
               std::cout << "Set inner update compute type to FP16" << std::endl;
            }
            else if ( !strcmp("--in-upd=fp32", argv[i]) ) {
               in_upd = remifa::compute_type::FP32;
               std::cout << "Set inner update compute type to FP32" << std::endl;
            }
            else if ( !strcmp("--in-upd=tc16", argv[i]) ) {
               in_upd = remifa::compute_type::TC16;
               std::cout << "Set inner update compute type to TC16" << std::endl;
            }
            else if ( !strcmp("--in-upd=tc32", argv[i]) ) {
               in_upd = remifa::compute_type::TC32;
               std::cout << "Set inner update compute type to TC32" << std::endl;
            }

            //
            // Outer panel prec (precision of the panel used to
            // accumulate the update which could be different from the
            // woking precision in which case it is necessary to use a
            // buffer)
            //
            else if ( !strcmp("--out-pnl=fp16", argv[i]) ) {
               out_pnl = remifa::tests::prec::FP16;
               std::cout << "Set outer panel prec to FP16" << std::endl;
            }
            else if ( !strcmp("--out-pnl=fp32", argv[i]) ) {
               out_pnl = remifa::tests::prec::FP32;
               std::cout << "Set outer panel prec to FP32" << std::endl;
            }

            //
            // Inner panel prec (precision of the panel used to
            // accumulate the update which could be different from the
            // woking precision in which case it is necessary to use a
            // buffer)
            //
            else if ( !strcmp("--in-pnl=fp16", argv[i]) ) {
               in_pnl = remifa::tests::prec::FP16;
               std::cout << "Set inner panel prec to FP16" << std::endl;
            }
            else if ( !strcmp("--in-pnl=fp32", argv[i]) ) {
               in_pnl = remifa::tests::prec::FP32;
               std::cout << "Set inner panel prec to FP32" << std::endl;
            }

            // Precision used in the factorization of inner block
            // column. If it is different from the working precision
            // then the factorization requires the use of a workspace.
            else if ( !strcmp("--in-fac=fp16", argv[i]) ) {
               in_fac = remifa::tests::prec::FP16;
               std::cout << "Set inner panel factorization prec to FP16" << std::endl;
            }
            else if ( !strcmp("--in-fac=fp32", argv[i]) ) {
               in_fac = remifa::tests::prec::FP32;
               std::cout << "Set inner panel factorization prec to FP32" << std::endl;
            }

            // Precision used in the factorization of inner block
            // column. If it is different from the working precision
            // then the factorization requires the use of a workspace.
            else if ( !strcmp("--out-fac=fp16", argv[i]) ) {
               out_fac = remifa::tests::prec::FP16;
               std::cout << "Set outer panel factorization prec to FP16" << std::endl;
            }
            else if ( !strcmp("--out-fac=fp32", argv[i]) ) {
               out_fac = remifa::tests::prec::FP32;
               std::cout << "Set outer panel factorization prec to FP32" << std::endl;
            }
            
#if defined(HAVE_CUTLASS)
            else if ( !strcmp("--cutlass", argv[i]) ) {
               usecutlass = true;
               std::cout << "Use GEMM implementation from CUTLASS library" << std::endl;
            }
            else if ( !strcmp("--cast-f16", argv[i]) ) {
               cast_f16 = true;
               std::cout << "Cast generated input matrix to fp16" << std::endl;
            }

            else if ( !strcmp("--scale", argv[i]) ) {
               scale = true;
               std::cout << "Scale matrix" << std::endl;
            }

            // Number of slices for Cutlass SplitK implementation
            else if ( !strcmp("--splitk", argv[i]) && i+1 < argc ) {
               split_k = true;
               split_k_slices = std::atoi( argv[++i] );
               std::cout << "Enable SplitK implementation, Number of slices = " << split_k_slices << std::endl;
            }

#endif

            //
            // Solve
            //
            else if ( !strcmp("--slv=fp16", argv[i]) ) {
               slv = remifa::compute_type::FP16;
               std::cout << "Set solve compute type to FP16" << std::endl;
            }

            
         }
         
         return ret;
      }

      int ncpu; // Number of CPUs
      int ngpu; // Number of GPUs
      int m; // no rows in matrix
      int n; // no columns in matrix
      int k;
      int nb; // block-size
      bool usetc; // Use tensor cores on GPU
      bool check; // Compute backward erro
      bool check_conv; // Compute fp32 to fp16 convertion error
      enum remifa::tests::prec prec; // Working precision
      enum remifa::tests::algo algo; // Algorithm to be tested
      enum remifa::tests::mat mat; // Test matrix type
      double cond; // Condition number of test matrix
      enum remifa::compute_type out_upd; // Compute type for outer update
      enum remifa::compute_type in_upd; // Compute type for inner update
      enum remifa::tests::prec out_pnl; // Outer panel 
      enum remifa::tests::prec in_pnl; // Inner panel
      enum remifa::tests::prec out_fac; // Outer panel 
      enum remifa::tests::prec in_fac; // Inner panel
      bool usecutlass; // Use CUTLASS implementation for GEMM operations
      double gamma;
      int latmr_mode;
      int latms_mode;
      bool cast_f16;
      bool scale;
      bool split_k;
      int split_k_slices;
      // Solve
      enum remifa::compute_type slv;
      // Matrix file
      std::string matfile;
   };
}}
