/// @file
/// @copyright 2019- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#include "TestOpts.hxx"
#include "testing_lu_gpu.hxx"

// STD
#include <string>
#include <iostream>

using namespace remifa::tests;

int main(int argc, char** argv) {

   int ret = 0;
   std::string context = "testing_lu_gpu";

   // Parse command line options for this test
   remifa::tests::TestOpts opts;
   int hlp = opts.parse_opts(context, argc, argv);

   // std::cout << "[" << context << "]" << std::endl;
   
   // Run test unless help was required
   if (!hlp) {
      switch (opts.prec) {
      case remifa::tests::prec::FP16:
      case remifa::tests::prec::FP32:
         if (opts.usecutlass) {
         ret = lu_test
            <float, // Presision used for generating original matrix
             true // CUTLASS enabled
             >(opts.m, // Matrix dimensions
               opts.algo, // Algo to be tested
               opts.prec, // Working precision (fp16 or fp32)
               opts.in_upd, // Inner update compute type (fp16, fp32,
                            // TC16 or TC32)
               opts.out_upd, // Outer update compute type (fp16, fp32,
                             // TC16 or TC32)
               opts.in_pnl, // Outer panel precision (fp16 or fp32)
               opts.out_pnl, // Outer panel precision (fp16 or fp32)
               opts.in_fac, // Outer factor precision (fp16 or fp32)
               opts.out_fac, // Outer factor precision (fp16 or fp32)
               // opts.usetc,
               opts.mat, (float)opts.cond, (float)opts.gamma,
               opts.check,
               opts);
      //    break;
      // case remifa::tests::prec::FP64:
      //    ret = lu_test<double>(
      //          opts.m, opts.algo, opts.prec, opts.in_upd, opts.out_upd, opts.usetc,
      //          opts.mat, (float)opts.cond, opts.check);
         }
         else {
            ret = lu_test
               <float, // Presision used for generating original matrix
                false // CUTLASS enabled
                >(opts.m, // Matrix dimensions
                  opts.algo, // Algo to be tested
                  opts.prec, // Working precision (fp16 or fp32)
                  opts.in_upd, // Inner update compute type (fp16, fp32,
                  // TC16 or TC32)
                  opts.out_upd, // Outer update compute type (fp16, fp32,
                  // TC16 or TC32)
                  opts.in_pnl, // Outer panel precision (fp16 or fp32)
                  opts.out_pnl, // Outer panel precision (fp16 or fp32)
                  opts.in_fac, // Outer factor precision (fp16 or fp32)
                  opts.out_fac, // Outer factor precision (fp16 or fp32)
                  // opts.usetc,
                  opts.mat, (float)opts.cond, (float)opts.gamma,
                  opts.check,
                  opts);
         }
         break;
      default:
         std::cout << "[" <<  context << "]" <<  " Requested working precision NOT available" << std::endl;
         break;
      }
   }

   return ret;
}
