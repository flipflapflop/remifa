/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "lu_nopiv_ll.hxx"

namespace remifa {

   // Helper functions

   //
   // W=fp32 O=OP=OF=fp32 I=IP=IF=fp32
   // IB=8 and OB=256
   void lu_nopiv_ll_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::FP32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp32 O=OP=OF=fp32 I=IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::FP32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp32 O=OP=OF=fp32 I=IP=IF=fp32, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::FP32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);

   }
      
   //
   // W=fp32 O=OP=OF=fp32 I=IP=IF=fp32, CUTLASS SplitK
   // IB=8

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32
   // IB=8
   void lu_nopiv_ll_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info, half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a_f16, ld_a_f16);      
   }

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a_f16, ld_a_f16);
   }

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace,
         half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace, d_a_f16, ld_a_f16);      
   }

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32
   // IB=8
   void lu_nopiv_ll_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a_f16, ld_a_f16);
      
   }

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a_f16, ld_a_f16);
      
   }

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         float *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace,
         half *d_a_f16, int ld_a_f16) {

      lu_nopiv_ll<
         /*W=*/float,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace, d_a_f16, ld_a_f16);
   }
   
   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16
   // IB=8 and OB=256
   void lu_nopiv_ll_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::FP16,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::FP16,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::FP16,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16
   // IB=8 and OB=256
   void lu_nopiv_ll_f16_tc16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC16,/*O=*/compute_type::TC16,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16, CUTLASS
   // IB=8 and OB=256
   void lu_nopiv_ll_cutlass_f16_tc16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC16,/*O=*/compute_type::TC16,
         /*IB=*/8,///*IB=*/256,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16
   // IB=8
   void lu_nopiv_ll_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         // /*IB=*/32,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   // 
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=fp16, IP=IF=fp16
   // IB=8
   void lu_nopiv_ll_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=fp16, IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=fp16, IP=IF=fp16, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=fp16, IP=IF=fp16
   // IB=8
   void lu_nopiv_ll_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=fp16, IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=fp16, IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }
   
   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32
   // IB=8
   void lu_nopiv_ll_f16_f32xt32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         // /*IB=*/2,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_f32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      // std::cout << "[lu_nopiv_ll_cutlass_f16_f32xtc32]" << std::endl;

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         // /*IB=*/2,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_f32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {
      
      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }
   
   //
   // W=fp16 O=TC16, OP=OF=fp16 I=fp16, IP=IF=fp16
   // IB=8
   void lu_nopiv_ll_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC16,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);

   }

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC16,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::FP16,/*O=*/compute_type::TC16,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/half,
         /*OF=*/half, /*IF=*/half,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32, cuBLAS
   // IB=8
   void lu_nopiv_ll_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {
      
      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/float,
         /*OF=*/half, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/float,
         /*OF=*/half, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info);
   }
   
   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32, CUTLASS SplitK
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/half, /*IP=*/float,
         /*OF=*/half, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace);
   }

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32, cuBLAS
   // IB=8
   void lu_nopiv_ll_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      // std::cout << "TETETETETET" << std::endl;

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CublasImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a, ldda);
   }   

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, 0, nullptr, d_a, ldda);
   }   

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32, CUTLASS
   // IB=8
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace) {

      lu_nopiv_ll<
         /*W=*/half,
         /*I=*/compute_type::TC32,/*O=*/compute_type::TC32,
         /*IB=*/8,
         /*OP=*/float, /*IP=*/float,
         /*OF=*/float, /*IF=*/float,
         remifa::CutlassSplitKImpl>
         (cuhandle, m, n, nb, d_a, ldda, d_info, slices, workspace, d_a, ldda);
   }   
}
