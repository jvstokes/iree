// RUN: iree-opt --iree-flow-set-encoding --cse --split-input-file %s | FileCheck %s

func.func @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, original_type = tensor<100x250xf32>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, original_type = tensor<250x500xf32>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f32f32f32_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> -> index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>> -> index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_i8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
    %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xi8>, tensor<250x500xi8>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  return %0 : tensor<100x500xi32>
}
//      CHECK: func @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xi32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xi8> to tensor<?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = LHS, original_type = tensor<100x250xi8>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xi8> to tensor<?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RHS, original_type = tensor<250x500xi8>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xi32, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xi32> to tensor<?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RESULT, original_type = tensor<100x500xi32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f32(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = LHS, original_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RHS, original_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RESULT, original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  return %0 : tensor<100x500xf16>
}
//      CHECK: func @matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = LHS, original_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RHS, original_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RESULT, original_type = tensor<100x500xf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16f32(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = LHS, original_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RHS, original_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RESULT, original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  return %0 : tensor<100x500xbf16>
}
//      CHECK: func @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xbf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = LHS>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = LHS, original_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RHS>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RHS, original_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RESULT>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RESULT, original_type = tensor<100x500xbf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
  return %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
}
//      CHECK: func @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>)
//      CHECK:   return %[[FILL]]

// -----

func.func @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
  return %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
}
//      CHECK: func @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   return %[[FILL]]
