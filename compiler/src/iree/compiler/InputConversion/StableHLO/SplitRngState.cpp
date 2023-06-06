// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific logic for lowering StableHLO/CHLO dialects to
// LinalgExt dialect.

#include <cmath>
#include <complex>
#include <memory>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/InputConversion/StableHLO/MapStableHLOToScalarOp.h"
#include "iree/compiler/InputConversion/StableHLO/PassDetail.h"
#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_SPLITRNGSTATE
#include "iree/compiler/InputConversion/StableHLO/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct SplitRngState final
    : impl::SplitRngStateBase<SplitRngState> {
  using SplitRngStateBase::SplitRngStateBase;

  void runOnOperation() {
    llvm::errs() <<"\n Hitting rng input split pass\n";
  }
};


}  // namespace mlir::iree_compiler::stablehlo
