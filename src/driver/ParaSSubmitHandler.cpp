/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "paras/ParaSSubmitHandler.hpp"
#include "paras/ParaSLog.hpp"

#include <clang/Lex/Lexer.h>
#include <clang/AST/AST.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Expr.h>
#include <algorithm>

namespace {

const clang::Expr* stripExpr(const clang::Stmt* stmt) {
    if (!stmt)
        return nullptr;
    const auto* expr = llvm::dyn_cast<clang::Expr>(stmt);
    if (!expr)
        return nullptr;
    expr = expr->IgnoreParenImpCasts();
    while (true) {
        if (const auto* cleanups = llvm::dyn_cast<clang::ExprWithCleanups>(expr))
            expr = cleanups->getSubExpr()->IgnoreParenImpCasts();
        else if (const auto* materialized = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr))
            expr = materialized->getSubExpr()->IgnoreParenImpCasts();
        else if (const auto* bound = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr))
            expr = bound->getSubExpr()->IgnoreParenImpCasts();
        else
            return expr;
    }
}

const clang::DeclRefExpr* simpleArrayBase(const clang::Expr* expr) {
    expr = stripExpr(expr);
    if (const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr))
        return ref;
    return nullptr;
}

const clang::DeclRefExpr* simpleIndexRef(const clang::Expr* expr,
                                         const clang::ParmVarDecl* param) {
    if (!expr)
        return nullptr;
    expr = stripExpr(expr);
    if (const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
        if (!param || ref->getDecl() == param)
            return ref;
    }
    if (const auto* cast = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(expr)) {
        if (cast->getCastKind() == clang::CK_UserDefinedConversion)
            expr = stripExpr(cast->getSubExpr());
    }
    if (const auto* memberCall = llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(expr)) {
        const auto* obj = stripExpr(memberCall->getImplicitObjectArgument());
        if (const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(obj)) {
            if (!param || ref->getDecl() == param)
                return ref;
        }
    }
    if (const auto* opCall = llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getNumArgs() >= 1) {
            const auto* obj = stripExpr(opCall->getArg(0));
            if (const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(obj)) {
                if (!param || ref->getDecl() == param)
                    return ref;
            }
        }
    }
    return nullptr;
}

bool isFloatElementPointer(clang::QualType type, bool allowConstPointee = true) {
    type = type.getCanonicalType();
    if (type->isReferenceType())
        type = type->getPointeeType().getCanonicalType();
    if (const auto* pointer = type->getAs<clang::PointerType>()) {
        clang::QualType pointee = pointer->getPointeeType().getCanonicalType();
        if (!allowConstPointee && pointee.isConstQualified())
            return false;
        return pointee.getUnqualifiedType()->isSpecificBuiltinType(clang::BuiltinType::Float);
    }
    if (const auto* array = type->getAsArrayTypeUnsafe()) {
        clang::QualType elem = array->getElementType().getCanonicalType();
        if (!allowConstPointee && elem.isConstQualified())
            return false;
        return elem.getUnqualifiedType()->isSpecificBuiltinType(clang::BuiltinType::Float);
    }
    return false;
}

bool isVectorAddKernel(const clang::CallExpr* parallelFor, std::string& count,
                       std::string& inputA, std::string& inputB, std::string& output,
                       const clang::SourceManager& SM, const clang::LangOptions& LangOpts) {
    if (!parallelFor || parallelFor->getNumArgs() < 2)
        return false;
    clang::QualType rangeType = parallelFor->getArg(0)->getType().getCanonicalType();
    if (rangeType->isReferenceType())
        rangeType = rangeType->getNonReferenceType().getCanonicalType();
    const auto* rangeDecl = rangeType->getAsCXXRecordDecl();
    const auto* rangeSpec = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(rangeDecl);
    if (!rangeSpec || rangeSpec->getSpecializedTemplate()->getQualifiedNameAsString() != "sycl::range" ||
        rangeSpec->getTemplateArgs().size() != 1 ||
        rangeSpec->getTemplateArgs()[0].getKind() != clang::TemplateArgument::Integral ||
        rangeSpec->getTemplateArgs()[0].getAsIntegral() != 1)
        return false;
    const auto* lambda = llvm::dyn_cast_or_null<clang::LambdaExpr>(stripExpr(parallelFor->getArg(1)));
    if (!lambda || lambda->getCallOperator()->getNumParams() != 1)
        return false;
    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(lambda->getBody());
    if (!body || body->size() != 1)
        return false;
    const auto* assignment = llvm::dyn_cast_or_null<clang::BinaryOperator>(stripExpr(*body->body_begin()));
    if (!assignment || assignment->getOpcode() != clang::BO_Assign ||
        !assignment->getType().getCanonicalType().getUnqualifiedType()->isSpecificBuiltinType(clang::BuiltinType::Float))
        return false;

    const auto* lhs = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(stripExpr(assignment->getLHS()));
    const auto* sum = llvm::dyn_cast_or_null<clang::BinaryOperator>(stripExpr(assignment->getRHS()));
    if (!lhs || !sum || sum->getOpcode() != clang::BO_Add ||
        !sum->getType().getCanonicalType().getUnqualifiedType()->isSpecificBuiltinType(clang::BuiltinType::Float))
        return false;
    const auto* paramDecl = lambda->getCallOperator()->getParamDecl(0);
    const auto* outRef = simpleArrayBase(lhs->getBase());
    const auto* lhsIndex = simpleIndexRef(lhs->getIdx(), paramDecl);
    if (!outRef || !lhsIndex)
        return false;

    const auto* addA = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(stripExpr(sum->getLHS()));
    const auto* addB = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(stripExpr(sum->getRHS()));
    if (!addA || !addB)
        return false;
    const auto* aRef = simpleArrayBase(addA->getBase());
    const auto* bRef = simpleArrayBase(addB->getBase());
    const auto* aIdx = simpleIndexRef(addA->getIdx(), paramDecl);
    const auto* bIdx = simpleIndexRef(addB->getIdx(), paramDecl);
    if (!aRef || !bRef || !aIdx || !bIdx ||
        !isFloatElementPointer(aRef->getType(), true) ||
        !isFloatElementPointer(bRef->getType(), true) ||
        !isFloatElementPointer(outRef->getType(), false))
        return false;

    const clang::Expr* range = stripExpr(parallelFor->getArg(0));
    std::string countStr;
    if (const auto* rangeCtor = llvm::dyn_cast_or_null<clang::CXXConstructExpr>(range)) {
        if (rangeCtor->getNumArgs() != 1)
            return false;
        clang::CharSourceRange countRange =
            clang::CharSourceRange::getTokenRange(rangeCtor->getArg(0)->getSourceRange());
        llvm::StringRef countText = clang::Lexer::getSourceText(countRange, SM, LangOpts);
        if (countText.empty())
            return false;
        if (rangeCtor->getConstructor()->isCopyOrMoveConstructor())
            countStr = "(" + countText.str() + ")[0]";
        else
            countStr = countText.str();
    } else if (const auto* rangeRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(range)) {
        countStr = "(" + rangeRef->getDecl()->getNameAsString() + ")[0]";
    } else {
        return false;
    }

    count = countStr;
    inputA = aRef->getDecl()->getNameAsString();
    inputB = bRef->getDecl()->getNameAsString();
    output = outRef->getDecl()->getNameAsString();
    return !inputA.empty() && !inputB.empty() && !output.empty();
}

} // namespace

static bool isAssignedToSyclEvent(const clang::CallExpr* CE, const clang::SourceManager& SM) {
    if (!CE)
        return false;

    clang::SourceLocation beginLoc = CE->getBeginLoc();
    if (beginLoc.isInvalid())
        return false;

    clang::FileID fileID = SM.getFileID(beginLoc);
    unsigned offset = SM.getFileOffset(beginLoc);

    llvm::StringRef code = SM.getBufferData(fileID);
    if (offset >= code.size())
        return false;

    size_t lineStart = code.rfind('\n', offset);
    if (lineStart == llvm::StringRef::npos)
        lineStart = 0;
    else
        lineStart++;

    llvm::StringRef beforeCall = code.slice(lineStart, offset);

    return beforeCall.contains("sycl::event");
}

void SubmitFunctionCallback::run(const clang::ast_matchers::MatchFinder::MatchResult& result) {
    if (const auto* waitCall =
            result.Nodes.getNodeAs<clang::CXXMemberCallExpr>("submit-result-wait")) {
        const auto* submitCall =
            result.Nodes.getNodeAs<clang::CXXMemberCallExpr>("submit-result-call");
        const bool gpuTarget = std::find(bkend_target.begin(), bkend_target.end(), "cuda") !=
                                   bkend_target.end() ||
                               std::find(bkend_target.begin(), bkend_target.end(), "hip") !=
                                   bkend_target.end();
        if (!gpuTarget || !submitCall || !result.SourceManager)
            return;

        const clang::SourceManager& SM = *result.SourceManager;
        const clang::Expr* queueExpr = submitCall->getImplicitObjectArgument();
        const clang::CXXMethodDecl* waitFn = waitCall->getMethodDecl();
        if (!queueExpr || queueExpr->isImplicitCXXThis() || !waitFn)
            return;

        const clang::SourceRange waitRange = waitCall->getSourceRange();
        const clang::SourceRange submitRange = submitCall->getSourceRange();
        const clang::SourceRange queueRange = queueExpr->getSourceRange();
        for (clang::SourceLocation L : {waitRange.getBegin(), waitRange.getEnd(),
                                        submitRange.getEnd(), queueRange.getEnd()})
            if (L.isInvalid() || L.isMacroID() || SM.isInSystemHeader(L))
                return;
        if (!SM.isWrittenInSameFile(waitRange.getBegin(), waitRange.getEnd()))
            return;

        for (const PendingSubmitWait& P : pendingSubmitWaits)
            if (P.waitRange == waitRange)
                return;
        pendingSubmitWaits.push_back({waitRange, submitRange, queueRange,
                                      waitFn->getNameAsString()});
        return;
    }

    std::string tpool_func_Call = "";

    if (const clang::TemplateArgument* DimArg =
            result.Nodes.getNodeAs<clang::TemplateArgument>("rangeDim")) {
        unsigned N = DimArg->getAsIntegral().getExtValue();

        if (N == 1)
            tpool_func_Call = "spawn_1D";
        else
            tpool_func_Call = "spawn_ND";
    }

    if (const clang::CallExpr* CE =
            result.Nodes.getNodeAs<clang::CallExpr>("submit-with-parallelfor")) {
        parasLog() << "[ParaS] matched submit-with-parallelfor at ";
        CE->getBeginLoc().print(parasLog(), *result.SourceManager);
        parasLog() << "\n";

        auto& SM = result.SourceManager;

        clang::SourceLocation spellingLoc = SM->getSpellingLoc(CE->getBeginLoc());
        std::string callSiteKey = spellingLoc.printToString(*SM);
        if (ProcessedCallSites.count(callSiteKey))
            return;

        const clang::Expr* Args = CE->getArg(0);
        auto langopts = result.Context->getLangOpts();

        clang::CharSourceRange args_range =
            clang::CharSourceRange::getTokenRange(Args->getSourceRange());

        llvm::StringRef kernel_function_text =
            clang::Lexer::getSourceText(args_range, *SM, langopts);

        if (kernel_function_text.empty())
            return;

        const clang::MemberExpr* ME = result.Nodes.getNodeAs<clang::MemberExpr>("memberexpr-pf");

        if (!ME)
            return;

        clang::Expr* base = ME->getBase();
        if (!base)
            return;

        clang::CharSourceRange baseRange =
            clang::CharSourceRange::getTokenRange(base->getSourceRange());

        llvm::StringRef objectName = clang::Lexer::getSourceText(baseRange, *SM, langopts);

        if (objectName.empty())
            return;

        const bool gaudiTarget = !bkend_target.empty() && bkend_target[0] == "gaudi";
        if (gaudiTarget) {
            std::string count, inputA, inputB, output;
            const auto* kernelCall =
                result.Nodes.getNodeAs<clang::CallExpr>("parallelforCall");
            const bool supported =
                isVectorAddKernel(kernelCall, count, inputA, inputB, output, *SM, langopts);
            std::string replacement;
            if (supported) {
                replacement = "::paras_gaudi::add_f32(" + count + ", " + inputA + ", " +
                              inputB + ", " + output + ")";
                parasLog() << "[ParaS] lowered Gaudi float vector addition to SynapseAI add_fwd_f32\n";
            } else {
                replacement = "::paras_gaudi::unsupported_kernel(" +
                              kernel_function_text.str() + ")";
                parasLog() << "[ParaS] Gaudi kernel is outside the v0.1 supported subset\n";
            }
            rewriter.ReplaceText(CE->getSourceRange(), replacement);
            ProcessedCallSites.insert(callSiteKey);
            HandleHeaders(*SM);
            return;
        }

        if (tpool_func_Call == "spawn_1D" && isAssignedToSyclEvent(CE, *SM)) {
            tpool_func_Call = "spawn_1D_event";
        }

        std::string new_text =
            objectName.str() + "." + tpool_func_Call + "(" + kernel_function_text.str() + ")";

        rewriter.ReplaceText(CE->getSourceRange(), new_text);
        ProcessedCallSites.insert(callSiteKey);
        HandleHeaders(*SM);
        return;
    }

    if (const clang::CallExpr* CE = result.Nodes.getNodeAs<clang::CallExpr>("submitCall")) {
        if (!CE || CE->getNumArgs() == 0)
            return;

        auto& SM = result.SourceManager;

        clang::SourceLocation spellingLoc = SM->getSpellingLoc(CE->getBeginLoc());
        std::string callSiteKey = spellingLoc.printToString(*SM);
        if (ProcessedCallSites.count(callSiteKey))
            return;

        const clang::MemberExpr* ME = result.Nodes.getNodeAs<clang::MemberExpr>("memberexpr");

        if (!ME)
            return;

        const clang::Expr* base = ME->getBase();
        if (!base)
            return;

        const clang::Expr* Args = CE->getArg(0);
        auto langopts = result.Context->getLangOpts();

        clang::CharSourceRange args_range =
            clang::CharSourceRange::getTokenRange(Args->getSourceRange());

        llvm::StringRef lambda_function_text =
            clang::Lexer::getSourceText(args_range, *SM, langopts);

        if (lambda_function_text.empty())
            return;

        if (!bkend_target.empty() && bkend_target[0] == "gaudi") {
            std::string replacement = "::paras_gaudi::unsupported_kernel(" +
                                      lambda_function_text.str() + ")";
            rewriter.ReplaceText(CE->getSourceRange(), replacement);
            ProcessedCallSites.insert(callSiteKey);
            HandleHeaders(*SM);
            return;
        }

        clang::CharSourceRange baseRange =
            clang::CharSourceRange::getTokenRange(base->getSourceRange());

        llvm::StringRef objectName = clang::Lexer::getSourceText(baseRange, *SM, langopts);

        if (objectName.empty())
            return;

        std::string submit_func = isAssignedToSyclEvent(CE, *SM) ? "spawn_1D_event" : "spawn_1D";

        std::string new_text =
            objectName.str() + "." + submit_func + "(" + lambda_function_text.str() + ")";

        rewriter.ReplaceText(CE->getSourceRange(), new_text);
        ProcessedCallSites.insert(callSiteKey);
        HandleHeaders(*SM);
    }
}

void SubmitFunctionCallback::onEndOfTranslationUnit() {

    for (const PendingSubmitWait& P : pendingSubmitWaits) {
        const std::string submitText = rewriter.getRewrittenText(P.submitRange);
        const std::string queueText = rewriter.getRewrittenText(P.queueRange);
        if (submitText.empty() || queueText.empty())
            continue;
        const std::string replacement =
            "(" + submitText + ", " + queueText + "." + P.waitName + "())";
        if (rewriter.ReplaceText(P.waitRange, replacement)) {
            parasLog() << "[ParaS] failed to rewrite " << P.waitName
                         << "() on submit() result\n";
            continue;
        }
        parasLog() << "[ParaS] " << P.waitName
                   << "() on submit() result rewritten to queue-level wait at "
                   << P.waitRange.getBegin().printToString(rewriter.getSourceMgr()) << "\n";
    }
    pendingSubmitWaits.clear();
}

void SubmitFunctionCallback::HandleHeaders(const clang::SourceManager& SM) {
    clang::FileID mainFileID = SM.getMainFileID();
    auto entry = SM.getFileEntryRefForID(mainFileID);

    if (!entry)
        return;

    llvm::StringRef filepath = entry->getName();
    const clang::FileEntry& FE = entry->getFileEntry();
    llvm::StringRef fullpath = FE.tryGetRealPathName();

    if (!fullpath.empty())
        filepath = fullpath;

    std::string filepathStr = filepath.str();

    if (ProcessedFiles.count(filepathStr))
        return;

    ProcessedFiles.insert(filepathStr);

    std::string new_header = "";

    for (std::string each_bkend_target : bkend_target) {
        if (each_bkend_target == "gaudi") {
            new_header = "#include <kem_gaudi/synapse_runtime.hpp>";
            llvm::StringRef code = SM.getBufferData(mainFileID);
            if (code.contains(new_header) ||
                code.contains("#include \"kem_gaudi/synapse_runtime.hpp\""))
                return;
            size_t insertPosition = code.rfind("#include");
            clang::SourceLocation insertlocation = SM.getLocForStartOfFile(mainFileID);
            if (insertPosition != llvm::StringRef::npos) {
                unsigned line = 1;
                for (size_t i = 0; i < insertPosition; i++)
                    if (code[i] == '\n')
                        line++;
                insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
            }
            rewriter.InsertText(insertlocation, new_header + "\n");
            return;
        }
        if (each_bkend_target == "cuda") {
            new_header = "#include <kem_gpu/gpu_threadpool.hpp>";
            llvm::StringRef code = SM.getBufferData(mainFileID);

            if (code.contains(new_header) ||
                code.contains("#include \"kem_gpu/gpu_threadpool.hpp\""))
                return;

            size_t insertPosition = code.rfind("#include");
            clang::SourceLocation insertlocation = SM.getLocForStartOfFile(mainFileID);

            if (insertPosition != llvm::StringRef::npos) {
                unsigned line = 1;

                for (size_t i = 0; i < insertPosition; i++)
                    if (code[i] == '\n')
                        line++;

                insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
            }

            rewriter.InsertText(insertlocation, "#include <kem_gpu/gpu_threadpool.hpp>\n");

            return;
        } else if (each_bkend_target == "hip") {
            new_header = "#include <kem_gpu/gpu_threadpool.hpp>";
            llvm::StringRef code = SM.getBufferData(mainFileID);

            if (code.contains(new_header) ||
                code.contains("#include \"kem_gpu/gpu_threadpool.hpp\""))
                return;

            size_t insertPosition = code.rfind("#include");
            clang::SourceLocation insertlocation = SM.getLocForStartOfFile(mainFileID);

            if (insertPosition != llvm::StringRef::npos) {
                unsigned line = 1;

                for (size_t i = 0; i < insertPosition; i++)
                    if (code[i] == '\n')
                        line++;

                insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
            }

            rewriter.InsertText(insertlocation, "#include <kem_gpu/gpu_threadpool.hpp>\n");

            return;
        }
    }

    llvm::StringRef code = SM.getBufferData(mainFileID);

    if (code.contains("#include <kem/threadpool.hpp>") ||
        code.contains("#include \"kem/threadpool.hpp\""))
        return;

    size_t insertPosition = code.rfind("#include");
    clang::SourceLocation insertlocation = SM.getLocForStartOfFile(mainFileID);

    if (insertPosition != llvm::StringRef::npos) {
        unsigned line = 1;

        for (size_t i = 0; i < insertPosition; i++)
            if (code[i] == '\n')
                line++;

        insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
    }

    rewriter.InsertText(insertlocation, "#include <kem/threadpool.hpp>\n");
}
