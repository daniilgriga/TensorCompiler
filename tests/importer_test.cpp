#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "importer/importer.hpp"

#ifndef TEST_MODELS_DIR
#error "TEST_MODELS_DIR must be defined"
#endif

namespace
{

    class ImporterTest : public ::testing::Test
    {
    protected:
        static void SetUpTestSuite()
        {
            std::string path = std::string (TEST_MODELS_DIR) + "/conv_relu_gemm.onnx";
            builder_ = std::make_unique<tc::GraphBuilder> (tc::import_onnx (path));
        }

        static void TearDownTestSuite()
        {
            builder_.reset();
        }

        static const tc::Node* find_node (const std::string& name)
        {
            for (const auto& n : builder_->nodes())
                if (n->name() == name)
                    return n.get();

            return nullptr;
        }

        static std::unique_ptr<tc::GraphBuilder> builder_;
    };

    std::unique_ptr<tc::GraphBuilder> ImporterTest::builder_ = nullptr;

    // integration
    TEST_F (ImporterTest, NodeCount)
    {
        EXPECT_EQ (builder_->nodes().size(), 4u);
    }

    TEST_F (ImporterTest, KeyValuesExist)
    {
        EXPECT_NE (builder_->find_value ("X"),          nullptr);
        EXPECT_NE (builder_->find_value ("W_conv"),     nullptr);
        EXPECT_NE (builder_->find_value ("B_conv"),     nullptr);
        EXPECT_NE (builder_->find_value ("conv1_out"),  nullptr);
        EXPECT_NE (builder_->find_value ("relu1_out"),  nullptr);
        EXPECT_NE (builder_->find_value ("flat_shape"), nullptr);
        EXPECT_NE (builder_->find_value ("flat"),       nullptr);
        EXPECT_NE (builder_->find_value ("W_gemm"),     nullptr);
        EXPECT_NE (builder_->find_value ("B_gemm"),     nullptr);
        EXPECT_NE (builder_->find_value ("Y"),          nullptr);
    }

    TEST_F (ImporterTest, GraphInputs)
    {
        ASSERT_EQ (builder_->graph_inputs().size(), 1u);
        EXPECT_EQ (builder_->graph_inputs()[0]->name(), "X");
    }

    TEST_F (ImporterTest, GraphOutputs)
    {
        ASSERT_EQ (builder_->graph_outputs().size(), 1u);
        EXPECT_EQ (builder_->graph_outputs()[0]->name(), "Y");
    }

    TEST_F (ImporterTest, InitializerCount)
    {
        EXPECT_EQ (builder_->graph_initializers().size(), 5u);
    }

    TEST_F (ImporterTest, InitializersAreNotGraphInputs)
    {
        for (const auto* init : builder_->graph_initializers())
            EXPECT_FALSE (init->is_graph_input())
                << "initializer " << init->name() << " is also graph input";
    }

    // unit
    TEST_F (ImporterTest, NodeOpTypes)
    {
        EXPECT_EQ (find_node ("conv1")->op_type(), "Conv");
        EXPECT_EQ (find_node ("relu1")->op_type(), "Relu");
        EXPECT_EQ (find_node ("reshape1")->op_type(), "Reshape");
        EXPECT_EQ (find_node ("gemm1")->op_type(), "Gemm");
    }

    TEST_F (ImporterTest, ConvKernelShape)
    {
        const tc::Node* conv = find_node ("conv1");
        ASSERT_NE (conv, nullptr);

        const auto* ks = conv->attribute ("kernel_shape");
        ASSERT_NE (ks, nullptr);

        auto val = std::get<std::vector<int64_t>> (*ks);
        EXPECT_EQ (val, (std::vector<int64_t>{3, 3}));
    }

    TEST_F (ImporterTest, ConvStrides)
    {
        const tc::Node* conv = find_node ("conv1");
        ASSERT_NE (conv, nullptr);

        const auto* s = conv->attribute ("strides");
        ASSERT_NE (s, nullptr);

        auto val = std::get<std::vector<int64_t>> (*s);
        EXPECT_EQ (val, (std::vector<int64_t>{1, 1}));
    }

    TEST_F (ImporterTest, ConvPads)
    {
        const tc::Node* conv = find_node ("conv1");
        ASSERT_NE (conv, nullptr);

        const auto* p = conv->attribute ("pads");
        ASSERT_NE (p, nullptr);

        auto val = std::get<std::vector<int64_t>> (*p);
        EXPECT_EQ (val, (std::vector<int64_t>{0, 0, 0, 0}));
    }

    TEST_F (ImporterTest, GemmTransB)
    {
        const tc::Node* gemm = find_node ("gemm1");
        ASSERT_NE (gemm, nullptr);

        const auto* tb = gemm->attribute ("transB");
        ASSERT_NE (tb, nullptr);

        EXPECT_EQ (std::get<int64_t> (*tb), 1);
    }

    TEST_F (ImporterTest, GemmAlpha)
    {
        const tc::Node* gemm = find_node ("gemm1");
        ASSERT_NE (gemm, nullptr);

        const auto* a = gemm->attribute ("alpha");
        ASSERT_NE (a, nullptr);

        EXPECT_FLOAT_EQ (std::get<float> (*a), 1.0f);
    }

    TEST_F (ImporterTest, GemmBeta)
    {
        const tc::Node* gemm = find_node ("gemm1");
        ASSERT_NE (gemm, nullptr);

        const auto* b = gemm->attribute ("beta");
        ASSERT_NE (b, nullptr);

        EXPECT_FLOAT_EQ (std::get<float> (*b), 1.0f);
    }

    TEST_F (ImporterTest, ReluNoAttributes)
    {
        const tc::Node* relu = find_node ("relu1");
        ASSERT_NE (relu, nullptr);

        EXPECT_TRUE (relu->attributes().empty());
    }

    TEST_F (ImporterTest, ConvOutputIsReluInput)
    {
        const tc::Node* conv = find_node ("conv1");
        const tc::Node* relu = find_node ("relu1");
        ASSERT_NE (conv, nullptr);
        ASSERT_NE (relu, nullptr);

        ASSERT_FALSE (conv->outputs().empty());
        ASSERT_FALSE (relu->inputs().empty());

        EXPECT_EQ (conv->outputs()[0], relu->inputs()[0]);
        EXPECT_EQ (conv->outputs()[0]->name(), "conv1_out");

        EXPECT_EQ (conv->outputs()[0]->producer(), conv);

        const auto& consumers = conv->outputs()[0]->consumers();
        bool relu_is_consumer = std::find (
            consumers.begin(), consumers.end(), relu) != consumers.end();
        EXPECT_TRUE (relu_is_consumer);
    }

    TEST_F (ImporterTest, InitializerShape)
    {
        const tc::Value* w = builder_->find_value ("W_conv");
        ASSERT_NE (w, nullptr);

        EXPECT_EQ (w->shape(), (std::vector<int64_t>{1, 1, 3, 3}));
        EXPECT_TRUE (w->is_initializer());
    }

    // error handling
    TEST (ImporterErrorTest, ThrowsOnMissingFile)
    {
        EXPECT_THROW (tc::import_onnx ("nonexistent.onnx"), std::runtime_error);
    }

} // namespace
