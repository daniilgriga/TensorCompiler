#include <stdexcept>

#include <gtest/gtest.h>

#include "graph/graph_builder.hpp"

namespace
{

TEST(GraphBuilderSmoke, GetOrCreateValueIsIdempotent)
{
    tc::GraphBuilder builder;

    tc::Value* v1 = builder.get_or_create_value("x");
    tc::Value* v2 = builder.get_or_create_value("x");
    tc::Value* missing = builder.find_value("missing");

    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(missing, nullptr);
}

TEST(GraphBuilderSmoke, MarkAsListsDoNotDuplicate)
{
    tc::GraphBuilder builder;
    tc::Value* v = builder.get_or_create_value("input");

    builder.mark_as_input(v);
    builder.mark_as_input(v);
    builder.mark_as_output(v);
    builder.mark_as_output(v);
    builder.mark_as_initializer(v);
    builder.mark_as_initializer(v);

    EXPECT_TRUE(v->is_graph_input());
    EXPECT_TRUE(v->is_graph_output());
    EXPECT_TRUE(v->is_initializer());
    EXPECT_EQ(builder.graph_inputs().size(), 1u);
    EXPECT_EQ(builder.graph_outputs().size(), 1u);
    EXPECT_EQ(builder.graph_initializers().size(), 1u);
}

TEST(GraphBuilderSmoke, AddNodeLinksProducerAndConsumers)
{
    tc::GraphBuilder builder;
    tc::Value* in = builder.get_or_create_value("x");
    tc::Value* out = builder.get_or_create_value("y");

    tc::Node* node = builder.add_node("Relu", {in}, {out}, {}, "");

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(out->producer(), node);
    ASSERT_EQ(in->consumers().size(), 1u);
    EXPECT_EQ(in->consumers().front(), node);
    EXPECT_EQ(node->name(), "n0");
}

TEST(GraphBuilderSmoke, AddNodeThrowsOnInvalidTopology)
{
    tc::GraphBuilder builder;
    tc::Value* in = builder.get_or_create_value("x");
    tc::Value* out = builder.get_or_create_value("y");

    EXPECT_THROW((void)builder.add_node("Relu", {nullptr}, {out}), std::runtime_error);

    (void)builder.add_node("Relu", {in}, {out});
    EXPECT_THROW((void)builder.add_node("Identity", {in}, {out}), std::runtime_error);
}

}  // namespace
