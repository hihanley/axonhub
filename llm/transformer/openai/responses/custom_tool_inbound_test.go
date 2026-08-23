package responses

import (
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

func TestConvertToResponsesAPIResponse_RestoresCustomToolCall(t *testing.T) {
	chatResp := &llm.Response{
		ID:    "resp-1",
		Model: "qwen3",
		TransformerMetadata: map[string]any{
			llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
		},
		Choices: []llm.Choice{
			{
				Message: &llm.Message{
					Role: "assistant",
					ToolCalls: []llm.ToolCall{
						{
							ID:   "call-1",
							Type: "function",
							Function: llm.FunctionCall{
								Name:      "apply_patch",
								Arguments: `{"patch":"*** Begin Patch"}`,
							},
							Index: 0,
						},
						{
							ID:   "call-2",
							Type: "function",
							Function: llm.FunctionCall{
								Name:      "exec_command",
								Arguments: `{"cmd":"ls"}`,
							},
							Index: 1,
						},
					},
				},
			},
		},
	}

	resp := convertToResponsesAPIResponse(chatResp)
	require.Len(t, resp.Output, 2)

	require.Equal(t, "custom_tool_call", resp.Output[0].Type)
	require.Equal(t, "apply_patch", resp.Output[0].Name)
	require.Equal(t, lo.ToPtr("*** Begin Patch"), resp.Output[0].Input)

	require.Equal(t, "function_call", resp.Output[1].Type)
	require.Equal(t, "exec_command", resp.Output[1].Name)
}

func TestInboundStream_RestoresCustomToolCall(t *testing.T) {
	trans := NewInboundTransformer()

	chunks := []*llm.Response{
		{
			ID:    "resp-1",
			Model: "qwen3",
			TransformerMetadata: map[string]any{
				llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
			},
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ollama-qwen3-0",
								Type:  "function",
								Index: 0,
								Function: llm.FunctionCall{
									Name:      "apply_patch",
									Arguments: `{"input":"*** Be`,
								},
							},
						},
					},
				},
			},
		},
		{
			ID:    "resp-1",
			Model: "qwen3",
			TransformerMetadata: map[string]any{
				llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
			},
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ollama-qwen3-0",
								Type:  "function",
								Index: 0,
								Function: llm.FunctionCall{
									Arguments: `gin Patch..."}`,
								},
							},
						},
					},
				},
			},
		},
		{
			ID:    "resp-1",
			Model: "qwen3",
			Choices: []llm.Choice{
				{
					Index:        0,
					FinishReason: lo.ToPtr("tool_calls"),
				},
			},
		},
		{
			ID:    "resp-1",
			Model: "qwen3",
			Usage: &llm.Usage{
				PromptTokens:     1,
				CompletionTokens: 1,
				TotalTokens:      2,
			},
		},
	}

	stream, err := trans.TransformStream(t.Context(), streams.SliceStream(chunks))
	require.NoError(t, err)

	var events []StreamEvent
	for stream.Next() {
		var ev StreamEvent
		require.NoError(t, json.Unmarshal(stream.Current().Data, &ev))
		events = append(events, ev)
	}
	require.NoError(t, stream.Err())

	var added *StreamEvent
	var done *StreamEvent
	for i := range events {
		switch events[i].Type {
		case StreamEventTypeOutputItemAdded:
			if events[i].Item != nil && events[i].Item.Type == "custom_tool_call" {
				added = &events[i]
			}
		case StreamEventTypeCustomToolCallInputDone:
			done = &events[i]
		}
	}

	require.NotNil(t, added, "expected a custom_tool_call output_item.added event")
	require.Equal(t, "apply_patch", added.Item.Name)
	require.NotNil(t, done, "expected a custom_tool_call_input.done event")
	require.Equal(t, "*** Begin Patch...", done.Input)
}
