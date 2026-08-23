package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

// TestCustomToolStreamRoundTrip_OllamaFunctionCall covers the real Codex ->
// AxonHub (pass-through) -> Ollama path: Ollama reports the custom tool as a
// function_call with JSON object arguments, and the Responses transformer must
// restore it to a custom_tool_call with the bare freeform input.
func TestCustomToolStreamRoundTrip_OllamaFunctionCall(t *testing.T) {
	outbound, err := NewOutboundTransformer("https://ollama.example", "test-api-key")
	require.NoError(t, err)

	req := &httpclient.Request{
		TransformerMetadata: map[string]any{
			llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
		},
	}

	upstreamEvents := []*httpclient.StreamEvent{
		{Data: []byte(`{"type":"response.created","response":{"id":"resp_ollama_1","object":"response","created_at":1700000000,"model":"ollama-qwen3","status":"in_progress","output":[]}}`)},
		{Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"apply_patch","arguments":""}}`)},
		{Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"patch\":\"*** Begin"}`)},
		{Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":" Patch...\"}"}`)},
		{Data: []byte(`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"name":"apply_patch","arguments":"{\"patch\":\"*** Begin Patch...\"}"}`)},
		{Data: []byte(`{"type":"response.completed","response":{"id":"resp_ollama_1","object":"response","created_at":1700000000,"model":"ollama-qwen3","status":"completed","output":[]}}`)},
	}

	llmStream, err := outbound.TransformStream(t.Context(), req, streams.SliceStream(upstreamEvents))
	require.NoError(t, err)

	clientStream, err := NewInboundTransformer().TransformStream(t.Context(), llmStream)
	require.NoError(t, err)

	clientEvents, err := streams.All(clientStream)
	require.NoError(t, err)

	var added *StreamEvent
	var inputDone *StreamEvent
	var itemDone *StreamEvent
	for _, clientEvent := range clientEvents {
		if string(clientEvent.Data) == "[DONE]" {
			continue
		}

		var event StreamEvent
		require.NoError(t, json.Unmarshal(clientEvent.Data, &event))

		switch event.Type {
		case StreamEventTypeOutputItemAdded:
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				added = &event
			}
		case StreamEventTypeCustomToolCallInputDone:
			inputDone = &event
		case StreamEventTypeOutputItemDone:
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				itemDone = &event
			}
		}
	}

	require.NotNil(t, added, "expected a custom_tool_call output_item.added event")
	require.Equal(t, "custom_tool_call", added.Item.Type)
	require.Equal(t, "apply_patch", added.Item.Name)
	require.Equal(t, "call_1", added.Item.CallID)

	require.NotNil(t, inputDone, "expected a custom_tool_call_input.done event")
	require.Equal(t, "*** Begin Patch...", inputDone.Input)

	require.NotNil(t, itemDone, "expected a custom_tool_call output_item.done event")
	require.Equal(t, "custom_tool_call", itemDone.Item.Type)
	require.Equal(t, "apply_patch", itemDone.Item.Name)
	require.Equal(t, "*** Begin Patch...", *itemDone.Item.Input)
}
