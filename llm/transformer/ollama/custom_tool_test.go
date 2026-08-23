package ollama

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

func TestTransformRequestRecordsCustomToolNamesMetadata(t *testing.T) {
	transformer, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL: "http://localhost:11434",
	})
	require.NoError(t, err)

	req, err := transformer.TransformRequest(context.Background(), &llm.Request{
		Model: "qwen3",
		Tools: []llm.Tool{
			{
				Type: llm.ToolTypeResponsesCustomTool,
				ResponseCustomTool: &llm.ResponseCustomTool{
					Name:        "apply_patch",
					Description: "Edit files with a freeform patch",
				},
			},
			{
				Type: llm.ToolTypeFunction,
				Function: llm.Function{
					Name:       "exec_command",
					Parameters: json.RawMessage(`{"type":"object"}`),
				},
			},
		},
		Messages: []llm.Message{
			{
				Role:    "user",
				Content: llm.MessageContent{Content: lo.ToPtr("hi")},
			},
		},
	})
	require.NoError(t, err)

	names, ok := req.TransformerMetadata[llm.TransformerMetadataKeyCustomToolNames].([]string)
	require.True(t, ok)
	require.Equal(t, []string{"apply_patch"}, names)

	// Custom tools are not advertised in the chat body: with pass-through enabled the
	// raw inbound body (which Ollama understands natively) replaces this body.
	var got ChatRequest
	require.NoError(t, json.Unmarshal(req.Body, &got))
	require.Len(t, got.Tools, 1)
	require.Equal(t, "exec_command", got.Tools[0].Function.Name)
}

func TestTransformResponseCopiesTransformerMetadata(t *testing.T) {
	transformer, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL: "http://localhost:11434",
	})
	require.NoError(t, err)

	resp, err := transformer.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: 200,
		Body:       []byte(`{"model":"qwen3","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`),
		Request: &httpclient.Request{
			TransformerMetadata: map[string]any{
				llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
			},
		},
	})
	require.NoError(t, err)

	names, ok := resp.TransformerMetadata[llm.TransformerMetadataKeyCustomToolNames].([]string)
	require.True(t, ok)
	require.Equal(t, []string{"apply_patch"}, names)
}

func TestTransformStreamAttachesTransformerMetadata(t *testing.T) {
	transformer, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL: "http://localhost:11434",
	})
	require.NoError(t, err)

	stream, err := transformer.TransformStream(context.Background(), &httpclient.Request{
		TransformerMetadata: map[string]any{
			llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch"},
		},
	}, streams.SliceStream([]*httpclient.StreamEvent{
		{Data: []byte(`{"model":"qwen3","message":{"role":"assistant","content":"hi"},"done":false}`)},
	}))
	require.NoError(t, err)

	require.True(t, stream.Next())
	chunk := stream.Current()
	require.NotNil(t, chunk)
	names, ok := chunk.TransformerMetadata[llm.TransformerMetadataKeyCustomToolNames].([]string)
	require.True(t, ok)
	require.Equal(t, []string{"apply_patch"}, names)
}
