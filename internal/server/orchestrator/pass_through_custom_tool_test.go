package orchestrator

import (
	"context"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestHasResponsesCustomTools(t *testing.T) {
	require.False(t, hasResponsesCustomTools(nil))
	require.False(t, hasResponsesCustomTools(&llm.Request{}))

	req := &llm.Request{
		Tools: []llm.Tool{
			{Type: llm.ToolTypeFunction, Function: llm.Function{Name: "exec_command"}},
		},
	}
	require.False(t, hasResponsesCustomTools(req))

	req.Tools = append(req.Tools, llm.Tool{
		Type:               llm.ToolTypeResponsesCustomTool,
		ResponseCustomTool: &llm.ResponseCustomTool{Name: "apply_patch"},
	})
	require.True(t, hasResponsesCustomTools(req))
}

func TestPassThrough_KeepsRequestBodyButSkipsResponseForCustomTools(t *testing.T) {
	ctx := context.Background()
	channel := &biz.Channel{
		Channel: &ent.Channel{
			ID:   1,
			Name: "test",
			Settings: &objects.ChannelSettings{
				PassThroughBody: lo.ToPtr(true),
			},
		},
	}
	state := &PersistenceState{
		CurrentCandidate: &ChannelModelsCandidate{Channel: channel},
		LlmRequest: &llm.Request{
			APIFormat: llm.APIFormatOpenAIResponse,
			Tools: []llm.Tool{
				{
					Type:               llm.ToolTypeResponsesCustomTool,
					ResponseCustomTool: &llm.ResponseCustomTool{Name: "apply_patch"},
				},
			},
		},
		RawProviderRequest: &httpclient.Request{
			APIFormat: string(llm.APIFormatOpenAIResponse),
		},
	}
	outbound := &PersistentOutboundTransformer{state: state}

	// Request-side pass-through stays enabled so Ollama receives the raw body.
	require.True(t, outbound.isPassThroughEnabled(ctx, nil))

	// Response-side pass-through is disabled so the transform chain can restore
	// custom tool calls.
	require.False(t, outbound.isPassThroughResponseEnabled(ctx, nil))

	mw := applyPassThroughResponse(outbound, nil)
	transformed := &httpclient.Response{StatusCode: 200, Body: []byte("transformed")}
	state.RawProviderResponse = &httpclient.Response{StatusCode: 200, Body: []byte("raw")}

	result, err := mw.OnInboundRawResponse(ctx, transformed)
	require.NoError(t, err)
	assert.Same(t, transformed, result)
}
