package responses

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestFreeformInputFromArguments(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		expected string
		ok       bool
	}{
		{name: "input key", raw: `{"input":"*** Begin Patch"}`, expected: "*** Begin Patch", ok: true},
		{name: "patch key", raw: `{"patch":"*** Begin Patch"}`, expected: "*** Begin Patch", ok: true},
		{name: "content key", raw: `{"content":"hello"}`, expected: "hello", ok: true},
		{name: "text key", raw: `{"text":"hello"}`, expected: "hello", ok: true},
		{name: "json string", raw: `"*** Begin Patch"`, expected: "*** Begin Patch", ok: true},
		{name: "sole string property", raw: `{"custom":"hello"}`, expected: "hello", ok: true},
		{name: "ambiguous object", raw: `{"a":"x","b":"y"}`, ok: false},
		{name: "empty object", raw: `{}`, ok: false},
		{name: "invalid json", raw: `{not json`, ok: false},
		{name: "empty", raw: "", ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := freeformInputFromArguments(tt.raw)
			require.Equal(t, tt.ok, ok)
			require.Equal(t, tt.expected, got)
		})
	}
}

func TestToCustomToolCall(t *testing.T) {
	names := map[string]struct{}{"apply_patch": {}}

	tc := llm.ToolCall{
		ID:   "call-1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "apply_patch",
			Arguments: `{"patch":"*** Begin Patch"}`,
		},
		Index: 0,
	}

	restored, ok := toCustomToolCall(tc, names)
	require.True(t, ok)
	require.Equal(t, llm.ToolTypeResponsesCustomTool, restored.Type)
	require.NotNil(t, restored.ResponseCustomToolCall)
	require.Equal(t, "call-1", restored.ResponseCustomToolCall.CallID)
	require.Equal(t, "apply_patch", restored.ResponseCustomToolCall.Name)
	require.Equal(t, "*** Begin Patch", restored.ResponseCustomToolCall.Input)

	// A tool not declared as custom stays a function call.
	other := llm.ToolCall{
		ID:   "call-2",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "exec_command",
			Arguments: `{"cmd":"ls"}`,
		},
		Index: 1,
	}
	_, ok = toCustomToolCall(other, names)
	require.False(t, ok)

	// Ambiguous arguments cannot be unwrapped and keep the function call.
	bad := llm.ToolCall{
		ID:   "call-3",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "apply_patch",
			Arguments: `{"a":"x","b":"y"}`,
		},
		Index: 2,
	}
	_, ok = toCustomToolCall(bad, names)
	require.False(t, ok)
}

func TestCustomToolNamesFromMetadata(t *testing.T) {
	require.Empty(t, customToolNamesFromMetadata(nil))
	require.Empty(t, customToolNamesFromMetadata(map[string]any{}))

	names := customToolNamesFromMetadata(map[string]any{
		llm.TransformerMetadataKeyCustomToolNames: []string{"apply_patch", "", "other"},
	})
	require.Len(t, names, 2)
	require.True(t, isCustomTool("apply_patch", names))
	require.True(t, isCustomTool("other", names))
	require.False(t, isCustomTool("missing", names))

	// JSON round-trips may yield []any.
	names = customToolNamesFromMetadata(map[string]any{
		llm.TransformerMetadataKeyCustomToolNames: []any{"apply_patch"},
	})
	require.True(t, isCustomTool("apply_patch", names))
}
