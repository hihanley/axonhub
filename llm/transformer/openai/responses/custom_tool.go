package responses

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// customToolNamesFromMetadata extracts the custom (freeform) tool name set from
// transformer metadata. The value may be []string or, after JSON round-trips, []any.
func customToolNamesFromMetadata(metadata map[string]any) map[string]struct{} {
	names := map[string]struct{}{}
	if len(metadata) == 0 {
		return names
	}

	switch raw := metadata[llm.TransformerMetadataKeyCustomToolNames].(type) {
	case []string:
		for _, name := range raw {
			if name != "" {
				names[name] = struct{}{}
			}
		}
	case []any:
		for _, item := range raw {
			if name, ok := item.(string); ok && name != "" {
				names[name] = struct{}{}
			}
		}
	}
	return names
}

// isCustomTool reports whether name is a custom (freeform) tool declared in names.
func isCustomTool(name string, names map[string]struct{}) bool {
	_, ok := names[name]
	return ok
}

// freeformInputFromArguments unwraps chat-style JSON object arguments into the
// freeform input string expected by a custom tool. Upstreams such as Ollama wrap
// the input in an object such as {"input": "..."} or {"patch": "..."}.
func freeformInputFromArguments(raw string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}

	// Some upstreams return the freeform text directly as a JSON string.
	if strings.HasPrefix(raw, `"`) {
		var text string
		if err := json.Unmarshal([]byte(raw), &text); err == nil {
			return text, true
		}
	}

	// Plain text without any JSON wrapper (not an object or quoted string).
	if !strings.HasPrefix(raw, "{") && !strings.HasPrefix(raw, "[") {
		return raw, true
	}

	var obj map[string]any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil || len(obj) == 0 {
		return "", false
	}

	// Prefer the conventional keys used by OpenAI-compatible chat converters.
	for _, key := range []string{"input", "patch", "content", "text"} {
		if value, ok := obj[key].(string); ok {
			return decodeFreeformString(value), true
		}
	}

	// Fall back to the sole string property when the upstream used a custom key.
	var found string
	for _, value := range obj {
		text, ok := value.(string)
		if !ok || found != "" {
			return "", false // ambiguous; never guess
		}
		found = text
	}
	if found != "" {
		return decodeFreeformString(found), true
	}
	return "", false
}

// decodeFreeformString returns value, decoding one extra JSON string layer when the
// upstream double-encoded the freeform input (e.g. `"\"*** Begin Patch\""`).
func decodeFreeformString(value string) string {
	if !strings.HasPrefix(value, `"`) {
		return value
	}
	var decoded string
	if err := json.Unmarshal([]byte(value), &decoded); err == nil {
		return decoded
	}
	return value
}

// toCustomToolCall converts a chat-style function call into a custom tool call when
// the tool was originally declared as custom. ok is false when the call should stay
// a function call.
func toCustomToolCall(tc llm.ToolCall, names map[string]struct{}) (llm.ToolCall, bool) {
	if tc.Type != "function" || tc.ResponseCustomToolCall != nil {
		return tc, false
	}
	if !isCustomTool(tc.Function.Name, names) {
		return tc, false
	}
	input, ok := freeformInputFromArguments(tc.Function.Arguments)
	if !ok {
		return tc, false
	}

	name := tc.Function.Name
	tc.Type = llm.ToolTypeResponsesCustomTool
	tc.Function = llm.FunctionCall{}
	tc.ResponseCustomToolCall = &llm.ResponseCustomToolCall{
		CallID: tc.ID,
		Name:   name,
		Input:  input,
	}
	if slog.Default().Enabled(context.Background(), slog.LevelDebug) {
		slog.DebugContext(context.Background(), "restored custom tool call", slog.String("tool", name))
	}
	return tc, true
}
