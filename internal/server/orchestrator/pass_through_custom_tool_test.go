package orchestrator

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func customToolNames(names ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(names))
	for _, name := range names {
		set[name] = struct{}{}
	}

	return set
}

func TestRewritePassThroughResponseBody(t *testing.T) {
	body := []byte(`{
		"id": "resp-1",
		"output": [
			{"type": "function_call", "call_id": "call_1", "name": "apply_patch", "arguments": "{\"patch\":\"*** Begin Patch\\n*** End Patch\"}"},
			{"type": "function_call", "call_id": "call_2", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"}
		]
	}`)

	rewritten, err := rewritePassThroughResponseBody(body, customToolNames("apply_patch"))
	require.NoError(t, err)
	require.NotNil(t, rewritten)

	var payload struct {
		Output []struct {
			Type      string `json:"type"`
			Name      string `json:"name"`
			Input     string `json:"input"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(rewritten, &payload))
	require.Len(t, payload.Output, 2)

	require.Equal(t, "custom_tool_call", payload.Output[0].Type)
	require.Equal(t, "apply_patch", payload.Output[0].Name)
	require.Equal(t, "*** Begin Patch\n*** End Patch", payload.Output[0].Input)
	require.Empty(t, payload.Output[0].Arguments)

	require.Equal(t, "function_call", payload.Output[1].Type)
	require.Equal(t, "exec_command", payload.Output[1].Name)
	require.Equal(t, `{"cmd":"ls"}`, payload.Output[1].Arguments)
}

func TestRewritePassThroughResponseBodyNoCustomTools(t *testing.T) {
	body := []byte(`{"id":"resp-1","output":[{"type":"function_call","name":"exec_command","arguments":"{}"}]}`)

	rewritten, err := rewritePassThroughResponseBody(body, customToolNames("apply_patch"))
	require.NoError(t, err)
	require.Nil(t, rewritten)
}

func TestCustomToolStreamRewriter(t *testing.T) {
	rewriter := newCustomToolStreamRewriter(customToolNames("apply_patch"))

	events := []*httpclient.StreamEvent{
		{Type: "event", Data: []byte(`{"type":"response.created","response":{"id":"resp-1","model":"qwen3","status":"in_progress"}}`)},
		{Type: "event", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"apply_patch","arguments":""}}`)},
		{Type: "event", Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"patch\":\"*** Begin"}`)},
		{Type: "event", Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":" Patch...\"}"}`)},
		{Type: "event", Data: []byte(`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"name":"apply_patch","arguments":"{\"patch\":\"*** Begin Patch...\"}"}`)},
		{Type: "event", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"apply_patch","arguments":"{\"patch\":\"*** Begin Patch...\"}"}}`)},
		{Type: "event", Data: []byte(`{"type":"response.completed","response":{"id":"resp-1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"apply_patch","arguments":"{\"patch\":\"*** Begin Patch...\"}"}]}}`)},
		{Type: "event", Data: []byte(`[DONE]`)},
	}

	var forwarded []string
	for _, event := range events {
		rewritten, err := rewriter.rewrite(event)
		require.NoError(t, err)

		for _, ev := range rewritten {
			forwarded = append(forwarded, string(ev.Data))
		}
	}

	require.Len(t, forwarded, 6)

	var added struct {
		Type string `json:"type"`
		Item struct {
			Type  string `json:"type"`
			Name  string `json:"name"`
			Input string `json:"input"`
		} `json:"item"`
	}
	require.NoError(t, json.Unmarshal([]byte(forwarded[1]), &added))
	require.Equal(t, "response.output_item.added", added.Type)
	require.Equal(t, "custom_tool_call", added.Item.Type)
	require.Equal(t, "apply_patch", added.Item.Name)

	// Argument deltas are buffered and replaced by a single input.done + item.done.
	var inputDone struct {
		Type string `json:"type"`
		Item string `json:"item_id"`
		In   string `json:"input"`
	}
	require.NoError(t, json.Unmarshal([]byte(forwarded[2]), &inputDone))
	require.Equal(t, "response.custom_tool_call_input.done", inputDone.Type)
	require.Equal(t, "fc_1", inputDone.Item)
	require.Equal(t, "*** Begin Patch...", inputDone.In)

	var itemDone struct {
		Type string `json:"type"`
		Item struct {
			Type  string `json:"type"`
			Name  string `json:"name"`
			Input string `json:"input"`
		} `json:"item"`
	}
	require.NoError(t, json.Unmarshal([]byte(forwarded[3]), &itemDone))
	require.Equal(t, "response.output_item.done", itemDone.Type)
	require.Equal(t, "custom_tool_call", itemDone.Item.Type)
	require.Equal(t, "*** Begin Patch...", itemDone.Item.Input)

	// The original function_call output_item.done is dropped.
	require.Equal(t, "response.completed", mustEventType(t, forwarded[4]))

	var completed struct {
		Type     string `json:"type"`
		Response struct {
			Output []struct {
				Type      string `json:"type"`
				Name      string `json:"name"`
				Input     string `json:"input"`
				Arguments string `json:"arguments"`
			} `json:"output"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal([]byte(forwarded[4]), &completed))
	require.Equal(t, "custom_tool_call", completed.Response.Output[0].Type)
	require.Equal(t, "*** Begin Patch...", completed.Response.Output[0].Input)
	require.Empty(t, completed.Response.Output[0].Arguments)

	require.Equal(t, "[DONE]", forwarded[5])
}

func TestCustomToolStreamRewriterLeavesOrdinaryFunctionCallsUntouched(t *testing.T) {
	rewriter := newCustomToolStreamRewriter(customToolNames("apply_patch"))

	events := []*httpclient.StreamEvent{
		{Type: "event", Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_2","type":"function_call","call_id":"call_2","name":"exec_command","arguments":""}}`)},
		{Type: "event", Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_2","output_index":0,"delta":"{\"cmd\":"}`)},
		{Type: "event", Data: []byte(`{"type":"response.function_call_arguments.done","item_id":"fc_2","output_index":0,"name":"exec_command","arguments":"{\"cmd\":\"ls\"}"}`)},
		{Type: "event", Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_2","type":"function_call","call_id":"call_2","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"}}`)},
	}

	var forwarded []string
	for _, event := range events {
		rewritten, err := rewriter.rewrite(event)
		require.NoError(t, err)

		for _, ev := range rewritten {
			forwarded = append(forwarded, string(ev.Data))
		}
	}

	require.Len(t, forwarded, len(events))
	for i, raw := range forwarded {
		require.JSONEq(t, string(events[i].Data), raw)
	}
}

func mustEventType(t *testing.T, raw string) string {
	t.Helper()

	var event struct {
		Type string `json:"type"`
	}
	require.NoError(t, json.Unmarshal([]byte(raw), &event))

	return event.Type
}

func TestResponsesCustomToolNames(t *testing.T) {
	req := &llm.Request{
		Tools: []llm.Tool{
			{Type: llm.ToolTypeFunction, Function: llm.Function{Name: "exec_command"}},
			{Type: llm.ToolTypeResponsesCustomTool, ResponseCustomTool: &llm.ResponseCustomTool{Name: "apply_patch"}},
			{Type: llm.ToolTypeResponsesCustomTool, ResponseCustomTool: &llm.ResponseCustomTool{Name: ""}},
		},
	}

	require.Equal(t, customToolNames("apply_patch"), responsesCustomToolNames(req))
	require.Empty(t, responsesCustomToolNames(nil))
}
