package orchestrator

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

// responsesCustomToolNames returns the names of Responses custom (freeform) tools
// declared in the inbound request. Ollama accepts these tools but reports their
// calls as function calls, so pass-through responses are translated back before
// they reach Codex.
func responsesCustomToolNames(req *llm.Request) map[string]struct{} {
	names := make(map[string]struct{})
	if req == nil {
		return names
	}

	for _, tool := range req.Tools {
		if tool.Type == llm.ToolTypeResponsesCustomTool && tool.ResponseCustomTool != nil && tool.ResponseCustomTool.Name != "" {
			names[tool.ResponseCustomTool.Name] = struct{}{}
		}
	}

	return names
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

// freeformInput returns arguments as the freeform input, falling back to the raw
// arguments when they cannot be unwrapped.
func freeformInput(arguments string) string {
	if input, ok := freeformInputFromArguments(arguments); ok {
		return input
	}

	return arguments
}

// rewritePassThroughResponseBody rewrites function_call output items back to
// custom_tool_call when the tool was declared as custom. It returns nil when
// nothing changed.
func rewritePassThroughResponseBody(body []byte, names map[string]struct{}) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("parse pass-through response: %w", err)
	}

	output, ok := payload["output"].([]any)
	if !ok {
		return nil, nil
	}

	changed := false
	for _, itemAny := range output {
		item, ok := itemAny.(map[string]any)
		if !ok {
			continue
		}

		if rewritePassThroughOutputItem(item, names) {
			changed = true
		}
	}

	if !changed {
		return nil, nil
	}

	rewritten, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal rewritten pass-through response: %w", err)
	}

	return rewritten, nil
}

// rewritePassThroughOutputItem rewrites one function_call item to custom_tool_call
// when name is a declared custom tool. It reports whether the item changed.
func rewritePassThroughOutputItem(item map[string]any, names map[string]struct{}) bool {
	if item["type"] != "function_call" {
		return false
	}

	name, _ := item["name"].(string)
	if _, ok := names[name]; !ok {
		return false
	}

	arguments, _ := item["arguments"].(string)
	item["type"] = "custom_tool_call"
	delete(item, "arguments")
	item["input"] = freeformInput(arguments)

	return true
}

// customToolCallState tracks one in-flight custom tool call while its arguments
// arrive in streamed chunks.
type customToolCallState struct {
	name      string
	itemID    string
	callID    string
	arguments strings.Builder
	done      bool
}

// customToolStreamRewriter translates Ollama-style function_call SSE events back
// into custom_tool_call events for the Responses API.
type customToolStreamRewriter struct {
	names    map[string]struct{}
	byCallID map[string]*customToolCallState
	byItemID map[string]*customToolCallState
}

func newCustomToolStreamRewriter(names map[string]struct{}) *customToolStreamRewriter {
	return &customToolStreamRewriter{
		names:    names,
		byCallID: make(map[string]*customToolCallState),
		byItemID: make(map[string]*customToolCallState),
	}
}

// rewrite rewrites one Responses SSE event and returns the events to forward to the
// client. It can return no events while buffering argument deltas, one event for
// unchanged or rewritten events, or two events when a custom call completes.
func (r *customToolStreamRewriter) rewrite(event *httpclient.StreamEvent) ([]*httpclient.StreamEvent, error) {
	if event == nil || len(event.Data) == 0 || string(event.Data) == "[DONE]" {
		return []*httpclient.StreamEvent{event}, nil
	}

	var payload map[string]any
	if err := json.Unmarshal(event.Data, &payload); err != nil {
		return nil, fmt.Errorf("parse pass-through stream event: %w", err)
	}

	eventType, _ := payload["type"].(string)
	switch eventType {
	case "response.output_item.added":
		return r.rewriteOutputItemAdded(event, payload)
	case "response.function_call_arguments.delta":
		return r.rewriteArgumentsDelta(event, payload)
	case "response.function_call_arguments.done":
		return r.rewriteArgumentsDone(event, payload)
	case "response.output_item.done":
		return r.rewriteOutputItemDone(event, payload)
	case "response.completed":
		return r.rewriteCompleted(event, payload)
	default:
		return []*httpclient.StreamEvent{event}, nil
	}
}

func (r *customToolStreamRewriter) lookup(itemID string) *customToolCallState {
	if state, ok := r.byItemID[itemID]; ok {
		return state
	}

	return r.byCallID[itemID]
}

func (r *customToolStreamRewriter) rewriteOutputItemAdded(event *httpclient.StreamEvent, payload map[string]any) ([]*httpclient.StreamEvent, error) {
	item, ok := payload["item"].(map[string]any)
	if !ok || item["type"] != "function_call" {
		return []*httpclient.StreamEvent{event}, nil
	}

	name, _ := item["name"].(string)
	if _, ok := r.names[name]; !ok {
		return []*httpclient.StreamEvent{event}, nil
	}

	callID, _ := item["call_id"].(string)
	itemID, _ := item["id"].(string)
	if itemID == "" {
		itemID = callID
	}

	state := &customToolCallState{name: name, itemID: itemID, callID: callID}
	r.byCallID[callID] = state
	if itemID != "" {
		r.byItemID[itemID] = state
	}

	item["type"] = "custom_tool_call"
	delete(item, "arguments")
	item["input"] = ""

	rewritten, err := marshalEvent(event, payload)
	if err != nil {
		return nil, err
	}

	return []*httpclient.StreamEvent{rewritten}, nil
}

func (r *customToolStreamRewriter) rewriteArgumentsDelta(event *httpclient.StreamEvent, payload map[string]any) ([]*httpclient.StreamEvent, error) {
	itemID, _ := payload["item_id"].(string)
	state := r.lookup(itemID)
	if state == nil {
		return []*httpclient.StreamEvent{event}, nil
	}

	delta, _ := payload["delta"].(string)
	state.arguments.WriteString(delta)

	return nil, nil
}

func (r *customToolStreamRewriter) rewriteArgumentsDone(event *httpclient.StreamEvent, payload map[string]any) ([]*httpclient.StreamEvent, error) {
	itemID, _ := payload["item_id"].(string)
	state := r.lookup(itemID)
	if state == nil {
		return []*httpclient.StreamEvent{event}, nil
	}

	fullArguments := state.arguments.String()
	if arguments, _ := payload["arguments"].(string); arguments != "" {
		fullArguments = arguments
	}
	state.done = true

	return r.completeCustomCall(event, payload, state, fullArguments)
}

func (r *customToolStreamRewriter) rewriteOutputItemDone(event *httpclient.StreamEvent, payload map[string]any) ([]*httpclient.StreamEvent, error) {
	item, ok := payload["item"].(map[string]any)
	if !ok {
		return []*httpclient.StreamEvent{event}, nil
	}

	callID, _ := item["call_id"].(string)
	itemID, _ := item["id"].(string)
	state := r.lookup(callID)
	if state == nil && itemID != "" {
		state = r.lookup(itemID)
	}
	if state == nil {
		return []*httpclient.StreamEvent{event}, nil
	}

	if state.done {
		// Already replaced by the custom output_item.done emitted from the
		// arguments.done handler; drop the original function_call done event.
		return nil, nil
	}

	arguments, _ := item["arguments"].(string)
	state.done = true

	return r.completeCustomCall(event, payload, state, arguments)
}

func (r *customToolStreamRewriter) rewriteCompleted(event *httpclient.StreamEvent, payload map[string]any) ([]*httpclient.StreamEvent, error) {
	response, ok := payload["response"].(map[string]any)
	if !ok {
		return []*httpclient.StreamEvent{event}, nil
	}

	output, _ := response["output"].([]any)
	for _, itemAny := range output {
		if item, ok := itemAny.(map[string]any); ok {
			rewritePassThroughOutputItem(item, r.names)
		}
	}

	rewritten, err := marshalEvent(event, payload)
	if err != nil {
		return nil, err
	}

	return []*httpclient.StreamEvent{rewritten}, nil
}

// completeCustomCall emits the custom_tool_call_input.done and output_item.done
// events that replace the upstream function_call completion.
func (r *customToolStreamRewriter) completeCustomCall(
	event *httpclient.StreamEvent,
	donePayload map[string]any,
	state *customToolCallState,
	fullArguments string,
) ([]*httpclient.StreamEvent, error) {
	input := freeformInput(fullArguments)

	inputDone := mapsClone(donePayload)
	inputDone["type"] = "response.custom_tool_call_input.done"
	inputDone["item_id"] = state.itemID
	inputDone["input"] = input
	delete(inputDone, "arguments")
	delete(inputDone, "delta")
	delete(inputDone, "name")
	delete(inputDone, "call_id")

	itemDone := mapsClone(donePayload)
	itemDone["type"] = "response.output_item.done"
	delete(itemDone, "arguments")
	delete(itemDone, "delta")
	delete(itemDone, "name")
	delete(itemDone, "call_id")
	itemDone["item"] = map[string]any{
		"id":      state.itemID,
		"type":    "custom_tool_call",
		"status":  "completed",
		"call_id": state.callID,
		"name":    state.name,
		"input":   input,
	}

	inputDoneEvent, err := marshalEvent(event, inputDone)
	if err != nil {
		return nil, err
	}
	itemDoneEvent, err := marshalEvent(event, itemDone)
	if err != nil {
		return nil, err
	}

	return []*httpclient.StreamEvent{inputDoneEvent, itemDoneEvent}, nil
}

func mapsClone(src map[string]any) map[string]any {
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}

	return dst
}

func marshalEvent(event *httpclient.StreamEvent, payload map[string]any) (*httpclient.StreamEvent, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal rewritten stream event: %w", err)
	}

	return &httpclient.StreamEvent{Type: event.Type, Data: data}, nil
}
